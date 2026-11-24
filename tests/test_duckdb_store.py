"""Unit tests for DuckDBStore concurrency, retry on lock conflicts, and read-only connections."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
from apostador.duckdb_store import DuckDBStore, get_store


def test_duckdb_store_read_only_concurrent(tmp_path: Path):
    """Test that multiple read_only connections can access DuckDB concurrently."""
    db_file = tmp_path / "test_concurrent.duckdb"
    
    # Initialize DB with write mode first
    store_init = DuckDBStore(path=db_file, read_only=False)
    store_init.put("raw", "test_key", {"msg": "hello"})
    
    # Now open read_only connections simultaneously
    store_ro1 = DuckDBStore(path=db_file, read_only=True)
    store_ro2 = DuckDBStore(path=db_file, read_only=True)
    
    val1 = store_ro1.get("raw", "test_key")
    val2 = store_ro2.get("raw", "test_key")
    assert val1 == {"msg": "hello"}
    assert val2 == {"msg": "hello"}


def test_duckdb_store_retry_on_lock(tmp_path: Path):
    """Test that DuckDBStore._connect automatically retries when a conflicting lock is held."""
    db_file = tmp_path / "test_lock.duckdb"
    store_init = DuckDBStore(path=db_file, read_only=False)
    store_init.put("raw", "test_key", {"data": 42})
    
    # Spawn a subprocess that holds an exclusive lock for 1.2 seconds
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import duckdb, time
c = duckdb.connect(r"{db_file}", read_only=False)
print("LOCKED", flush=True)
time.sleep(1.2)
c.close()
print("UNLOCKED", flush=True)
""",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    
    # Wait until subprocess confirms it acquired the lock
    line = proc.stdout.readline()
    assert "LOCKED" in line
    
    # Connect with retry using DuckDBStore
    t0 = time.time()
    store = DuckDBStore(path=db_file, read_only=True)
    with store.connect(read_only=True, max_retries=20, retry_delay=0.15) as con:
        row = con.execute("SELECT payload FROM raw.artifacts WHERE artifact_key = 'test_key'").fetchone()
    elapsed = time.time() - t0
    
    proc.wait()
    assert row is not None
    assert elapsed >= 1.0  # Proves it waited and retried until the lock was released


def test_duckdb_store_read_methods_use_read_only(tmp_path: Path):
    """Test that get, list, and query methods work in read-only store instances."""
    db_file = tmp_path / "test_methods.duckdb"
    store_writer = DuckDBStore(path=db_file, read_only=False)
    store_writer.put("raw", "item_1", {"v": 1})
    store_writer.put("raw", "item_2", {"v": 2})
    
    store_reader = DuckDBStore(path=db_file, read_only=True)
    assert store_reader.get("raw", "item_1") == {"v": 1}
    items = store_reader.list("raw", "item_")
    assert len(items) == 2


def test_load_duckdb_kpis_fallback(monkeypatch):
    """Test that load_duckdb_kpis returns fallback dict when DuckDB raises a lock error."""
    from apps.streamlit_app import load_duckdb_kpis

    def mock_get_db_con(*args, **kwargs):
        raise RuntimeError("Conflicting lock is held in another process")

    monkeypatch.setattr("apps.streamlit_app.get_db_con", mock_get_db_con)
    kpis = load_duckdb_kpis()
    assert isinstance(kpis, dict)
    assert kpis["total_bets"] == 0
    assert kpis["locked"] is True


def test_save_result_normalizes_string_fixture_id(tmp_path: Path):
    """Test that save_result sanitizes string fixture_ids such as 'espn_732194' into BIGINT."""
    db_file = tmp_path / "test_save_result.duckdb"
    store = DuckDBStore(path=db_file, read_only=False)
    store.save_result(
        run_id="daily_portfolio:2026-10-25:test",
        row_number=1,
        fixture_id="espn_732194",
        status="FT",
        home_goals=2,
        away_goals=1,
        bet_won=True,
        source="espn_open",
    )
    with store.connect(read_only=True) as con:
        row = con.execute(
            "SELECT fixture_id, status, home_goals, away_goals, bet_won, result_outcome, source FROM output.results WHERE run_id = ?",
            ["daily_portfolio:2026-10-25:test"],
        ).fetchone()
    assert row == (732194, "FT", 2, 1, True, "local", "espn_open")


def test_get_store_singleton_allows_writes():
    """Ensure get_store() ensures a read-write singleton even if called with read_only=True."""
    from apostador.duckdb_store import reset_store
    reset_store()
    store1 = get_store(read_only=True)
    assert store1.read_only is False
    store2 = get_store()
    assert store2.read_only is False


def test_recalculate_saved_stakes_syncs_recommendations(tmp_path: Path):
    """Ensure recalculate_saved_stakes adds/removes rows in output.recommendations when min_ev changes."""
    db_file = tmp_path / "test_recalc.duckdb"
    store = DuckDBStore(path=db_file, read_only=False)
    value_bets = [
        {"date": "2026-11-07", "home_team": "Team A", "away_team": "Team B", "market_outcome": "Home Win", "probability": 0.55, "odds": 2.00, "ev": 0.10, "market_implied_prob": 0.48},
        {"date": "2026-11-07", "home_team": "Team C", "away_team": "Team D", "market_outcome": "Home Win", "probability": 0.52, "odds": 2.00, "ev": 0.04, "market_implied_prob": 0.48},
        {"date": "2026-11-07", "home_team": "Team E", "away_team": "Team F", "market_outcome": "Home Win", "probability": 0.85, "odds": 1.20, "ev": 0.02, "market_implied_prob": 0.82},
    ]
    initial_portfolio = [value_bets[0], value_bets[1]]
    payload = {
        "date": "2026-11-07",
        "bankroll": 100000.0,
        "kelly_factor": 0.25,
        "value_bets": value_bets,
        "portfolio": initial_portfolio,
        "processed_matches": [],
    }
    store.put("processed", "portfolio_bets_2026-11-07.json", payload)
    store.save_run("daily_portfolio:2026-11-07:test", "daily_portfolio", "2026-11-07", payload)

    # Initially 2 rows in output.recommendations
    with store.connect(read_only=True) as con:
        count_initial = con.execute("SELECT count(*) FROM output.recommendations WHERE event_date='2026-11-07'").fetchone()[0]
    assert count_initial == 2

    # Recalculate with min_ev=0.015 under multi_market -> all 3 value_bets qualify!
    summary = store.recalculate_saved_stakes(
        "2026-11-07", "2026-11-07",
        bankroll=100000.0,
        fraction_factor=0.25,
        target_dates=["2026-11-07"],
        stake_mode="flat",
        flat_stake_pct=0.03,
        max_stake_pct=0.03,
        strategy_mode="multi_market",
        min_ev=0.015,
    )
    assert summary["recommendations_updated"] == 3
    with store.connect(read_only=True) as con:
        count_after = con.execute("SELECT count(*) FROM output.recommendations WHERE event_date='2026-11-07'").fetchone()[0]
    assert count_after == 3




