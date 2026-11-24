"""DuckDB persistence for raw inputs, processed artifacts, and model outputs."""
from __future__ import annotations

import json
import hashlib
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from .config import DATA_DIR, DUCKDB_PATH, ROOT_DIR
from .outcome_utils import compute_bet_won, normalize_selection_to_side

logger = logging.getLogger(__name__)

_CANONICAL_OUTCOME_LABEL = {"local": "Home Win", "empate": "Draw", "visita": "Away Win"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class DuckDBStore:
    """Small local store. Legacy files are imported once and remain as backup."""

    def __init__(self, path: Path | str = DUCKDB_PATH, read_only: bool = False):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB no está instalado. Ejecuta: pip install -r requirements.txt") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._duckdb = duckdb
        self.read_only = read_only
        if not read_only:
            self._initialize()

    def _connect(self, read_only: bool | None = None, max_retries: int = 25, retry_delay: float = 0.2):
        if read_only is None:
            read_only = self.read_only
        last_exc = None
        for attempt in range(max_retries):
            try:
                return self._duckdb.connect(str(self.path), read_only=read_only)
            except Exception as exc:
                msg = str(exc).lower()
                if ("lock" in msg or "conflicting lock" in msg or "could not set lock" in msg
                        or "different configuration" in msg):
                    last_exc = exc
                    sleep_time = retry_delay * (1.15 ** min(attempt, 8))
                    logger.debug("DuckDB bloqueado temporalmente. Reintento %d/%d en %.2fs...", attempt + 1, max_retries, sleep_time)
                    time.sleep(sleep_time)
                else:
                    raise
        raise last_exc

    def connect(self, read_only: bool | None = None, max_retries: int = 25, retry_delay: float = 0.2):
        """Retorna una nueva conexión con la configuración homogénea del store."""
        return self._connect(read_only=read_only, max_retries=max_retries, retry_delay=retry_delay)

    def _initialize(self):
        with self._connect(read_only=False) as con:
            for schema in ("raw", "processed", "output"):
                con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                con.execute(f"""CREATE TABLE IF NOT EXISTS {schema}.artifacts (
                    artifact_key VARCHAR PRIMARY KEY,
                    payload JSON NOT NULL,
                    content_type VARCHAR NOT NULL DEFAULT 'application/json',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
                )""")
            con.execute("""CREATE TABLE IF NOT EXISTS output.runs (
                run_id VARCHAR PRIMARY KEY, run_type VARCHAR NOT NULL,
                run_date VARCHAR, started_at TIMESTAMPTZ NOT NULL,
                summary JSON NOT NULL
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS output.recommendations (
                run_id VARCHAR NOT NULL, row_number INTEGER NOT NULL,
                event_date VARCHAR, home_team VARCHAR, away_team VARCHAR,
                outcome VARCHAR, probability DOUBLE, odds DOUBLE, ev DOUBLE,
                stake DOUBLE, payload JSON NOT NULL,
                PRIMARY KEY (run_id, row_number)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS output.results (
                run_id VARCHAR NOT NULL, row_number INTEGER NOT NULL,
                fixture_id BIGINT, status VARCHAR,
                home_goals INTEGER, away_goals INTEGER, bet_won BOOLEAN,
                result_outcome VARCHAR, source VARCHAR NOT NULL DEFAULT 'api',
                checked_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (run_id, row_number)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS output.impect_results (
                iteration_id INTEGER NOT NULL, match_id BIGINT NOT NULL,
                scheduled_at TIMESTAMPTZ, match_day_index INTEGER,
                home_squad_id BIGINT, home_team VARCHAR NOT NULL,
                away_squad_id BIGINT, away_team VARCHAR NOT NULL,
                home_goals INTEGER NOT NULL, away_goals INTEGER NOT NULL,
                result_outcome VARCHAR NOT NULL, result_type VARCHAR,
                fetched_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                payload JSON NOT NULL,
                PRIMARY KEY (iteration_id, match_id)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS output.team_crosswalk (
                provider VARCHAR NOT NULL, provider_team_id BIGINT NOT NULL,
                iteration_id INTEGER NOT NULL, provider_team_name VARCHAR,
                impect_squad_id BIGINT, impect_squad_name VARCHAR NOT NULL,
                match_score DOUBLE, mapping_source VARCHAR NOT NULL DEFAULT 'aligner',
                wyscout_id BIGINT,
                statsbomb_id BIGINT,
                sportradar_id VARCHAR,
                understat_id VARCHAR,
                skillcorner_id BIGINT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (provider, provider_team_id, iteration_id)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS raw.match_history (
                league VARCHAR NOT NULL,
                season VARCHAR NOT NULL,
                game_id VARCHAR NOT NULL,
                match_date TIMESTAMPTZ,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL,
                fthg INTEGER,
                ftag INTEGER,
                ftr VARCHAR,
                b365_home DOUBLE,
                b365_draw DOUBLE,
                b365_away DOUBLE,
                pinnacle_home DOUBLE,
                pinnacle_draw DOUBLE,
                pinnacle_away DOUBLE,
                avg_home DOUBLE,
                avg_draw DOUBLE,
                avg_away DOUBLE,
                max_home DOUBLE,
                max_draw DOUBLE,
                max_away DOUBLE,
                b365_close_home DOUBLE,
                b365_close_draw DOUBLE,
                b365_close_away DOUBLE,
                pinnacle_close_home DOUBLE,
                pinnacle_close_draw DOUBLE,
                pinnacle_close_away DOUBLE,
                avg_close_home DOUBLE,
                avg_close_draw DOUBLE,
                avg_close_away DOUBLE,
                max_close_home DOUBLE,
                max_close_draw DOUBLE,
                max_close_away DOUBLE,
                pinnacle_is_stale BOOLEAN NOT NULL DEFAULT FALSE,
                data_source VARCHAR NOT NULL DEFAULT 'football_data_co_uk',
                usage_scope VARCHAR NOT NULL DEFAULT 'personal_research_only',
                raw_payload JSON,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (league, season, game_id)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS raw.clubelo_ratings (
                rating_date DATE NOT NULL,
                team_name VARCHAR NOT NULL,
                country VARCHAR,
                level INTEGER,
                elo DOUBLE NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (rating_date, team_name)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS raw.player_valuations (
                provider VARCHAR NOT NULL,
                team_name VARCHAR NOT NULL,
                player_name VARCHAR NOT NULL,
                as_of_date DATE NOT NULL,
                market_value_eur DOUBLE,
                sofifa_overall INTEGER,
                minutes_played DOUBLE,
                position VARCHAR,
                raw_payload JSON,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (provider, team_name, player_name, as_of_date)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS raw.odds_snapshots (
                sport VARCHAR NOT NULL,
                league VARCHAR,
                fixture_id VARCHAR NOT NULL,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL,
                event_time TIMESTAMPTZ,
                odds_timestamp TIMESTAMPTZ,
                received_at TIMESTAMPTZ NOT NULL,
                provider VARCHAR NOT NULL,
                bookmaker VARCHAR NOT NULL,
                market VARCHAR NOT NULL DEFAULT 'h2h',
                line DOUBLE NOT NULL DEFAULT 0.0,
                selection VARCHAR NOT NULL,
                price DOUBLE NOT NULL,
                is_suspended BOOLEAN NOT NULL DEFAULT FALSE,
                is_closing BOOLEAN NOT NULL DEFAULT FALSE,
                raw_payload JSON,
                PRIMARY KEY (sport, fixture_id, provider, bookmaker, market, line, selection, received_at)
            )""")
            # Migrate databases created before manual result entry was added.
            con.execute("ALTER TABLE output.results ADD COLUMN IF NOT EXISTS result_outcome VARCHAR")
            con.execute("ALTER TABLE output.results ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'api'")
            # Migrate raw.match_history to closing odds (*C), Pinnacle staleness and license metadata.
            for col_sql in (
                "b365_close_home DOUBLE",
                "b365_close_draw DOUBLE",
                "b365_close_away DOUBLE",
                "pinnacle_close_home DOUBLE",
                "pinnacle_close_draw DOUBLE",
                "pinnacle_close_away DOUBLE",
                "avg_close_home DOUBLE",
                "avg_close_draw DOUBLE",
                "avg_close_away DOUBLE",
                "max_close_home DOUBLE",
                "max_close_draw DOUBLE",
                "max_close_away DOUBLE",
                "pinnacle_is_stale BOOLEAN DEFAULT FALSE",
                "data_source VARCHAR DEFAULT 'football_data_co_uk'",
                "usage_scope VARCHAR DEFAULT 'personal_research_only'",
            ):
                con.execute(f"ALTER TABLE raw.match_history ADD COLUMN IF NOT EXISTS {col_sql}")
            # Migrate output.team_crosswalk to support canonical B2B external IDs (TASK-14).
            for col_sql in (
                "wyscout_id BIGINT",
                "statsbomb_id BIGINT",
                "sportradar_id VARCHAR",
                "understat_id VARCHAR",
                "skillcorner_id BIGINT",
            ):
                con.execute(f"ALTER TABLE output.team_crosswalk ADD COLUMN IF NOT EXISTS {col_sql}")
        self._import_legacy_files()

    def _import_legacy_files(self):
        """One-time non-destructive import of existing raw/processed/output files."""
        roots = (("raw", DATA_DIR / "raw"), ("processed", DATA_DIR / "processed"),
                 ("output", ROOT_DIR / "outputs"))
        legacy_runs = []
        with self._connect(read_only=False) as con:
            con.execute("CREATE TABLE IF NOT EXISTS processed.store_meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
            imported = con.execute("SELECT 1 FROM processed.store_meta WHERE key = 'legacy_files_imported'").fetchone()
            if imported:
                return
            for layer, root in roots:
                if not root.exists():
                    continue
                for file in root.rglob("*"):
                    if not file.is_file() or file.suffix.lower() not in {".json", ".csv", ".md"}:
                        continue
                    key = file.relative_to(root).as_posix()
                    exists = con.execute(f"SELECT 1 FROM {layer}.artifacts WHERE artifact_key=?", [key]).fetchone()
                    if exists:
                        continue
                    try:
                        if file.suffix.lower() == ".json":
                            payload, kind = json.loads(file.read_text(encoding="utf-8")), "application/json"
                        elif file.suffix.lower() == ".csv":
                            import pandas as pd
                            payload, kind = pd.read_csv(file).to_dict(orient="records"), "text/csv"
                        else:
                            payload, kind = {"text": file.read_text(encoding="utf-8")}, "text/markdown"
                        con.execute(f"INSERT INTO {layer}.artifacts (artifact_key,payload,content_type) VALUES (?,?::JSON,?)",
                                    [key, _json(payload), kind])
                        if layer == "processed" and key.startswith(("portfolio_bets_", "black_swan_")) and isinstance(payload, dict):
                            legacy_runs.append((key, payload))
                    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                        continue
            con.execute("INSERT OR REPLACE INTO processed.store_meta VALUES ('legacy_files_imported', current_timestamp::VARCHAR)")
        for key, payload in legacy_runs:
            stem = Path(key).stem
            run_type = "black_swan" if stem.startswith("black_swan") else ("max_kelly" if "max_kelly" in stem else "daily_portfolio")
            run_date = next((part for part in stem.split("_") if len(part) == 10 and part[4] == "-"), None)
            self.save_run(key, run_type, run_date, payload)

    def put(self, layer: str, key: str, payload: Any, content_type: str = "application/json"):
        if layer not in {"raw", "processed", "output"}:
            raise ValueError(f"Capa inválida: {layer}")
        with self._connect(read_only=False) as con:
            con.execute(f"""INSERT INTO {layer}.artifacts(artifact_key,payload,content_type,updated_at)
                VALUES (?,?::JSON,?,current_timestamp)
                ON CONFLICT(artifact_key) DO UPDATE SET payload=excluded.payload,
                content_type=excluded.content_type, updated_at=now()""",
                [str(key), _json(payload), content_type])

    def get(self, layer: str, key: str, default=None):
        if layer not in {"raw", "processed", "output"}:
            raise ValueError(f"Capa inválida: {layer}")
        with self._connect() as con:
            row = con.execute(f"SELECT payload FROM {layer}.artifacts WHERE artifact_key=?", [str(key)]).fetchone()
        return json.loads(row[0]) if row else default

    def list(self, layer: str, prefix: str = ""):
        if layer not in {"raw", "processed", "output"}:
            raise ValueError(f"Capa inválida: {layer}")
        with self._connect() as con:
            rows = con.execute(f"SELECT artifact_key,payload FROM {layer}.artifacts WHERE starts_with(artifact_key, ?) ORDER BY artifact_key",
                               [prefix]).fetchall()
        return [(key, json.loads(payload)) for key, payload in rows]

    def save_run(self, run_id: str, run_type: str, run_date: str | None, payload: dict):
        started = datetime.now(timezone.utc)
        list_keys = ("recommendations", "portfolio", "legs", "bets", "value_bets")
        row_key = next((key for key in list_keys if isinstance(payload.get(key), list) and payload[key]), None)
        if row_key is None:
            row_key = next((key for key in list_keys if isinstance(payload.get(key), list)), None)
        if row_key is None and isinstance(payload.get("parlay"), dict):
            row_key = "parlay.legs"
            rows = payload["parlay"].get("legs") or []
        else:
            rows = payload.get(row_key, []) if row_key else []
        summary = {key: value for key, value in payload.items()
                   if key not in {"recommendations", "portfolio", "legs", "bets", "value_bets"}}
        if isinstance(summary.get("parlay"), dict):
            summary["parlay"] = {key: value for key, value in summary["parlay"].items() if key != "legs"}
        with self._connect(read_only=False) as con:
            prior_results = []
            if run_type == "daily_portfolio":
                prior_results = con.execute("""
                    SELECT r.event_date, r.home_team, r.away_team, res.fixture_id,
                           res.status, res.home_goals, res.away_goals,
                           res.result_outcome, res.source, res.checked_at
                    FROM output.results res
                    JOIN output.recommendations r
                      ON r.run_id=res.run_id AND r.row_number=res.row_number
                    WHERE r.run_id=?
                """, [run_id]).fetchall()
            con.execute("INSERT OR REPLACE INTO output.runs VALUES (?,?,?,?,?::JSON)",
                        [run_id, run_type, run_date, started, _json(summary)])
            con.execute("DELETE FROM output.recommendations WHERE run_id=?", [run_id])
            if run_type == "daily_portfolio":
                con.execute("DELETE FROM output.results WHERE run_id=?", [run_id])
            result_by_match = {}
            for old in prior_results:
                key = (old[0], (old[1] or "").strip().casefold(), (old[2] or "").strip().casefold())
                result_by_match.setdefault(key, old[3:])
            new_rows_by_match = {}
            for idx, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    continue
                match = row.get("match", "")
                home, away = row.get("home_team"), row.get("away_team")
                if (not home or not away) and " vs " in match:
                    home, away = match.split(" vs ", 1)
                raw_outcome = row.get("market_outcome") or row.get("outcome") or row.get("selection")
                side = normalize_selection_to_side(raw_outcome, home, away)
                outcome = _CANONICAL_OUTCOME_LABEL.get(side, raw_outcome) if side else raw_outcome
                con.execute("""INSERT INTO output.recommendations VALUES (?,?,?,?,?,?,?,?,?,?,?::JSON)""",
                    [run_id, idx, row.get("date") or run_date, home, away,
                     outcome,
                     _number(row.get("probability") or row.get("p_adj_leg") or row.get("model_probability")),
                     _number(row.get("odds") or row.get("market_odds")),
                     _number(row.get("ev") or row.get("ev_adj") or row.get("expected_value")),
                     _number(row.get("stake") or row.get("stake_amount")), _json(row)])
                if run_type == "daily_portfolio":
                    match_key = (row.get("date") or run_date,
                                 (home or "").strip().casefold(), (away or "").strip().casefold())
                    new_rows_by_match.setdefault(match_key, []).append((idx, outcome, home, away))
            for match_key, new_rows in new_rows_by_match.items():
                old_result = result_by_match.get(match_key)
                if old_result is None:
                    continue
                fixture_id, status, home_goals, away_goals, result_outcome, source, checked_at = old_result
                for idx, outcome, home, away in new_rows:
                    bet_won = compute_bet_won(outcome, result_outcome, home, away)
                    con.execute("""INSERT INTO output.results
                        (run_id,row_number,fixture_id,status,home_goals,away_goals,
                         bet_won,result_outcome,source,checked_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        [run_id, idx, fixture_id, status, home_goals, away_goals,
                         bet_won, result_outcome, source or "api", checked_at])

    def recalculate_saved_stakes(self, start_date: str, end_date: str,
                                 bankroll: float, fraction_factor: float,
                                 target_dates: list[str] | None = None,
                                 stake_mode: str = "flat",
                                 flat_stake_pct: float = 0.03,
                                 max_stake_pct: float = 0.03,
                                 strategy_mode: str = "hybrid",
                                 min_ev: float = 0.03,
                                 min_stake_amount: float = 1.0) -> dict:
        """Update stake amounts on each date's latest saved run without rerunning predictions."""
        if bankroll <= 0 or fraction_factor < 0:
            raise ValueError("El bankroll debe ser positivo y Kelly no puede ser negativo.")
        with self._connect(read_only=False) as con:
            date_filter = "run_date BETWEEN ? AND ?"
            date_params = [str(start_date), str(end_date)]
            if target_dates:
                unique_dates = sorted({str(value) for value in target_dates})
                date_filter = f"run_date IN ({','.join('?' for _ in unique_dates)})"
                date_params = unique_dates
            latest_runs = con.execute(f"""
                SELECT run_id, run_date, summary
                FROM output.runs
                WHERE run_type='daily_portfolio' AND {date_filter}
                QUALIFY row_number() OVER (PARTITION BY run_date ORDER BY started_at DESC, run_id DESC)=1
                ORDER BY run_date
            """, date_params).fetchall()

            updated_dates = []
            updated_rows = 0
            try:
                import importlib
                from . import optimizer as _opt_mod
                importlib.reload(_opt_mod)
                KellyOptimizer = _opt_mod.KellyOptimizer
            except Exception:
                from .optimizer import KellyOptimizer

            for run_id, run_date, summary_raw in latest_runs:
                summary = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
                artifact_key = f"portfolio_bets_{run_date}.json"
                artifact_row = con.execute(
                    "SELECT payload FROM processed.artifacts WHERE artifact_key=?", [artifact_key]
                ).fetchone()
                if not artifact_row:
                    continue
                artifact = json.loads(artifact_row[0]) if isinstance(artifact_row[0], str) else artifact_row[0]
                value_bets = artifact.get("value_bets") or []
                try:
                    optimizer = KellyOptimizer(
                        bankroll=bankroll,
                        fraction_factor=fraction_factor,
                        strategy_mode=strategy_mode,
                        stake_mode=stake_mode,
                        flat_stake_pct=flat_stake_pct,
                        max_stake_pct=max_stake_pct,
                        min_ev=min_ev,
                        min_stake_amount=min_stake_amount,
                    )
                except TypeError as exc:
                    if "min_stake_amount" in str(exc):
                        optimizer = KellyOptimizer(
                            bankroll=bankroll,
                            fraction_factor=fraction_factor,
                            strategy_mode=strategy_mode,
                            stake_mode=stake_mode,
                            flat_stake_pct=flat_stake_pct,
                            max_stake_pct=max_stake_pct,
                            min_ev=min_ev,
                        )
                        setattr(optimizer, "min_stake_amount", float(min_stake_amount))
                    else:
                        raise
                new_portfolio = optimizer.optimize_portfolio(value_bets)

                prior_results = con.execute("""
                    SELECT r.event_date, r.home_team, r.away_team, res.fixture_id,
                           res.status, res.home_goals, res.away_goals,
                           res.result_outcome, res.source, res.checked_at
                    FROM output.results res
                    JOIN output.recommendations r
                      ON r.run_id=res.run_id AND r.row_number=res.row_number
                    WHERE r.run_id=?
                """, [run_id]).fetchall()
                result_by_match = {}
                for old in prior_results:
                    key = (old[0], (old[1] or "").strip().casefold(), (old[2] or "").strip().casefold())
                    result_by_match.setdefault(key, old[3:])

                con.execute("DELETE FROM output.recommendations WHERE run_id=?", [run_id])
                con.execute("DELETE FROM output.results WHERE run_id=?", [run_id])

                date_updates = 0
                new_rows_by_match = {}
                for idx, bet in enumerate(new_portfolio, 1):
                    home, away = bet.get("home_team"), bet.get("away_team")
                    raw_outcome = bet.get("market_outcome") or bet.get("outcome") or bet.get("selection")
                    side = normalize_selection_to_side(raw_outcome, home, away)
                    outcome = _CANONICAL_OUTCOME_LABEL.get(side, raw_outcome) if side else raw_outcome
                    event_date_val = bet.get("date") or str(run_date)
                    con.execute(
                        "INSERT INTO output.recommendations VALUES (?,?,?,?,?,?,?,?,?,?,?::JSON)",
                        [
                            run_id, idx, event_date_val, home, away, outcome,
                            _number(bet.get("probability")),
                            _number(bet.get("odds")),
                            _number(bet.get("ev")),
                            _number(bet.get("stake")),
                            _json(bet),
                        ],
                    )
                    match_key = (
                        event_date_val,
                        (home or "").strip().casefold(),
                        (away or "").strip().casefold(),
                    )
                    new_rows_by_match.setdefault(match_key, []).append((idx, outcome, home, away))
                    date_updates += 1

                for match_key, new_rows in new_rows_by_match.items():
                    old_result = result_by_match.get(match_key)
                    if old_result is None:
                        continue
                    fixture_id, status, home_goals, away_goals, result_outcome, source, checked_at = old_result
                    for idx, outcome, home, away in new_rows:
                        bet_won = compute_bet_won(outcome, result_outcome, home, away)
                        con.execute("""INSERT INTO output.results
                            (run_id,row_number,fixture_id,status,home_goals,away_goals,
                             bet_won,result_outcome,source,checked_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            [run_id, idx, fixture_id, status, home_goals, away_goals,
                             bet_won, result_outcome, source or "api", checked_at])

                artifact.update({"bankroll": float(bankroll), "kelly_factor": float(fraction_factor),
                                 "portfolio": new_portfolio})
                con.execute("""
                    UPDATE processed.artifacts SET payload=?::JSON, updated_at=now()
                    WHERE artifact_key=?
                """, [_json(artifact), artifact_key])
                summary.update({"bankroll": float(bankroll), "kelly_factor": float(fraction_factor),
                                "stake_updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                con.execute("UPDATE output.runs SET summary=?::JSON WHERE run_id=?", [_json(summary), run_id])
                updated_rows += date_updates
                updated_dates.append(str(run_date))

        return {"dates": updated_dates, "recommendations_updated": updated_rows}

    def pending_recommendations(self):
        """Recommendations still lacking a finished match result."""
        with self._connect() as con:
            return con.execute("""
                SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team, r.outcome
                FROM output.recommendations r
                LEFT JOIN output.results res ON res.run_id = r.run_id AND res.row_number = r.row_number
                WHERE res.run_id IS NULL OR res.status NOT IN ('FT', 'AET', 'PEN', 'MANUAL')
                ORDER BY r.event_date, r.run_id, r.row_number
            """).fetchall()

    def pending_result_recommendations(self):
        """Pending recommendations with payload metadata for provider matching."""
        with self._connect() as con:
            return con.execute("""
                SELECT r.run_id,r.row_number,r.event_date,r.home_team,r.away_team,r.outcome,r.payload
                FROM output.recommendations r
                LEFT JOIN output.results res ON res.run_id=r.run_id AND res.row_number=r.row_number
                WHERE res.run_id IS NULL OR res.status NOT IN ('FT','AET','PEN','MANUAL')
                ORDER BY r.event_date,r.home_team,r.away_team
            """).fetchall()

    def save_result(self, run_id: str, row_number: int, fixture_id, status: str,
                     home_goals, away_goals, bet_won, source: str = "api"):
        if fixture_id is not None and not isinstance(fixture_id, int):
            digits = "".join(ch for ch in str(fixture_id) if ch.isdigit())
            fixture_id = int(digits) if digits else None
        result_outcome = None
        if home_goals is not None and away_goals is not None:
            result_outcome = 'local' if home_goals > away_goals else ('empate' if home_goals == away_goals else 'visita')
        with self._connect(read_only=False) as con:
            con.execute("""INSERT INTO output.results
                (run_id,row_number,fixture_id,status,home_goals,away_goals,bet_won,result_outcome,source,checked_at)
                VALUES (?,?,?,?,?,?,?,?, ?, current_timestamp)
                ON CONFLICT(run_id,row_number) DO UPDATE SET
                fixture_id=excluded.fixture_id, status=excluded.status,
                home_goals=excluded.home_goals, away_goals=excluded.away_goals,
                bet_won=excluded.bet_won, result_outcome=excluded.result_outcome,
                source=excluded.source, checked_at=now()""",
                [run_id, row_number, fixture_id, status, home_goals, away_goals, bet_won, result_outcome, source])

    def save_impect_results(self, records: list[dict]):
        """Upsert finished match scores returned by Impect, preserving the API payload."""
        if not records:
            return
        with self._connect(read_only=False) as con:
            con.executemany("""INSERT INTO output.impect_results
                (iteration_id,match_id,scheduled_at,match_day_index,home_squad_id,home_team,
                 away_squad_id,away_team,home_goals,away_goals,result_outcome,result_type,payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?::JSON)
                ON CONFLICT(iteration_id,match_id) DO UPDATE SET
                  scheduled_at=excluded.scheduled_at,match_day_index=excluded.match_day_index,
                  home_squad_id=excluded.home_squad_id,home_team=excluded.home_team,
                  away_squad_id=excluded.away_squad_id,away_team=excluded.away_team,
                  home_goals=excluded.home_goals,away_goals=excluded.away_goals,
                  result_outcome=excluded.result_outcome,result_type=excluded.result_type,
                  fetched_at=now(),payload=excluded.payload""",
                [[r["iteration_id"],r["match_id"],r.get("scheduled_at"),r.get("match_day_index"),
                  r.get("home_squad_id"),r["home_team"],r.get("away_squad_id"),r["away_team"],
                  r["home_goals"],r["away_goals"],r["result_outcome"],r.get("result_type"),_json(r)]
                 for r in records])

    def list_impect_results(self):
        with self._connect() as con:
            return con.execute("""SELECT iteration_id,match_id,scheduled_at,match_day_index,
                home_squad_id,home_team,away_squad_id,away_team,home_goals,away_goals,
                result_outcome,result_type FROM output.impect_results""").fetchall()

    @staticmethod
    def _infer_pinnacle_is_stale(record: dict) -> bool:
        if record.get("pinnacle_is_stale") is not None:
            return bool(record["pinnacle_is_stale"])
        match_day = str(record.get("match_date") or "")[:10]
        return len(match_day) == 10 and "2025-07-23" <= match_day < "9000-01-01"

    def save_match_history(self, records: list[dict]):
        """Upsert historical match games and odds from soccerdata / football-data.co.uk."""
        if not records:
            return
        with self._connect(read_only=False) as con:
            con.executemany("""INSERT INTO raw.match_history
                (league, season, game_id, match_date, home_team, away_team,
                 fthg, ftag, ftr,
                 b365_home, b365_draw, b365_away,
                 pinnacle_home, pinnacle_draw, pinnacle_away,
                 avg_home, avg_draw, avg_away,
                 max_home, max_draw, max_away,
                 b365_close_home, b365_close_draw, b365_close_away,
                 pinnacle_close_home, pinnacle_close_draw, pinnacle_close_away,
                 avg_close_home, avg_close_draw, avg_close_away,
                 max_close_home, max_close_draw, max_close_away,
                 pinnacle_is_stale, data_source, usage_scope,
                 raw_payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?::JSON)
                ON CONFLICT(league, season, game_id) DO UPDATE SET
                  match_date=excluded.match_date, home_team=excluded.home_team, away_team=excluded.away_team,
                  fthg=excluded.fthg, ftag=excluded.ftag, ftr=excluded.ftr,
                  b365_home=excluded.b365_home, b365_draw=excluded.b365_draw, b365_away=excluded.b365_away,
                  pinnacle_home=excluded.pinnacle_home, pinnacle_draw=excluded.pinnacle_draw, pinnacle_away=excluded.pinnacle_away,
                  avg_home=excluded.avg_home, avg_draw=excluded.avg_draw, avg_away=excluded.avg_away,
                  max_home=excluded.max_home, max_draw=excluded.max_draw, max_away=excluded.max_away,
                  b365_close_home=excluded.b365_close_home, b365_close_draw=excluded.b365_close_draw, b365_close_away=excluded.b365_close_away,
                  pinnacle_close_home=excluded.pinnacle_close_home, pinnacle_close_draw=excluded.pinnacle_close_draw, pinnacle_close_away=excluded.pinnacle_close_away,
                  avg_close_home=excluded.avg_close_home, avg_close_draw=excluded.avg_close_draw, avg_close_away=excluded.avg_close_away,
                  max_close_home=excluded.max_close_home, max_close_draw=excluded.max_close_draw, max_close_away=excluded.max_close_away,
                  pinnacle_is_stale=excluded.pinnacle_is_stale, data_source=excluded.data_source, usage_scope=excluded.usage_scope,
                  raw_payload=excluded.raw_payload, updated_at=now()""",
                [[
                    r["league"], r["season"], r["game_id"], r.get("match_date"),
                    r["home_team"], r["away_team"], r.get("fthg"), r.get("ftag"), r.get("ftr"),
                    r.get("b365_home"), r.get("b365_draw"), r.get("b365_away"),
                    r.get("pinnacle_home"), r.get("pinnacle_draw"), r.get("pinnacle_away"),
                    r.get("avg_home"), r.get("avg_draw"), r.get("avg_away"),
                    r.get("max_home"), r.get("max_draw"), r.get("max_away"),
                    r.get("b365_close_home"), r.get("b365_close_draw"), r.get("b365_close_away"),
                    r.get("pinnacle_close_home"), r.get("pinnacle_close_draw"), r.get("pinnacle_close_away"),
                    r.get("avg_close_home"), r.get("avg_close_draw"), r.get("avg_close_away"),
                    r.get("max_close_home"), r.get("max_close_draw"), r.get("max_close_away"),
                    self._infer_pinnacle_is_stale(r),
                    r.get("data_source") or "football_data_co_uk",
                    r.get("usage_scope") or "personal_research_only",
                    _json(r)
                ] for r in records])

    def list_match_history(self, league: str | None = None, season: str | None = None) -> list[dict]:
        query = "SELECT * FROM raw.match_history WHERE 1=1"
        params = []
        if league:
            query += " AND league = ?"
            params.append(league)
        if season:
            query += " AND season = ?"
            params.append(season)
        query += " ORDER BY match_date DESC"
        with self._connect() as con:
            return con.execute(query, params).fetch_df().to_dict(orient="records")

    def save_clubelo_ratings(self, records: list[dict]):
        """Upsert ClubElo ratings into raw.clubelo_ratings."""
        if not records:
            return
        with self._connect(read_only=False) as con:
            con.executemany("""INSERT INTO raw.clubelo_ratings
                (rating_date, team_name, country, level, elo)
                VALUES (?::DATE, ?, ?, ?, ?)
                ON CONFLICT(rating_date, team_name) DO UPDATE SET
                  country=excluded.country, level=excluded.level, elo=excluded.elo, updated_at=now()""",
                [[
                    r["rating_date"], r["team_name"], r.get("country"),
                    r.get("level"), float(r["elo"])
                ] for r in records])

    def get_clubelo_rating(self, team_name: str, rating_date: str) -> float | None:
        """Get point-in-time safe Elo rating for a team on or before a given date."""
        query = """
            SELECT elo FROM raw.clubelo_ratings
            WHERE team_name = ? AND rating_date <= ?::DATE
            ORDER BY rating_date DESC
            LIMIT 1
        """
        with self._connect() as con:
            row = con.execute(query, [team_name, rating_date]).fetchone()
            if row:
                return float(row[0])
        return None

    def save_player_valuations(self, records: list[dict]):
        """Upsert player market valuations / ratings into raw.player_valuations."""
        if not records:
            return
        with self._connect(read_only=False) as con:
            con.executemany("""INSERT INTO raw.player_valuations
                (provider, team_name, player_name, as_of_date, market_value_eur,
                 sofifa_overall, minutes_played, position, raw_payload)
                VALUES (?, ?, ?, ?::DATE, ?, ?, ?, ?, ?::JSON)
                ON CONFLICT(provider, team_name, player_name, as_of_date) DO UPDATE SET
                  market_value_eur=excluded.market_value_eur,
                  sofifa_overall=excluded.sofifa_overall,
                  minutes_played=excluded.minutes_played,
                  position=excluded.position,
                  raw_payload=excluded.raw_payload,
                  updated_at=now()""",
                [[
                    str(r.get("provider") or "transfermarkt"),
                    str(r["team_name"]),
                    str(r["player_name"]),
                    str(r["as_of_date"])[:10],
                    float(r["market_value_eur"]) if r.get("market_value_eur") is not None else None,
                    int(r["sofifa_overall"]) if r.get("sofifa_overall") is not None else None,
                    float(r["minutes_played"]) if r.get("minutes_played") is not None else None,
                    r.get("position"),
                    _json(r),
                ] for r in records])

    def list_valuation_teams(self) -> list[str]:
        """List distinct team names stored in raw.player_valuations."""
        with self._connect() as con:
            rows = con.execute("SELECT DISTINCT team_name FROM raw.player_valuations ORDER BY team_name").fetchall()
            return [str(r[0]) for r in rows]

    def get_team_player_valuations(
        self,
        team_name: str,
        as_of_date: str | None = None,
        provider: str | None = None,
    ) -> list[dict]:
        """Get the most recent point-in-time squad valuations for a team."""
        date_filter = "AND as_of_date <= ?::DATE" if as_of_date else ""
        prov_filter = "AND provider = ?" if provider else ""
        params: list[Any] = [team_name]
        if as_of_date:
            params.append(str(as_of_date)[:10])
        if provider:
            params.append(provider)

        with self._connect() as con:
            latest_row = con.execute(
                f"""SELECT provider, MAX(as_of_date) AS snap_date
                    FROM raw.player_valuations
                    WHERE team_name = ? {date_filter} {prov_filter}
                    GROUP BY provider
                    ORDER BY snap_date DESC
                    LIMIT 1""",
                params,
            ).fetchone()
            if not latest_row:
                return []
            chosen_provider, snap_date = latest_row[0], str(latest_row[1])[:10]
            rows = con.execute(
                """SELECT provider, team_name, player_name, as_of_date::VARCHAR AS as_of_date,
                          market_value_eur, sofifa_overall, minutes_played, position
                   FROM raw.player_valuations
                   WHERE provider = ? AND team_name = ? AND as_of_date = ?::DATE""",
                [chosen_provider, team_name, snap_date],
            ).fetch_df()
            return rows.to_dict(orient="records") if not rows.empty else []

    def save_odds_snapshots(self, records: list[dict]) -> int:
        """Persist odds snapshots with triple timestamps (event_time, odds_timestamp, received_at)."""
        if not records:
            return 0
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for r in records:
            price = r.get("price")
            if price is None or float(price) <= 1.0:
                continue
            home = str(r.get("home_team") or "").strip()
            away = str(r.get("away_team") or "").strip()
            if not home or not away:
                continue
            ev_time = r.get("event_time")
            fid = str(r.get("fixture_id") or f"{str(ev_time or '')[:10]}_{home}_{away}")
            rows.append([
                str(r.get("sport") or "soccer"),
                r.get("league"),
                fid,
                home,
                away,
                ev_time,
                r.get("odds_timestamp"),
                r.get("received_at") or now_iso,
                str(r.get("provider") or "the_odds_api"),
                str(r.get("bookmaker") or "Consensus"),
                str(r.get("market") or "h2h"),
                float(r["line"]) if r.get("line") is not None else 0.0,
                str(r["selection"]),
                float(price),
                bool(r.get("is_suspended", False)),
                bool(r.get("is_closing", False)),
                _json(r),
            ])
        if not rows:
            return 0
        with self._connect(read_only=False) as con:
            con.executemany("""INSERT INTO raw.odds_snapshots
                (sport, league, fixture_id, home_team, away_team, event_time,
                 odds_timestamp, received_at, provider, bookmaker, market, line,
                 selection, price, is_suspended, is_closing, raw_payload)
                VALUES (?, ?, ?, ?, ?, ?::TIMESTAMPTZ, ?::TIMESTAMPTZ, ?::TIMESTAMPTZ,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON)
                ON CONFLICT(sport, fixture_id, provider, bookmaker, market, line, selection, received_at)
                DO UPDATE SET
                  price=excluded.price,
                  is_suspended=excluded.is_suspended,
                  is_closing=(excluded.is_closing OR raw.odds_snapshots.is_closing),
                  odds_timestamp=excluded.odds_timestamp,
                  raw_payload=excluded.raw_payload""", rows)
        return len(rows)

    def get_odds_trajectory(
        self,
        home_team: str,
        away_team: str,
        event_date: str | None = None,
        market: str = "h2h",
        bookmaker: str | None = None,
    ) -> list[dict]:
        """Return chronological odds snapshots for a fixture."""
        query = """
            SELECT sport, league, fixture_id, home_team, away_team,
                   event_time::VARCHAR AS event_time,
                   odds_timestamp::VARCHAR AS odds_timestamp,
                   received_at::VARCHAR AS received_at,
                   provider, bookmaker, market, line, selection, price,
                   is_suspended, is_closing
            FROM raw.odds_snapshots
            WHERE LOWER(home_team) = LOWER(?) AND LOWER(away_team) = LOWER(?)
              AND market = ?
        """
        params: list[Any] = [home_team, away_team, market]
        if event_date:
            query += " AND (event_time IS NULL OR event_time::DATE BETWEEN (?::DATE - INTERVAL '1 day') AND (?::DATE + INTERVAL '1 day'))"
            d_str = str(event_date)[:10]
            params.extend([d_str, d_str])
        if bookmaker:
            query += " AND LOWER(bookmaker) = LOWER(?)"
            params.append(bookmaker)
        query += " ORDER BY received_at ASC"
        with self._connect() as con:
            df = con.execute(query, params).fetch_df()
        return df.to_dict(orient="records") if not df.empty else []

    def get_closing_odds_snapshot(
        self,
        home_team: str,
        away_team: str,
        event_date: str | None = None,
        market: str = "h2h",
    ) -> dict[str, Any] | None:
        """Extract closing (or latest pre-kickoff) odds map by selection from raw.odds_snapshots."""
        traj = self.get_odds_trajectory(home_team, away_team, event_date=event_date, market=market)
        if not traj:
            return None
        active = [r for r in traj if not r.get("is_suspended")]
        if not active:
            return None
        closing_rows = [r for r in active if r.get("is_closing")]
        pool = closing_rows if closing_rows else active

        sharp_prio = ("pinnacle", "betfair", "matchbook", "market consensus", "consensus", "bet365")
        books_present = sorted({str(r.get("bookmaker") or "") for r in pool})
        chosen_book = None
        for prio in sharp_prio:
            for b in books_present:
                if prio in b.lower():
                    chosen_book = b
                    break
            if chosen_book:
                break
        if chosen_book is None:
            chosen_book = books_present[-1]

        latest_by_sel: dict[str, dict[str, Any]] = {}
        for r in pool:
            if r.get("bookmaker") != chosen_book:
                continue
            latest_by_sel[str(r["selection"])] = r
        if not latest_by_sel:
            return None
        out: dict[str, Any] = {sel: float(row["price"]) for sel, row in latest_by_sel.items()}
        sample = next(iter(latest_by_sel.values()))
        out["bookmaker"] = chosen_book
        out["is_closing"] = bool(closing_rows)
        out["odds_timestamp"] = sample.get("odds_timestamp")
        out["received_at"] = sample.get("received_at")
        return out

    def get_team_crosswalk(self, provider: str, provider_team_id: int, iteration_id: int):
        with self._connect() as con:
            row = con.execute("""SELECT provider_team_name,impect_squad_id,impect_squad_name,
                match_score,mapping_source,wyscout_id,statsbomb_id,sportradar_id,
                understat_id,skillcorner_id FROM output.team_crosswalk
                WHERE provider=? AND provider_team_id=? AND iteration_id=?""",
                [provider, int(provider_team_id), int(iteration_id)]).fetchone()
        if not row:
            return None
        return {
            "provider_team_name": row[0],
            "impect_squad_id": row[1],
            "impect_squad_name": row[2],
            "match_score": row[3],
            "mapping_source": row[4],
            "wyscout_id": row[5],
            "statsbomb_id": row[6],
            "sportradar_id": row[7],
            "understat_id": row[8],
            "skillcorner_id": row[9],
        }

    def save_team_crosswalk(self, rows: list[dict]):
        if not rows:
            return
        with self._connect(read_only=False) as con:
            con.executemany("""INSERT INTO output.team_crosswalk
                (provider,provider_team_id,iteration_id,provider_team_name,impect_squad_id,
                 impect_squad_name,match_score,mapping_source,
                 wyscout_id,statsbomb_id,sportradar_id,understat_id,skillcorner_id,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,current_timestamp)
                ON CONFLICT(provider,provider_team_id,iteration_id) DO UPDATE SET
                  provider_team_name=excluded.provider_team_name,
                  impect_squad_id=COALESCE(excluded.impect_squad_id, output.team_crosswalk.impect_squad_id),
                  impect_squad_name=excluded.impect_squad_name,
                  match_score=excluded.match_score,
                  mapping_source=excluded.mapping_source,
                  wyscout_id=COALESCE(excluded.wyscout_id, output.team_crosswalk.wyscout_id),
                  statsbomb_id=COALESCE(excluded.statsbomb_id, output.team_crosswalk.statsbomb_id),
                  sportradar_id=COALESCE(excluded.sportradar_id, output.team_crosswalk.sportradar_id),
                  understat_id=COALESCE(excluded.understat_id, output.team_crosswalk.understat_id),
                  skillcorner_id=COALESCE(excluded.skillcorner_id, output.team_crosswalk.skillcorner_id),
                  updated_at=now()""",
                [[
                    r["provider"],
                    int(r["provider_team_id"]),
                    int(r["iteration_id"]),
                    r.get("provider_team_name"),
                    r.get("impect_squad_id"),
                    r["impect_squad_name"],
                    r.get("match_score"),
                    r.get("mapping_source", "aligner"),
                    int(r["wyscout_id"]) if r.get("wyscout_id") is not None else None,
                    int(r["statsbomb_id"]) if r.get("statsbomb_id") is not None else None,
                    str(r["sportradar_id"]) if r.get("sportradar_id") is not None else None,
                    str(r["understat_id"]) if r.get("understat_id") is not None else None,
                    int(r["skillcorner_id"]) if r.get("skillcorner_id") is not None else None,
                ] for r in rows])

    def ingest_hudl_id_mapping(self, mappings: list[dict], iteration_id: int = 0) -> int:
        """Persiste correspondencias de IDs StatsBomb <-> Wyscout (y otros IDs B2B) en output.team_crosswalk."""
        rows: list[dict] = []
        for m in mappings:
            wy_id = m.get("wyscout_team_id") or m.get("wyscout_id")
            sb_id = m.get("statsbomb_team_id") or m.get("statsbomb_id")
            sc_id = m.get("skillcorner_id")
            sr_id = m.get("sportradar_id")
            team_name = str(m.get("team_name") or m.get("impect_squad_name") or "").strip()
            if not team_name:
                continue
            pid = wy_id if wy_id is not None else (sb_id if sb_id is not None else sc_id)
            if pid is None:
                pid = int(hashlib.sha1((sr_id or team_name).encode("utf-8")).hexdigest()[:12], 16)
            rows.append({
                "provider": str(m.get("provider") or "hudl_mapping"),
                "provider_team_id": int(pid),
                "iteration_id": int(m.get("iteration_id", iteration_id)),
                "provider_team_name": team_name,
                "impect_squad_id": m.get("impect_squad_id"),
                "impect_squad_name": str(m.get("impect_squad_name") or team_name),
                "match_score": float(m.get("match_score") or 1.0),
                "mapping_source": str(m.get("mapping_source") or "hudl_id_mapping_api"),
                "wyscout_id": wy_id,
                "statsbomb_id": sb_id,
                "sportradar_id": sr_id,
                "understat_id": m.get("understat_id"),
                "skillcorner_id": sc_id,
            })
        if rows:
            self.save_team_crosswalk(rows)
        return len(rows)

    def manual_result_candidates(self):
        """Recommendations whose match outcome is not yet recorded."""
        with self._connect() as con:
            return con.execute("""
                SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team,
                        res.status, res.result_outcome, res.source
                FROM output.recommendations r
                LEFT JOIN output.results res ON res.run_id=r.run_id AND res.row_number=r.row_number
                WHERE res.run_id IS NULL OR res.status NOT IN ('FT','AET','PEN','MANUAL')
                ORDER BY r.event_date DESC, r.home_team, r.away_team
            """).fetchall()

    def result_comparison_rows(self):
        """Predictions joined with their recorded actual outcomes."""
        with self._connect() as con:
            return con.execute("""
                SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team,
                        r.outcome, r.probability, res.status, res.result_outcome,
                        res.source, res.bet_won
                FROM output.recommendations r
                LEFT JOIN output.results res ON res.run_id=r.run_id AND res.row_number=r.row_number
                ORDER BY r.event_date DESC, r.home_team, r.away_team, r.row_number
            """).fetchall()

    def save_manual_result(self, run_id: str, row_number: int, result_outcome: str):
        if result_outcome not in {'local', 'empate', 'visita'}:
            raise ValueError("El resultado debe ser local, empate o visita")
        with self._connect(read_only=False) as con:
            rec = con.execute("SELECT outcome, home_team, away_team FROM output.recommendations WHERE run_id=? AND row_number=?",
                              [run_id, row_number]).fetchone()
            if not rec:
                raise ValueError("No existe esa recomendación")
            bet_won = compute_bet_won(rec[0], result_outcome, rec[1], rec[2])
            con.execute("""INSERT INTO output.results
                (run_id,row_number,status,result_outcome,source,bet_won,checked_at)
                VALUES (?,?,'MANUAL',?,'manual',?,current_timestamp)
                ON CONFLICT(run_id,row_number) DO UPDATE SET fixture_id=NULL,status='MANUAL',
                home_goals=NULL,away_goals=NULL,result_outcome=excluded.result_outcome,
                source='manual',bet_won=excluded.bet_won,checked_at=now()""",
                [run_id, row_number, result_outcome, bet_won])

    def results_summary(self, run_ids: list[str] | None = None):
        """Joined view of recommendations + results, optionally filtered by run_id."""
        with self._connect() as con:
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                query = f"""
                    SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team,
                            r.outcome, r.odds, res.status, res.home_goals, res.away_goals, res.bet_won,
                            res.result_outcome, res.source
                    FROM output.recommendations r
                    LEFT JOIN output.results res ON res.run_id = r.run_id AND res.row_number = r.row_number
                    WHERE r.run_id IN ({placeholders})
                    ORDER BY r.run_id, r.row_number
                """
                return con.execute(query, run_ids).fetchall()
            return con.execute("""
                SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team,
                        r.outcome, r.odds, res.status, res.home_goals, res.away_goals, res.bet_won,
                        res.result_outcome, res.source
                FROM output.recommendations r
                LEFT JOIN output.results res ON res.run_id = r.run_id AND res.row_number = r.row_number
                ORDER BY r.run_id, r.row_number
            """).fetchall()


def _number(value):
    try:
        if isinstance(value, str) and value.endswith("%"):
            return float(value[:-1]) / 100
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_store: DuckDBStore | None = None


def get_store(read_only: bool = False) -> DuckDBStore:
    global _store
    if _store is None or (_store.read_only and not read_only):
        _store = DuckDBStore(read_only=False)
    return _store


def reset_store():
    global _store
    _store = None


def get_connection(read_only: bool = False):
    """Retorna una nueva conexión con la configuración homogénea del store."""
    return get_store().connect(read_only=read_only)


def artifact_key(path: Path | str) -> tuple[str, str]:
    path = Path(path)
    try:
        rel = path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        rel = path.name
    if rel.startswith("data/raw/"):
        return "raw", rel.removeprefix("data/raw/")
    if rel.startswith("data/processed/"):
        return "processed", rel.removeprefix("data/processed/")
    if rel.startswith("outputs/"):
        return "output", rel.removeprefix("outputs/")
    return "processed", rel


def read_json(path: Path | str, default=None):
    layer, key = artifact_key(path)
    value = get_store().get(layer, key, None)
    if value is not None:
        return value
    path = Path(path)
    if not path.exists():
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    get_store().put(layer, key, value)
    return value


def write_json(path: Path | str, payload: Any, run_type: str | None = None, run_date: str | None = None):
    layer, key = artifact_key(path)
    store = get_store()
    store.put(layer, key, payload)
    if isinstance(payload, dict) and (run_type or layer == "output"):
        kind = run_type or Path(path).stem.split("_")[0]
        effective_date = run_date or datetime.now(timezone.utc).date().isoformat()
        if kind == "daily_portfolio":
            # A single updatable run identity per match date prevents repeated
            # executions from appending duplicate predictions in DuckDB.
            date_hash = hashlib.sha256(effective_date.encode("utf-8")).hexdigest()[:12]
            run_id = f"{kind}:{effective_date}:{date_hash}"
        else:
            run_id = f"{kind}:{effective_date}:{uuid.uuid4().hex[:12]}"
        store.save_run(run_id, kind, run_date, payload)


def read_text(path: Path | str, default=None):
    layer, key = artifact_key(path)
    payload = get_store().get(layer, key, None)
    if isinstance(payload, dict) and "text" in payload:
        return payload["text"]
    path = Path(path)
    if not path.exists():
        return default
    value = path.read_text(encoding="utf-8")
    get_store().put(layer, key, {"text": value}, "text/markdown")
    return value


def write_text(path: Path | str, content: str):
    layer, key = artifact_key(path)
    get_store().put(layer, key, {"text": content}, "text/markdown")


@contextmanager
def report_writer(path):
    """Collect a legacy Markdown report and persist it as an output artifact."""
    buffer = StringIO()
    try:
        yield buffer
    finally:
        write_text(path, buffer.getvalue())
