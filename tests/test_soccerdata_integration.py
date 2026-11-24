"""Tests de integración para los módulos de SoccerData y Backtesting con Cuotas Históricas."""

import pytest
from datetime import datetime
from pathlib import Path

from apostador.config import SOCCERDATA_RAW_DIR, ensure_runtime_directories
from apostador.duckdb_store import get_store
from apostador.soccerdata_match_history import (
    PINNACLE_STALE_CUTOFF_DATE,
    SoccerDataHistoryLoader,
    validate_usage_compliance,
)
from apostador.soccerdata_clubelo import ClubEloLoader
from apostador.soccerdata_fbref import (
    FBrefTacticalLoader,
    UnderstatTacticalLoader,
    WhoScoredMissingPlayersLoader,
    resolve_current_season_code,
)
from apostador.duckdb_store import DuckDBStore
from analysis.backtest_with_historical_odds import HistoricalOddsBacktester


def test_config_and_runtime_dirs():
    """Valida la configuración de SOCCERDATA_RAW_DIR y la creación de directorios."""
    ensure_runtime_directories()
    assert SOCCERDATA_RAW_DIR.exists()
    assert SOCCERDATA_RAW_DIR.is_dir()


def test_teamname_replacements_config():
    """Valida que el archivo de reemplazo de nombres exista y sea JSON válido."""
    repl_file = SOCCERDATA_RAW_DIR / "config" / "teamname_replacements.json"
    assert repl_file.exists()
    import json
    data = json.loads(repl_file.read_text(encoding="utf-8"))
    assert "Real Madrid" in data
    assert "Barcelona" in data
    assert "Atlético de Madrid" in data


def test_match_history_duckdb_persistence():
    """Valida la persistencia y consulta en raw.match_history en DuckDB."""
    store = get_store()
    test_record = {
        "league": "TEST-League",
        "season": "9900",
        "game_id": "9999-01-01-Alpha-Beta",
        "match_date": "9999-01-01T12:00:00",
        "home_team": "Alpha FC",
        "away_team": "Beta FC",
        "fthg": 2,
        "ftag": 1,
        "ftr": "H",
        "b365_home": 1.95,
        "b365_draw": 3.40,
        "b365_away": 4.10,
        "pinnacle_home": 1.98,
        "pinnacle_draw": 3.45,
        "pinnacle_away": 4.15,
        "avg_home": 1.96,
        "avg_draw": 3.42,
        "avg_away": 4.12,
        "max_home": 2.02,
        "max_draw": 3.55,
        "max_away": 4.30,
    }
    store.save_match_history([test_record])

    results = store.list_match_history(league="TEST-League", season="9900")
    assert len(results) >= 1
    found = [r for r in results if r["game_id"] == "9999-01-01-Alpha-Beta"][0]
    assert found["home_team"] == "Alpha FC"
    assert found["b365_home"] == 1.95
    assert found["pinnacle_home"] == 1.98


def test_backtest_with_historical_odds_execution():
    """Ejecuta el backtester sobre recomendaciones de DuckDB y verifica métricas."""
    backtester = HistoricalOddsBacktester()
    res = backtester.run_backtest()
    assert "resumen_general" in res
    gen = res["resumen_general"]
    assert gen["total_apuestas"] > 0
    assert "roi_real_pct" in gen
    assert "win_rate_pct" in gen
    assert "por_rango_cuotas" in res
    assert "por_seleccion" in res
    assert "por_umbral_ev" in res


def test_clubelo_probabilities_and_fallback():
    """Verifica el cálculo de probabilidades Elo y la coherencia matemática."""
    probs = ClubEloLoader.calculate_probabilities(home_elo=1800, away_elo=1600, home_advantage=65.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert probs["home_win"] > probs["away_win"]

    loader = ClubEloLoader()
    match_info = loader.get_match_elos("Real Madrid", "Getafe", "2024-05-10")
    assert "prob_home" in match_info
    assert "prob_draw" in match_info
    assert "prob_away" in match_info


def test_fbref_and_whoscored_loaders(monkeypatch):
    """Verifica la instanciación y comportamiento de fallback de FBref y WhoScored."""
    fb = FBrefTacticalLoader()
    assert fb.cache_dir.exists()

    ws = WhoScoredMissingPlayersLoader()
    # Mock fetch_missing_players to test formatting and alignment without launching live headless browser
    import pandas as pd
    mock_df = pd.DataFrame([
        {"team": "Real Madrid", "player": "Courtois", "reason": "Cruciate Ligament", "status": "Out"},
        {"team": "Barcelona", "player": "Gavi", "reason": "Knee Injury", "status": "Out"},
    ])
    monkeypatch.setattr(ws, "fetch_missing_players", lambda **kwargs: mock_df)

    res = ws.get_missing_for_match("Real Madrid", "Barcelona")
    assert "home_missing" in res
    assert "away_missing" in res
    assert len(res["home_missing"]) == 1
    assert res["home_missing"][0]["player"] == "Courtois"
    assert len(res["away_missing"]) == 1
    assert res["away_missing"][0]["player"] == "Gavi"


def test_resolve_current_season_code_and_understat_aggregation(monkeypatch):
    """TASK-01: Valida cálculo dinámico de temporada, agregación Understat y fallback desde FBref."""
    import pandas as pd

    assert resolve_current_season_code("2026-11-20") == "2627"
    assert resolve_current_season_code("2026-02-15") == "2526"

    sample_matches = pd.DataFrame([
        {
            "date": "2026-09-12 19:00:00",
            "home_team": "Real Madrid",
            "away_team": "Getafe",
            "home_goals": 2,
            "away_goals": 0,
            "home_xg": 2.15,
            "away_xg": 0.45,
            "home_np_xg": 1.40,
            "away_np_xg": 0.45,
            "home_expected_points": 2.60,
            "away_expected_points": 0.25,
            "home_ppda": 8.5,
            "away_ppda": 16.0,
            "home_deep_completions": 14,
            "away_deep_completions": 3,
        },
        {
            "date": "2026-09-20 16:00:00",
            "home_team": "Valencia",
            "away_team": "Real Madrid",
            "home_goals": 1,
            "away_goals": 1,
            "home_xg": 1.10,
            "away_xg": 1.65,
            "home_np_xg": 1.10,
            "away_np_xg": 1.65,
            "home_expected_points": 0.90,
            "away_expected_points": 1.85,
            "home_ppda": 11.0,
            "away_ppda": 7.5,
            "home_deep_completions": 6,
            "away_deep_completions": 11,
        },
    ])

    agg_all = UnderstatTacticalLoader.aggregate_team_stats(sample_matches)
    rm_all = agg_all[agg_all["team"] == "Real Madrid"].iloc[0]
    assert rm_all["matches_played"] == 2
    assert pytest.approx(rm_all["xg"], 1e-6) == 3.80
    assert pytest.approx(rm_all["xga"], 1e-6) == 1.55
    assert pytest.approx(rm_all["xg_per_match"], 1e-6) == 1.90

    # Filtro as-of debe excluir el partido del 2026-09-20
    agg_asof = UnderstatTacticalLoader.aggregate_team_stats(sample_matches, as_of_date="2026-09-15")
    rm_asof = agg_asof[agg_asof["team"] == "Real Madrid"].iloc[0]
    assert rm_asof["matches_played"] == 1
    assert pytest.approx(rm_asof["xg"], 1e-6) == 2.15

    # FBrefTacticalLoader debe priorizar Understat para ligas soportadas
    fb = FBrefTacticalLoader()
    monkeypatch.setattr(fb.understat, "get_team_match_stats", lambda **kwargs: sample_matches)
    ctx = fb.get_team_context("Real Madrid", league="ESP-La Liga", season="2627")
    assert ctx is not None
    assert ctx["source"] == "understat"
    assert pytest.approx(ctx["xg"], 1e-6) == 3.80


def test_closing_odds_staleness_and_usage_compliance(tmp_path):
    """TASK-02 & TASK-04: Valida columnas *C, corte Pinnacle post-2025-07-23 y guard comercial."""
    import pandas as pd

    loader = SoccerDataHistoryLoader()
    df_sample = pd.DataFrame([
        {
            "league": "ESP-La Liga",
            "season": "2425",
            "game_id": "2025-05-10-RM-BAR",
            "match_date": pd.Timestamp("2025-05-10"),
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "fthg": 2,
            "ftag": 1,
            "ftr": "H",
            "b365_home": 2.10, "b365_draw": 3.50, "b365_away": 3.40,
            "pinnacle_home": 2.12, "pinnacle_draw": 3.55, "pinnacle_away": 3.45,
            "avg_home": 2.08, "avg_draw": 3.48, "avg_away": 3.38,
            "max_home": 2.18, "max_draw": 3.65, "max_away": 3.55,
            "b365_close_home": 2.00, "b365_close_draw": 3.60, "b365_close_away": 3.65,
            "pinnacle_close_home": 2.05, "pinnacle_close_draw": 3.62, "pinnacle_close_away": 3.70,
            "avg_close_home": 2.01, "avg_close_draw": 3.55, "avg_close_away": 3.60,
            "max_close_home": 2.08, "max_close_draw": 3.70, "max_close_away": 3.75,
            "pinnacle_is_stale": False,
            "data_source": "football_data_co_uk",
            "usage_scope": "personal_research_only",
        },
        {
            "league": "ESP-La Liga",
            "season": "2526",
            "game_id": "2025-09-14-RM-ATM",
            "match_date": pd.Timestamp("2025-09-14"),
            "home_team": "Real Madrid",
            "away_team": "Atletico Madrid",
            "fthg": 1,
            "ftag": 0,
            "ftr": "H",
            "b365_home": 1.90, "b365_draw": 3.60, "b365_away": 4.00,
            "pinnacle_home": 2.50, "pinnacle_draw": 3.40, "pinnacle_away": 2.90,
            "avg_home": 1.88, "avg_draw": 3.55, "avg_away": 3.95,
            "max_home": 1.95, "max_draw": 3.70, "max_away": 4.10,
            "b365_close_home": 1.85, "b365_close_draw": 3.65, "b365_close_away": 4.20,
            "pinnacle_close_home": 2.55, "pinnacle_close_draw": 3.35, "pinnacle_close_away": 2.85,
            "avg_close_home": 1.84, "avg_close_draw": 3.60, "avg_close_away": 4.10,
            "max_close_home": 1.90, "max_close_draw": 3.75, "max_close_away": 4.25,
            "pinnacle_is_stale": True,
            "data_source": "football_data_co_uk",
            "usage_scope": "personal_research_only",
        },
    ])
    records = loader.to_records(df_sample)
    assert records[0]["pinnacle_is_stale"] is False
    assert records[1]["pinnacle_is_stale"] is True
    assert PINNACLE_STALE_CUTOFF_DATE == "2025-07-23"

    # TASK-04: modo comercial debe bloquear exportación de personal_research_only
    assert len(validate_usage_compliance(records, commercial_mode=False)) == 2
    with pytest.raises(PermissionError):
        validate_usage_compliance(records, commercial_mode=True, raise_on_violation=True)
    assert validate_usage_compliance(records, commercial_mode=True, raise_on_violation=False) == []

    # Backtest con DuckDB temporal debe usar PSCH pre-2025-07-23 y AvgCH post-2025-07-23
    tmp_store = DuckDBStore(tmp_path / "test_phase1.duckdb")
    tmp_store.save_match_history(records)
    with tmp_store._connect(read_only=False) as con:
        con.execute("""INSERT INTO output.recommendations VALUES
            ('run-1', 1, '2025-05-10', 'Real Madrid', 'Barcelona', 'Home', 0.52, 2.20, 0.144, 100.0, '{}'),
            ('run-1', 2, '2025-09-14', 'Real Madrid', 'Atletico Madrid', 'Home', 0.56, 2.00, 0.120, 100.0, '{}')
        """)
        con.execute("""INSERT INTO output.results (run_id, row_number, fixture_id, status, home_goals, away_goals, bet_won, result_outcome, source) VALUES
            ('run-1', 1, 1001, 'FT', 2, 1, TRUE, 'H', 'api'),
            ('run-1', 2, 1002, 'FT', 1, 0, TRUE, 'H', 'api')
        """)

    backtester = HistoricalOddsBacktester(store=tmp_store)
    df_bt = backtester.load_data()
    assert df_bt.loc[0, "clv_source"] == "pinnacle_closing"
    assert pytest.approx(df_bt.loc[0, "closing_benchmark"], 1e-6) == 2.05
    assert df_bt.loc[1, "clv_source"] == "market_avg_closing"
    assert pytest.approx(df_bt.loc[1, "closing_benchmark"], 1e-6) == 1.84


def test_colombia_odds_source_enabled():
    """TASK-03: Valida que Primera A de Colombia (239) tenga odds_source='apifootball'."""
    from commands.main import SUPPORTED_LEAGUES

    assert 239 in SUPPORTED_LEAGUES
    assert SUPPORTED_LEAGUES[239]["impect_iteration"] == 1752
    assert SUPPORTED_LEAGUES[239]["odds_source"] == "apifootball"


def test_open_data_calibration_lab_samples(tmp_path):
    """TASK-07: Valida ingesta offline StatsBomb/360, IMPECT Open Data y SkillCorner 10Hz."""
    import json
    from apostador.open_data_lab import OpenDataCalibrationLab

    sb_ev = tmp_path / "sb_events.json"
    sb_360 = tmp_path / "sb_360.json"
    sb_ev.write_text(json.dumps([
        {"id": "ev-1", "type": {"name": "Shot"}, "team": {"name": "Team X"}, "shot": {"statsbomb_xg": 0.35}},
        {"id": "ev-2", "type": {"name": "Pass"}, "team": {"name": "Team X"}},
        {"id": "ev-3", "type": {"name": "Pressure"}, "team": {"name": "Team Y"}},
    ]), encoding="utf-8")
    sb_360.write_text(json.dumps([{"event_uuid": "ev-1", "freeze_frame": []}]), encoding="utf-8")

    imp_open = tmp_path / "impect_bundesliga.json"
    imp_open.write_text(json.dumps([
        {"squadName": "Bayer Leverkusen", "BYPASSED_DEFENDERS": 60, "BYPASSED_OPPONENTS": 150, "PXT": 1.2, "SHOT_XG": 2.0},
    ]), encoding="utf-8")

    sc_track = tmp_path / "skillcorner_tracking.jsonl"
    lines = []
    for i in range(15):
        lines.append(json.dumps({
            "frame": i,
            "player_data": [
                {"player_id": 10, "x": i * 0.65, "y": 0.0, "is_detected": True},
                {"player_id": 99, "x": -10.0, "y": 5.0, "is_detected": False},
            ],
        }))
    sc_track.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sc_dyn = tmp_path / "skillcorner_dynamic.csv"
    sc_dyn.write_text("event_id,type\n1,off_ball_run\n2,passing_option\n", encoding="utf-8")

    lab = OpenDataCalibrationLab()
    sb_res = lab.load_statsbomb_match(sb_ev, sb_360)
    imp_res = lab.load_impect_open_bundesliga(imp_open)
    sc_res = lab.load_skillcorner_match(sc_track, sc_dyn)
    priors = lab.calibrate_open_data_priors(sb_res, imp_res, sc_res)

    assert sb_res["events_with_360"] == 1
    assert sb_res["team_xg"]["Team X"] == 0.35
    assert imp_res["teams"]["Bayer Leverkusen"]["pxt"] == 1.2
    assert sc_res["detected_ratio"] == 0.5
    assert sc_res["high_speed_running_m"] > 0.0
    assert sc_res["dynamic_events_count"] == 2
    assert priors["packing_per_xg_ratio"] == 30.0


def test_scrapers_resilience_tos_and_apifootball_fallback(tmp_path, monkeypatch):
    """TASK-08: Valida cumplimiento ToS, auditoría de esquema y fallback hacia API-Football."""
    import pandas as pd
    from apostador.api_health import check_scrapers_resilience
    from apostador.soccerdata_fbref import check_scraper_tos_compliance

    assert check_scraper_tos_compliance("fotmob", production_mode=True)["allowed"] is False
    assert check_scraper_tos_compliance("sofascore", production_mode=True)["allowed"] is False
    assert check_scraper_tos_compliance("understat", production_mode=True)["allowed"] is True

    tmp_store = DuckDBStore(tmp_path / "scrapers.duckdb")
    ws = WhoScoredMissingPlayersLoader(store=tmp_store)
    assert ws.validate_schema(pd.DataFrame([{"wrong_col": 1}])) is False

    monkeypatch.setattr(ws, "fetch_missing_players", lambda **kwargs: pd.DataFrame())

    class _StubAFClient:
        @staticmethod
        def injuries_for_date(match_date):
            return {
                "records": [
                    {"team": {"name": "Real Madrid"}, "player": {"name": "Courtois", "reason": "Knee", "type": "Out"}},
                ],
                "error": None,
            }

    res = ws.get_missing_for_match(
        "Real Madrid",
        "Getafe",
        match_date="2026-11-20",
        apifootball_client=_StubAFClient(),
    )
    assert res["source"] == "api_football_fallback"
    assert len(res["home_missing"]) == 1
    assert res["home_missing"][0]["player"] == "Courtois"

    ws._record_audit("schema_error", rows=0, detail="missing column player")
    health = check_scrapers_resilience(store=tmp_store)
    assert health["status"] == "warning"
    assert "fotmob" in health["tos_blocked_for_automated_production"]
    assert "sofascore" in health["tos_blocked_for_automated_production"]


def test_odds_snapshots_clv_and_phase3_connectors(tmp_path):
    """TASK-09, TASK-11, TASK-12: Valida raw.odds_snapshots, Betfair, Sportmonks y OpticOdds."""
    from apostador.betfair_exchange import BetfairExchangeClient
    from apostador.opticodds_client import OpticOddsClient
    from apostador.sportmonks_client import SportmonksFootballClient

    tmp_store = DuckDBStore(tmp_path / "phase3.duckdb")
    snaps = [
        {
            "sport": "soccer", "league": "Primera A (Colombia)", "fixture_id": "9001",
            "home_team": "Atletico Nacional", "away_team": "Millonarios",
            "event_time": "2026-11-20T20:00:00Z",
            "odds_timestamp": "2026-11-20T12:00:00Z",
            "received_at": "2026-11-20T12:01:00Z",
            "provider": "api_football", "bookmaker": "Pinnacle",
            "market": "h2h", "selection": "Home", "price": 2.25, "is_closing": False,
        },
        {
            "sport": "soccer", "league": "Primera A (Colombia)", "fixture_id": "9001",
            "home_team": "Atletico Nacional", "away_team": "Millonarios",
            "event_time": "2026-11-20T20:00:00Z",
            "odds_timestamp": "2026-11-20T19:54:00Z",
            "received_at": "2026-11-20T19:55:00Z",
            "provider": "api_football", "bookmaker": "Pinnacle",
            "market": "h2h", "selection": "Home", "price": 1.95, "is_closing": True,
        },
    ]
    assert tmp_store.save_odds_snapshots(snaps) == 2
    traj = tmp_store.get_odds_trajectory("Atletico Nacional", "Millonarios", event_date="2026-11-20")
    assert len(traj) == 2
    assert traj[0]["price"] == 2.25
    assert traj[1]["price"] == 1.95

    with tmp_store._connect(read_only=False) as con:
        con.execute("""INSERT INTO output.recommendations VALUES
            ('r-col-1', 1, '2026-11-20', 'Atletico Nacional', 'Millonarios', 'Home Win', 0.52, 2.20, 0.144, 100.0, '{}')
        """)
        con.execute("""INSERT INTO output.results (run_id, row_number, fixture_id, status, home_goals, away_goals, bet_won, result_outcome, source) VALUES
            ('r-col-1', 1, 9001, 'FT', 2, 0, TRUE, 'H', 'api')
        """)
    bt = HistoricalOddsBacktester(store=tmp_store).load_data()
    assert bt.loc[0, "clv_source"] == "own_closing_snapshot"
    assert pytest.approx(bt.loc[0, "closing_benchmark"], 1e-6) == 1.95

    # Betfair Exchange: política Live read-only, chunking <= 200 pts, microestructura
    with pytest.raises(PermissionError):
        BetfairExchangeClient(app_key="k", session_token="s", is_live_key=True, read_only_only=True)
    batches = BetfairExchangeClient.chunk_market_ids([f"1.{i}" for i in range(25)], price_data=["EX_BEST_OFFERS", "EX_TRADED"])
    assert all(len(b) * 22 <= 200 for b in batches)
    micro = BetfairExchangeClient.parse_market_book_microstructure(
        {
            "marketId": "1.234",
            "status": "OPEN",
            "totalMatched": 15000.0,
            "runners": [
                {"selectionId": 1, "ex": {"availableToBack": [{"price": 2.00, "size": 300.0}], "availableToLay": [{"price": 2.04, "size": 250.0}]}, "totalMatched": 8000.0},
                {"selectionId": 2, "ex": {"availableToBack": [{"price": 3.40, "size": 150.0}], "availableToLay": [{"price": 3.50, "size": 120.0}]}, "totalMatched": 4000.0},
                {"selectionId": 3, "ex": {"availableToBack": [{"price": 4.20, "size": 100.0}], "availableToLay": [{"price": 4.40, "size": 90.0}]}, "totalMatched": 3000.0},
            ],
        },
        runner_names={1: "Home", 2: "Draw", 3: "Away"},
    )
    assert micro["runners"]["Home"]["top3_back_liquidity"] == 300.0
    assert abs(sum(micro["mid_novig_probs"].values()) - 1.0) < 1e-4

    # Sportmonks v3 & OpticOdds v3 normalizers
    sm_rows = SportmonksFootballClient.normalize_odds_to_snapshots(
        {"data": [{"label": "1", "value": 2.15, "bookmaker": {"name": "Bet365"}, "market_description": "Fulltime Result"}]},
        home_team="Home FC", away_team="Away FC", fixture_id=555,
    )
    assert len(sm_rows) == 1 and sm_rows[0]["selection"] == "Home" and sm_rows[0]["market"] == "h2h"

    sse = OpticOddsClient.parse_sse_event('id: e-99\nevent: locked-odds\ndata: {"home_team": "Arsenal", "away_team": "Chelsea", "odds": [{"selection": "Arsenal", "price": 1.85, "sportsbook": "Pinnacle", "market": "Moneyline"}]}')
    assert sse is not None and sse["is_locked"] is True and sse["last_entry_id"] == "e-99"
    oo_rows = OpticOddsClient.normalize_odds_to_snapshots([sse["data"]], is_locked=sse["is_locked"])
    assert len(oo_rows) == 1 and oo_rows[0]["is_suspended"] is True and oo_rows[0]["selection"] == "Home"
