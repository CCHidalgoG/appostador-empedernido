import pytest
import numpy as np
from src.apostador.predictor import SoccerPredictor, _dixon_coles_probs


def test_dixon_coles_probs_sums_to_one():
    ph, pd, pa = _dixon_coles_probs(1.65, 1.10, rho=-0.06)
    assert 0.0 < ph < 1.0
    assert 0.0 < pd < 1.0
    assert 0.0 < pa < 1.0
    assert np.isclose(ph + pd + pa, 1.0, atol=1e-4)
    # With higher home xG (1.65 vs 1.10), home prob should exceed away prob
    assert ph > pa


def test_dixon_coles_probs_handles_invalid_lambda():
    ph, pd, pa = _dixon_coles_probs(0.0, -1.0)
    assert ph == pytest.approx(0.44)
    assert pd == pytest.approx(0.28)
    assert pa == pytest.approx(0.28)


def test_soccer_predictor_fallback_when_no_iteration():
    predictor = SoccerPredictor()
    pred = predictor.predict_match("Unknown Home", "Unknown Away", iteration_id=None)
    assert "P_H" in pred
    assert "P_D" in pred
    assert "P_A" in pred
    assert np.isclose(pred["P_H"] + pred["P_D"] + pred["P_A"], 1.0, atol=1e-4)
    assert pred["home_rating"] == 0.68
    assert pred["away_rating"] == 0.68
    assert pred["poisson_available"] is False


def test_soccer_predictor_with_market_shrinkage():
    predictor = SoccerPredictor()
    market_odds = {"Home": 2.20, "Draw": 3.30, "Away": 3.40}
    pred_no_market = predictor.predict_match("Team A", "Team B", iteration_id=None)
    pred_with_market = predictor.predict_match(
        "Team A", "Team B", iteration_id=None, market_odds=market_odds
    )

    # Both must sum to 1.0
    assert np.isclose(pred_with_market["P_H"] + pred_with_market["P_D"] + pred_with_market["P_A"], 1.0, atol=1e-4)

    # Market implied no-vig for Home is ~(1/2.2) / (1/2.2 + 1/3.3 + 1/3.4) ~= 0.454 / 1.05 ~= 0.432
    # The shrunk probability should reflect market pull
    assert "P_ratings_base" in pred_with_market
    assert "P_tactical" in pred_with_market


def test_power_novig_probs_corrects_favorite_longshot_bias():
    from src.apostador.predictor import _power_novig_probs

    oh, od, oa = 1.45, 4.40, 8.00
    ph, pd_, pa = _power_novig_probs(oh, od, oa)
    assert np.isclose(ph + pd_ + pa, 1.0, atol=1e-6)

    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    prop = inv / inv.sum()
    assert ph > prop[0]
    assert pa < prop[2]


def test_available_player_talent_and_injury_impact(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    def _local_read_json(p, default=None):
        fp = Path(p)
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else default

    monkeypatch.setattr("src.apostador.predictor.read_json", _local_read_json)

    (tmp_path / "ratings_999.json").write_text(json.dumps([
        {"date": "2026-11-10T00:00:00Z", "squadName": "Home FC", "squadId": 1, "value": 0.76},
        {"date": "2026-11-10T00:00:00Z", "squadName": "Away FC", "squadId": 2, "value": 0.66},
    ]), encoding="utf-8")

    home_players = [
        {"squadName": "Home FC", "playerName": "David Ospina", "firstname": "David", "lastname": "Ospina", "matchShare": 0.95, "IMPECT_SCORE_PACKING": 0.45, "OFFENSIVE_IMPECT_SCORE_PACKING": 0.10, "DEFENSIVE_IMPECT_SCORE_PACKING": 0.35, "PXT": 0.08},
        {"squadName": "Home FC", "playerName": "James Rodríguez", "firstname": "James", "lastname": "Rodríguez", "matchShare": 0.90, "IMPECT_SCORE_PACKING": 0.50, "OFFENSIVE_IMPECT_SCORE_PACKING": 0.40, "DEFENSIVE_IMPECT_SCORE_PACKING": 0.10, "PXT": 0.12},
    ]
    for i in range(9):
        home_players.append({
            "squadName": "Home FC",
            "playerName": f"Starter Home {i}",
            "firstname": "Starter",
            "lastname": f"Home{i}",
            "matchShare": 0.85,
            "IMPECT_SCORE_PACKING": 0.38,
            "OFFENSIVE_IMPECT_SCORE_PACKING": 0.22,
            "DEFENSIVE_IMPECT_SCORE_PACKING": 0.16,
            "PXT": 0.07,
        })
    for i in range(2):
        home_players.append({
            "squadName": "Home FC",
            "playerName": f"Bench Home {i}",
            "firstname": "Bench",
            "lastname": f"HBench{i}",
            "matchShare": 0.15,
            "IMPECT_SCORE_PACKING": 0.10,
            "OFFENSIVE_IMPECT_SCORE_PACKING": 0.05,
            "DEFENSIVE_IMPECT_SCORE_PACKING": 0.05,
            "PXT": 0.01,
        })
    away_players = [
        {
            "squadName": "Away FC",
            "playerName": f"Away Starter {i}",
            "firstname": "Away",
            "lastname": f"Starter{i}",
            "matchShare": 0.85,
            "IMPECT_SCORE_PACKING": 0.25,
            "OFFENSIVE_IMPECT_SCORE_PACKING": 0.15,
            "DEFENSIVE_IMPECT_SCORE_PACKING": 0.10,
            "PXT": 0.05,
        }
        for i in range(11)
    ]
    (tmp_path / "player_scores_999.json").write_text(json.dumps(home_players + away_players), encoding="utf-8")

    pred = SoccerPredictor(raw_impect_dir=str(tmp_path))
    base_tal = pred.get_available_player_talent(999, "Home FC", target_date="2026-11-15")
    assert base_tal["missing_share_impact"] == 0.0

    inj_tal = pred.get_available_player_talent(
        999,
        "Home FC",
        target_date="2026-11-15",
        injured_players=["D. Ospina", "J. Rodriguez"],
    )
    assert inj_tal["missing_share_impact"] > 0.15
    assert inj_tal["top11_packing"] < base_tal["top11_packing"]

    xi_names = [f"Starter Home {i}" for i in range(9)] + ["Bench Home 0", "Bench Home 1"]
    xi_tal = pred.get_available_player_talent(999, "Home FC", target_date="2026-11-15", starting_xi=xi_names)
    assert xi_tal["top11_packing"] < base_tal["top11_packing"]

    p_clean = pred.predict_match("Home FC", "Away FC", iteration_id=999, match_date="2026-11-15")
    p_inj = pred.predict_match(
        "Home FC",
        "Away FC",
        iteration_id=999,
        match_date="2026-11-15",
        home_injuries=["D. Ospina", "J. Rodriguez"],
    )
    assert p_inj["home_missing_impact"] > 0.0
    assert p_inj["P_H"] < p_clean["P_H"]


def test_predict_match_apply_market_shrinkage_flag():
    pred = SoccerPredictor()
    market = {"Home": 1.60, "Draw": 3.80, "Away": 5.50}
    p_no_shrink = pred.predict_match(
        "Team A", "Team B", iteration_id=None, market_odds=market, apply_market_shrinkage=False
    )
    p_with_shrink = pred.predict_match(
        "Team A", "Team B", iteration_id=None, market_odds=market, apply_market_shrinkage=True
    )
    assert np.isclose(p_no_shrink["P_H"], p_no_shrink["P_ensemble"]["H"], atol=1e-6)
    assert p_with_shrink["P_H"] > p_no_shrink["P_H"]


def test_tactical_decay_attenuates_future_leakage(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    def _local_read_json(p, default=None):
        fp = Path(p)
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else default

    monkeypatch.setattr("src.apostador.predictor.read_json", _local_read_json)

    (tmp_path / "ratings_888.json").write_text(json.dumps([
        {"date": "2026-12-01T00:00:00Z", "squadName": "Alpha FC", "squadId": 1, "value": 0.78},
        {"date": "2026-12-01T00:00:00Z", "squadName": "Beta FC", "squadId": 2, "value": 0.62},
    ]), encoding="utf-8")
    (tmp_path / "averages_888.json").write_text(json.dumps([
        {"squadName": "Alpha FC", "SHOT_XG": 2.4},
        {"squadName": "Beta FC", "SHOT_XG": 0.8},
    ]), encoding="utf-8")

    from src.apostador.aligner import normalize_name

    pred = SoccerPredictor(raw_impect_dir=str(tmp_path))
    assert pred._tactical_decay_factor(888, "2026-12-01") == 1.0
    assert pred._tactical_decay_factor(888, "2026-10-01") < 0.20

    avg_past = pred.load_tactical_averages(888, target_date="2026-10-01")
    alpha_xg = avg_past[normalize_name("Alpha FC")]["SHOT_XG"]
    assert 1.6 < alpha_xg < 2.0


def test_clubelo_and_understat_fallback_when_no_iteration(tmp_path):
    """TASK-05: Valida fallback ClubElo + Understat xG para ligas sin Impect."""
    from src.apostador.duckdb_store import DuckDBStore
    from src.apostador.soccerdata_clubelo import ClubEloLoader

    tmp_store = DuckDBStore(tmp_path / "fallback.duckdb")
    tmp_store.save_clubelo_ratings([
        {"rating_date": "2026-10-01", "team_name": "Arsenal", "country": "ENG", "level": 1, "elo": 1920.0},
        {"rating_date": "2026-10-01", "team_name": "Chelsea", "country": "ENG", "level": 1, "elo": 1780.0},
        {"rating_date": "2026-10-01", "team_name": "Ajax", "country": "NED", "level": 1, "elo": 1750.0},
        {"rating_date": "2026-10-01", "team_name": "Utrecht", "country": "NED", "level": 1, "elo": 1610.0},
    ])

    class _StubUnderstat:
        @staticmethod
        def get_team_context(team_name, league=None, as_of_date=None, cache_only=True):
            data = {
                "Arsenal": {"xg_per_match": 2.10, "xga_per_match": 0.75},
                "Chelsea": {"xg_per_match": 1.60, "xga_per_match": 1.25},
            }
            return data.get(team_name)

    pred = SoccerPredictor()
    pred._clubelo_loader = ClubEloLoader(store=tmp_store)
    pred._understat_loader = _StubUnderstat()

    res_eng = pred.predict_match("Arsenal", "Chelsea", iteration_id=None, match_date="2026-10-10", league="England")
    assert res_eng["clubelo_covered"] is True
    assert res_eng["poisson_available"] is True
    assert res_eng["signal_source"] == "clubelo_understat_fallback"
    assert res_eng["home_rating"] > res_eng["away_rating"]
    assert res_eng["P_H"] > res_eng["P_A"]
    assert np.isclose(res_eng["P_H"] + res_eng["P_D"] + res_eng["P_A"], 1.0, atol=1e-4)

    res_ned = pred.predict_match("Ajax", "Utrecht", iteration_id=None, match_date="2026-10-10", league="Netherlands")
    assert res_ned["clubelo_covered"] is True
    assert res_ned["poisson_available"] is False
    assert res_ned["signal_source"] == "clubelo_fallback"
    assert res_ned["home_rating"] > res_ned["away_rating"]


def test_player_valuation_fallback_and_jev_state(tmp_path):
    """TASK-06: Valida ponderación de bajas con raw.player_valuations y bloque JEV."""
    from src.apostador.duckdb_store import DuckDBStore
    from src.apostador.player_valuation_loader import PlayerValuationLoader, parse_market_value_eur
    from src.apostador.jev_decision import build_match_state

    assert parse_market_value_eur("€45.00m") == 45_000_000.0
    assert parse_market_value_eur("€500k") == 500_000.0

    tmp_store = DuckDBStore(tmp_path / "valuations.duckdb")
    squad = [
        {"provider": "transfermarkt", "team_name": "Club Test", "player_name": "Star Player", "as_of_date": "2026-10-01", "market_value_eur": 80_000_000.0, "minutes_played": 900.0},
    ] + [
        {"provider": "transfermarkt", "team_name": "Club Test", "player_name": f"Squad Player {i}", "as_of_date": "2026-10-01", "market_value_eur": 10_000_000.0, "minutes_played": 850.0}
        for i in range(10)
    ]
    tmp_store.save_player_valuations(squad)

    pred = SoccerPredictor()
    pred._valuation_loader = PlayerValuationLoader(store=tmp_store)

    tal = pred.get_available_player_talent(
        None,
        "Club Test",
        target_date="2026-10-10",
        injured_players=["Star Player"],
    )
    assert tal["valuation_source"] == "transfermarkt"
    assert tal["missing_share_impact"] == pytest.approx(80_000_000.0 / 180_000_000.0, rel=1e-4)
    assert tal["missing_value_pct"] > 40.0

    state = build_match_state(
        "Club Test",
        "Rival FC",
        "England",
        {"local": 0.45, "empate": 0.28, "visita": 0.27},
        0.72,
        0.66,
        0.02,
        home_missing_impact=tal["missing_share_impact"],
        home_missing_value_pct=tal["missing_value_pct"],
    )
    assert "impacto_bajas_valoracion" in state
    assert state["impacto_bajas_valoracion"]["local_missing_value_pct"] == tal["missing_value_pct"]


def test_steam_move_and_stale_quote_increase_shrinkage():
    """TASK-10: Valida deriva sin margen (line_drift_pp) y aumento dinámico de w_shrink."""
    from src.apostador.predictor import compute_steam_move_metrics

    opening = {"Home": 2.30, "Draw": 3.30, "Away": 3.10}
    current = {"Home": 1.75, "Draw": 3.60, "Away": 4.80}
    metrics = compute_steam_move_metrics(opening, current, odds_age_minutes=20.0)
    assert metrics["steam_move_detected"] is True
    assert metrics["line_drift_pp"]["Home"] > 0.08
    assert metrics["w_shrink_bonus"] == 0.15

    pred = SoccerPredictor()
    res_base = pred.predict_match("Team A", "Team B", iteration_id=None, market_odds=current)
    res_steam = pred.predict_match(
        "Team A", "Team B", iteration_id=None,
        market_odds=current, opening_odds=opening, odds_age_minutes=20.0,
    )
    assert res_steam["steam_move_detected"] is True
    assert res_steam["market_shrinkage_used"] > res_base["market_shrinkage_used"]
    assert res_steam["P_H"] > res_base["P_H"]

    res_stale = pred.predict_match(
        "Team A", "Team B", iteration_id=None,
        market_odds=current, odds_age_minutes=120.0,
    )
    assert res_stale["market_shrinkage_used"] > res_base["market_shrinkage_used"]

