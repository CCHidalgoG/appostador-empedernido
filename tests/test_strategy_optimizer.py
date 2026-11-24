"""Tests for StrategyEngine and KellyOptimizer stake sizing."""

import pytest
from src.apostador.optimizer import KellyOptimizer


def test_evaluate_bet_niche_valor_local_medio():
    optimizer = KellyOptimizer(strategy_mode="hybrid", min_ev=0.03)
    eligible, niche = optimizer.evaluate_bet_niche(
        outcome="Home Win",
        odds=2.10,
        p=0.62,
        ev=0.302,
    )
    assert eligible is True
    assert niche == "valor_local_medio"


def test_evaluate_bet_niche_rejects_implausible_ev():
    """EV > max_ev es un artefacto del modelo, no valor (FC Iberia 2026-10-22: EV +103%)."""
    optimizer = KellyOptimizer(strategy_mode="hybrid", min_ev=0.03)
    eligible, niche = optimizer.evaluate_bet_niche(
        outcome="Home Win",
        odds=3.13,
        p=0.649,
        ev=1.03,
    )
    assert eligible is False
    assert niche is None


def test_evaluate_bet_niche_rejects_extreme_market_divergence():
    """Modelo +20 pp sobre el consenso: el modelo está mal calibrado, no es valor."""
    optimizer = KellyOptimizer(strategy_mode="hybrid", min_ev=0.03)
    eligible, niche = optimizer.evaluate_bet_niche(
        outcome="Home Win",
        odds=1.90,
        p=0.60,
        ev=0.14,
        market_implied_p=0.40,
    )
    assert eligible is False
    assert niche is None


def test_evaluate_bet_niche_accepts_moderate_market_divergence():
    optimizer = KellyOptimizer(strategy_mode="hybrid", min_ev=0.03)
    eligible, niche = optimizer.evaluate_bet_niche(
        outcome="Home Win",
        odds=2.10,
        p=0.62,
        ev=0.302,
        market_implied_p=0.50,
    )
    assert eligible is True
    assert niche == "valor_local_medio"


def test_evaluate_bet_niche_favorito_alta_certeza():
    optimizer = KellyOptimizer(strategy_mode="hybrid", min_ev=0.03)
    eligible, niche = optimizer.evaluate_bet_niche(
        outcome="Home Win",
        odds=1.30,
        p=0.82,
        ev=0.066,
    )
    assert eligible is True
    assert niche == "favorito_alta_certeza"


def test_evaluate_bet_niche_unconstrained():
    optimizer = KellyOptimizer(strategy_mode="unconstrained", min_ev=0.03)
    eligible, niche = optimizer.evaluate_bet_niche(
        outcome="Draw",
        odds=3.40,
        p=0.35,
        ev=0.19,
    )
    assert eligible is True
    assert niche == "ev_positivo"


def test_flat_staking_mode():
    bankroll = 100000.0
    optimizer = KellyOptimizer(
        bankroll=bankroll,
        strategy_mode="hybrid",
        stake_mode="flat",
        flat_stake_pct=0.03,
        min_ev=0.03,
    )
    bets = [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "outcome": "Arsenal",
            "odds": 2.10,
            "probability": 0.62,
            "ev": 0.302,
        },
        {
            "home_team": "Real Madrid",
            "away_team": "Alaves",
            "outcome": "Real Madrid",
            "odds": 1.35,
            "probability": 0.80,
            "ev": 0.08,
        },
    ]
    portfolio = optimizer.optimize_portfolio(bets)
    assert len(portfolio) == 2
    for b in portfolio:
        assert b["stake"] == pytest.approx(bankroll * 0.03)
        assert b["nicho"] in ["valor_local_medio", "favorito_alta_certeza"]


def test_hard_cap_kelly_mode():
    bankroll = 100000.0
    # Overconfident bet that would generate a huge Kelly fraction without cap
    optimizer = KellyOptimizer(
        bankroll=bankroll,
        fraction_factor=1.0,  # full kelly
        strategy_mode="unconstrained",
        stake_mode="kelly",
        max_stake_pct=0.03,   # hard cap 3%
        min_ev=0.01,
    )
    bets = [
        {
            "home_team": "Team A",
            "away_team": "Team B",
            "outcome": "Team A",
            "odds": 1.45,
            "probability": 0.75,  # Kelly pure = (0.0875)/0.45 = 0.194 (19% of bankroll!)
            "ev": 0.0875,
        }
    ]
    portfolio = optimizer.optimize_portfolio(bets)
    assert len(portfolio) == 1
    # Must be capped at exactly max_stake_pct * bankroll
    assert portfolio[0]["stake"] == pytest.approx(3000.0)
    assert portfolio[0]["kelly_scaled"] == pytest.approx(0.03)


def test_strategy_filter_value_local_only():
    bankroll = 50000.0
    optimizer = KellyOptimizer(
        bankroll=bankroll,
        strategy_mode="value_local",
        stake_mode="flat",
        min_ev=0.03,
    )
    bets = [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "outcome": "Arsenal",
            "odds": 2.15,
            "probability": 0.62,
            "ev": 0.333,
        },
        {
            "home_team": "Bayern",
            "away_team": "Bochum",
            "outcome": "Bayern",
            "odds": 1.25,
            "probability": 0.85,
            "ev": 0.0625,
        },
    ]
    portfolio = optimizer.optimize_portfolio(bets)
    # Only the local medium bet should be retained
    assert len(portfolio) == 1
    assert portfolio[0]["home_team"] == "Arsenal"
    assert portfolio[0]["nicho"] == "valor_local_medio"


def test_optimize_portfolio_vetoes_divergent_bet():
    """El portfolio registra la divergencia vs. mercado y veta los desacuerdos extremos."""
    bankroll = 100000.0
    optimizer = KellyOptimizer(
        bankroll=bankroll,
        strategy_mode="hybrid",
        stake_mode="flat",
        min_ev=0.03,
    )
    bets = [
        {   # EV plausible pero +27 pp sobre el consenso -> vetado (patrón FC Thun 2026-10-22)
            "home_team": "FC Thun",
            "away_team": "Hajduk Split",
            "outcome": "Home Win",
            "odds": 2.05,
            "probability": 0.62,
            "ev": 0.271,
            "market_implied_prob": 0.35,
        },
        {   # Divergencia moderada (+12 pp) -> aceptado
            "home_team": "Llaneros FC",
            "away_team": "Deportivo Cali",
            "outcome": "Home Win",
            "odds": 2.10,
            "probability": 0.62,
            "ev": 0.302,
            "market_implied_prob": 0.50,
        },
    ]
    portfolio = optimizer.optimize_portfolio(bets)
    assert len(portfolio) == 1
    assert portfolio[0]["home_team"] == "Llaneros FC"
    assert portfolio[0]["market_implied_prob"] == 0.50
    assert portfolio[0]["market_divergence"] == pytest.approx(0.12)


def test_multi_market_strategy_mode():
    """Verifica que el modo multi_market capture valor equilibrado en 1X2."""
    optimizer = KellyOptimizer(strategy_mode="multi_market", min_ev=0.03)

    # Home Win calibrado
    ok_h, niche_h = optimizer.evaluate_bet_niche("Home Win", odds=2.10, p=0.55, ev=0.155, market_implied_p=0.48)
    assert ok_h is True
    assert niche_h == "valor_local_calibrado"

    # Draw de alto margen (odds >= 2.80, ev >= 0.04)
    ok_d, niche_d = optimizer.evaluate_bet_niche("Draw", odds=3.40, p=0.33, ev=0.122, market_implied_p=0.29)
    assert ok_d is True
    assert niche_d == "valor_empate"

    # Draw con divergencia excesiva (> 8 pp)
    ok_d_bad, _ = optimizer.evaluate_bet_niche("Draw", odds=3.40, p=0.40, ev=0.36, market_implied_p=0.29)
    assert ok_d_bad is False

    # Away Win con valor (odds >= 2.10, ev >= 0.04)
    ok_a, niche_a = optimizer.evaluate_bet_niche("Away Win", odds=2.85, p=0.40, ev=0.14, market_implied_p=0.35)
    assert ok_a is True
    assert niche_a == "valor_visitante"


def test_calibrated_value_strategy_mode():
    optimizer = KellyOptimizer(strategy_mode="calibrated_value", min_ev=0.03, max_ev=0.35)
    ok_valid, niche_valid = optimizer.evaluate_bet_niche("Away Win", odds=3.20, p=0.35, ev=0.12)
    assert ok_valid is True
    assert niche_valid == "valor_calibrado"

    # Rechaza EV excesivo (> max_ev)
    ok_high, _ = optimizer.evaluate_bet_niche("Home Win", odds=3.00, p=0.60, ev=0.80)
    assert ok_high is False


def test_min_stake_amount_discards_under_minimum():
    """Valida que apuestas con stake calculado menor a min_stake_amount sean descartadas."""
    bets = [
        {
            "home_team": "Team A",
            "away_team": "Team B",
            "outcome": "Home Win",
            "odds": 2.10,
            "probability": 0.62,
            "market_implied_prob": 0.50,
        }
    ]
    # Con bankroll 50 y flat 3%, stake = $1.50
    # Con min_stake_amount = 2.0, debe ser descartada sin sobre-apostar
    opt_strict = KellyOptimizer(
        bankroll=50.0,
        strategy_mode="hybrid",
        stake_mode="flat",
        flat_stake_pct=0.03,
        min_stake_amount=2.0,
    )
    res_strict = opt_strict.optimize_portfolio(bets)
    assert len(res_strict) == 0

    # Con min_stake_amount = 1.0, el stake de $1.50 es viable y se conserva
    opt_permissive = KellyOptimizer(
        bankroll=50.0,
        strategy_mode="hybrid",
        stake_mode="flat",
        flat_stake_pct=0.03,
        min_stake_amount=1.0,
    )
    res_permissive = opt_permissive.optimize_portfolio(bets)
    assert len(res_permissive) == 1
    assert res_permissive[0]["stake"] == 1.50


def test_optimize_portfolio_league_cap_and_daily_exposure_scaling():
    bets_same_league = [
        {"home_team": f"H{i}", "away_team": f"A{i}", "league": "Premier League",
         "market_outcome": "Home Win", "probability": p, "odds": 1.60, "market_implied_prob": p - 0.06}
        for i, p in enumerate([0.68, 0.78, 0.74, 0.70])
    ]
    opt_league = KellyOptimizer(
        bankroll=1000.0,
        strategy_mode="hybrid",
        stake_mode="flat",
        flat_stake_pct=0.03,
        max_bets_per_league=2,
        max_daily_exposure_pct=0.15,
    )
    res_league = opt_league.optimize_portfolio(bets_same_league)
    assert len(res_league) == 2
    # Conserva las 2 con mayor EV ordenadas descendentemente
    assert [b["home_team"] for b in res_league] == ["H1", "H2"]

    bets_multi_league = [
        {"home_team": f"M{i}", "away_team": f"V{i}", "league": f"League {i}",
         "market_outcome": "Home Win", "probability": 0.72, "odds": 1.60, "market_implied_prob": 0.63}
        for i in range(5)
    ]
    opt_exposure = KellyOptimizer(
        bankroll=1000.0,
        strategy_mode="hybrid",
        stake_mode="flat",
        flat_stake_pct=0.03,
        max_bets_per_league=3,
        max_daily_exposure_pct=0.10,
    )
    res_exp = opt_exposure.optimize_portfolio(bets_multi_league)
    assert len(res_exp) == 5
    assert sum(b["kelly_scaled"] for b in res_exp) <= 0.10 + 1e-6
    assert all(b.get("exposure_scaled") is True for b in res_exp)
    assert all(b["stake"] == 20.0 for b in res_exp)


def test_steam_move_veto_and_betfair_liquidity_cap():
    """TASK-10 & TASK-11: Veto por steam move adverso y tope 25% por liquidez disponible."""
    opt = KellyOptimizer(
        bankroll=10000.0,
        strategy_mode="calibrated_value",
        stake_mode="flat",
        flat_stake_pct=0.03,
        max_stake_pct=0.03,
    )
    bets = [
        {
            "home_team": "Vetoed Home",
            "away_team": "Away 1",
            "market_outcome": "Home Win",
            "probability": 0.60,
            "odds": 2.00,
            "market_implied_prob": 0.50,
            "line_drift_pp": {"Home": -0.06, "Draw": 0.02, "Away": 0.04},
        },
        {
            "home_team": "Liquid Home",
            "away_team": "Away 2",
            "market_outcome": "Home Win",
            "probability": 0.60,
            "odds": 2.00,
            "market_implied_prob": 0.50,
            "line_drift_pp": {"Home": 0.01, "Draw": -0.01, "Away": 0.0},
            "available_liquidity": 400.0,  # cap 25% = 100.0 vs stake normal 300.0
        },
    ]
    res = opt.optimize_portfolio(bets)
    assert len(res) == 1
    assert res[0]["home_team"] == "Liquid Home"
    assert res[0]["liquidity_capped"] is True
    assert res[0]["stake"] == 100.0



