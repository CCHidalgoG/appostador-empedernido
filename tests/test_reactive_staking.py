import pandas as pd
import pytest

from apps.streamlit_app import apply_reactive_staking


def test_apply_reactive_staking_flat_mode():
    sample_recs = pd.DataFrame([
        {"cuota": 2.10, "probabilidad": 0.60, "ev": 0.26, "stake": 4140.0},
        {"cuota": 1.50, "probabilidad": 0.75, "ev": 0.125, "stake": 4140.0},
    ])

    # Bankroll 50,000 con 3% flat
    updated = apply_reactive_staking(
        sample_recs,
        bankroll=50000.0,
        stake_mode="flat",
        flat_stake_pct=0.03,
        max_stake_pct=0.03,
        kelly_factor=0.10,
    )

    assert len(updated) == 2
    for _, row in updated.iterrows():
        assert row["stake"] == 1500.0
        assert row["stake_pct"] == pytest.approx(3.0)


def test_apply_reactive_staking_kelly_mode():
    sample_recs = pd.DataFrame([
        # Bet 1: high edge -> f* = 0.26 / 1.10 = 0.2363 -> 10% Kelly = 2.363%
        {"cuota": 2.10, "probabilidad": 0.60, "ev": 0.26, "stake": 4140.0},
        # Bet 2: moderate edge -> f* = 0.10 / 0.50 = 0.20 -> 10% Kelly = 2.0%
        {"cuota": 1.50, "probabilidad": 0.7333, "ev": 0.10, "stake": 4140.0},
    ])

    # Bankroll 100,000 con Kelly factor 0.10 y tope 5%
    updated = apply_reactive_staking(
        sample_recs,
        bankroll=100000.0,
        stake_mode="kelly",
        flat_stake_pct=0.03,
        max_stake_pct=0.05,
        kelly_factor=0.10,
    )

    stakes = updated["stake"].tolist()
    assert stakes[0] > stakes[1]
    assert stakes[0] == pytest.approx(2363.6, abs=1.0)
    assert stakes[1] == pytest.approx(2000.0, abs=1.0)


def test_apply_reactive_staking_empty():
    empty_df = pd.DataFrame()
    res = apply_reactive_staking(empty_df, bankroll=100000.0)
    assert res.empty


def test_apply_reactive_staking_min_stake_filter():
    sample_recs = pd.DataFrame([
        {"cuota": 2.00, "probabilidad": 0.60, "ev": 0.20, "stake": 5.0},
        {"cuota": 1.50, "probabilidad": 0.75, "ev": 0.125, "stake": 5.0},
    ])

    # Bankroll $100 con flat 1% -> stake $1.00 por apuesta
    # Con min_stake_amount=2.0, ambas apuestas quedan descartadas (stake=0.0)
    filtered = apply_reactive_staking(
        sample_recs,
        bankroll=100.0,
        stake_mode="flat",
        flat_stake_pct=0.01,
        max_stake_pct=0.05,
        min_stake_amount=2.0,
    )
    assert (filtered["stake"] == 0.0).all()
    assert (filtered["retorno_potencial"] == 0.0).all()

    # Con min_stake_amount=1.0, las apuestas de $1.00 se conservan
    accepted = apply_reactive_staking(
        sample_recs,
        bankroll=100.0,
        stake_mode="flat",
        flat_stake_pct=0.01,
        max_stake_pct=0.05,
        min_stake_amount=1.0,
    )
    assert (accepted["stake"] == 1.0).all()
    assert accepted["retorno_potencial"].tolist() == [2.0, 1.5]


def test_build_dynamic_black_swan_min_stake():
    from apps.streamlit_app import build_dynamic_black_swan

    recs = pd.DataFrame([
        {"local": "Team A", "visitante": "Team B", "seleccion": "Home Win", "cuota": 5.0, "probabilidad": 0.30, "ev": 0.50, "fecha": "2026-11-03"},
        {"local": "Team C", "visitante": "Team D", "seleccion": "Home Win", "cuota": 6.0, "probabilidad": 0.25, "ev": 0.50, "fecha": "2026-11-03"},
    ])
    # Bankroll pequeño ($100) generaría micro-stake < $1.0
    parlay = build_dynamic_black_swan(
        recs,
        target_odds=30.0,
        max_legs=2,
        bankroll=100.0,
        min_stake_amount=5.0,
    )
    assert parlay is not None
    assert parlay["stake"] == 5.0
    assert parlay["payout"] == round(5.0 * parlay["total_odds"], 2)

