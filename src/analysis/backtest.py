import numpy as np
import pandas as pd
from src.apostador.config import BACKTESTS_DIR
from src.apostador.duckdb_store import get_store

def run_simulation(days=100, matches_per_day=5, initial_bankroll=1000.0, margin=0.05, fraction_factor=0.25):
    """
    Simulates betting over a series of days to compare Kelly Puro vs. Kelly Fraccionario vs. Parlay.
    """
    bankroll_puro = initial_bankroll
    bankroll_frac = initial_bankroll
    bankroll_parlay = initial_bankroll
    
    history_puro = [bankroll_puro]
    history_frac = [bankroll_frac]
    history_parlay = [bankroll_parlay]
    
    ruined_puro = False
    ruined_frac = False
    ruined_parlay = False
    
    np.random.seed(42) # Set seed for reproducibility
    
    for day in range(days):
        day_bets_puro = []
        day_bets_frac = []
        day_parlay_legs = []
        
        for match in range(matches_per_day):
            # 1. Generate true probability of home win (between 30% and 80%)
            p_true = np.random.uniform(0.3, 0.8)
            
            # 2. Generate bookmaker odds with a margin
            # Odds = (1 - margin) / p_implied
            # We simulate that bookmaker has slightly different probabilities (implied)
            # Let's say implied prob is true prob + some noise
            p_implied = p_true + np.random.normal(0, 0.05)
            p_implied = np.clip(p_implied, 0.1, 0.9)
            
            odds = (1.0 - margin) / p_implied
            
            # EV according to our model (assuming our model is 100% correct, i.e., true probability)
            ev = (p_true * odds) - 1.0
            
            # 3. Simulate match outcome (1 = Win, 0 = Loss)
            outcome = np.random.binomial(1, p_true)
            
            if ev > 0.0:
                # Kelly Pure fraction
                f_star = ev / (odds - 1.0)
                
                if not ruined_puro:
                    day_bets_puro.append({"f": f_star, "odds": odds, "outcome": outcome})
                if not ruined_frac:
                    day_bets_frac.append({"f": f_star * fraction_factor, "odds": odds, "outcome": outcome})
                
                day_parlay_legs.append({"p": p_true, "odds": odds, "outcome": outcome})
                
        # --- Update Pure Kelly Bankroll ---
        if not ruined_puro and day_bets_puro:
            total_f_puro = sum([b["f"] for b in day_bets_puro])
            # Scale if total fraction > 1.0
            scale = 1.0 / total_f_puro if total_f_puro > 1.0 else 1.0
            
            payout = 0.0
            for bet in day_bets_puro:
                f_stake = bet["f"] * scale
                stake = f_stake * bankroll_puro
                if bet["outcome"] == 1:
                    payout += stake * bet["odds"]
                else:
                    payout += 0.0
            
            bankroll_puro = bankroll_puro - (total_f_puro * scale * bankroll_puro) + payout
            if bankroll_puro <= 1.0:
                bankroll_puro = 0.0
                ruined_puro = True
                
        # --- Update Fractional Kelly Bankroll ---
        if not ruined_frac and day_bets_frac:
            # Scale if total fraction > 0.20 (max daily exposure)
            total_f_frac = sum([b["f"] for b in day_bets_frac])
            scale = 0.20 / total_f_frac if total_f_frac > 0.20 else 1.0
            
            payout = 0.0
            for bet in day_bets_frac:
                f_stake = bet["f"] * scale
                stake = f_stake * bankroll_frac
                if bet["outcome"] == 1:
                    payout += stake * bet["odds"]
                else:
                    payout += 0.0
                    
            bankroll_frac = bankroll_frac - (total_f_frac * scale * bankroll_frac) + payout
            if bankroll_frac <= 1.0:
                bankroll_frac = 0.0
                ruined_frac = True
                
        # --- Update Parlay Bankroll (combining top 2 EV bets) ---
        if not ruined_parlay and len(day_parlay_legs) >= 2:
            # Sort by EV
            day_parlay_legs = sorted(day_parlay_legs, key=lambda x: (x["p"] * x["odds"]) - 1.0, reverse=True)
            leg1 = day_parlay_legs[0]
            leg2 = day_parlay_legs[1]
            
            p_parlay = leg1["p"] * leg2["p"]
            odds_parlay = leg1["odds"] * leg2["odds"]
            ev_parlay = (p_parlay * odds_parlay) - 1.0
            
            if ev_parlay > 0.0:
                # Kelly for parlay
                f_parlay = (ev_parlay / (odds_parlay - 1.0)) * fraction_factor
                f_parlay = min(f_parlay, 0.20) # cap at 20% exposure
                
                stake = f_parlay * bankroll_parlay
                won_parlay = (leg1["outcome"] == 1) and (leg2["outcome"] == 1)
                
                if won_parlay:
                    bankroll_parlay = bankroll_parlay - stake + (stake * odds_parlay)
                else:
                    bankroll_parlay = bankroll_parlay - stake
                    
            if bankroll_parlay <= 1.0:
                bankroll_parlay = 0.0
                ruined_parlay = True
                
        history_puro.append(bankroll_puro)
        history_frac.append(bankroll_frac)
        history_parlay.append(bankroll_parlay)
        
    print("=== Backtesting Simulation Results ===")
    print(f"Days simulated: {days} | Matches per day: {matches_per_day} | Bankroll: ${initial_bankroll}")
    print(f"1. Kelly Puro: Final Bankroll: ${bankroll_puro:.2f} | Ruined: {ruined_puro}")
    print(f"2. Kelly Fraccionario (0.25): Final Bankroll: ${bankroll_frac:.2f} | Ruined: {ruined_frac}")
    print(f"3. Combinadas (Parlay 2-legs): Final Bankroll: ${bankroll_parlay:.2f} | Ruined: {ruined_parlay}")
    
    # Calculate stats
    df = pd.DataFrame({
        "Day": list(range(days + 1)),
        "Pure_Kelly": history_puro,
        "Fractional_Kelly": history_frac,
        "Parlay": history_parlay
    })
    
    # Save results CSV
    csv_path = BACKTESTS_DIR / "backtest_results.csv"
    rows = df.to_dict(orient="records")
    store = get_store()
    store.put("output", "backtests/backtest_results.csv", rows, "text/csv")
    store.save_run("backtest_results", "backtest", None, {"rows": rows})
    print(f"Saved simulation history to {csv_path}")

if __name__ == "__main__":
    run_simulation(days=100, matches_per_day=6)
