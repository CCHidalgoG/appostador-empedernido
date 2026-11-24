import numpy as np

class KellyOptimizer:
    def __init__(
        self,
        bankroll: float = 1000.0,
        fraction_factor: float = 0.10,
        strategy_mode: str = "hybrid",
        stake_mode: str = "flat",
        flat_stake_pct: float = 0.03,
        max_stake_pct: float = 0.03,
        min_ev: float = 0.03,
        max_ev: float = 0.35,
        max_divergence: float = 0.15,
        min_stake_amount: float = 1.0,
        max_daily_exposure_pct: float = 0.15,
        max_bets_per_league: int | None = 3,
        **kwargs,
    ):
        """
        Optimizer to allocate betting capital with risk bounding and strategy filtering.

        bankroll: Total betting bankroll in currency
        fraction_factor: Fractional Kelly scaling (e.g. 0.10 for Decile Kelly)
        strategy_mode: Filtering mode for value niches:
            - 'hybrid': Combines Value in Medium Odds (2.0-4.5) with High Certainty Favorites (1.2-1.8)
            - 'value_local': Locales in medium odds [2.00 - 4.50] with P >= 60% (Top ROI +31.1%)
            - 'favorites': High certainty favorites [1.20 - 1.80] (Top Win Rate 75%)
            - 'unconstrained': All positive EV bets (legacy mode)
        stake_mode: 'flat' (fixed percentage of bankroll) or 'kelly' (fractional Kelly)
        flat_stake_pct: Percentage of bankroll for flat stake (default: 0.03 = 3%)
        max_stake_pct: Hard ceiling cap per bet (default: 0.03 = 3%)
        min_ev: Minimum expected value threshold to qualify (default: 0.03 = +3%)
        max_ev: Plausibility cap on expected value (default: 0.35 = +35%). Higher edges
            are almost always artifacts of a miscalibrated probability, not real value
            (e.g. 2026-10-22: EV +103% on a Conference League match with default ratings).
        max_divergence: Maximum allowed gap between the model probability and the
            vig-free implied market probability, in probability points (default:
            0.15 = 15 pp). A model claiming +30 pp over a 15-20 bookmaker consensus
            is broken, not smart (2026-10-22: Iberia model 65% vs market 33%).
        min_stake_amount: Minimum currency amount required to place a bet (default: 1.0).
            Bets with calculated stakes below this threshold are discarded to prevent
            overbetting while respecting minimum bookmaker ticket constraints.
        max_daily_exposure_pct: Maximum total fraction of bankroll exposed in a single day
            across all selected bets (default: 0.15 = 15%). Excess stakes are rescaled proportionally.
        max_bets_per_league: Maximum number of recommended bets per league per day (default: 3),
            keeping only top-EV picks when a league name is provided.
        """
        self.bankroll = float(bankroll)
        self.fraction_factor = float(fraction_factor)
        self.strategy_mode = str(strategy_mode).lower()
        self.stake_mode = str(stake_mode).lower()
        self.flat_stake_pct = float(flat_stake_pct)
        self.max_stake_pct = float(max_stake_pct)
        self.min_ev = float(min_ev)
        self.max_ev = float(max_ev)
        self.max_divergence = float(max_divergence)
        self.min_stake_amount = float(min_stake_amount)
        self.max_daily_exposure_pct = (
            float(max_daily_exposure_pct) if max_daily_exposure_pct is not None else None
        )
        self.max_bets_per_league = (
            int(max_bets_per_league) if max_bets_per_league is not None else None
        )

    def calculate_kelly_fraction(self, p: float, odds: float) -> float:
        """Calculates pure Kelly fraction: f* = (p * odds - 1) / (odds - 1)"""
        if odds <= 1.0:
            return 0.0
        ev = (p * odds) - 1.0
        if ev <= 0.0:
            return 0.0
        return float(ev / (odds - 1.0))

    def evaluate_bet_niche(self, outcome: str, odds: float, p: float, ev: float,
                           market_implied_p: float | None = None,
                           line_drift_pp: float | None = None) -> tuple[bool, str | None]:
        """Verifica si una apuesta clasifica dentro de los nichos de margen positivo validados.

        Vetos de sanidad aplicados antes de los nichos:
        - ``ev > max_ev``: edge implausible; casi siempre un artefacto del modelo.
        - ``p - market_implied_p > max_divergence``: desacuerdo extremo con el consenso
          del mercado (cuando se conoce la probabilidad implícita sin margen).
        - ``line_drift_pp <= -0.05``: steam move fuerte en contra de la selección.
        """
        if ev < self.min_ev or odds <= 1.0:
            return False, None
        if ev > self.max_ev:
            return False, None
        if market_implied_p is not None and (p - float(market_implied_p)) > self.max_divergence:
            return False, None
        if line_drift_pp is not None and float(line_drift_pp) <= -0.05:
            return False, None

        if self.strategy_mode in ("multi_market", "multi_market_value"):
            # Nicho 1: Locales con valor calibrado
            if outcome == "Home Win" and self.min_ev <= ev <= self.max_ev:
                return True, "valor_local_calibrado"
            # Nicho 2: Empates de alto margen (cuotas competitivas >= 2.80)
            if outcome == "Draw" and ev >= max(self.min_ev, 0.04) and odds >= 2.80:
                if market_implied_p is None or (p - float(market_implied_p)) <= 0.08:
                    return True, "valor_empate"
            # Nicho 3: Visitantes de valor (cuotas >= 2.10)
            if outcome == "Away Win" and ev >= max(self.min_ev, 0.04) and odds >= 2.10:
                if market_implied_p is None or (p - float(market_implied_p)) <= 0.10:
                    return True, "valor_visitante"
            return False, None

        elif self.strategy_mode in ("calibrated_value", "calibrated"):
            if self.min_ev <= ev <= self.max_ev:
                return True, "valor_calibrado"
            return False, None

        elif self.strategy_mode in ("hybrid", "hybrid_optimal"):
            # Nicho 1: Locales en cuotas medias (2.00 - 4.50) con alta probabilidad (P >= 60%)
            if outcome == "Home Win" and 2.00 <= odds <= 4.50 and p >= 0.60:
                return True, "valor_local_medio"
            # Nicho 2: Favoritos de alta certeza (1.20 - 1.80) con valor
            if outcome == "Home Win" and 1.20 <= odds <= 1.80 and ev >= self.min_ev:
                return True, "favorito_alta_certeza"
            return False, None

        elif self.strategy_mode in ("value_local", "value_local_medium"):
            if outcome == "Home Win" and 2.00 <= odds <= 4.50 and p >= 0.60:
                return True, "valor_local_medio"
            return False, None

        elif self.strategy_mode in ("favorites", "high_certainty"):
            if outcome == "Home Win" and 1.20 <= odds <= 1.80 and ev >= self.min_ev:
                return True, "favorito_alta_certeza"
            return False, None

        # Modo no restringido (legado)
        return (ev > 0.0), "ev_positivo"

    def optimize_portfolio(self, matches_with_odds: list[dict]) -> list[dict]:
        """
        Allocates bankroll across multiple matches for a given day applying
        niche filtering and hard stake ceilings.
        """
        portfolio = []

        for match in matches_with_odds:
            p = float(match.get("probability", 0.0))
            odds = float(match.get("odds", 0.0))
            raw_outcome = match.get("market_outcome") or match.get("outcome", "")
            home_team = match.get("home_team", "")
            away_team = match.get("away_team", "")
            if raw_outcome == "Home Win" or (home_team and raw_outcome == home_team):
                outcome = "Home Win"
            elif raw_outcome == "Away Win" or (away_team and raw_outcome == away_team):
                outcome = "Away Win"
            elif raw_outcome in ("Draw", "Empate", "X"):
                outcome = "Draw"
            else:
                outcome = raw_outcome

            ev = (p * odds) - 1.0
            market_implied_p = match.get("market_implied_prob")
            raw_drift = match.get("line_drift_pp")
            if isinstance(raw_drift, dict):
                side_key = "Home" if outcome == "Home Win" else ("Away" if outcome == "Away Win" else "Draw")
                sel_drift = raw_drift.get(side_key)
            else:
                sel_drift = raw_drift

            eligible, niche = self.evaluate_bet_niche(
                outcome, odds, p, ev, market_implied_p=market_implied_p, line_drift_pp=sel_drift)
            if not eligible:
                continue

            f_star = self.calculate_kelly_fraction(p, odds)

            if self.stake_mode == "flat":
                f_stake = min(self.flat_stake_pct, self.max_stake_pct)
            else:
                f_frac = f_star * self.fraction_factor
                f_stake = min(f_frac, self.max_stake_pct)

            if f_stake > 0.0:
                stake_amount = round(f_stake * self.bankroll, 2)
                liquidity_capped = False
                avail_liq = match.get("available_liquidity")
                if avail_liq is not None and float(avail_liq) > 0.0:
                    max_liq_stake = round(float(avail_liq) * 0.25, 2)
                    if stake_amount > max_liq_stake:
                        stake_amount = max_liq_stake
                        f_stake = stake_amount / self.bankroll if self.bankroll > 0 else 0.0
                        liquidity_capped = True
                if self.min_stake_amount > 0.0 and stake_amount < self.min_stake_amount:
                    continue
                match_entry = {
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "league": match.get("league"),
                    "outcome": outcome,
                    "probability": p,
                    "odds": odds,
                    "ev": ev,
                    "market_implied_prob": market_implied_p,
                    "market_divergence": (round(p - float(market_implied_p), 4)
                                          if market_implied_p is not None else None),
                    "line_drift_pp": round(float(sel_drift), 4) if sel_drift is not None else None,
                    "odds_age_minutes": match.get("odds_age_minutes"),
                    "liquidity_capped": liquidity_capped,
                    "niche": niche,
                    "nicho": niche,
                    "strategy_mode": self.strategy_mode,
                    "stake_mode": self.stake_mode,
                    "kelly_pure": f_star,
                    "kelly_fractional": f_star * self.fraction_factor,
                    "kelly_scaled": f_stake,
                    "stake": stake_amount,
                }
                portfolio.append(match_entry)

        # Deduplicación: Si varios signos (1, X, 2) del mismo partido tienen valor positivo,
        # seleccionar únicamente el resultado con mayor EV para evitar canibalización y doble exposición.
        best_per_match = {}
        for bet in portfolio:
            match_key = (bet.get("home_team"), bet.get("away_team"))
            if match_key not in best_per_match or bet["ev"] > best_per_match[match_key]["ev"]:
                best_per_match[match_key] = bet

        deduped = sorted(best_per_match.values(), key=lambda b: float(b.get("ev", 0.0)), reverse=True)

        # Tope de concentración por liga (mantiene las de mayor EV cuando la liga está especificada)
        if self.max_bets_per_league is not None and self.max_bets_per_league > 0:
            selected = []
            league_counts: dict[str, int] = {}
            for bet in deduped:
                lg = bet.get("league")
                if isinstance(lg, str) and lg.strip():
                    key_lg = lg.strip()
                    if league_counts.get(key_lg, 0) >= self.max_bets_per_league:
                        continue
                    league_counts[key_lg] = league_counts.get(key_lg, 0) + 1
                selected.append(bet)
        else:
            selected = list(deduped)

        # Tope de exposición diaria agregada: reescala stakes proporcionalmente sin perder diversificación
        if self.max_daily_exposure_pct is not None and self.max_daily_exposure_pct > 0.0 and selected:
            total_pct = sum(float(b["kelly_scaled"]) for b in selected)
            if total_pct > self.max_daily_exposure_pct + 1e-9:
                scale = self.max_daily_exposure_pct / total_pct
                rescaled = []
                for bet in selected:
                    scaled_pct = float(bet["kelly_scaled"]) * scale
                    scaled_stake = round(scaled_pct * self.bankroll, 2)
                    if self.min_stake_amount > 0.0 and scaled_stake < self.min_stake_amount:
                        continue
                    bet["kelly_scaled"] = scaled_pct
                    bet["stake"] = scaled_stake
                    bet["exposure_scaled"] = True
                    rescaled.append(bet)
                selected = rescaled

        return selected

    def analyze_parlay_vs_singles(self, p1, odds1, p2, odds2):
        """
        Provides a mathematical comparison between betting on two matches
        as separate singles vs. combining them into a Parlay (combinada).
        """
        # Single Bet 1 Details
        ev1 = (p1 * odds1) - 1.0
        f_star1 = self.calculate_kelly_fraction(p1, odds1)
        
        # Single Bet 2 Details
        ev2 = (p2 * odds2) - 1.0
        f_star2 = self.calculate_kelly_fraction(p2, odds2)
        
        # Parlay Details
        p_parlay = p1 * p2
        odds_parlay = odds1 * odds2
        ev_parlay = (p_parlay * odds_parlay) - 1.0
        f_star_parlay = self.calculate_kelly_fraction(p_parlay, odds_parlay)
        
        # Risk of losing entire capital
        # Assume equal stake size S for singles (total 2S) vs. 2S on parlay
        prob_lose_all_singles = (1.0 - p1) * (1.0 - p2)
        prob_lose_all_parlay = 1.0 - p_parlay
        
        return {
            "single_1": {"p": p1, "odds": odds1, "ev": ev1, "kelly": f_star1},
            "single_2": {"p": p2, "odds": odds2, "ev": ev2, "kelly": f_star2},
            "parlay": {"p": p_parlay, "odds": odds_parlay, "ev": ev_parlay, "kelly": f_star_parlay},
            "risk": {
                "prob_lose_all_singles": prob_lose_all_singles,
                "prob_lose_all_parlay": prob_lose_all_parlay,
                "increase_in_ruin_prob": prob_lose_all_parlay - prob_lose_all_singles
            }
        }
