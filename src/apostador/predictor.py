from __future__ import annotations

import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .aligner import TeamAligner, clean_name, normalize_name
from .config import IMPECT_RAW_DIR, PROCESSED_DIR
from .duckdb_store import read_json
from .impect_team_metrics import resolve_as_of_impect_records
from .skillcorner_fatigue import compute_soccer_fatigue_delta

COUNTRY_TO_UNDERSTAT_LEAGUE = {
    "Spain": "ESP-La Liga",
    "England": "ENG-Premier League",
    "Italy": "ITA-Serie A",
    "Germany": "GER-Bundesliga",
    "France": "FRA-Ligue 1",
}


def _power_novig_probs(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    """Calcula probabilidades implícitas 1X2 sin margen corrigiendo favorite-longshot bias (Power method).

    Resuelve sum(inv_i ** k) = 1 mediante bisección cuando hay overround > 1,
    asignando mayor probabilidad real al favorito respecto a la división proporcional.
    """
    inv = np.array([1.0 / float(oh), 1.0 / float(od), 1.0 / float(oa)], dtype=float)
    overround = float(inv.sum())
    if overround <= 1.0 + 1e-6:
        probs = inv / overround
        return float(probs[0]), float(probs[1]), float(probs[2])

    lo, hi = 0.5, 5.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if float(np.sum(inv ** mid)) > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    probs = inv ** k
    probs = probs / float(probs.sum())
    return float(probs[0]), float(probs[1]), float(probs[2])


def compute_steam_move_metrics(
    opening_odds: dict[str, float] | None,
    current_odds: dict[str, float] | None,
    odds_age_minutes: float | None = None,
    is_2way: bool = False,
) -> dict[str, Any]:
    """Calcula deriva sin margen (line_drift_pp) y detecta steam moves (TASK-10)."""
    drift_by_side: dict[str, float] = {}
    max_abs_drift = 0.0
    required_keys = {"Home", "Away"} if is_2way else {"Home", "Draw", "Away"}

    if (
        opening_odds
        and current_odds
        and required_keys.issubset(opening_odds.keys())
        and required_keys.issubset(current_odds.keys())
    ):
        if is_2way:
            oh0, oa0 = float(opening_odds["Home"]), float(opening_odds["Away"])
            oh1, oa1 = float(current_odds["Home"]), float(current_odds["Away"])
            if oh0 > 1.0 and oa0 > 1.0 and oh1 > 1.0 and oa1 > 1.0:
                inv0 = np.array([1.0 / oh0, 1.0 / oa0], dtype=float)
                inv1 = np.array([1.0 / oh1, 1.0 / oa1], dtype=float)
                p_open = inv0 / float(inv0.sum())
                p_curr = inv1 / float(inv1.sum())
                drift_by_side = {
                    "Home": round(float(p_curr[0] - p_open[0]), 4),
                    "Away": round(float(p_curr[1] - p_open[1]), 4),
                }
        else:
            oh0, od0, oa0 = float(opening_odds["Home"]), float(opening_odds["Draw"]), float(opening_odds["Away"])
            oh1, od1, oa1 = float(current_odds["Home"]), float(current_odds["Draw"]), float(current_odds["Away"])
            if oh0 > 1.0 and od0 > 1.0 and oa0 > 1.0 and oh1 > 1.0 and od1 > 1.0 and oa1 > 1.0:
                p_open = _power_novig_probs(oh0, od0, oa0)
                p_curr = _power_novig_probs(oh1, od1, oa1)
                drift_by_side = {
                    "Home": round(p_curr[0] - p_open[0], 4),
                    "Draw": round(p_curr[1] - p_open[1], 4),
                    "Away": round(p_curr[2] - p_open[2], 4),
                }
        if drift_by_side:
            max_abs_drift = max(abs(v) for v in drift_by_side.values())

    steam_detected = max_abs_drift >= 0.04
    age_val = float(odds_age_minutes) if odds_age_minutes is not None else None
    is_stale = age_val is not None and age_val > 90.0
    w_bonus = 0.15 if (steam_detected or is_stale) else 0.0
    return {
        "line_drift_pp": drift_by_side,
        "max_abs_drift_pp": round(max_abs_drift, 4),
        "steam_move_detected": steam_detected,
        "odds_age_minutes": age_val,
        "is_stale_quote": is_stale,
        "w_shrink_bonus": w_bonus,
    }


def _dixon_coles_probs(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.06,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Calcula probabilidades 1X2 mediante distribución Poisson bivariada ajustada por Dixon-Coles."""
    if pd.isna(lambda_home) or pd.isna(lambda_away) or lambda_home <= 0 or lambda_away <= 0:
        return 0.44, 0.28, 0.28

    p_h = np.array([poisson.pmf(i, lambda_home) for i in range(max_goals)])
    p_a = np.array([poisson.pmf(j, lambda_away) for j in range(max_goals)])
    matrix = np.outer(p_h, p_a)

    # Ajuste de Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)
    matrix[0, 0] *= max(0.0, 1.0 - lambda_home * lambda_away * rho)
    matrix[0, 1] *= max(0.0, 1.0 + lambda_away * rho)
    matrix[1, 0] *= max(0.0, 1.0 + lambda_home * rho)
    matrix[1, 1] *= max(0.0, 1.0 - rho)

    total = matrix.sum()
    if total > 0:
        matrix /= total

    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    return p_home, p_draw, p_away


class SoccerPredictor:
    """Modelo Predictivo Cuantitativo Avanzado de Fútbol.

    Integra múltiples capas analíticas potenciadas por Impect API V5:
    1. Expected Goals Dixon-Coles Poisson (predModelHome / predModelAway a nivel partido).
    2. Ensamble Táctico Multivariable (xG, xGA, Packing defensores/rivales, Field Tilt).
    3. Talento individual agregado de plantilla (Top 11 Packing Score, PxT, Defensivo).
    4. Calibración probabilística y contracción bayesiana al consenso de mercado (Shrinkage).
    5. Fallback seguro a modelo Softmax paramétrico calibrado por MLE cuando hay datos parciales.
    """

    _DEFAULT_PARAMS_PATH = os.path.join(PROCESSED_DIR, "calibrated_params.json")
    _DEFAULT_IMPECT_DIR = os.path.join(IMPECT_RAW_DIR)

    def __init__(self, raw_impect_dir=None, params_path=None):
        self.raw_impect_dir = raw_impect_dir or self._DEFAULT_IMPECT_DIR
        self.params_path = params_path or self._DEFAULT_PARAMS_PATH
        self.ratings_cache: dict[tuple, dict] = {}
        self.squad_ids_cache: dict[tuple, dict] = {}
        self.max_rating_date_cache: dict[int, datetime | None] = {}
        self.predictions_cache: dict[int, list[dict]] = {}
        self.averages_cache: dict[tuple, dict[str, dict]] = {}
        self.scores_cache: dict[tuple, dict[str, dict]] = {}
        self.player_scores_cache: dict[tuple, dict[str, dict]] = {}
        self.player_raw_cache: dict[int, list[dict]] = {}
        self.aligner = TeamAligner()
        self.params = self._load_params()
        self._clubelo_loader = None
        self._understat_loader = None
        self._valuation_loader = None

    def _get_clubelo_loader(self):
        if self._clubelo_loader is None:
            from .soccerdata_clubelo import ClubEloLoader
            self._clubelo_loader = ClubEloLoader()
        return self._clubelo_loader

    def _get_understat_loader(self):
        if self._understat_loader is None:
            from .soccerdata_fbref import UnderstatTacticalLoader
            self._understat_loader = UnderstatTacticalLoader()
        return self._understat_loader

    def _get_valuation_loader(self):
        if self._valuation_loader is None:
            from .player_valuation_loader import PlayerValuationLoader
            self._valuation_loader = PlayerValuationLoader()
        return self._valuation_loader

    def resolve_clubelo_pair(
        self,
        home_team: str,
        away_team: str,
        match_date=None,
    ) -> tuple[float | None, float | None]:
        """Consulta Elos point-in-time en DuckDB para local y visita."""
        loader = self._get_clubelo_loader()
        elo_h = loader.find_team_elo(home_team, match_date=match_date, allow_network=False)
        elo_a = loader.find_team_elo(away_team, match_date=match_date, allow_network=False)
        return elo_h, elo_a

    def get_understat_tactical_summary(
        self,
        league_country: str | None,
        team_name: str,
        target_date=None,
    ) -> dict[str, Any]:
        """Devuelve un resumen compacto desde Understat cacheado cuando falta Impect."""
        us_league = COUNTRY_TO_UNDERSTAT_LEAGUE.get(str(league_country or ""))
        if not us_league or not team_name:
            return {}
        ctx = self._get_understat_loader().get_team_context(
            team_name,
            league=us_league,
            as_of_date=target_date,
            cache_only=True,
        )
        if not ctx:
            return {}
        matches = max(1, int(ctx.get("matches_played") or 1))
        return {
            "goles_partido": round(float(ctx.get("goals_scored") or 0.0) / matches, 2),
            "xg_partido": round(float(ctx.get("xg_per_match") or 0.0), 2),
            "goles_concedidos": round(float(ctx.get("goals_conceded") or 0.0) / matches, 2),
            "xg_concedido": round(float(ctx.get("xga_per_match") or 0.0), 2),
            "ppda": round(float(ctx.get("ppda") or 0.0), 2) if ctx.get("ppda") is not None else None,
            "source": "understat",
        }

    def _load_params(self) -> dict[str, Any]:
        """Carga parámetros calibrados MLE por liga con fallbacks heurísticos robustos."""
        defaults = {
            "alpha": 4.322,
            "gamma_d": -0.168,
            "gamma_d_per_league": {
                "England": -0.18,
                "Germany": -0.20,
                "France": -0.22,
                "Europe": -0.20,
                "Netherlands": -0.19,
                "Portugal": -0.19,
                "Saudi Arabia": -0.18,
                "Colombia": -0.14,
                "Argentina": -0.15,
                "Mexico": -0.17,
                "Brazil": -0.16,
                "Spain": -0.18,
                "Italy": -0.17,
                "USA": -0.19,
            },
            "home_advantage_per_league": {
                "Brazil": 0.097,
                "Argentina": 0.092,
                "Colombia": 0.081,
                "Mexico": 0.070,
                "Spain": 0.064,
                "Italy": 0.057,
                "USA": 0.052,
                "England": 0.055,
                "Germany": 0.058,
                "France": 0.056,
                "Europe": 0.060,
                "Saudi Arabia": 0.050,
            },
            "home_advantage_global": 0.065,
        }
        try:
            loaded = read_json(self.params_path)
            if loaded is not None:
                merged_gamma = {**defaults["gamma_d_per_league"], **loaded.get("gamma_d_per_league", {})}
                merged_ha = {**defaults["home_advantage_per_league"], **loaded.get("home_advantage_per_league", {})}
                return {**defaults, **loaded, "gamma_d_per_league": merged_gamma, "home_advantage_per_league": merged_ha}
        except Exception as e:
            print(f"[SoccerPredictor] Warning: could not load params ({e}). Using defaults.")
        return defaults

    def load_ratings(self, iteration_id: int | None, target_date=None) -> dict[str, float]:
        if not iteration_id:
            return {}
        cache_key = (iteration_id, str(target_date)[:10] if target_date else None)
        if cache_key in self.ratings_cache:
            return self.ratings_cache[cache_key]

        ratings_path = os.path.join(self.raw_impect_dir, f"ratings_{iteration_id}.json")
        data = read_json(ratings_path, default=[])
        if not data:
            return {}

        df = pd.DataFrame(data)
        if df.empty or "value" not in df.columns:
            return {}

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        if target_date is not None:
            t_date = pd.to_datetime(target_date, utc=True).tz_localize(None)
            available_dates = df.loc[df["date"] <= t_date, "date"].dropna().unique()
            if len(available_dates):
                closest_date = max(available_dates)
                days_diff = max(0, (t_date - closest_date).days)
                latest_df = df[df["date"] == closest_date]
            else:
                days_diff = 0
                latest_df = df.iloc[0:0]
        else:
            closest_date = df["date"].max()
            days_diff = 0
            latest_df = df[df["date"] == closest_date]

        default_rating = 0.68
        mean_rating = float(latest_df["value"].mean()) if not latest_df.empty else default_rating

        decay_factor = 1.0
        if days_diff > 7 and target_date is not None:
            decay_factor = np.exp(-0.05 * (days_diff - 7))

        ratings_map = {}
        squad_ids = {}
        for _, row in latest_df.iterrows():
            val = float(row["value"])
            if days_diff > 7 and target_date is not None:
                val = mean_rating + (val - mean_rating) * decay_factor
            ratings_map[row["squadName"]] = val
            squad_id = row.get("squadId")
            if pd.notna(squad_id):
                squad_ids[row["squadName"]] = int(squad_id)

        self.ratings_cache[cache_key] = ratings_map
        self.squad_ids_cache[cache_key] = squad_ids
        return ratings_map

    def load_squad_ids(self, iteration_id: int | None, target_date=None) -> dict[str, int]:
        cache_key = (iteration_id, str(target_date)[:10] if target_date else None)
        if cache_key not in self.squad_ids_cache:
            self.load_ratings(iteration_id, target_date)
        return dict(self.squad_ids_cache.get(cache_key, {}))

    def load_match_predictions(self, iteration_id: int | None) -> list[dict]:
        """Carga las predicciones directas de partido de Impect (λ_H, λ_A)."""
        if not iteration_id:
            return []
        if iteration_id in self.predictions_cache:
            return self.predictions_cache[iteration_id]
        path = os.path.join(self.raw_impect_dir, f"predictions_{iteration_id}.json")
        preds = read_json(path, default=[])
        self.predictions_cache[iteration_id] = preds if isinstance(preds, list) else []
        return self.predictions_cache[iteration_id]

    def _get_max_rating_date(self, iteration_id: int) -> datetime | None:
        if iteration_id in self.max_rating_date_cache:
            return self.max_rating_date_cache[iteration_id]
        ratings_path = os.path.join(self.raw_impect_dir, f"ratings_{iteration_id}.json")
        data = read_json(ratings_path, default=[])
        max_dt = None
        if data:
            df = pd.DataFrame(data)
            if not df.empty and "date" in df.columns:
                parsed = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
                candidate = parsed.max()
                if pd.notna(candidate):
                    max_dt = candidate.to_pydatetime() if hasattr(candidate, "to_pydatetime") else candidate
        self.max_rating_date_cache[iteration_id] = max_dt
        return max_dt

    def _tactical_decay_factor(self, iteration_id: int | None, target_date=None) -> float:
        """Atenúa métricas agregadas de temporada cuando target_date antecede al último dato conocido (>14d)."""
        if not iteration_id or target_date is None:
            return 1.0
        t_date = pd.to_datetime(target_date, utc=True, errors="coerce")
        if pd.isna(t_date):
            return 1.0
        if getattr(t_date, "tzinfo", None) is not None:
            t_date = t_date.tz_localize(None)
        max_date = self._get_max_rating_date(iteration_id)
        if max_date is None:
            return 1.0
        days_ahead = (pd.Timestamp(max_date) - pd.Timestamp(t_date)).days
        if days_ahead > 14:
            return float(np.exp(-0.05 * (days_ahead - 14)))
        return 1.0

    @staticmethod
    def _attenuate_lookup(lookup: dict[str, dict], decay: float) -> dict[str, dict]:
        """Contrae valores numéricos de una tabla por equipo hacia la media de liga según decay."""
        if decay >= 0.9999 or not lookup:
            return lookup
        skip_keys = {"squadId", "squadName", "matches", "iterationId"}
        col_values: dict[str, list[float]] = {}
        for row in lookup.values():
            for k, v in row.items():
                if k in skip_keys:
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool) and not pd.isna(v):
                    col_values.setdefault(k, []).append(float(v))
        means = {k: float(np.mean(vals)) for k, vals in col_values.items() if vals}
        attenuated = {}
        for squad, row in lookup.items():
            new_row = {}
            for k, v in row.items():
                if k in means and isinstance(v, (int, float)) and not isinstance(v, bool) and not pd.isna(v):
                    new_row[k] = means[k] + (float(v) - means[k]) * decay
                else:
                    new_row[k] = v
            attenuated[squad] = new_row
        return attenuated

    def _load_tactical_json_lookup(
        self,
        iteration_id: int | None,
        target_date,
        cache: dict[tuple, dict[str, dict]],
        prefix: str,
    ) -> dict[str, dict]:
        if not iteration_id:
            return {}
        cache_key = (iteration_id, str(target_date)[:10] if target_date else None)
        if cache_key in cache:
            return cache[cache_key]
        data, is_as_of_safe = resolve_as_of_impect_records(
            self.raw_impect_dir,
            prefix,
            iteration_id,
            target_date=target_date,
            reader=read_json,
        )
        lookup = {}
        if isinstance(data, list):
            for row in data:
                name = row.get("squadName")
                if name:
                    lookup[normalize_name(name)] = row
        decay = 1.0 if is_as_of_safe else self._tactical_decay_factor(iteration_id, target_date)
        result = self._attenuate_lookup(lookup, decay)
        cache[cache_key] = result
        return result

    def load_tactical_averages(self, iteration_id: int | None, target_date=None) -> dict[str, dict]:
        """Carga promedios tácticos por equipo (xG, xGA, Packing) con control as-of."""
        return self._load_tactical_json_lookup(iteration_id, target_date, self.averages_cache, "averages")

    def load_tactical_scores(self, iteration_id: int | None, target_date=None) -> dict[str, dict]:
        """Carga índices de rendimiento táctico por equipo (Field Tilt, PPDA) con control as-of."""
        return self._load_tactical_json_lookup(iteration_id, target_date, self.scores_cache, "scores")

    def _load_raw_player_scores(self, iteration_id: int | None) -> list[dict]:
        if not iteration_id:
            return []
        if iteration_id in self.player_raw_cache:
            return self.player_raw_cache[iteration_id]
        path = os.path.join(self.raw_impect_dir, f"player_scores_{iteration_id}.json")
        data = read_json(path, default=[])
        rows = data if isinstance(data, list) else []
        self.player_raw_cache[iteration_id] = rows
        return rows

    @staticmethod
    def _compute_player_group_stats(top_df: pd.DataFrame) -> dict[str, float]:
        metric_specs = {
            "top11_packing": ("IMPECT_SCORE_PACKING", 0.25),
            "top11_off": ("OFFENSIVE_IMPECT_SCORE_PACKING", 0.15),
            "top11_def": ("DEFENSIVE_IMPECT_SCORE_PACKING", 0.10),
            "top11_pxt": ("PXT", 0.05),
        }
        stats: dict[str, float] = {}
        for out_key, (col, default_val) in metric_specs.items():
            if not top_df.empty and col in top_df.columns:
                val = pd.to_numeric(top_df[col], errors="coerce").mean()
                stats[out_key] = float(val) if pd.notna(val) else default_val
            else:
                stats[out_key] = default_val
        return stats

    @staticmethod
    def _match_player_mask(squad_df: pd.DataFrame, query_names: list[str] | None) -> pd.Series:
        if squad_df.empty or not query_names:
            return pd.Series(False, index=squad_df.index)
        compiled = []
        for q in query_names:
            q_clean = clean_name(str(q).replace(".", " "))
            q_tokens = q_clean.split()
            if q_clean:
                compiled.append((q_clean, q_tokens))
        if not compiled:
            return pd.Series(False, index=squad_df.index)

        matches = []
        for _, row in squad_df.iterrows():
            first_raw = str(row.get("firstname") or "")
            last_raw = str(row.get("lastname") or "")
            full_raw = str(row.get("playerName") or f"{first_raw} {last_raw}".strip())
            full = clean_name(full_raw.replace(".", " "))
            first = clean_name(first_raw.replace(".", " "))
            last = clean_name(last_raw.replace(".", " "))
            full_tokens = full.split()
            last_tokens = last.split() if last else (full_tokens[1:] if len(full_tokens) >= 2 else [])
            first_tok = first.split()[0] if first else (full_tokens[0] if full_tokens else "")

            is_hit = False
            for q_clean, q_tokens in compiled:
                if not full:
                    continue
                if q_clean == full:
                    is_hit = True
                    break
                if len(q_tokens) >= 2 and last_tokens and first_tok:
                    q_first, q_last = q_tokens[0], q_tokens[-1]
                    last_match = (q_last in last_tokens) or (last and (last in q_clean or q_last in last))
                    first_match = (
                        q_first == first_tok
                        or (len(q_first) == 1 and first_tok.startswith(q_first))
                        or (len(first_tok) == 1 and q_first.startswith(first_tok))
                        or (len(q_first) > 1 and first_tok.startswith(q_first))
                    )
                    if last_match and first_match:
                        is_hit = True
                        break
                if SequenceMatcher(None, q_clean, full).ratio() >= 0.86:
                    is_hit = True
                    break
            matches.append(is_hit)
        return pd.Series(matches, index=squad_df.index)

    def load_player_talent(self, iteration_id: int | None, target_date=None) -> dict[str, dict]:
        """Carga talento agregado de los 11 titulares habituales (Top 11 Packing y PxT) con control as-of."""
        if not iteration_id:
            return {}
        cache_key = (iteration_id, str(target_date)[:10] if target_date else None)
        if cache_key in self.player_scores_cache:
            return self.player_scores_cache[cache_key]
        data = self._load_raw_player_scores(iteration_id)
        lookup = {}
        if data:
            df = pd.DataFrame(data)
            if not df.empty and "squadName" in df.columns:
                df = df.copy()
                if "matchShare" in df.columns:
                    df["matchShare"] = pd.to_numeric(df["matchShare"], errors="coerce").fillna(0.0)
                else:
                    df["matchShare"] = 0.0
                for squad, grp in df.groupby("squadName"):
                    top11 = grp.sort_values(by="matchShare", ascending=False).head(11)
                    lookup[normalize_name(squad)] = self._compute_player_group_stats(top11)
        decay = self._tactical_decay_factor(iteration_id, target_date)
        result = self._attenuate_lookup(lookup, decay)
        self.player_scores_cache[cache_key] = result
        return result

    def get_available_player_talent(
        self,
        iteration_id: int | None,
        squad_name: str,
        target_date=None,
        injured_players: list[str] | None = None,
        starting_xi: list[str] | None = None,
    ) -> dict[str, float]:
        """Calcula talento ajustado por alineación confirmada o bajas y cuantifica pérdida de packing."""
        defaults = {
            "top11_packing": 0.25,
            "top11_off": 0.15,
            "top11_def": 0.10,
            "top11_pxt": 0.05,
            "missing_share_impact": 0.0,
            "missing_value_pct": 0.0,
            "valuation_source": None,
        }
        if not squad_name:
            return defaults

        # Sin contexto de bajas ni alineación, reutilizar tabla cacheada (ya atenuada as-of)
        if iteration_id and not injured_players and not starting_xi:
            base_lookup = self.load_player_talent(iteration_id, target_date=target_date)
            norm_sq = normalize_name(squad_name)
            row = base_lookup.get(norm_sq)
            if row is None and base_lookup:
                best_k, _ = self.aligner.find_best_match(squad_name, list(base_lookup.keys()), threshold=0.75)
                if best_k:
                    row = base_lookup.get(best_k)
            if row is None:
                return defaults
            return {**defaults, **row, "missing_share_impact": 0.0, "missing_value_pct": 0.0, "valuation_source": "impect"}

        data = self._load_raw_player_scores(iteration_id) if iteration_id else []
        squad_rows = []
        if data:
            norm_target = normalize_name(squad_name)
            squad_rows = [r for r in data if normalize_name(r.get("squadName", "")) == norm_target]
            if not squad_rows:
                all_squads = list({r.get("squadName", "") for r in data if r.get("squadName")})
                best_sq, _ = self.aligner.find_best_match(squad_name, all_squads, threshold=0.75)
                if best_sq:
                    squad_rows = [r for r in data if r.get("squadName") == best_sq]

        if not squad_rows:
            if injured_players or starting_xi:
                val_impact = self._get_valuation_loader().compute_missing_valuation_impact(
                    squad_name,
                    injured_players=injured_players,
                    starting_xi=starting_xi,
                    as_of_date=str(target_date)[:10] if target_date else None,
                )
                if val_impact.get("valuation_source") is not None:
                    miss_imp = float(val_impact["missing_share_impact"])
                    return {
                        **defaults,
                        "top11_packing": defaults["top11_packing"] * (1.0 - 0.50 * miss_imp),
                        "missing_share_impact": miss_imp,
                        "missing_value_pct": float(val_impact["missing_value_pct"]),
                        "valuation_source": val_impact["valuation_source"],
                    }
            return defaults

        grp = pd.DataFrame(squad_rows)
        if "matchShare" in grp.columns:
            grp["matchShare"] = pd.to_numeric(grp["matchShare"], errors="coerce").fillna(0.0)
        else:
            grp["matchShare"] = 0.0

        base_top11 = grp.sort_values(by="matchShare", ascending=False).head(11)
        in_xi = self._match_player_mask(grp, starting_xi)
        is_injured = self._match_player_mask(grp, injured_players)

        if starting_xi and int(in_xi.sum()) >= 7:
            selected_df = grp[in_xi]
        elif injured_players and int(is_injured.sum()) > 0:
            avail_df = grp[~is_injured]
            selected_df = avail_df.sort_values(by="matchShare", ascending=False).head(11) if not avail_df.empty else base_top11
        else:
            selected_df = base_top11

        stats = self._compute_player_group_stats(selected_df)

        missing_share_impact = 0.0
        if injured_players and int(is_injured.sum()) > 0:
            base_pack = (
                pd.to_numeric(base_top11["IMPECT_SCORE_PACKING"], errors="coerce").fillna(0.0).clip(lower=0.0)
                if "IMPECT_SCORE_PACKING" in base_top11.columns
                else pd.Series(0.0, index=base_top11.index)
            )
            base_weighted = float((base_pack * base_top11["matchShare"].clip(lower=0.0)).sum())
            inj_df = grp[is_injured]
            inj_pack = (
                pd.to_numeric(inj_df["IMPECT_SCORE_PACKING"], errors="coerce").fillna(0.0).clip(lower=0.0)
                if "IMPECT_SCORE_PACKING" in inj_df.columns
                else pd.Series(0.0, index=inj_df.index)
            )
            inj_weighted = float((inj_pack * inj_df["matchShare"].clip(lower=0.0)).sum())
            if base_weighted > 0:
                missing_share_impact = float(np.clip(inj_weighted / base_weighted, 0.0, 1.0))

        decay = self._tactical_decay_factor(iteration_id, target_date)
        if decay < 0.9999:
            league_lookup = self.load_player_talent(iteration_id, target_date=None)
            if league_lookup:
                for key in list(stats.keys()):
                    vals = [row[key] for row in league_lookup.values() if key in row]
                    if vals:
                        mean_v = float(np.mean(vals))
                        stats[key] = mean_v + (stats[key] - mean_v) * decay

        return {
            **stats,
            "missing_share_impact": missing_share_impact,
            "missing_value_pct": round(missing_share_impact * 100.0, 2),
            "valuation_source": "impect",
        }

    def _find_impect_match_prediction(
        self,
        home_team: str,
        away_team: str,
        iteration_id: int | None,
        match_date=None,
    ) -> dict[str, Any] | None:
        """Busca predicción oficial Impect con λ_H y λ_A."""
        predictions = self.load_match_predictions(iteration_id)
        if not predictions:
            return None

        norm_home = normalize_name(home_team)
        norm_away = normalize_name(away_team)
        target_day = str(match_date)[:10] if match_date else None

        candidates = []
        for p in predictions:
            pH = normalize_name(p.get("homeSquadName", ""))
            pA = normalize_name(p.get("awaySquadName", ""))
            if (norm_home in pH or pH in norm_home) and (norm_away in pA or pA in norm_away):
                candidates.append(p)

        if not candidates:
            # Fuzzy match si no hubo coincidencia directa
            all_homes = list({p.get("homeSquadName", "") for p in predictions})
            best_h, score_h = self.aligner.find_best_match(home_team, all_homes, threshold=0.75)
            if best_h:
                h_preds = [p for p in predictions if p.get("homeSquadName") == best_h]
                all_aways = list({p.get("awaySquadName", "") for p in h_preds})
                best_a, score_a = self.aligner.find_best_match(away_team, all_aways, threshold=0.75)
                if best_a:
                    candidates = [p for p in h_preds if p.get("awaySquadName") == best_a]

        if not candidates:
            return None

        if target_day:
            day_matches = [c for c in candidates if str(c.get("scheduledDate", ""))[:10] == target_day]
            if day_matches:
                return day_matches[0]

        return candidates[0]

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        iteration_id: int | None = None,
        match_date=None,
        league: str | None = None,
        form_delta: float = 0.0,
        market_odds: dict[str, float] | None = None,
        market_implied_prob: float | None = None,
        home_injuries: list[str] | None = None,
        away_injuries: list[str] | None = None,
        home_lineup: list[str] | None = None,
        away_lineup: list[str] | None = None,
        apply_market_shrinkage: bool = True,
        opening_odds: dict[str, float] | None = None,
        odds_age_minutes: float | None = None,
        fatigue_delta: float | None = None,
        home_physical: dict[str, Any] | None = None,
        away_physical: dict[str, Any] | None = None,
        home_rest_days: float | None = None,
        away_rest_days: float | None = None,
    ) -> dict[str, Any]:
        """Estima probabilidades 1X2 utilizando Impect o el fallback ClubElo + Understat xG."""
        ratings_map = self.load_ratings(iteration_id, match_date) if iteration_id else {}
        default_rating = 0.68
        impect_ratings_found = bool(iteration_id and home_team in ratings_map and away_team in ratings_map)
        clubelo_covered = False

        ha = self.params["home_advantage_per_league"].get(league, self.params["home_advantage_global"])
        gamma_d = self.params.get("gamma_d_per_league", {}).get(league, self.params["gamma_d"])

        if fatigue_delta is not None:
            eff_fatigue_delta = float(np.clip(float(fatigue_delta), -0.06, 0.06))
        else:
            eff_fatigue_delta = compute_soccer_fatigue_delta(
                home_physical=home_physical,
                away_physical=away_physical,
                home_rest_days=home_rest_days,
                away_rest_days=away_rest_days,
            )["fatigue_delta"]

        if impect_ratings_found:
            r_h = ratings_map[home_team]
            r_a = ratings_map[away_team]
        else:
            elo_h, elo_a = self.resolve_clubelo_pair(home_team, away_team, match_date=match_date)
            if elo_h is not None and elo_a is not None:
                clubelo_covered = True
                r_h = float(np.clip(0.68 + (elo_h - 1500.0) / 1500.0, 0.35, 0.98))
                r_a = float(np.clip(0.68 + (elo_a - 1500.0) / 1500.0, 0.35, 0.98))
            else:
                r_h = ratings_map.get(home_team, default_rating)
                r_a = ratings_map.get(away_team, default_rating)

        diff_rating = r_h - r_a
        x = diff_rating + ha + form_delta + eff_fatigue_delta

        # 1. Componente Softmax Clásico (Ratings Impect o Elo ClubElo)
        if clubelo_covered and not impect_ratings_found:
            from .soccerdata_clubelo import ClubEloLoader

            elo_ha = float(np.clip(ha * 1000.0, 35.0, 100.0))
            elo_probs = ClubEloLoader.calculate_probabilities(elo_h, elo_a, home_advantage=elo_ha)
            p_ratings = (float(elo_probs["home_win"]), float(elo_probs["draw"]), float(elo_probs["away_win"]))
        else:
            u_h = self.params["alpha"] * x
            u_a = -self.params["alpha"] * x
            u_d = gamma_d
            exp_h, exp_a, exp_d = np.exp(u_h), np.exp(u_a), np.exp(u_d)
            sum_exp = exp_h + exp_a + exp_d
            p_ratings = (float(exp_h / sum_exp), float(exp_d / sum_exp), float(exp_a / sum_exp))

        # Talento disponible de plantilla (Impect o fallback Transfermarkt/SoFIFA)
        tal_h = self.get_available_player_talent(
            iteration_id,
            home_team,
            target_date=match_date,
            injured_players=home_injuries,
            starting_xi=home_lineup,
        )
        tal_a = self.get_available_player_talent(
            iteration_id,
            away_team,
            target_date=match_date,
            injured_players=away_injuries,
            starting_xi=away_lineup,
        )
        home_missing_impact = float(tal_h.get("missing_share_impact", 0.0))
        away_missing_impact = float(tal_a.get("missing_share_impact", 0.0))

        # 2. Componente Dixon-Coles Poisson (Impect o fallback Understat xG)
        impect_pred = self._find_impect_match_prediction(home_team, away_team, iteration_id, match_date)
        lh = float(impect_pred["predModelHome"]) if impect_pred and pd.notna(impect_pred.get("predModelHome")) else None
        la = float(impect_pred["predModelAway"]) if impect_pred and pd.notna(impect_pred.get("predModelAway")) else None

        us_h, us_a = None, None
        understat_used = False
        if (lh is None or la is None) and league in COUNTRY_TO_UNDERSTAT_LEAGUE:
            us_league = COUNTRY_TO_UNDERSTAT_LEAGUE[str(league)]
            us_loader = self._get_understat_loader()
            us_h = us_loader.get_team_context(home_team, league=us_league, as_of_date=match_date, cache_only=True)
            us_a = us_loader.get_team_context(away_team, league=us_league, as_of_date=match_date, cache_only=True)
            if (
                us_h and us_a
                and us_h.get("xg_per_match") is not None and us_h.get("xga_per_match") is not None
                and us_a.get("xg_per_match") is not None and us_a.get("xga_per_match") is not None
            ):
                lh = float(np.clip(0.5 * (us_h["xg_per_match"] + us_a["xga_per_match"]) * 1.08, 0.30, 3.50))
                la = float(np.clip(0.5 * (us_a["xg_per_match"] + us_h["xga_per_match"]) * 0.94, 0.30, 3.50))
                understat_used = True

        if lh is not None and la is not None and lh > 0 and la > 0:
            lh = max(0.05, lh * (1.0 - 0.12 * home_missing_impact))
            la = max(0.05, la * (1.0 - 0.12 * away_missing_impact))
            p_poisson = _dixon_coles_probs(lh, la, rho=-0.06)
            poisson_available = True
        else:
            p_poisson = p_ratings
            poisson_available = False

        # 3. Componente Táctico y de Jugadores (KPIs Impect o fallback Understat xG)
        averages = self.load_tactical_averages(iteration_id, target_date=match_date)
        scores = self.load_tactical_scores(iteration_id, target_date=match_date)

        avg_h = averages.get(normalize_name(home_team), {})
        avg_a = averages.get(normalize_name(away_team), {})
        score_h = scores.get(normalize_name(home_team), {})
        score_a = scores.get(normalize_name(away_team), {})

        if avg_h and avg_a:
            xg_diff = float((avg_h.get("SHOT_XG") or avg_h.get("EXPECTED_GOALS") or 1.3) -
                            (avg_a.get("SHOT_XG") or avg_a.get("EXPECTED_GOALS") or 1.1))
            xga_diff = float((avg_a.get("CONCEDED_SHOT_XG") or 1.1) -
                             (avg_h.get("CONCEDED_SHOT_XG") or 1.3))
        elif understat_used and us_h and us_a:
            xg_diff = float(us_h["xg_per_match"] - us_a["xg_per_match"])
            xga_diff = float(us_a["xga_per_match"] - us_h["xga_per_match"])
        else:
            xg_diff = 0.2
            xga_diff = -0.2

        pack_diff = float((avg_h.get("BYPASSED_DEFENDERS") or 40.0) -
                          (avg_a.get("BYPASSED_DEFENDERS") or 40.0)) / 20.0
        tilt_diff = float((score_h.get("FIELD_TILT") or 0.50) - (score_a.get("FIELD_TILT") or 0.50))
        talent_diff = float((tal_h.get("top11_packing") or 0.25) - (tal_a.get("top11_packing") or 0.25)) * 2.0

        pxt_h = avg_h.get("PXT") if avg_h.get("PXT") is not None else tal_h.get("top11_pxt")
        pxt_a = avg_a.get("PXT") if avg_a.get("PXT") is not None else tal_a.get("top11_pxt")
        pxt_diff = float(pxt_h - pxt_a) if (pxt_h is not None and pxt_a is not None and (pxt_h != 0.05 or pxt_a != 0.05)) else 0.0

        psxg_h = avg_h.get("POST_SHOT_XG")
        psxg_a = avg_a.get("POST_SHOT_XG")
        psxg_diff = float(psxg_h - psxg_a) if (psxg_h is not None and psxg_a is not None) else 0.0

        tactical_score = (
            0.35 * diff_rating * 10.0 +
            0.25 * xg_diff +
            0.20 * xga_diff +
            0.10 * pack_diff +
            0.10 * tilt_diff +
            0.10 * talent_diff +
            0.08 * pxt_diff +
            0.07 * psxg_diff +
            ha * 5.0 +
            form_delta * 3.0 +
            eff_fatigue_delta * 3.0 +
            (away_missing_impact - home_missing_impact) * 1.5
        )

        u_tactical_h = tactical_score
        u_tactical_a = -tactical_score
        exp_th, exp_ta, exp_td = np.exp(np.clip(u_tactical_h, -5, 5)), np.exp(np.clip(u_tactical_a, -5, 5)), np.exp(gamma_d)
        sum_tactical = exp_th + exp_ta + exp_td
        p_tactical = (float(exp_th / sum_tactical), float(exp_td / sum_tactical), float(exp_ta / sum_tactical))

        # 4. Ensamble de Modelos (Ponderación óptima validada empíricamente)
        if poisson_available:
            w_poisson = 0.50
            w_tactical = 0.35
            w_ratings = 0.15
        else:
            w_poisson = 0.0
            w_tactical = 0.65
            w_ratings = 0.35

        p_ensemble_h = w_poisson * p_poisson[0] + w_tactical * p_tactical[0] + w_ratings * p_ratings[0]
        p_ensemble_d = w_poisson * p_poisson[1] + w_tactical * p_tactical[1] + w_ratings * p_ratings[1]
        p_ensemble_a = w_poisson * p_poisson[2] + w_tactical * p_tactical[2] + w_ratings * p_ratings[2]
        tot_ens = p_ensemble_h + p_ensemble_d + p_ensemble_a
        p_ensemble = (p_ensemble_h / tot_ens, p_ensemble_d / tot_ens, p_ensemble_a / tot_ens)

        if impect_ratings_found:
            signal_source = "impect"
        elif clubelo_covered and understat_used:
            signal_source = "clubelo_understat_fallback"
        elif clubelo_covered:
            signal_source = "clubelo_fallback"
        else:
            signal_source = "uninformed_prior"

        # 5. Contracción Bayesiana al Mercado (mayor shrinkage si solo hay Elo sin xG o ante steam move/cuota envejecida)
        base_shrink = 0.35 if signal_source == "clubelo_fallback" else 0.20
        steam = compute_steam_move_metrics(opening_odds, market_odds, odds_age_minutes=odds_age_minutes)
        w_shrink = min(0.85, base_shrink + steam["w_shrink_bonus"])
        if not apply_market_shrinkage:
            p_final = p_ensemble
        elif market_odds and {"Home", "Draw", "Away"}.issubset(market_odds.keys()):
            oh = float(market_odds["Home"])
            od = float(market_odds["Draw"])
            oa = float(market_odds["Away"])
            if oh > 1.0 and od > 1.0 and oa > 1.0:
                novig_p = _power_novig_probs(oh, od, oa)
                p_final_h = (1.0 - w_shrink) * p_ensemble[0] + w_shrink * novig_p[0]
                p_final_d = (1.0 - w_shrink) * p_ensemble[1] + w_shrink * novig_p[1]
                p_final_a = (1.0 - w_shrink) * p_ensemble[2] + w_shrink * novig_p[2]
                tot_f = p_final_h + p_final_d + p_final_a
                p_final = (p_final_h / tot_f, p_final_d / tot_f, p_final_a / tot_f)
            else:
                p_final = p_ensemble
        elif market_implied_prob is not None:
            p_final_h = (1.0 - w_shrink) * p_ensemble[0] + w_shrink * float(market_implied_prob)
            rem = (1.0 - p_final_h) / max(0.01, (p_ensemble[1] + p_ensemble[2]))
            p_final = (p_final_h, p_ensemble[1] * rem, p_ensemble[2] * rem)
        else:
            p_final = p_ensemble

        model_version = "v2_tactical_xg_ensemble" if signal_source == "impect" else f"v2_{signal_source}"

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_rating": r_h,
            "away_rating": r_a,
            "rating_difference": diff_rating,
            "home_advantage_used": ha,
            "form_delta": form_delta,
            "fatigue_delta": float(eff_fatigue_delta),
            "model_alpha": float(self.params["alpha"]),
            "model_gamma_d": float(gamma_d),
            "adjusted_difference": x,
            "model_version": model_version,
            "signal_source": signal_source,
            "clubelo_covered": clubelo_covered,
            "poisson_available": poisson_available,
            "lambda_home": lh,
            "lambda_away": la,
            "home_missing_impact": home_missing_impact,
            "away_missing_impact": away_missing_impact,
            "home_missing_value_pct": float(tal_h.get("missing_value_pct", round(home_missing_impact * 100.0, 2))),
            "away_missing_value_pct": float(tal_a.get("missing_value_pct", round(away_missing_impact * 100.0, 2))),
            "valuation_source": tal_h.get("valuation_source") or tal_a.get("valuation_source"),
            "line_drift_pp": steam["line_drift_pp"],
            "steam_move_detected": steam["steam_move_detected"],
            "odds_age_minutes": steam["odds_age_minutes"],
            "market_shrinkage_used": float(w_shrink) if apply_market_shrinkage else 0.0,
            "P_poisson": {"H": p_poisson[0], "D": p_poisson[1], "A": p_poisson[2]},
            "P_tactical": {"H": p_tactical[0], "D": p_tactical[1], "A": p_tactical[2]},
            "P_ratings_base": {"H": p_ratings[0], "D": p_ratings[1], "A": p_ratings[2]},
            "P_ensemble": {"H": float(p_ensemble[0]), "D": float(p_ensemble[1]), "A": float(p_ensemble[2])},
            "P_H": float(p_final[0]),
            "P_D": float(p_final[1]),
            "P_A": float(p_final[2]),
        }
