"""Interfaz Streamlit moderna para 'App-ostador Empedernido'.

Permite ejecutar el pipeline de apuestas, consultar pronósticos de JEV y Core,
explorar cuotas históricas reales (Pinnacle/Bet365), evaluar el backtesting
financiero y calcular probabilidades con ratings ClubElo.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging

logger = logging.getLogger(__name__)

import importlib
import apostador.duckdb_store
import apostador.optimizer
importlib.reload(apostador.duckdb_store)
importlib.reload(apostador.optimizer)

from apostador.config import DUCKDB_PATH, ROOT_DIR
from apostador.duckdb_store import get_store


def get_db_con(read_only: bool = False):
    """Retorna una conexión activa a DuckDB compartiendo la configuración de DuckDBStore."""
    store = get_store()
    if hasattr(store, "connect"):
        return store.connect(read_only=read_only)
    return store._connect(read_only=read_only)
from apostador.api_health import (
    check_apifootball,
    check_brave_api,
    check_clubelo_api,
    check_impect_api,
    check_jev_api,
    check_odds_api,
    get_duckdb_backup_inventory,
    run_all_api_checks,
)
from apostador.outcome_utils import compute_bet_won, normalize_selection_to_side
from apostador.soccerdata_clubelo import ClubEloLoader
from analysis.backtest_with_historical_odds import HistoricalOddsBacktester

ICON_PATH = PROJECT_ROOT / "apps" / "assets" / "app_icon.png"


def inject_custom_css():
    """Inyecta estilos CSS modernos, dark-mode y glassmorphism."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Banner Hero */
        .hero-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 2rem;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.08) 50%, rgba(15, 23, 42, 0.7) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 18px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(16px);
        }
        .hero-left {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #10B981, #06B6D4, #38BDF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            color: #94A3B8;
            font-size: 0.95rem;
            margin-top: 0.25rem;
            font-weight: 500;
        }

        /* Tarjetas de Métricas */
        .metric-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(16, 185, 129, 0.4);
            box-shadow: 0 8px 15px -3px rgba(16, 185, 129, 0.15);
        }
        .metric-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.2;
        }
        .metric-subtext {
            font-size: 0.82rem;
            color: #10B981;
            font-weight: 600;
            margin-top: 0.25rem;
        }

        /* Estilo de pestañas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0px 0px;
            padding: 8px 16px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_recommendations(target_dates: list[date]) -> pd.DataFrame:
    if not DUCKDB_PATH.exists() or not target_dates:
        return pd.DataFrame()
    try:
        con = get_db_con()
    except Exception as exc:
        logger.warning("No se pudo conectar a DuckDB para recomendaciones: %s", exc)
        return pd.DataFrame()
    try:
        date_values = sorted({day.isoformat() for day in target_dates})
        marks = ",".join("?" for _ in date_values)
        runs = con.execute(f"""SELECT run_id, run_date, started_at
               FROM output.runs
               WHERE run_type = 'daily_portfolio' AND run_date IN ({marks})
               QUALIFY row_number() OVER (PARTITION BY run_date
                                          ORDER BY started_at DESC, run_id DESC)=1
               ORDER BY run_date""", date_values).fetchdf()
        if runs.empty:
            return pd.DataFrame()
        run_ids = runs["run_id"].tolist()
        run_marks = ",".join("?" for _ in run_ids)
        df = con.execute(
            f"""SELECT r.run_id, r.event_date AS fecha, r.home_team AS local,
                       r.away_team AS visitante, r.outcome AS seleccion,
                       r.probability AS probabilidad, r.odds AS cuota,
                       r.ev, r.stake, r.payload
                FROM output.recommendations r
                JOIN output.runs run ON run.run_id=r.run_id
                WHERE r.run_id IN ({run_marks})
                  AND r.event_date IN ({marks})
                ORDER BY r.event_date, r.ev DESC""",
            [*run_ids, *date_values],
        ).fetchdf()
        if df.empty:
            return df

        def _get_nicho_label(p):
            if not p:
                return "⚡ General EV+"
            try:
                data = json.loads(p) if isinstance(p, str) else p
                nicho = (data.get("nicho") or data.get("niche")) if isinstance(data, dict) else None
                if nicho == "valor_local_calibrado":
                    return "⭐ Local Calibrado (Multi-Capa)"
                elif nicho == "valor_empate":
                    return "🤝 Empate de Valor (Cuota ≥ 2.80)"
                elif nicho == "valor_visitante":
                    return "🚀 Visitante de Valor (Cuota ≥ 2.10)"
                elif nicho == "valor_calibrado":
                    return "💎 Valor Calibrado"
                elif nicho == "valor_local_medio":
                    return "🎯 Local Medio (+31% ROI)"
                elif nicho == "favorito_alta_certeza":
                    return "🛡️ Favorito Certero (75% WR)"
                elif nicho:
                    return f"📌 {nicho}"
                return "⚡ General EV+"
            except Exception:
                return "⚡ General EV+"

        def _get_divergence(p):
            if not p:
                return None
            try:
                data = json.loads(p) if isinstance(p, str) else p
                return data.get("market_divergence") if isinstance(data, dict) else None
            except Exception:
                return None

        df["nicho"] = df["payload"].map(_get_nicho_label)
        df["divergencia_mercado"] = df["payload"].map(_get_divergence)
        return df
    except Exception as exc:
        logger.warning("Error consultando recomendaciones en DuckDB: %s", exc)
        return pd.DataFrame()
    finally:
        con.close()


def load_ev_funnel_audit(target_dates: list[date], min_ev_threshold: float = 0.03) -> dict[str, Any]:
    """Calcula el embudo de filtrado y el desglose de EV = (P * Cuota) - 1 por partido elegible."""
    empty_res = {
        "total_matches": 0,
        "coverage_ok": 0,
        "odds_ok": 0,
        "eligible_both": 0,
        "raw_ev_positive": 0,
        "audit_df": pd.DataFrame(),
    }
    if not DUCKDB_PATH.exists() or not target_dates:
        return empty_res
    try:
        con = get_db_con()
    except Exception:
        return empty_res
    try:
        date_values = sorted({day.isoformat() for day in target_dates})
        marks = ",".join("?" for _ in date_values)
        rows = con.execute(
            f"""SELECT run_date, summary
                FROM output.runs
                WHERE run_type = 'daily_portfolio' AND run_date IN ({marks})
                QUALIFY row_number() OVER (PARTITION BY run_date ORDER BY started_at DESC, run_id DESC)=1
                ORDER BY run_date DESC""",
            date_values,
        ).fetchall()
        total_matches = 0
        coverage_ok = 0
        odds_ok = 0
        eligible_both = 0
        raw_ev_positive = 0
        audit_records = []

        for run_date, summary in rows:
            if isinstance(summary, str):
                summary = json.loads(summary)
            for match in (summary or {}).get("processed_matches", []):
                total_matches += 1
                cov = bool(match.get("data_coverage_ok", True))
                has_odds = bool(match.get("odds_available") and isinstance(match.get("odds"), dict))
                if cov:
                    coverage_ok += 1
                if has_odds:
                    odds_ok += 1
                if not (cov and has_odds):
                    continue
                eligible_both += 1
                odds_map = match.get("odds") or {}
                for label, p_key, o_key in (
                    ("Home Win", "P_H", "Home"),
                    ("Draw", "P_D", "Draw"),
                    ("Away Win", "P_A", "Away"),
                ):
                    prob = match.get(p_key)
                    price = odds_map.get(o_key)
                    if prob is None or price is None:
                        continue
                    prob_f = float(prob)
                    price_f = float(price)
                    ev_val = (prob_f * price_f) - 1.0
                    if ev_val > 0.0:
                        raw_ev_positive += 1
                    if ev_val >= min_ev_threshold:
                        status = "✅ Cumple EV Mínimo"
                    elif ev_val > 0.0:
                        status = f"⚠️ EV > 0% pero < {min_ev_threshold:.1%}"
                    else:
                        status = "❌ EV ≤ 0% (Sin valor)"
                    audit_records.append({
                        "fecha": str(run_date),
                        "liga": match.get("league_name", ""),
                        "partido": f"{match.get('home_team')} vs {match.get('visitante') or match.get('away_team')}",
                        "seleccion": label,
                        "P(modelo)": f"{prob_f:.1%}",
                        "cuota": f"{price_f:.2f}",
                        "EV = P×Cuota - 1": f"{ev_val:+.2%}",
                        "ev_num": ev_val,
                        "casa": match.get("bookmaker") or "—",
                        "estado": status,
                    })
        audit_df = pd.DataFrame(audit_records)
        if not audit_df.empty:
            audit_df = audit_df.sort_values("ev_num", ascending=False).drop(columns=["ev_num"]).reset_index(drop=True)
        return {
            "total_matches": total_matches,
            "coverage_ok": coverage_ok,
            "odds_ok": odds_ok,
            "eligible_both": eligible_both,
            "raw_ev_positive": raw_ev_positive,
            "audit_df": audit_df,
        }
    except Exception as exc:
        logger.warning("Error en auditoría de embudo EV: %s", exc)
        return empty_res
    finally:
        con.close()


def load_probabilities(target_dates: list[date], only_impect: bool = True) -> pd.DataFrame:
    if not DUCKDB_PATH.exists() or not target_dates:
        return pd.DataFrame()
    try:
        con = get_db_con()
    except Exception as exc:
        logger.warning("No se pudo conectar a DuckDB para probabilidades: %s", exc)
        return pd.DataFrame()
    try:
        date_values = sorted({day.isoformat() for day in target_dates})
        marks = ",".join("?" for _ in date_values)
        rows = con.execute(f"""SELECT run_id, run_date, started_at, summary
               FROM output.runs WHERE run_type = 'daily_portfolio'
                 AND run_date IN ({marks})
               QUALIFY row_number() OVER (PARTITION BY run_date
                                          ORDER BY started_at DESC, run_id DESC)=1
               ORDER BY run_date DESC""", date_values).fetchall()
        records = []
        for run_id, run_date, started_at, summary in rows:
            if isinstance(summary, str):
                summary = json.loads(summary)
            configured_jev_weight = (summary or {}).get("jev_weight_config")
            for match in (summary or {}).get("processed_matches", []):
                flags = list(match.get("data_coverage_flags") or [])
                r_h = match.get("home_rating")
                r_a = match.get("away_rating")

                # Inner filter: identificar si el partido tiene cobertura real de Impect
                has_impect = True
                if "liga_sin_ratings_impect" in flags or "equipos_sin_rating_impect" in flags:
                    has_impect = False
                elif r_h is not None and r_a is not None and round(float(r_h), 4) == 0.68 and round(float(r_a), 4) == 0.68:
                    has_impect = False

                if only_impect and not has_impect:
                    continue

                provider_context = match.get("provider_context") or {}
                signals = list(match.get("news_categories") or [])
                if provider_context.get("injuries"):
                    signals.append("bajas reportadas por API-Football")
                if provider_context.get("lineups_available"):
                    signals.append("alineación oficial API-Football")

                p_poiss = match.get("P_poisson") or {}
                p_tact = match.get("P_tactical") or {}
                p_rate = match.get("P_ratings_base") or {}
                lh = match.get("lambda_home")
                la = match.get("lambda_away")
                poiss_avail = match.get("poisson_available", False)

                records.append({
                    "fecha": run_date,
                    "liga": match.get("league_name") or "",
                    "local": match.get("home_team"),
                    "visitante": match.get("away_team"),
                    "corrida": int(hashlib.sha256(str(run_date).encode("utf-8")).hexdigest()[:12], 16),
                    "ejecutada": str(started_at)[:16],
                    "P(local)": match.get("P_H"), "P(empate)": match.get("P_D"),
                    "P(visita)": match.get("P_A"),
                    "origen": "JEV + core" if float(match.get("jev_weight") or 0) > 0 else "Core (Impect V5)",
                    "peso JEV": float(match.get("jev_weight") or 0),
                    "peso JEV config": configured_jev_weight,
                    "señales": ", ".join(signals),
                    "motivo": match.get("jev_reason", ""),
                    "evidencias": match.get("news_evidence") or [],
                    "forma": match.get("form_context") or {},
                    "proveedor": provider_context,
                    "has_impect": has_impect,
                    "rating_local": r_h,
                    "rating_visita": r_a,
                    "poisson_disponible": poiss_avail,
                    "lambda_local": lh,
                    "lambda_visita": la,
                    "P_poisson": p_poiss,
                    "P_tactical": p_tact,
                    "P_ratings_base": p_rate,
                })
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = frame.drop_duplicates(
                subset=["fecha", "local", "visitante"], keep="first"
            ).reset_index(drop=True)
        return frame
    except Exception as exc:
        logger.warning("Error consultando probabilidades en DuckDB: %s", exc)
        return pd.DataFrame()
    finally:
        con.close()


def load_black_swan_runs(target_dates: list[date]) -> list[dict[str, Any]]:
    """Carga ejecuciones registradas de Black Swan desde DuckDB para las fechas dadas."""
    if not DUCKDB_PATH.exists() or not target_dates:
        return []
    try:
        con = get_db_con()
    except Exception as exc:
        logger.warning("No se pudo conectar a DuckDB para Black Swan: %s", exc)
        return []
    try:
        date_values = sorted({day.isoformat() for day in target_dates})
        marks = ",".join("?" for _ in date_values)
        rows = con.execute(f"""
            SELECT run_id, run_type, run_date, started_at, summary
            FROM output.runs
            WHERE run_type IN ('black_swan_v2', 'black_swan')
              AND run_date IN ({marks})
            ORDER BY started_at DESC, run_id DESC
        """, date_values).fetchall()

        results = []
        for run_id, r_type, r_date, started_at, summary in rows:
            if isinstance(summary, str):
                try:
                    summary = json.loads(summary)
                except Exception:
                    summary = {}
            legs_df = con.execute("""
                SELECT row_number, event_date AS fecha, home_team AS local, away_team AS visitante,
                       outcome AS seleccion, probability AS probabilidad, odds AS cuota, ev, stake, payload
                FROM output.recommendations
                WHERE run_id = ?
                ORDER BY row_number ASC
            """, [run_id]).fetchdf()
            results.append({
                "run_id": run_id,
                "run_type": r_type,
                "run_date": r_date,
                "started_at": started_at,
                "summary": summary or {},
                "legs": legs_df,
            })
        return results
    except Exception as exc:
        logger.warning("Error consultando Black Swan en DuckDB: %s", exc)
        return []
    finally:
        con.close()


def apply_reactive_staking(
    recommendations: pd.DataFrame,
    bankroll: float,
    stake_mode: str = "flat",
    flat_stake_pct: float = 0.03,
    max_stake_pct: float = 0.03,
    kelly_factor: float = 0.10,
    min_stake_amount: float = 1.0,
) -> pd.DataFrame:
    """Recalcula reactivamente el stake y % de banca para cada recomendación según los controles de la UI."""
    if recommendations.empty or bankroll <= 0:
        return recommendations

    df = recommendations.copy()
    new_stakes = []
    new_stake_pcts = []
    new_payouts = []

    for _, row in df.iterrows():
        try:
            cuota = float(row.get("cuota", 0.0) or 0.0)
            prob = float(row.get("probabilidad", 0.0) or 0.0)
        except (ValueError, TypeError):
            cuota, prob = 0.0, 0.0

        if cuota <= 1.0 or prob <= 0.0:
            new_stakes.append(0.0)
            new_stake_pcts.append(0.0)
            new_payouts.append(0.0)
            continue

        ev = (prob * cuota) - 1.0
        if stake_mode == "flat":
            f_stake = min(flat_stake_pct, max_stake_pct)
        else:
            # Kelly fraccionario: f* = ev / (cuota - 1.0)
            f_star = (ev / (cuota - 1.0)) if ev > 0.0 else 0.0
            f_frac = f_star * kelly_factor
            f_stake = min(f_frac, max_stake_pct)

        stake_val = round(f_stake * bankroll, 2)
        if min_stake_amount > 0.0 and stake_val < min_stake_amount:
            stake_val = 0.0
        new_stakes.append(stake_val)
        new_stake_pcts.append((stake_val / bankroll * 100.0) if bankroll > 0 else 0.0)
        new_payouts.append(round(stake_val * cuota, 2))

    df["stake"] = new_stakes
    df["stake_pct"] = new_stake_pcts
    df["retorno_potencial"] = new_payouts
    return df


def build_dynamic_black_swan(
    recommendations: pd.DataFrame,
    target_odds: float = 30.0,
    max_legs: int = 5,
    bankroll: float = 40000.0,
    micro_stake_pct: float = 0.0001,
    min_stake_amount: float = 1.0,
) -> dict[str, Any] | None:
    """Construye una combinada óptima Cisne Negro a partir de las recomendaciones disponibles."""
    if recommendations.empty:
        return None

    from collections import defaultdict
    import itertools

    by_match = defaultdict(list)
    for _, row in recommendations.iterrows():
        try:
            cuota = float(row.get("cuota", 0.0) or 0.0)
            prob = float(row.get("probabilidad", 0.0) or 0.0)
            ev_val = float(row.get("ev", 0.0) or 0.0)
        except (ValueError, TypeError):
            continue
        if cuota <= 1.0 or prob <= 0.0:
            continue
        m_key = f"{row.get('local', '')} vs {row.get('visitante', '')}"
        by_match[m_key].append({
            "fecha": row.get("fecha", ""),
            "local": row.get("local", ""),
            "visitante": row.get("visitante", ""),
            "seleccion": row.get("seleccion", ""),
            "probabilidad": prob,
            "cuota": cuota,
            "ev": ev_val,
        })

    keys = list(by_match.keys())
    if len(keys) < 2:
        return None

    min_acceptable_odds = target_odds * 0.65
    best = None
    best_score = -1e9

    for n in range(2, min(max_legs + 1, len(keys) + 1)):
        for combo_matches in itertools.combinations(keys, n):
            for combo in itertools.product(*[by_match[k] for k in combo_matches]):
                odds = 1.0
                prob = 1.0
                for leg in combo:
                    odds *= leg["cuota"]
                    prob *= leg["probabilidad"]

                if odds >= min_acceptable_odds:
                    distance_penalty = (abs(odds - target_odds) / target_odds) * 0.1
                    score = prob - distance_penalty
                    if score > best_score:
                        best_score = score
                        best = {
                            "legs": list(combo),
                            "total_odds": odds,
                            "total_probability": prob,
                            "n_legs": n,
                            "score": score,
                        }

    # Si ninguna alcanzó la cuota mínima, seleccionar la combinación con cuota más alta
    if not best:
        max_odds_found = 0.0
        for n in range(2, min(max_legs + 1, len(keys) + 1)):
            for combo_matches in itertools.combinations(keys, n):
                for combo in itertools.product(*[by_match[k] for k in combo_matches]):
                    odds = 1.0
                    prob = 1.0
                    for leg in combo:
                        odds *= leg["cuota"]
                        prob *= leg["probabilidad"]
                    if odds > max_odds_found:
                        max_odds_found = odds
                        best = {
                            "legs": list(combo),
                            "total_odds": odds,
                            "total_probability": prob,
                            "n_legs": n,
                            "score": prob,
                        }

    if not best:
        return None

    p = best["total_probability"]
    odds = best["total_odds"]
    ev = (p * odds) - 1.0
    f_star = (ev / (odds - 1.0)) if (odds > 1.0 and ev > 0.0) else 0.0
    calculated_stake = max(f_star * 0.001 * bankroll, bankroll * micro_stake_pct)
    stake = max(round(calculated_stake, 2), float(min_stake_amount))

    best["stake"] = stake
    best["stake_pct"] = (stake / bankroll) * 100.0 if bankroll > 0 else 0.0
    best["ev"] = ev
    best["payout"] = round(stake * odds, 2)
    return best


def load_result_comparison() -> list[tuple]:
    """Lee filas de predicciones y resultados directamente de DuckDB."""
    if not DUCKDB_PATH.exists():
        return []
    try:
        con = get_db_con()
    except Exception as exc:
        logger.warning("No se pudo conectar a DuckDB para comparativa de resultados: %s", exc)
        return []
    try:
        return con.execute("""
            SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team,
                   r.outcome, r.probability, res.status, res.result_outcome,
                   res.source, res.bet_won
            FROM output.recommendations r
            LEFT JOIN output.results res ON res.run_id=r.run_id AND res.row_number=r.row_number
            LEFT JOIN output.runs run ON run.run_id=r.run_id
            QUALIFY row_number() OVER (
                PARTITION BY r.event_date, lower(trim(r.home_team)),
                             lower(trim(r.away_team)), r.outcome
                ORDER BY (res.status IN ('FT','AET','PEN','MANUAL')) DESC,
                         (res.result_outcome IS NOT NULL) DESC,
                         run.started_at DESC NULLS LAST, r.run_id DESC, r.row_number DESC
            ) = 1
            ORDER BY r.event_date DESC, r.home_team, r.away_team, r.row_number
        """).fetchall()
    except Exception as exc:
        logger.warning("Error consultando comparativa de resultados en DuckDB: %s", exc)
        return []
    finally:
        con.close()


def load_impect_results() -> pd.DataFrame:
    """Carga los marcadores importados de Impect."""
    if not DUCKDB_PATH.exists():
        return pd.DataFrame()
    try:
        con = get_db_con()
    except Exception as exc:
        logger.warning("No se pudo conectar a DuckDB para marcadores Impect: %s", exc)
        return pd.DataFrame()
    try:
        return con.execute("""
            SELECT iteration_id AS iteracion, match_id AS id_partido,
                   CAST(scheduled_at AS DATE) AS fecha, home_team AS local,
                   home_goals AS goles_local, away_goals AS goles_visitante,
                   away_team AS visitante, result_outcome AS resultado,
                   result_type AS tipo_resultado
            FROM output.impect_results
            ORDER BY scheduled_at DESC, iteration_id, match_id
        """).fetchdf()
    except Exception as exc:
        logger.warning("Error consultando marcadores Impect en DuckDB: %s", exc)
        return pd.DataFrame()
    finally:
        con.close()


def load_duckdb_kpis() -> dict[str, Any]:
    """Obtiene métricas agregadas globales para las tarjetas superiores."""
    fallback = {
        "total_bets": 0, "win_rate": 0.0, "pnl": 0.0, "roi": 0.0,
        "history_matches": 0, "impect_matches": 0, "total_matches": 0,
        "locked": False,
    }
    if not DUCKDB_PATH.exists():
        return fallback

    try:
        con = get_db_con()
    except Exception as exc:
        logger.warning("No se pudo conectar a DuckDB para KPIs: %s", exc)
        fallback["locked"] = True
        return fallback

    try:
        row = con.execute("""
            SELECT 
                count(*) as total,
                avg(CASE WHEN res.bet_won = true THEN 1.0 ELSE 0.0 END) as win_rate,
                sum(CASE WHEN res.bet_won = true THEN r.stake * (r.odds - 1.0) ELSE -r.stake END) as pnl,
                sum(CASE WHEN res.bet_won = true THEN r.stake * (r.odds - 1.0) ELSE -r.stake END) / NULLIF(sum(r.stake), 0) as roi
            FROM output.recommendations r
            JOIN output.results res ON r.run_id = res.run_id AND r.row_number = res.row_number
            WHERE res.bet_won IS NOT NULL AND r.stake > 0
        """).fetchone()

        hist_count = con.execute("""
            SELECT count(*) FROM raw.match_history WHERE league != 'TEST-League'
        """).fetchone()[0]

        impect_count = con.execute("""
            SELECT count(*) FROM output.impect_results
        """).fetchone()[0]

        return {
            "total_bets": row[0] if row else 0,
            "win_rate": (row[1] * 100.0) if row and row[1] is not None else 0.0,
            "pnl": row[2] if row and row[2] is not None else 0.0,
            "roi": (row[3] * 100.0) if row and row[3] is not None else 0.0,
            "history_matches": hist_count,
            "impect_matches": impect_count,
            "total_matches": hist_count + impect_count,
            "locked": False,
        }
    except Exception as exc:
        logger.warning("Error consultando KPIs en DuckDB: %s", exc)
        fallback["locked"] = True
        return fallback
    finally:
        con.close()


@st.cache_data(ttl=180)
def load_benchmark_report_df() -> pd.DataFrame:
    """Carga y parsea la tabla comparativa de modelos del reporte Markdown."""
    report_file = ROOT_DIR / "outputs" / "reports" / "benchmark_modelos_impect.md"
    if not report_file.exists():
        return pd.DataFrame()
    try:
        lines = report_file.read_text(encoding="utf-8").splitlines()
        tbl = [l for l in lines if l.strip().startswith("|")]
        if not tbl:
            return pd.DataFrame()
        import io
        df = pd.read_csv(io.StringIO("\n".join(tbl)), sep="|", skipinitialspace=True).dropna(how="all", axis=1)
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"\*\*", "", regex=True).str.strip()
        df = df[df["Modelo"] != "---"].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_tactical_dataframe(iteration_id: int) -> pd.DataFrame:
    """Extrae métricas tácticas consolidadas por equipo para una iteración de Impect."""
    try:
        from apostador.predictor import SoccerPredictor
        p = SoccerPredictor()
        averages = p.load_tactical_averages(iteration_id)
        scores = p.load_tactical_scores(iteration_id)
        talent = p.load_player_talent(iteration_id)
        ratings = p.load_ratings(iteration_id)

        rows = []
        for team_norm, avg in averages.items():
            sc = scores.get(team_norm, {})
            tal = talent.get(team_norm, {})
            r_val = ratings.get(team_norm)
            if r_val is None:
                for r_name, val in ratings.items():
                    if team_norm in r_name.lower() or r_name.lower() in team_norm:
                        r_val = val
                        break

            xg_val = float(avg.get("SHOT_XG") or avg.get("EXPECTED_GOALS") or 0.0)
            xga_val = float(avg.get("CONCEDED_SHOT_XG") or 0.0)
            pack_val = float(avg.get("BYPASSED_DEFENDERS") or 0.0)
            tilt_val = float(sc.get("FIELD_TILT") or 0.50)
            tal_pack = float(tal.get("top11_packing") or 0.0)
            tal_pxt = float(tal.get("top11_pxt") or 0.0)

            rows.append({
                "Equipo": team_norm.title(),
                "Rating Impect": f"{r_val:.3f}" if r_val is not None else "0.680",
                "xG a Favor": round(xg_val, 2),
                "xGA Concedido": round(xga_val, 2),
                "Packing Defensores": round(pack_val, 1),
                "Field Tilt (%)": f"{tilt_val:.1%}",
                "Talento Top 11": round(tal_pack, 2),
                "Talento PxT": round(tal_pxt, 3),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="xG a Favor", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def get_cached_backtest():
    """Ejecuta el backtester y cachea el resultado en memoria."""
    backtester = HistoricalOddsBacktester()
    return backtester.run_backtest()


@st.cache_data(ttl=300)
def get_cached_elo_match(h: str, a: str, d_str: str, advantage: float) -> dict[str, Any]:
    """Calcula ratings Elo y probabilidades 1X2 para el enfrentamiento."""
    loader = ClubEloLoader()
    h_elo = loader.get_team_elo(h, d_str)
    a_elo = loader.get_team_elo(a, d_str)
    probs = loader.calculate_probabilities(h_elo, a_elo, home_advantage=advantage)
    return {
        "home_team": h,
        "away_team": a,
        "match_date": d_str,
        "home_elo": round(h_elo, 1),
        "away_elo": round(a_elo, 1),
        "elo_diff": round(h_elo - a_elo, 1),
        "prob_home": round(probs["home_win"], 4),
        "prob_draw": round(probs["draw"], 4),
        "prob_away": round(probs["away_win"], 4),
    }


def main() -> None:
    page_icon = str(ICON_PATH) if ICON_PATH.exists() else "⚽"
    st.set_page_config(
        page_title="App-ostador Empedernido",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        if ICON_PATH.exists():
            st.image(str(ICON_PATH), width=100)
        st.markdown("### ⚙️ Parámetros de Operación")

        bankroll = st.number_input("Bankroll ($)", min_value=100.0, value=138000.0, step=10000.0)
        start_date = st.date_input("Fecha inicial", value=date.today())
        date_options = [start_date + timedelta(days=offset) for offset in range(14)]
        selected_dates = st.multiselect(
            "Fechas de partidos", options=date_options, default=[start_date],
            format_func=lambda day: day.strftime("%Y-%m-%d (%A)"),
            help="Selecciona los días a incluir en el análisis diario.",
        )
        selected_dates = sorted(set(selected_dates))
        if not selected_dates:
            st.warning("Selecciona al menos una fecha.")

        st.markdown("### 🎯 Estrategia & Riesgo")
        strategy_mode_label = st.selectbox(
            "Perfil de Estrategia",
            options=[
                "⭐ Multi-Mercado Calibrado (1X2 Equilibrado - ROI +14.9%)",
                "💎 Valor Calibrado (EV Sanidad ≤ 35%)",
                "🎯 Híbrida Optimizada (Locales Medios + Favoritos)",
                "📈 Locales en Cuotas Medias (ROI +31%)",
                "🛡️ Favoritos de Alta Convicción (75% WR)",
                "🌐 No Restringida (Todo EV > umbral)",
            ],
            index=0,
            help="Multi-Mercado explota valor en Local, Empate y Visitante con deduplicación por partido, contracción bayesiana al consenso y filtros de divergencia.",
        )
        strategy_mode_map = {
            "⭐ Multi-Mercado Calibrado (1X2 Equilibrado - ROI +14.9%)": "multi_market",
            "💎 Valor Calibrado (EV Sanidad ≤ 35%)": "calibrated_value",
            "🎯 Híbrida Optimizada (Locales Medios + Favoritos)": "hybrid",
            "📈 Locales en Cuotas Medias (ROI +31%)": "value_local",
            "🛡️ Favoritos de Alta Convicción (75% WR)": "favorites",
            "🌐 No Restringida (Todo EV > umbral)": "unconstrained",
        }
        strategy_mode_code = strategy_mode_map[strategy_mode_label]

        stake_mode_label = st.radio(
            "Gestión de Capital (Staking)",
            options=["Fijo (Flat Stake)", "Kelly Fraccionario"],
            index=0,
            help="Flat Stake previene drawdowns severos fijando un % estricto de la banca por apuesta.",
        )
        stake_mode_code = "flat" if "Flat" in stake_mode_label else "kelly"

        c_st1, c_st2 = st.columns(2)
        with c_st1:
            if stake_mode_code == "flat":
                flat_stake_pct = st.slider("Stake Fijo (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.5, help="% del bankroll por apuesta")
                max_stake_pct = flat_stake_pct
            else:
                flat_stake_pct = 3.0
                max_stake_pct = st.slider("Tope Máx (%)", min_value=1.0, max_value=15.0, value=3.0, step=0.5, help="Hard Cap máximo por apuesta")
        with c_st2:
            min_ev_pct = st.slider("EV Mínimo (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.5, help="Filtrar apuestas con EV menor a este valor")

        min_stake_val = st.number_input(
            "Stake Mínimo ($)",
            min_value=0.0,
            value=1.0,
            step=1.0,
            help="Descartar apuestas individuales cuyo stake sea inferior a este umbral (evita micro-apuestas inviables en casas de apuestas)",
        )

        kelly = st.select_slider("Factor Kelly", options=[0.05, 0.1, 0.25, 0.5, 1.0], value=0.10 if stake_mode_code == "kelly" else 0.25)
        jev_weight = st.slider("Peso JEV (0 = Core, 1 = JEV)", min_value=0.0, max_value=1.0, value=0.50, step=0.05, help="0.50 balancea el valor empírico óptimo descubierto en backtesting")
        bookmaker = st.text_input("Casa preferida", value="Pinnacle")
        odds_mode = st.selectbox(
            "Modo de Cuotas",
            options=["Automático (caché + API)", "Solo caché local", "Actualizar desde API"],
        )

        st.markdown("---")
        run_clicked = st.button("🚀 Ejecutar Análisis", type="primary", width="stretch", disabled=not selected_dates)
        update_stakes_clicked = st.button(
            "🔄 Recalcular Montos Guardados", width="stretch", disabled=not selected_dates,
            help="Actualiza los stakes con el nuevo bankroll o Kelly sin llamar a APIs ni crear otra predicción.",
        )

    # Hero Banner
    col_hero1, col_hero2 = st.columns([0.7, 0.3])
    with col_hero1:
        st.markdown(
            """
            <div class="hero-banner">
                <div class="hero-left">
                    <div>
                        <h1 class="hero-title">App-ostador Empedernido</h1>
                        <p class="hero-subtitle">Sistema Cuantitativo de Estimación de Probabilidades, Valor Esperado (EV) y Gestión de Capital</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # KPIs Globales
    kpis = load_duckdb_kpis()
    if kpis.get("locked"):
        st.warning("⚠️ DuckDB está sincronizando datos con otro proceso en segundo plano. Los KPIs se actualizarán al completarse la operación.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Apuestas Validadas</div>
                <div class="metric-value">{kpis['total_bets']}</div>
                <div class="metric-subtext">Histórico resuelto</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tasa de Acierto</div>
                <div class="metric-value">{kpis['win_rate']:.1f}%</div>
                <div class="metric-subtext">Cuota media ~2.85</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        pnl_color = "#10B981" if kpis['pnl'] >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Beneficio Neto (P&L)</div>
                <div class="metric-value" style="color: {pnl_color};">${kpis['pnl']:,.0f}</div>
                <div class="metric-subtext">Yield real: <b>+{kpis['roi']:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Partidos en DuckDB</div>
                <div class="metric-value">{kpis.get('total_matches', kpis['history_matches']):,}</div>
                <div class="metric-subtext">{kpis.get('impect_matches', 0):,} Impect V5 + {kpis['history_matches']:,} Históricos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Lógica de Ejecución y Recálculo
    if update_stakes_clicked:
        try:
            end_stake_date = max(selected_dates) if selected_dates else start_date
            update_summary = get_store().recalculate_saved_stakes(
                min(selected_dates).isoformat() if selected_dates else start_date.isoformat(),
                end_stake_date.isoformat(),
                float(bankroll),
                float(kelly),
                target_dates=[day.isoformat() for day in selected_dates] or None,
                stake_mode=stake_mode_code,
                flat_stake_pct=flat_stake_pct / 100.0,
                max_stake_pct=max_stake_pct / 100.0,
                strategy_mode=strategy_mode_code,
                min_ev=min_ev_pct / 100.0,
                min_stake_amount=float(min_stake_val),
            )
            if update_summary["dates"]:
                st.success(
                    f"Monto actualizado para {len(update_summary['dates'])} fecha(s) y "
                    f"{update_summary['recommendations_updated']} recomendación(es) bajo estrategia '{strategy_mode_label}'."
                )
            else:
                st.info("No hay corridas guardadas para las fechas seleccionadas.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al recalcular stakes: {exc}")

    if run_clicked:
        command = [
            sys.executable, "-m", "src.commands.main",
            "--date", min(selected_dates).isoformat(), "--target-dates",
            *[day.isoformat() for day in selected_dates],
            "--bankroll", str(bankroll),
            "--strategy-mode", strategy_mode_code,
            "--stake-mode", stake_mode_code,
            "--flat-stake-pct", str(flat_stake_pct / 100.0),
            "--max-stake-pct", str(max_stake_pct / 100.0),
            "--min-ev", str(min_ev_pct / 100.0),
            "--min-stake", str(min_stake_val),
            "--kelly-factor", str(kelly),
            "--jev-weight", str(jev_weight),
            "--bookmaker", bookmaker,
        ]
        if odds_mode == "Solo caché local":
            command.extend(["--no-refresh-odds", "--no-refresh-af-odds"])
        elif odds_mode == "Actualizar desde API":
            command.append("--refresh-odds")

        st.session_state["analysis_started"] = datetime.now(timezone.utc)
        with st.spinner("Ejecutando pipeline predictivo y optimización de cartera..."):
            try:
                result = subprocess.run(
                    command, cwd=ROOT_DIR, env=os.environ.copy(), text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900,
                )
                st.session_state["analysis_output"] = result.stdout
                st.session_state["analysis_success"] = result.returncode == 0
            except Exception as exc:  # noqa: BLE001
                st.session_state["analysis_output"] = str(exc)
                st.session_state["analysis_success"] = False

    if "analysis_success" in st.session_state:
        if st.session_state["analysis_success"]:
            st.success("Análisis completado y registrado exitosamente en DuckDB.")
        else:
            st.error("Ocurrió un error en la ejecución. Revisa el registro.")
        with st.expander("Registro de ejecución"):
            st.code(st.session_state.get("analysis_output", ""), language="text")

    # Pestañas principales
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Pronósticos & Portafolio",
        "🔬 Modelos & Benchmark Impect",
        "📈 Backtesting & ROI Real",
        "🏟️ Cuotas Históricas & Mercados",
        "⚡ ClubElo & Probabilidades 1X2",
        "🎯 Resultados & Conciliación",
        "🔌 Estado de APIs & Vigencia",
    ])

    # TAB 1: PRONÓSTICOS & PORTAFOLIO
    with tab1:
        col_hdr1, col_hdr2 = st.columns([0.70, 0.30])
        with col_hdr1:
            st.subheader("Portafolio Recomendado y Probabilidades")
            st.caption("Predicciones generadas por el Ensamble Multi-Capa Impect V5 (Poisson xG + Táctico Packing + Talento Top 11) con contracción al mercado y ajuste contextual de JEV.")
        with col_hdr2:
            solo_impect = st.checkbox(
                "Solo con cobertura Impect",
                value=True,
                key="chk_solo_impect",
                help="Filtra y oculta partidos de ligas sin ratings reales de Impect (ratings default 0.68)",
            )

        try:
            recommendations = load_recommendations(selected_dates)
            probabilities = load_probabilities(selected_dates, only_impect=solo_impect)
            saved_bs_runs = load_black_swan_runs(selected_dates)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Error cargando recomendaciones: {exc}")
            recommendations = pd.DataFrame()
            probabilities = pd.DataFrame()
            saved_bs_runs = []

        if not recommendations.empty:
            recommendations = apply_reactive_staking(
                recommendations,
                bankroll=float(bankroll),
                stake_mode=stake_mode_code,
                flat_stake_pct=flat_stake_pct / 100.0,
                max_stake_pct=max_stake_pct / 100.0,
                kelly_factor=float(kelly),
                min_stake_amount=float(min_stake_val),
            )

        if probabilities.empty and recommendations.empty and not saved_bs_runs:
            st.info("No hay datos para las fechas seleccionadas en DuckDB. Selecciona otra fecha o ejecuta el análisis.")
        else:
            if not probabilities.empty:
                st.markdown("#### Matriz de Probabilidades 1X2")
                if solo_impect:
                    st.caption("Mostrando únicamente partidos con ratings reales y cobertura de Impect.")
                prob_display = probabilities.copy()
                for col in ("P(local)", "P(empate)", "P(visita)"):
                    prob_display[col] = prob_display[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
                cols_to_show = ["fecha"]
                if "liga" in prob_display.columns:
                    cols_to_show.append("liga")
                cols_to_show.extend(["local", "visitante", "P(local)", "P(empate)", "P(visita)", "origen", "peso JEV", "señales"])
                st.dataframe(
                    prob_display[cols_to_show],
                    width="stretch", hide_index=True,
                )

                with st.expander("🔬 Desglose Multi-Capa Impect (Poisson xG + Táctico Packing + Ratings)"):
                    breakdown_list = []
                    for _, row in probabilities.iterrows():
                        poiss = row.get("P_poisson") or {}
                        tact = row.get("P_tactical") or {}
                        rate = row.get("P_ratings_base") or {}
                        lh = row.get("lambda_local")
                        la = row.get("lambda_visita")
                        breakdown_list.append({
                            "Fecha": row["fecha"],
                            "Partido": f"{row['local']} vs {row['visitante']}",
                            "xG Impect (L - V)": f"{lh:.2f} - {la:.2f}" if lh is not None and la is not None else "—",
                            "Poisson 1X2": f"{poiss.get('H', 0):.0%} / {poiss.get('D', 0):.0%} / {poiss.get('A', 0):.0%}" if poiss else "—",
                            "Táctico Packing 1X2": f"{tact.get('H', 0):.0%} / {tact.get('D', 0):.0%} / {tact.get('A', 0):.0%}" if tact else "—",
                            "Ratings Base 1X2": f"{rate.get('H', 0):.0%} / {rate.get('D', 0):.0%} / {rate.get('A', 0):.0%}" if rate else "—",
                            "Prob Final": f"{row['P(local)']} / {row['P(empate)']} / {row['P(visita)']}",
                        })
                    if breakdown_list:
                        st.dataframe(pd.DataFrame(breakdown_list), width="stretch", hide_index=True)
                    else:
                        st.caption("No hay desglose multi-capa registrado para este conjunto de partidos.")

                with st.expander("🔍 Detalle de Evidencias y Noticias JEV por Partido"):
                    ev_list = []
                    for _, row in probabilities.iterrows():
                        for ev in row.get("evidencias", []):
                            ev_list.append({
                                "Fecha": row["fecha"],
                                "Partido": f"{row['local']} vs {row['visitante']}",
                                "Titular": ev.get("title", ""),
                                "Categoría": ", ".join(ev.get("categories", [])),
                                "Fuente": ev.get("domain", ""),
                                "Certeza": ev.get("certainty", ""),
                            })
                    if ev_list:
                        st.dataframe(pd.DataFrame(ev_list), width="stretch", hide_index=True)
                    else:
                        st.caption("No hay noticias específicas registradas para este conjunto de partidos.")

            st.markdown("#### Apuestas con Valor Esperado Positivo ($EV > 0$)")
            if not recommendations.empty:
                # Métricas inmediatas de la cartera reactiva
                total_stake = float(recommendations["stake"].sum())
                total_pct = (total_stake / bankroll * 100.0) if bankroll > 0 else 0.0
                total_payout = float(recommendations["retorno_potencial"].sum()) if "retorno_potencial" in recommendations.columns else 0.0
                ganancia_neta = total_payout - total_stake

                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                kpi_col1.metric("Apuestas Sugeridas", f"{len(recommendations)}")
                kpi_col2.metric("Capital Comprometido", f"${total_stake:,.0f}", f"{total_pct:.2f}% de banca")
                kpi_col3.metric(
                    "Stake Promedio",
                    f"${recommendations['stake'].mean():,.0f}" if len(recommendations) > 0 else "$0",
                    stake_mode_label,
                )
                kpi_col4.metric("Retorno Potencial", f"${total_payout:,.0f}", f"+${ganancia_neta:,.0f} ganancia")

                if stake_mode_code == "flat":
                    st.caption(
                        f"⚡ **Dimensionamiento reactivo:** Montos calculados en tiempo real para un bankroll de **${bankroll:,.0f}** "
                        f"({flat_stake_pct:.1f}% fijo por apuesta). Para fijarlos en DuckDB, usa '🔄 Recalcular Montos Guardados'."
                    )
                else:
                    st.caption(
                        f"⚡ **Dimensionamiento reactivo:** Montos calculados en tiempo real para un bankroll de **${bankroll:,.0f}** "
                        f"(Kelly {kelly}x, tope máx {max_stake_pct:.1f}%). Para fijarlos en DuckDB, usa '🔄 Recalcular Montos Guardados'."
                    )

                rec_display = recommendations.copy()
                rec_display["probabilidad"] = rec_display["probabilidad"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
                rec_display["ev"] = rec_display["ev"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
                rec_display["cuota"] = rec_display["cuota"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
                rec_display["stake"] = rec_display["stake"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
                rec_display["% banca"] = rec_display["stake_pct"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
                if "divergencia_mercado" in rec_display.columns:
                    rec_display["divergencia_mercado"] = rec_display["divergencia_mercado"].map(
                        lambda v: f"{v:+.1%}" if pd.notna(v) else "—"
                    )
                cols_to_show = ["fecha", "local", "visitante", "seleccion"]
                if "nicho" in rec_display.columns:
                    cols_to_show.append("nicho")
                cols_to_show.extend(["probabilidad", "cuota", "ev", "stake", "% banca"])
                if "divergencia_mercado" in rec_display.columns:
                    cols_to_show.append("divergencia_mercado")
                st.dataframe(
                    rec_display[cols_to_show],
                    width="stretch", hide_index=True,
                )
            else:
                st.info(
                    "💡 **No hay apuestas con Valor Esperado Positivo ($EV > 0$) para las fechas seleccionadas.**\n\n"
                    "- Para los partidos de la fecha, o bien el mercado no ofreció cuotas con margen de valor ($EV \\le 0$), "
                    "o bien los encuentros (por ejemplo copas UEFA sin ratings base Impect o con historial de forma insuficiente) "
                    "fueron excluidos por el guard de cobertura para proteger el capital.\n"
                    "- Puedes consultar fechas anteriores con valor validado (ej. **2026-11-04**) en el selector de la barra lateral "
                    "o esperar a la publicación y actualización de nuevas cuotas de mercado."
                )

            funnel_info = load_ev_funnel_audit(selected_dates, min_ev_threshold=min_ev_pct / 100.0)
            if funnel_info["total_matches"] > 0:
                st.caption(
                    f"📊 **Embudo de Selección:** {funnel_info['total_matches']} partidos analizados → "
                    f"**{funnel_info['coverage_ok']}** con cobertura Impect + forma → "
                    f"**{funnel_info['eligible_both']}** con cobertura y cuotas activas → "
                    f"**{funnel_info['raw_ev_positive']}** opciones con $EV > 0\\%$ bruto → "
                    f"**{len(recommendations)}** recomendadas en portafolio."
                )
                with st.expander("🧮 Auditoría de Cálculo de EV ($EV = P \\times \\text{Cuota} - 1$) en Partidos Elegibles"):
                    if not funnel_info["audit_df"].empty:
                        st.dataframe(funnel_info["audit_df"], width="stretch", hide_index=True)
                    else:
                        st.caption("Ningún partido de la fecha seleccionada reúne simultáneamente cobertura Impect + forma y cuotas de mercado.")

            # SECCIÓN BLACK SWAN PARLAY (SUGERENCIA CISNE NEGRO)
            st.markdown("---")
            st.markdown("#### 🦢 Sugerencia Cisne Negro (Black Swan Parlay)")
            st.caption(
                "Combinada asimétrica de alta cuota y micro-stake (0.01%–0.1% de la banca). "
                "Maximiza la probabilidad conjunta de éxito combinando selecciones de valor ($EV > 0$) para alcanzar una cuota objetivo."
            )

            col_bs1, col_bs2, col_bs3 = st.columns(3)
            with col_bs1:
                target_odds_val = st.select_slider(
                    "Cuota Objetivo (~x)",
                    options=[10, 20, 30, 50, 70, 100, 300, 700],
                    value=30,
                    key="bs_target_odds_slider",
                    help="Cuota combinada aproximada que se busca construir.",
                )
            with col_bs2:
                max_legs_val = st.slider(
                    "Máximo de Piernas",
                    min_value=2, max_value=6, value=4,
                    key="bs_max_legs_slider",
                    help="Número máximo de partidos combinados en el ticket.",
                )
            with col_bs3:
                stake_pct_label = st.selectbox(
                    "Gestión de Micro-Stake",
                    options=["0.01% (Conservador)", "0.05% (Moderado)", "0.10% (Agresivo)"],
                    index=0,
                    key="bs_stake_pct_sel",
                )
                pct_map = {
                    "0.01% (Conservador)": 0.0001,
                    "0.05% (Moderado)": 0.0005,
                    "0.10% (Agresivo)": 0.0010,
                }
                micro_stake_pct = pct_map.get(stake_pct_label, 0.0001)

            mode_bs = "Dinámica"
            if saved_bs_runs:
                first_saved = saved_bs_runs[0]
                mode_bs = st.radio(
                    "Modo de visualización:",
                    options=[
                        "⚡ Sugerencia Dinámica (Optimizada al vuelo)",
                        f"💾 Corrida Guardada ({first_saved['run_type']} - {first_saved['run_id'][:24]}...)",
                    ],
                    horizontal=True,
                    key="bs_mode_selection",
                )

            if "Sugerencia Dinámica" in mode_bs or not saved_bs_runs:
                bs_data = build_dynamic_black_swan(
                    recommendations,
                    target_odds=float(target_odds_val),
                    max_legs=int(max_legs_val),
                    bankroll=float(bankroll),
                    micro_stake_pct=micro_stake_pct,
                    min_stake_amount=float(min_stake_val),
                )

                if bs_data is None:
                    st.info("💡 Se requieren al menos 2 partidos distintos con $EV > 0$ en la fecha seleccionada para construir una combinada Cisne Negro.")
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Cuota Combinada", f"{bs_data['total_odds']:.2f}x")
                    m2.metric("Probabilidad Conjunta", f"{bs_data['total_probability']:.2%}")
                    m3.metric("EV Combinado", f"{bs_data['ev']:+.1%}")
                    m4.metric("Micro-Stake", f"${bs_data['stake']:,.0f}", f"{bs_data['stake_pct']:.4f}% banco")
                    m5.metric("Retorno Potencial", f"${bs_data['payout']:,.0f}")

                    legs_df = pd.DataFrame(bs_data["legs"])
                    legs_display = legs_df.copy()
                    legs_display["probabilidad"] = legs_display["probabilidad"].map(lambda v: f"{v:.1%}")
                    legs_display["cuota"] = legs_display["cuota"].map(lambda v: f"{v:.2f}")
                    legs_display["ev"] = legs_display["ev"].map(lambda v: f"{v:+.2%}")
                    st.dataframe(
                        legs_display[["fecha", "local", "visitante", "seleccion", "cuota", "probabilidad", "ev"]],
                        width="stretch", hide_index=True,
                    )
            else:
                chosen_run = saved_bs_runs[0]
                summary = chosen_run["summary"]
                parlay_info = summary.get("parlay", {})
                stake_info = summary.get("stake", {})
                legs_df = chosen_run["legs"]

                tot_odds = float(parlay_info.get("total_odds", 0.0) or 0.0)
                tot_prob = float(parlay_info.get("total_probability", 0.0) or 0.0)
                stake_val = float(stake_info.get("stake", 4.0) or 4.0)
                ev_val = float(stake_info.get("ev", 0.0) or 0.0)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Cuota Combinada", f"{tot_odds:.2f}x")
                m2.metric("Probabilidad Conjunta", f"{tot_prob:.4%}")
                m3.metric("EV Combinado", f"{ev_val:+.1%}")
                m4.metric("Stake Guardado", f"${stake_val:,.0f}", f"{stake_info.get('stake_pct', 0.01):.4f}% banco")
                m5.metric("Retorno Potencial", f"${stake_val * tot_odds:,.0f}")

                if not legs_df.empty:
                    display_legs = legs_df.copy()
                    display_legs["probabilidad"] = display_legs["probabilidad"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
                    display_legs["cuota"] = display_legs["cuota"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
                    display_legs["ev"] = display_legs["ev"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
                    st.dataframe(
                        display_legs[["fecha", "local", "visitante", "seleccion", "cuota", "probabilidad", "ev"]],
                        width="stretch", hide_index=True,
                    )

            with st.expander("🚀 Ejecutar Black Swan v2 con Brave Search + JEV (CLI)"):
                st.caption("Consulta noticias frescas en tiempo real y evalúa riesgos cualitativos con JEV antes de guardar el ticket en DuckDB.")
                if st.button("Ejecutar Black Swan v2 ahora", key="btn_run_cli_black_swan"):
                    target_date_str = min(selected_dates).isoformat()
                    cli_cmd = [
                        sys.executable, "-m", "src.commands.black_swan_v2",
                        "--input", target_date_str,
                        "--target-odds", str(target_odds_val),
                        "--bankroll", str(bankroll),
                        "--max-legs", str(max_legs_val),
                    ]
                    with st.spinner("Ejecutando Black Swan v2 con noticias y JEV..."):
                        try:
                            res = subprocess.run(
                                cli_cmd, cwd=ROOT_DIR, env=os.environ.copy(), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
                            )
                            if res.returncode == 0:
                                st.success("Black Swan v2 ejecutado y guardado exitosamente en DuckDB.")
                                st.rerun()
                            else:
                                st.warning("Black Swan v2 finalizó con avisos:")
                                st.code(res.stdout, language="text")
                        except Exception as exc:
                            st.error(f"Error al ejecutar: {exc}")

    # TAB 2: MODELOS & BENCHMARK IMPECT
    with tab2:
        st.subheader("🔬 Benchmarking Cuantitativo de Modelos & Datos Impect V5")
        st.caption(
            "Comparación rigurosa de 5 arquitecturas predictivas sobre 3,640 partidos históricos "
            "con cuotas reales de cierre y métricas avanzadas (xG Dixon-Coles, Packing y Talento Top 11)."
        )

        # Tarjetas de Highlights del Benchmark
        bm_c1, bm_c2, bm_c3, bm_c4 = st.columns(4)
        with bm_c1:
            st.metric("Mejor Log Loss", "0.9992", "Modelo Talento Top 11")
        with bm_c2:
            st.metric("Mejor Brier Score", "0.1988", "Calibración superior")
        with bm_c3:
            st.metric("Máximo ROI Kelly", "+14.88%", "Táctico Multi-KPI (Cap 2.5%)")
        with bm_c4:
            st.metric("Control de Drawdown", "14.4%", "-77% vs modelo previo")

        # 1. Tabla de Benchmark Completo
        st.markdown("---")
        st.markdown("#### 🏆 Tabla Comparativa Multimodelo (3,640 Partidos)")
        df_bench = load_benchmark_report_df()
        if not df_bench.empty:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                available_models = ["Todos"] + sorted(list(df_bench["Modelo"].unique()))
                filter_model = st.selectbox("Filtrar por Modelo", options=available_models, key="bm_filter_model")
            with col_b2:
                available_strats = ["Todas"] + sorted(list(df_bench["Estrategia"].unique()))
                filter_strat = st.selectbox("Filtrar por Estrategia", options=available_strats, key="bm_filter_strat")

            df_bench_filtered = df_bench.copy()
            if filter_model != "Todos":
                df_bench_filtered = df_bench_filtered[df_bench_filtered["Modelo"] == filter_model]
            if filter_strat != "Todas":
                df_bench_filtered = df_bench_filtered[df_bench_filtered["Estrategia"] == filter_strat]

            st.dataframe(df_bench_filtered, width="stretch", hide_index=True)
        else:
            st.info("Reporte de benchmark no encontrado. Puedes generarlo ejecutando el backtest abajo.")

        # 2. Explorador Táctico Impect
        st.markdown("---")
        st.markdown("#### ⚽ Explorador Táctico de Equipos (Packing, xG y Field Tilt)")
        st.caption("Métricas profundas de Impect API V5 acumuladas por equipo en la temporada 2026.")

        leagues_list = [
            ("LaLiga (España)", 2110),
            ("Serie A (Italia)", 2111),
            ("Liga MX (México)", 2176),
            ("Primera A (Colombia)", 1752),
            ("Série A (Brasil)", 1799),
            ("Major League Soccer (USA)", 1864),
            ("Liga Profesional (Argentina)", 1930),
            ("Jupiler Pro League (Bélgica)", 2143),
            ("Premier Liga (Rusia)", 2155),
            ("Primera División (Chile)", 1802),
        ]
        col_t1, col_t2 = st.columns([0.4, 0.6])
        with col_t1:
            sel_league_tuple = st.selectbox(
                "Seleccionar Competición Impect",
                options=leagues_list,
                format_func=lambda x: x[0],
                key="tactical_league_sel",
            )
        iter_id = sel_league_tuple[1]

        df_tactical = load_tactical_dataframe(iter_id)
        if not df_tactical.empty:
            st.dataframe(df_tactical, width="stretch", hide_index=True)
        else:
            st.info(f"No hay métricas tácticas cacheadas para la iteración {iter_id}.")

        # 3. Acciones de Re-Backfill y Benchmark
        st.markdown("---")
        st.markdown("#### ⚙️ Acciones Cuantitativas y Actualización de Modelos")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🚀 Re-ejecutar Benchmark Completo (3,640 partidos)", key="btn_run_benchmark", width="stretch"):
                with st.spinner("Ejecutando backtest comparativo sobre 3,640 partidos..."):
                    cmd = [sys.executable, "-m", "src.analysis.comprehensive_model_backtest"]
                    res = subprocess.run(cmd, cwd=ROOT_DIR, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
                    if res.returncode == 0:
                        st.success("Benchmark actualizado con éxito.")
                        st.rerun()
                    else:
                        st.error("Error en benchmark:")
                        st.code(res.stdout, language="text")
        with col_act2:
            if st.button("📥 Re-ejecutar Backfill Impect (17 Ligas)", key="btn_run_backfill", width="stretch"):
                with st.spinner("Descargando métricas y marcadores de Impect API..."):
                    cmd = [sys.executable, "-m", "src.commands.backfill_impect_data"]
                    res = subprocess.run(cmd, cwd=ROOT_DIR, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
                    if res.returncode == 0:
                        st.success("Backfill de Impect completado con éxito.")
                        st.rerun()
                    else:
                        st.error("Error en backfill:")
                        st.code(res.stdout, language="text")

    # TAB 3: BACKTESTING FINANCIERO & ROI REAL
    with tab3:
        st.subheader("Auditoría de Rendimiento Financiero con Cuotas Reales")
        st.caption("Backtesting out-of-sample que cruza el histórico de recomendaciones con cuotas reales de cierre de Pinnacle y Bet365.")

        backtest_data = get_cached_backtest()
        if "error" in backtest_data:
            st.warning(backtest_data["error"])
        else:
            gen = backtest_data["resumen_general"]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Yield / ROI Real", f"{gen['roi_real_pct']}%", delta=f"{gen['roi_real_pct']}%")
            b2.metric("Beneficio Neto", f"${gen['beneficio_neto']:,.2f}")
            b3.metric("Tasa de Acierto", f"{gen['win_rate_pct']}%", f"{gen['aciertos']}W - {gen['fallos']}L")
            b4.metric("Máximo Drawdown", f"-{gen['max_drawdown_pct']}%", delta_color="inverse")

            col_bt_left, col_bt_right = st.columns(2)
            with col_bt_left:
                st.markdown("##### Rendimiento por Rango de Cuotas")
                df_buckets = pd.DataFrame(backtest_data["por_rango_cuotas"])
                if not df_buckets.empty:
                    df_buckets["staked"] = df_buckets["staked"].map(lambda v: f"${v:,.0f}")
                    df_buckets["pnl"] = df_buckets["pnl"].map(lambda v: f"${v:,.0f}")
                    df_buckets["roi_pct"] = df_buckets["roi_pct"].map(lambda v: f"{v:+.2f}%")
                    df_buckets["win_rate_pct"] = df_buckets["win_rate_pct"].map(lambda v: f"{v:.1f}%")
                    st.dataframe(df_buckets, width="stretch", hide_index=True)

            with col_bt_right:
                st.markdown("##### Rendimiento por Selección (1X2)")
                df_outcomes = pd.DataFrame(backtest_data["por_seleccion"])
                if not df_outcomes.empty:
                    df_outcomes["staked"] = df_outcomes["staked"].map(lambda v: f"${v:,.0f}")
                    df_outcomes["pnl"] = df_outcomes["pnl"].map(lambda v: f"${v:,.0f}")
                    df_outcomes["roi_pct"] = df_outcomes["roi_pct"].map(lambda v: f"{v:+.2f}%")
                    df_outcomes["win_rate_pct"] = df_outcomes["win_rate_pct"].map(lambda v: f"{v:.1f}%")
                    st.dataframe(df_outcomes, width="stretch", hide_index=True)

            st.markdown("##### Sensibilidad al Filtro de Valor Esperado ($EV$)")
            df_ev = pd.DataFrame(backtest_data["por_umbral_ev"])
            if not df_ev.empty:
                df_ev["staked"] = df_ev["staked"].map(lambda v: f"${v:,.0f}")
                df_ev["pnl"] = df_ev["pnl"].map(lambda v: f"${v:,.0f}")
                df_ev["roi_pct"] = df_ev["roi_pct"].map(lambda v: f"{v:+.2f}%")
                df_ev["win_rate_pct"] = df_ev["win_rate_pct"].map(lambda v: f"{v:.1f}%")
                st.dataframe(df_ev, width="stretch", hide_index=True)

            st.info(r"💡 **Conclusión:** Las apuestas a **Victoria Local** con cuotas moderadas (1.80 a 2.50) y $EV \ge 10\%$ entregan la curva de capital más sólida, mientras que los empates generan alta dispersión negativa.")

    # TAB 4: CUOTAS HISTÓRICAS & MERCADOS
    with tab4:
        st.subheader("Explorador de Partidos y Cuotas de Cierre")
        st.caption("Base relacional de cuotas históricas ingeridas vía SoccerData (`raw.match_history` en DuckDB).")

        try:
            con = get_db_con()
            try:
                available_leagues = [
                    r[0] for r in con.execute(
                        "SELECT DISTINCT league FROM raw.match_history WHERE league != 'TEST-League' ORDER BY league"
                    ).fetchall()
                ]
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    sel_league = st.selectbox("Liga", options=["Todas"] + available_leagues)
                with col_f2:
                    sel_season = st.selectbox("Temporada", options=["Todas", "2425", "2324"])

                query = "SELECT match_date::DATE AS fecha, league AS liga, home_team AS local, away_team AS visitante, fthg AS g_loc, ftag AS g_vis, ftr AS res, pinnacle_home AS pin_loc, pinnacle_draw AS pin_emp, pinnacle_away AS pin_vis, b365_home AS b365_loc, b365_draw AS b365_emp, b365_away AS b365_vis FROM raw.match_history WHERE league != 'TEST-League'"
                params = []
                if sel_league != "Todas":
                    query += " AND league = ?"
                    params.append(sel_league)
                if sel_season != "Todas":
                    query += " AND season = ?"
                    params.append(sel_season)
                query += " ORDER BY match_date DESC LIMIT 200"

                df_matches = con.execute(query, params).fetchdf()
                st.dataframe(df_matches, width="stretch", hide_index=True)
                st.caption(f"Mostrando los últimos {len(df_matches)} partidos encontrados.")
            finally:
                con.close()
        except Exception as exc:
            st.warning(f"No se pudieron cargar las cuotas históricas en este momento: {exc}")

    # TAB 5: CLUBELO & PROBABILIDADES 1X2
    with tab5:
        st.subheader("Calculadora de Ratings Elo Point-in-Time")
        st.caption("Calcula probabilidades teóricas 1X2 basadas en ratings Elo independientes y ventaja de campo.")

        c_elo1, c_elo2, c_elo3 = st.columns(3)
        with c_elo1:
            team_a = st.text_input("Equipo Local", value="Real Madrid")
        with c_elo2:
            team_b = st.text_input("Equipo Visitante", value="Barcelona")
        with c_elo3:
            adv = st.number_input("Ventaja de Local (Elo pts)", min_value=0.0, max_value=120.0, value=65.0, step=5.0)
        elo_info = get_cached_elo_match(team_a, team_b, date.today().strftime("%Y-%m-%d"), float(adv))

        st.markdown("##### Distribución Estimada")
        p_loc, p_emp, p_vis = st.columns(3)
        p_loc.metric(f"Victoria {team_a}", f"{elo_info['prob_home']:.1%}", f"Elo: {elo_info['home_elo']:.0f}")
        p_emp.metric("Empate", f"{elo_info['prob_draw']:.1%}", "Equilibrio")
        p_vis.metric(f"Victoria {team_b}", f"{elo_info['prob_away']:.1%}", f"Elo: {elo_info['away_elo']:.0f}")

        if elo_info.get("home_elo") == 1500.0 and elo_info.get("away_elo") == 1500.0:
            st.warning(
                "⚠️ **Aviso de contingencia ClubElo:** La API pública `api.clubelo.com` no está disponible en este momento "
                "(HTTP 502 Bad Gateway / origen caído). Se están utilizando ratings neutrales por defecto (Elo 1500).\n\n"
                "💡 **Importante:** Esto **no** afecta las predicciones ni la selección de cartera en el Tab 1, las cuales operan con "
                "el Ensamble Multi-Capa Impect V5 + Poisson xG + JEV de manera totalmente independiente a ClubElo."
            )

        st.caption("Formulación logística con decalaje paramétrico de empates conforme al modelo ClubElo de fútbol europeo.")

    # TAB 6: RESULTADOS & CONCILIACIÓN
    with tab6:
        st.subheader("Auditoría de Resultados Reales")
        st.caption("Verifica o introduce marcadores reales para alimentar la retroalimentación de la cartera.")

        # Acciones de sincronización y subproceso
        col_f1, col_f2, col_f3 = st.columns([0.25, 0.40, 0.35])
        with col_f1:
            fill_results_clicked = st.button(
                "⚡ Fill results",
                type="primary",
                key="btn_fill_results",
                help="Ejecuta el subproceso para consultar marcadores reales de partidos en las APIs y conciliar automáticamente las recomendaciones.",
            )
        with col_f2:
            fill_provider = st.selectbox(
                "Fuente de marcadores",
                options=["API-Football (En vivo / Finales)", "Impect (Marcadores Oficiales)", "Ambos Proveedores"],
                index=0,
                key="fill_provider_sel",
            )
        with col_f3:
            fill_scope = st.selectbox(
                "Alcance de verificación",
                options=["Todas las pendientes", "Solo fechas seleccionadas en barra lateral"],
                index=0,
                key="fill_scope_sel",
            )

        if fill_results_clicked:
            outputs = []
            success = True
            with st.spinner("Ejecutando subproceso para llenar y conciliar resultados..."):
                if fill_provider in ["API-Football (En vivo / Finales)", "Ambos Proveedores"]:
                    cmd_af = [sys.executable, "-m", "src.commands.fetch_results"]
                    if fill_scope == "Solo fechas seleccionadas en barra lateral" and selected_dates:
                        for d in selected_dates:
                            cmd_af.extend(["--date", d.isoformat()])
                    try:
                        res_af = subprocess.run(
                            cmd_af, cwd=ROOT_DIR, env=os.environ.copy(), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
                        )
                        outputs.append(f"=== API-Football ===\n{res_af.stdout}")
                        if res_af.returncode != 0:
                            success = False
                    except Exception as exc:  # noqa: BLE001
                        outputs.append(f"Error ejecutando fetch_results: {exc}")
                        success = False

                if fill_provider in ["Impect (Marcadores Oficiales)", "Ambos Proveedores"]:
                    cmd_imp = [sys.executable, "-m", "src.commands.fetch_impect_results"]
                    try:
                        res_imp = subprocess.run(
                            cmd_imp, cwd=ROOT_DIR, env=os.environ.copy(), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
                        )
                        outputs.append(f"=== Impect ===\n{res_imp.stdout}")
                        if res_imp.returncode != 0:
                            success = False
                    except Exception as exc:  # noqa: BLE001
                        outputs.append(f"Error ejecutando fetch_impect_results: {exc}")
                        success = False

            st.session_state["fill_results_output"] = "\n\n".join(outputs)
            st.session_state["fill_results_success"] = success
            st.rerun()

        if "fill_results_success" in st.session_state:
            if st.session_state["fill_results_success"]:
                st.success("¡Resultados llenados y conciliados exitosamente en DuckDB!")
            else:
                st.warning("El subproceso de conciliación finalizó con observaciones.")
            with st.expander("Registro de ejecución de 'Fill results'"):
                st.code(st.session_state.get("fill_results_output", ""), language="text")
            del st.session_state["fill_results_success"]

        st.markdown("---")

        try:
            result_rows = load_result_comparison()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Error cargando resultados: {exc}")
            result_rows = []

        if not result_rows:
            st.info("No hay resultados pendientes de revisión.")
        else:
            outcome_labels = {"local": "Gana local", "empate": "Empate", "visita": "Gana visitante"}
            label_to_outcome = {value: key for key, value in outcome_labels.items()}

            def _acierto(row: tuple) -> str:
                bet_won = row[10]
                if bet_won is None:
                    bet_won = compute_bet_won(row[5], row[8], row[3], row[4])
                return ("Sí" if bet_won else "No") if bet_won is not None else "—"

            display = pd.DataFrame([
                {
                    "run_id": row[0], "row_number": row[1], "Fecha": row[2] or "", "Local": row[3] or "",
                    "Visitante": row[4] or "", "Selección": row[5] or "",
                    "Prob. predicha": f"{row[6]:.1%}" if pd.notna(row[6]) else "",
                    "Resultado real": outcome_labels.get(row[8], "Pendiente"),
                    "Acierto": _acierto(row),
                    "Fuente": row[9] or "—",
                }
                for row in result_rows
            ])

            won_count = sum(1 for r in result_rows if r[10] is True)
            lost_count = sum(1 for r in result_rows if r[10] is False)
            pending_count = sum(1 for r in result_rows if r[10] is None)

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Total Recomendaciones", f"{len(result_rows)}")
            win_rate_str = f"{won_count / (won_count + lost_count):.1%}" if (won_count + lost_count) > 0 else "0.0%"
            m_c2.metric("Acertadas", f"{won_count}", f"{win_rate_str} acierto")
            m_c3.metric("Falladas", f"{lost_count}")
            m_c4.metric("Pendientes", f"{pending_count}")

            edited = st.data_editor(
                display,
                column_config={
                    "run_id": None,
                    "row_number": None,
                    "Resultado real": st.column_config.SelectboxColumn(
                        options=["Pendiente", "Gana local", "Empate", "Gana visitante"], required=True,
                    ),
                    "Acierto": st.column_config.TextColumn(disabled=True),
                    "Fuente": st.column_config.TextColumn(disabled=True),
                },
                disabled=["Fecha", "Local", "Visitante", "Selección", "Prob. predicha", "Acierto", "Fuente"],
                width="stretch", hide_index=True, key="result_comparison_editor",
            )
            if st.button("💾 Guardar Resultados Editados", type="primary", key="save_results_table"):
                store = get_store()
                saved = 0
                for original, updated in zip(display.to_dict("records"), edited.to_dict("records")):
                    selected = updated["Resultado real"]
                    if selected == "Pendiente" or selected == original["Resultado real"]:
                        continue
                    store.save_manual_result(original["run_id"], int(original["row_number"]), label_to_outcome[selected])
                    saved += 1
                st.success(f"Resultados guardados: {saved}")
                st.rerun()

        with st.expander("Marcadores Recibidos de Impect"):
            try:
                impect_results = load_impect_results()
                if impect_results.empty:
                    st.info("Aún no se han importado marcadores de Impect.")
                else:
                    st.caption(f"{len(impect_results):,} marcadores guardados en `output.impect_results`.")
                    st.dataframe(impect_results, width="stretch", hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Error consultando output.impect_results: {exc}")

    # TAB 7: ESTADO DE APIS & VIGENCIA (MONITOR IMPECT + PROVEEDORES)
    with tab7:
        st.subheader("🔌 Monitor de Estado y Vigencia de Conexiones API")
        st.caption(
            "Valida en tiempo real si las credenciales, tokens OAuth2 y suscripciones de las APIs siguen vigentes, "
            "con monitoreo prioritario de **Impect API V5** ante su próxima desconexión y auditoría del respaldo local en DuckDB."
        )

        col_btn_all, col_btn_imp, col_chk_opt = st.columns([0.34, 0.34, 0.32])
        with col_chk_opt:
            include_paid_pings = st.checkbox(
                "Incluir ping a Brave & JEV",
                value=True,
                key="chk_include_paid_pings",
                help="Realiza también una consulta mínima de validación a Brave Search y JEV (TypeSafe). Impect, API-Football (/status) y The Odds API (/sports) no consumen cuota de predicción.",
            )
        with col_btn_all:
            check_all_clicked = st.button(
                "🔄 Validar Todas las APIs Ahora",
                type="primary",
                width="stretch",
                key="btn_validate_all_apis",
            )
        with col_btn_imp:
            check_impect_only_clicked = st.button(
                "⚡ Verificar Solo Vigencia de Impect",
                width="stretch",
                key="btn_validate_impect_only",
            )

        if check_all_clicked:
            with st.spinner("Verificando vigencia, tokens y cuotas en tiempo real..."):
                st.session_state["api_health_results"] = run_all_api_checks(
                    include_jev=include_paid_pings,
                    include_brave=include_paid_pings,
                )
                st.session_state["api_health_last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if check_impect_only_clicked:
            with st.spinner("Validando autenticación OAuth2 y acceso a iteraciones en Impect API V5..."):
                imp_res = check_impect_api(verify_data_access=True)
                existing = st.session_state.get("api_health_results", [])
                updated_list = [imp_res] + [r for r in existing if r.get("key") != "impect"]
                st.session_state["api_health_results"] = updated_list
                st.session_state["api_health_last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Si aún no se ha ejecutado en la sesión, ofrecer estado inicial rápido o ejecutar validación ligera
        api_results: list[dict[str, Any]] = st.session_state.get("api_health_results", [])
        last_check_ts = st.session_state.get("api_health_last_run")

        if not api_results:
            st.info(
                "👆 Haz clic en **'🔄 Validar Todas las APIs Ahora'** o **'⚡ Verificar Solo Vigencia de Impect'** "
                "para comprobar en vivo el estado de las credenciales y conexiones."
            )
        else:
            st.caption(f"⏱️ Última comprobación en vivo: **{last_check_ts}**")

            # Foco especial en Impect API
            impect_diag = next((r for r in api_results if r.get("key") == "impect"), None)
            if impect_diag:
                if impect_diag.get("is_active"):
                    st.success(
                        f"✅ **Impect API V5 SIGUE VIGENTE:** {impect_diag.get('message')} "
                        f"({impect_diag.get('expiration_info')} | Latencia: {impect_diag.get('latency_ms')} ms). "
                        "Se recomienda mantener actualizado el respaldo local en DuckDB antes de la fecha de desconexión."
                    )
                else:
                    st.error(
                        f"🚨 **ALERTA IMPECT API V5 ({impect_diag.get('status_label')}):** {impect_diag.get('message')} — "
                        "El sistema operará automáticamente utilizando los ratings, KPIs tácticos y marcadores respaldados en DuckDB."
                    )

            # Tarjetas resumen de estado
            active_count = sum(1 for r in api_results if r.get("is_active"))
            total_checked = len(api_results)
            avg_lat_vals = [r["latency_ms"] for r in api_results if r.get("latency_ms") is not None]
            avg_lat = round(sum(avg_lat_vals) / len(avg_lat_vals), 1) if avg_lat_vals else 0.0

            k_c1, k_c2, k_c3, k_c4 = st.columns(4)
            k_c1.metric("APIs Operativas", f"{active_count} / {total_checked}", "En línea" if active_count == total_checked else "Atención requerida")
            k_c2.metric(
                "Estado Impect V5",
                "VIGENTE" if (impect_diag and impect_diag.get("is_active")) else "INACTIVA / ERROR",
                impect_diag.get("quota_info", "—") if impect_diag else "No probado",
            )
            af_diag = next((r for r in api_results if r.get("key") == "apifootball"), None)
            k_c3.metric(
                "API-Football (Hoy)",
                af_diag.get("status_label", "—") if af_diag else "No probado",
                af_diag.get("quota_info", "—") if af_diag else "—",
            )
            k_c4.metric("Latencia Media", f"{avg_lat:.0f} ms", "Tiempo de respuesta")

            # Tabla consolidada de estado de APIs
            st.markdown("#### 📋 Resumen de Vigencia y Cuotas por Proveedor")
            summary_rows = []
            for r in api_results:
                summary_rows.append({
                    "Proveedor API": r.get("api", ""),
                    "Estado": r.get("status_label", ""),
                    "Vigencia / Suscripción": r.get("expiration_info", "—"),
                    "Cuota / Cobertura": r.get("quota_info", "—"),
                    "Latencia": f"{r['latency_ms']:.0f} ms" if r.get("latency_ms") is not None else "—",
                    "Credencial (.env)": r.get("credential_hint", "—"),
                    "Diagnóstico Detallado": r.get("message", ""),
                })
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

            with st.expander("🔍 Inspeccionar Payload Técnico de Diagnóstico por API"):
                for r in api_results:
                    st.markdown(f"**{r.get('api')}** — `{r.get('status_label')}`")
                    st.json(r.get("details", {}))

        # Pruebas individuales rápidas por proveedor
        st.markdown("---")
        st.markdown("#### 🧪 Pruebas Individuales por Proveedor")
        st.caption("Verifica un servicio específico sin consultar los demás:")
        p_c1, p_c2, p_c3, p_c4, p_c5, p_c6 = st.columns(6)
        single_check = None
        if p_c1.button("Impect V5", key="btn_single_impect", width="stretch"):
            single_check = check_impect_api(verify_data_access=True)
        if p_c2.button("API-Football", key="btn_single_af", width="stretch"):
            single_check = check_apifootball()
        if p_c3.button("The Odds API", key="btn_single_odds", width="stretch"):
            single_check = check_odds_api()
        if p_c4.button("Brave News", key="btn_single_brave", width="stretch"):
            single_check = check_brave_api()
        if p_c5.button("JEV TypeSafe", key="btn_single_jev", width="stretch"):
            single_check = check_jev_api()
        if p_c6.button("ClubElo", key="btn_single_elo", width="stretch"):
            single_check = check_clubelo_api()

        if single_check is not None:
            existing = st.session_state.get("api_health_results", [])
            replaced = False
            new_list = []
            for item in existing:
                if item.get("key") == single_check.get("key"):
                    new_list.append(single_check)
                    replaced = True
                else:
                    new_list.append(item)
            if not replaced:
                new_list.append(single_check)
            st.session_state["api_health_results"] = new_list
            st.session_state["api_health_last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

        # SECCIÓN DE CONTINGENCIA Y RESPALDO LOCAL EN DUCKDB (ANTI-DESCONEXIÓN IMPECT)
        st.markdown("---")
        st.markdown("#### 🛡️ Contingencia ante Desconexión de Impect: Respaldo Local en DuckDB")
        st.caption(
            "Dado que **Impect API se desconectará próximamente**, aquí puedes auditar cuántos datos tácticos, "
            "ratings y marcadores ya están asegurados localmente en `apostador.duckdb` y ejecutar una sincronización "
            "preventiva mientras la cuenta siga vigente."
        )

        backup_inv = get_duckdb_backup_inventory()
        inv_c1, inv_c2, inv_c3, inv_c4 = st.columns(4)
        inv_c1.metric(
            "Artefactos Impect en DuckDB",
            f"{backup_inv['impect_artifacts_total']:,}",
            f"Últ. sync: {backup_inv['impect_last_sync'] or 'N/D'}",
        )
        inv_c2.metric(
            "Marcadores Oficiales Impect",
            f"{backup_inv['impect_results_count']:,}",
            f"{backup_inv['impect_iterations_cached']} iteraciones cubiertas",
        )
        inv_c3.metric(
            "Caché API-Football / Odds",
            f"{backup_inv['apifootball_artifacts']:,} / {backup_inv['odds_artifacts']:,}",
            f"Brave News: {backup_inv['brave_artifacts']:,}",
        )
        inv_c4.metric(
            "Partidos Históricos (SoccerData)",
            f"{backup_inv['historical_matches']:,}",
            "Cuotas Pinnacle & Bet365",
        )

        if backup_inv["impect_by_type"]:
            df_inv_impect = pd.DataFrame([
                {
                    "Categoría de Datos Impect": k,
                    "Archivos / Iteraciones en DuckDB": v["count"],
                    "Última Actualización (UTC)": v["last_updated"],
                }
                for k, v in backup_inv["impect_by_type"].items()
            ])
            st.dataframe(df_inv_impect, width="stretch", hide_index=True)

        col_sync1, col_sync2 = st.columns(2)
        with col_sync1:
            if st.button(
                "📥 Respaldar Ahora Ligas Activas Impect (Backfill Preventivo)",
                key="btn_preventive_impect_backfill",
                width="stretch",
            ):
                with st.spinner("Descargando y asegurando ratings, KPIs, scores, talento y predicciones de Impect en DuckDB..."):
                    cmd = [sys.executable, "-m", "src.commands.backfill_impect_data"]
                    res = subprocess.run(
                        cmd, cwd=ROOT_DIR, env=os.environ.copy(), text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
                    )
                    if res.returncode == 0:
                        st.success("¡Respaldo preventivo de Impect completado y guardado en DuckDB!")
                        st.rerun()
                    else:
                        st.error("El respaldo preventivo reportó errores (verifica si Impect sigue vigente):")
                        st.code(res.stdout, language="text")
        with col_sync2:
            if st.button(
                "⚽ Sincronizar Últimos Marcadores Impect",
                key="btn_preventive_impect_results",
                width="stretch",
            ):
                with st.spinner("Actualizando marcadores oficiales desde Impect hacia output.impect_results..."):
                    cmd = [sys.executable, "-m", "src.commands.fetch_impect_results"]
                    res = subprocess.run(
                        cmd, cwd=ROOT_DIR, env=os.environ.copy(), text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
                    )
                    if res.returncode == 0:
                        st.success("¡Marcadores de Impect actualizados en DuckDB!")
                        st.rerun()
                    else:
                        st.error("Error sincronizando marcadores de Impect:")
                        st.code(res.stdout, language="text")


if __name__ == "__main__":
    main()
