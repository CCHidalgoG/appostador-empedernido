"""black_swan_v2.py — Black Swan ajustado con noticias (Brave) + decision (JEV).

Pipeline por partido:
  P_base (SoccerPredictor) -> Brave digest -> state JEV -> JEV decide
  -> fuse_probabilities -> P_adj -> seleccion de piernas EV+ con P_adj
  -> combinatoria de parlay (reusa logica v1) penalizada por riesgo JEV.

Entradas:
  --input JSON de portfolio (el que genera main.py: portfolio_bets_<fecha>.json)
  --date  fecha para modo fixtures en vivo (requiere main.SUPPORTED_LEAGUES)
Modos noticia/JEV desactivables (--no-news, --no-jev) para comparar v1 vs v2.

Solo usa requests/python-dotenv (ya en el proyecto). JEV/Brave nunca rompen
el pipeline: ante fallo se usa P_base.
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.apostador.config import PROCESSED_DIR, REPORTS_DIR, ensure_runtime_directories
from src.apostador.duckdb_store import (
    artifact_key,
    get_store,
    read_json,
    write_json,
    write_text,
)
from src.apostador.jev_decision import (
    JevDecisionClient,
    build_match_state,
    fallback_decision,
)
from src.apostador.news_search import BraveNewsClient
from src.apostador.optimizer import KellyOptimizer
from src.apostador.probability_fusion import OUTCOME_KEYS, fuse_probabilities

OUTCOME_TO_FUSION = {"Home Win": "local", "Draw": "empate", "Away Win": "visita"}
FUSION_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_FUSION.items()}


def enrich_value_bets(value_bets, use_news=True, use_jev=True):
    """Agrega P_adj + diagnostico JEV a cada value bet. Retorna (enriched, stats)."""
    brave = BraveNewsClient() if use_news else None
    jev = JevDecisionClient() if use_jev else None
    enriched, stats = [], {
        "n": 0, "con_noticias": 0, "candidatas_noticias": 0,
        "noticias_pertinentes": 0, "jev_llamadas": 0, "jev_ok": 0,
        "jev_prioridad": 0, "jev_no_relevante": 0,
        "jev_saltado_sin_evidencia": 0, "vetadas": 0, "w_medio": 0.0,
    }
    for bet in value_bets:
        home, away = bet["home_team"], bet["away_team"]
        date = bet.get("date")
        news = {"digest": "", "sources": [], "n_sources": 0,
                "news_relevant": False, "news_evidence": [], "news_categories": [],
                "news_candidate_count": 0, "errors": []}
        if brave:
            try:
                news = brave.search_match_news(home, away, date)
            except Exception as exc:  # noqa: BLE001
                news = {"digest": "", "sources": [], "n_sources": 0,
                        "news_relevant": False, "news_evidence": [], "news_categories": [],
                        "news_candidate_count": 0, "errors": [str(exc)]}
        base_p = {k: float(bet.get(f"P_{k[0].upper()}", 0.0)) or 0.0 for k in OUTCOME_KEYS}
        if sum(base_p.values()) <= 0:  # portfolio v1 solo trae la proba de la pierna
            p_leg = float(bet.get("probability", 0.33))
            base_p = {"local": 0.33, "empate": 0.33, "visita": 0.33}
            base_p[OUTCOME_TO_FUSION.get(bet.get("market_outcome") or bet.get("outcome"), "local")] = p_leg
            s = sum(base_p.values())
            base_p = {k: v / s for k, v in base_p.items()}
        if not jev:
            fallback_reason = "jev_desactivado"
        elif news.get("errors"):
            fallback_reason = "error_busqueda_noticias"
        elif not news.get("news_relevant"):
            fallback_reason = ("sin_evidencia_noticiosa_pertinente"
                               if news.get("digest") else "sin_noticias")
        else:
            fallback_reason = "jev_no_ejecutado"
        decision = fallback_decision(fallback_reason)
        if jev and news.get("news_relevant"):
            state = build_match_state(home, away, bet.get("league", ""), base_p,
                                      bet.get("home_rating", 0.68), bet.get("away_rating", 0.68),
                                      bet.get("form_delta", 0.0),
                                      {"cuota_pierna": bet.get("odds")}, news["digest"],
                                      news["sources"], news.get("news_evidence", []),
                                      structured_context=bet.get("form_context", {}))
            stats["jev_llamadas"] += 1
            try:
                decision = jev.decide(state)
            except Exception as exc:  # noqa: BLE001
                decision = fallback_decision(f"jev_error: {exc}")
        elif jev:
            stats["jev_saltado_sin_evidencia"] += 1
        fused = fuse_probabilities(base_p, decision, news_relevant=news.get("news_relevant", False))
        key = OUTCOME_TO_FUSION.get(bet.get("market_outcome") or bet.get("outcome"), "local")
        p_adj = fused["P_adj"][key]
        ev_adj = p_adj * float(bet["odds"]) - 1.0
        row = dict(bet, P_base=fused.get("P_base", base_p), P_jev=decision.get("choice_probs"),
                   P_adj=fused["P_adj"], p_adj_leg=p_adj, ev_adj=ev_adj,
                   w=fused.get("w", 0.0), vetada=fused.get("vetada", False),
                   jev_choice=decision.get("choice"), jev_conf=decision.get("confidence", 0.0),
                   riesgo=decision.get("riesgo_favorito", 0.0),
                   volatilidad=decision.get("volatilidad", "estable"),
                   n_fuentes=news.get("n_sources", 0), fuentes=news.get("sources", [])[:5],
                   candidatas_noticias=news.get("news_candidate_count", 0),
                   noticias_categorias=news.get("news_categories", []),
                   noticias_evidencia=news.get("news_evidence", [])[:5],
                   noticias_pertinentes=fused.get("news_relevant", False),
                   noticias_relevancia_jev=fused.get("news_relevance_score", 0.0),
                   decision_source=fused.get("decision_source", "softmax"),
                   motivo_decision=fused.get("motivo"))
        enriched.append(row)
        stats["n"] += 1
        stats["con_noticias"] += bool(news.get("n_sources"))
        stats["candidatas_noticias"] += int(news.get("news_candidate_count", 0) > 0)
        stats["noticias_pertinentes"] += bool(fused.get("news_relevant"))
        stats["jev_ok"] += bool(not decision.get("fallback"))
        stats["jev_no_relevante"] += bool(
            not decision.get("fallback") and not fused.get("news_relevant")
        )
        stats["jev_prioridad"] += fused.get("decision_source") == "jev_priority"
        stats["vetadas"] += bool(row["vetada"])
        stats["w_medio"] += row["w"]
    if stats["n"]:
        stats["w_medio"] = round(stats["w_medio"] / stats["n"], 4)
    return enriched, stats

def select_adjusted_legs(enriched, min_ev_adj=0.0):
    legs = [b for b in enriched if not b.get("vetada") and b.get("ev_adj", -9) > min_ev_adj]
    legs.sort(key=lambda b: b["ev_adj"], reverse=True)
    return legs


def build_adjusted_parlay(legs, target_odds=700.0, max_legs=6, bankroll=40000.0):
    by_match = defaultdict(list)
    for b in legs:
        by_match[f"{b['home_team']} vs {b['away_team']}"].append(b)
    keys = list(by_match.keys())
    min_odds = target_odds * 0.7
    best, best_score = None, -1.0
    for n in range(2, min(max_legs + 1, len(keys) + 1)):
        for combo_matches in itertools.combinations(keys, n):
            for combo in itertools.product(*[by_match[k] for k in combo_matches]):
                odds = 1.0
                prob = 1.0
                riesgo = 0.0
                for leg in combo:
                    odds *= float(leg["odds"])
                    prob *= float(leg["p_adj_leg"])
                    riesgo += float(leg.get("riesgo", 0.0))
                if odds < min_odds:
                    continue
                riesgo_medio = riesgo / len(combo)
                penalty = abs(odds - target_odds) / target_odds * 0.1
                score = prob * (1 - 0.5 * riesgo_medio) - penalty
                if score > best_score:
                    best_score = score
                    best = {"legs": list(combo), "total_odds": odds,
                            "total_probability": prob, "n_legs": n,
                            "riesgo_medio": riesgo_medio, "score": score}
    return best


def black_swan_stake(parlay, bankroll, kelly_factor=0.001, min_stake_amount=1.0):
    opt = KellyOptimizer(bankroll=bankroll, fraction_factor=kelly_factor, min_stake_amount=min_stake_amount)
    p, odds = parlay["total_probability"], parlay["total_odds"]
    f_star = opt.calculate_kelly_fraction(p, odds)
    ev = p * odds - 1.0
    stake = max(f_star * kelly_factor * bankroll, bankroll * 0.0001)
    stake = max(round(stake, 2), float(min_stake_amount))
    return {"kelly_pure": f_star, "stake": stake,
            "stake_pct": stake / bankroll * 100, "ev": ev,
            "payout": round(stake * odds, 2)}


def write_report(parlay, stake_info, stats, bankroll, path, enriched):
    path = Path(path)
    L = [f"# Black Swan Ajustado (Brave + JEV) — {datetime.now(timezone.utc):%Y-%m-%d}",
         f"Cuota: **{parlay['total_odds']:.2f}x** | P_adj: **{parlay['total_probability']*100:.4f}%**",
         (f"Banco: ${bankroll:,.2f} | Stake: ${stake_info['stake']:,.2f} "
          f"({stake_info['stake_pct']:.4f}%) | EV: {stake_info['ev']*100:.2f}%"),
         (f"Riesgo medio JEV: {parlay.get('riesgo_medio', 0):.2f} | "
          f"Piernas: {parlay['n_legs']} | w_medio: {stats.get('w_medio', 0)}"),
         "", "## Piernas (con P_base -> P_adj)",
         "| # | Partido | Apuesta | Cuota | P_base | P_adj | EV_adj | JEV | Riesgo | Fuentes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, leg in enumerate(parlay["legs"], 1):
        out = leg.get("market_outcome") or leg.get("outcome")
        key = OUTCOME_TO_FUSION.get(out, "local")
        pb = (leg.get("P_base") or {}).get(key, leg.get("probability", 0))
        L.append(f"| {i} | {leg['home_team']} vs {leg['away_team']} | {out} | "
                 f"{leg['odds']:.2f} | {pb*100:.1f}% | {leg['p_adj_leg']*100:.1f}% | "
                 f"{leg['ev_adj']*100:.1f}% | {leg.get('jev_choice')}/{leg.get('jev_conf', 0):.2f} | "
                 f"{leg.get('riesgo', 0):.2f} | {leg.get('n_fuentes', 0)} |")
    L += ["", "## Cobertura JEV/Brave",
          (f"- Evaluadas: {stats.get('n', 0)}, con noticias: {stats.get('con_noticias', 0)}, "
           f"JEV ok: {stats.get('jev_ok', 0)}, JEV prioridad: {stats.get('jev_prioridad', 0)}, "
           f"sin evidencia pertinente: {stats.get('jev_saltado_sin_evidencia', 0)}, "
           f"vetadas: {stats.get('vetadas', 0)}."),
          "- Decisión: evidencia pertinente + confirmación JEV ≥ 0.50 → mezcla predeterminada JEV 0.20 / Softmax 0.80.",
          "- Sin evidencia, sin confirmación JEV o error de API → P_base Softmax (w=0).",
          "", "## Fuentes por pierna"]
    for leg in parlay["legs"]:
        L.append(f"- **{leg['home_team']} vs {leg['away_team']}**:")
        for u in leg.get("fuentes", []) or ["(sin fuentes: se uso P_base)"]:
            L.append(f"  - {u}")
    write_text(path, "\n".join(L) + "\n")
    return path


def load_value_bets(path_or_date):
    p = Path(str(path_or_date))
    data = read_json(p) if p.exists() else get_store().get(*artifact_key(p), None)
    if data is None:
        store = get_store()
        prefix = f"portfolio_bets_{Path(str(path_or_date)).stem if str(path_or_date).endswith('.json') else path_or_date}"
        candidates = [(key, value) for key, value in store.list("processed", "portfolio_bets_") if key.startswith(prefix)]
        if not candidates and str(path_or_date).count("-") != 2:
            candidates = store.list("processed", "portfolio_bets_")
        if not candidates:
            raise FileNotFoundError("Sin portfolios en DuckDB ni archivos históricos. Ejecuta main.py primero.")
        data = max(candidates, key=lambda item: item[0])[1]
    bets = data.get("value_bets") or data.get("bets") or []
    norm = []
    for b in bets:
        if "odds" not in b and "market_odds" in b:
            b = dict(b, odds=b["market_odds"])
        if "market_outcome" not in b and "outcome" in b:
            b = dict(b, market_outcome=b["outcome"])
        norm.append(b)
    return norm


def main(argv=None):
    ap = argparse.ArgumentParser(description="Black Swan v2 ajustado con Brave + JEV")
    ap.add_argument("--input", default=None, help="JSON portfolio de main.py o fecha YYYY-MM-DD")
    ap.add_argument("--date", default=None, help="Alias de --input cuando es fecha")
    ap.add_argument("--target-odds", type=float, default=700.0)
    ap.add_argument("--bankroll", type=float, default=40000.0)
    ap.add_argument("--min-stake", type=float, default=1.0, help="Stake mínimo por apuesta en unidades monetarias")
    ap.add_argument("--max-legs", type=int, default=6)
    ap.add_argument("--min-ev", type=float, default=0.0)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-jev", action="store_true")
    a = ap.parse_args(argv)
    ensure_runtime_directories()
    src = a.input or a.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    value_bets = load_value_bets(src)
    print(f"Apuestas de valor cargadas: {len(value_bets)} (fuente: {src})")
    enriched, stats = enrich_value_bets(value_bets, use_news=not a.no_news, use_jev=not a.no_jev)
    legs = select_adjusted_legs(enriched, min_ev_adj=a.min_ev)
    print(f"Piernas EV+ ajustadas: {len(legs)} | vetadas: {stats['vetadas']} | "
          f"JEV ok: {stats['jev_ok']}/{stats['n']} | w_medio: {stats['w_medio']}")
    if len(legs) < 2:
        print("Sin piernas suficientes tras el ajuste.")
        return None
    parlay = build_adjusted_parlay(legs, a.target_odds, a.max_legs, a.bankroll)
    if not parlay:
        print("Sin combinacion sobre la cuota minima (70% del objetivo).")
        return None
    stake = black_swan_stake(parlay, a.bankroll, min_stake_amount=a.min_stake)
    tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_json = PROCESSED_DIR / f"black_swan_v2_{tag}.json"
    output_payload = {"parlay": parlay, "stake": stake, "stats": stats, "legs": enriched}
    write_json(out_json, output_payload, run_type="black_swan_v2", run_date=tag)
    report_path = REPORTS_DIR / f"black_swan_v2_{tag}.md"
    write_report(parlay, stake, stats, a.bankroll, report_path, enriched)
    rep = report_path
    print(f"Cuota {parlay['total_odds']:.2f}x | P_adj {parlay['total_probability']*100:.4f}% | "
          f"stake ${stake['stake']:,.2f} | reporte {rep}")
    return parlay


if __name__ == "__main__":
    main()
