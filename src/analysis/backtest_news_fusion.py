"""backtest_news_fusion.py — Backtest temporal Brave+JEV vs Softmax.

Compara P_base vs P_adj (Brier + LogLoss) sobre partidos FT conocidos y barre
threshold x weight con split temporal (mitad calibra, mitad evalua).
Fuentes DuckDB: processed/black_swan_v2_*, processed/jev_decisions_*,
raw/apifootball/fixtures_result_*. Si n<30 lo informa igual (ver temp/03 §9).
Uso: python -m src.analysis.backtest_news_fusion
"""
from __future__ import annotations
import argparse
import math
from datetime import datetime, timezone
from src.apostador.aligner import TeamAligner
from src.apostador.config import BACKTESTS_DIR
from src.apostador.duckdb_store import get_store, write_json
KEYS = ("local", "empate", "visita")
THRESHOLDS = (0.50, 0.60, 0.70)
WEIGHTS = (0.20, 0.50, 0.70, 0.85)
MIN_SAMPLE = 30
EPS = 1e-9

def _renorm(probs):
    total = sum(float(probs[k]) for k in KEYS) or 1.0
    return {k: float(probs[k]) / total for k in KEYS}

def fuse_vector(base, jev, news_relevant, relevance, threshold, weight):
    """Replica el gate con threshold/weight variables."""
    base = _renorm(base)
    if not jev or not news_relevant or float(relevance or 0.0) < threshold:
        return dict(base), 0.0
    try:
        jv = _renorm({k: float(jev[k]) for k in KEYS})
    except (KeyError, TypeError, ValueError):
        return dict(base), 0.0
    return {k: (1 - weight) * base[k] + weight * jv[k] for k in KEYS}, weight

def brier_score(probs, outcome_idx):
    target = [1.0 if i == outcome_idx else 0.0 for i in range(3)]
    return sum((float(probs[k]) - t) ** 2 for k, t in zip(KEYS, target))

def log_loss_1x2(probs, outcome_idx):
    return -math.log(max(float(probs[KEYS[outcome_idx]]), EPS))

def outcome_key(gh, ga):
    return "local" if gh > ga else ("empate" if gh == ga else "visita")

def _find_fixture(aligner, fixtures, home_team, away_team):
    names = [f["teams"]["home"]["name"] for f in fixtures if "teams" in f]
    home_match, _ = aligner.find_best_match(home_team, names)
    if not home_match:
        return None
    for f in fixtures:
        if f["teams"]["home"]["name"] == home_match:
            away_match, _ = aligner.find_best_match(away_team, [f["teams"]["away"]["name"]])
            return f if away_match else None
    return None
def handle_scored(scored, pending, seen, aligner, results_by_date, home, away, date,
                  base, jev, news_relevant, relevance, source):
    if (date, home, away) in seen:
        return
    seen.add((date, home, away))
    row = {"home_team": home, "away_team": away, "date": date,
           "P_base": _renorm(base), "P_jev": None, "source": source,
           "news_relevant": bool(news_relevant), "relevance": float(relevance or 0.0)}
    if jev:
        try:
            row["P_jev"] = _renorm({k: float(jev[k]) for k in KEYS})
        except (KeyError, TypeError, ValueError):
            row["P_jev"] = None
    fixture = _find_fixture(aligner, results_by_date.get(date, []), home, away)
    if fixture is None or fixture["goals"]["home"] is None:
        row["result"] = None
        pending.append(row)
        return
    row["result"] = outcome_key(fixture["goals"]["home"], fixture["goals"]["away"])
    row["fixture_id"] = fixture["fixture"]["id"]
    scored.append(row)

def collect_rows(store=None):
    """Une piernas/decisiones con resultados FT. Retorna (scored, pending)."""
    store = store or get_store()
    legs = []
    for _, payload in store.list("processed", "black_swan_v2_"):
        if isinstance(payload, dict):
            legs.extend(payload.get("legs", []))
    decisions = []
    for _, payload in store.list("processed", "jev_decisions_"):
        if isinstance(payload, dict):
            decisions.extend(payload.get("decisions", []))
    results_by_date = {}
    for key, payload in store.list("raw", "apifootball/fixtures_result_"):
        date = key.removeprefix("apifootball/fixtures_result_").removesuffix(".json")
        resp = payload.get("response", []) if isinstance(payload, dict) else []
        results_by_date[date] = [f for f in resp
                                 if f.get("fixture", {}).get("status", {}).get("short")
                                 in ("FT", "AET", "PEN")]
    by_match = {(d.get("date"), d.get("home_team"), d.get("away_team")): d for d in decisions}
    aligner = TeamAligner()
    scored, pending, seen = [], [], set()
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        home, away, date = leg.get("home_team"), leg.get("away_team"), leg.get("date")
        if not home or not away or not date:
            continue
        base = leg.get("P_base") or {k: leg.get(f"P_{k[0].upper()}", 1 / 3) for k in KEYS}
        jev, rel = leg.get("P_jev"), leg.get("noticias_relevancia_jev", 0.0)
        news = bool(leg.get("candidatas_noticias", 0))
        extra = by_match.get((date, home, away))
        if jev is None and extra:
            jev, rel, news = extra.get("P_jev"), extra.get("relevance", rel), True
        handle_scored(scored, pending, seen, aligner, results_by_date, home, away,
                      date, base, jev, news, rel, "black_swan_v2")
    for d in decisions:
        if not d.get("P_base") or not d.get("home_team") or not d.get("date"):
            continue
        handle_scored(scored, pending, seen, aligner, results_by_date,
                      d["home_team"], d["away_team"], d["date"], d["P_base"], d.get("P_jev"),
                      d.get("news_relevant", True), d.get("relevance", 0.0), "jev_decision")
    scored.sort(key=lambda r: r["date"] or "")
    return scored, pending
def evaluate_rows(rows, threshold, weight):
    """Metricas medias base vs fusionada con (threshold, weight)."""
    n = nb = llb = nf = llf = 0
    for r in rows:
        if r["result"] is None or r["P_jev"] is None:
            continue
        idx = KEYS.index(r["result"])
        fused, _ = fuse_vector(r["P_base"], r["P_jev"], r["news_relevant"],
                               r["relevance"], threshold, weight)
        nb += brier_score(r["P_base"], idx)
        llb += log_loss_1x2(r["P_base"], idx)
        nf += brier_score(fused, idx)
        llf += log_loss_1x2(fused, idx)
        n += 1
    if not n:
        return {"n": 0}
    return {"n": n, "brier_base": nb / n, "logloss_base": llb / n,
            "brier_fused": nf / n, "logloss_fused": llf / n}

def run_backtest(thresholds=THRESHOLDS, weights=WEIGHTS):
    scored, pending = collect_rows()
    with_jev = [r for r in scored if r["P_jev"] is not None]
    cut = len(with_jev) // 2
    cal, eva = with_jev[:cut], with_jev[cut:]
    grid = [{"threshold": t, "weight": w, **evaluate_rows(cal, t, w)}
            for t in thresholds for w in weights]
    usable = [g for g in grid if g["n"] > 0]
    best = min(usable, key=lambda g: g["logloss_fused"]) if usable else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scored": len(scored), "n_with_jev": len(with_jev),
        "n_pending": len(pending), "n_cal": len(cal), "n_eval": len(eva),
        "sufficient_sample": len(with_jev) >= MIN_SAMPLE,
        "min_sample_required": MIN_SAMPLE,
        "baseline_softmax_vs_gate_0.50_0.20": evaluate_rows(with_jev, 0.50, 0.20),
        "grid_calibration": grid,
        "best_calibration": best,
        "eval_best": evaluate_rows(eva, best["threshold"], best["weight"]) if best else {"n": 0},
        "pending": [(r["date"], r["home_team"], r["away_team"]) for r in pending],
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest temporal Brave+JEV vs Softmax")
    ap.parse_args(argv)
    summary = run_backtest()
    tag = datetime.now().strftime("%Y-%m-%d")
    out = BACKTESTS_DIR / "news_fusion_backtest.json"
    write_json(out, summary, run_type="backtest_news_fusion", run_date=tag)
    b = summary["baseline_softmax_vs_gate_0.50_0.20"]
    print(f"Puntuables: {summary['n_scored']} (con JEV: {summary['n_with_jev']}) | "
          f"pendientes: {summary['n_pending']} | suficiente: {summary['sufficient_sample']}")
    if b.get("n"):
        print(f"Base: Brier {b['brier_base']:.4f} LogLoss {b['logloss_base']:.4f} | "
              f"Gate: Brier {b['brier_fused']:.4f} LogLoss {b['logloss_fused']:.4f}")
    if summary["best_calibration"]:
        bc = summary["best_calibration"]
        print(f"Mejor: thr={bc['threshold']} w={bc['weight']} "
              f"(LogLoss {bc['logloss_fused']:.4f}, n={bc['n']})")
    else:
        print("Sin filas con JEV+resultado: barrido pendiente a mas muestra.")
    print("Reporte en DuckDB: output.artifacts (backtests/news_fusion_backtest.json)")
    return summary

if __name__ == "__main__":
    main()


