"""Backtest stored core/JEV predictions and simulate alternative JEV weights."""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apostador.jev_decision import NEWS_RELEVANCE_THRESHOLD
from src.apostador.probability_fusion import JEV_PRIORITY_WEIGHT


DB_PATH = ROOT / "data" / "apostador.duckdb"
REPORT_PATH = ROOT / "documentacion" / "backtesting" / "jev_backtesting.md"
PARAMS_PATH = ROOT / "data" / "processed" / "calibrated_params.json"
OUTCOMES = ("local", "empate", "visita")


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def probabilities(value: object) -> dict[str, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    try:
        result = {key: float(value[key]) for key in OUTCOMES}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(p) or p < 0 for p in result.values()):
        return None
    total = sum(result.values())
    if total <= 0:
        return None
    return {key: p / total for key, p in result.items()}


def metric(rows: list[dict], field: str) -> dict[str, float | int]:
    if not rows:
        return {"n": 0, "accuracy": math.nan, "log_loss": math.nan, "brier": math.nan, "ece": math.nan}
    correct = loss = brier = 0.0
    confidence_correct = []
    for row in rows:
        probs = row[field]
        actual = row["actual"]
        predicted = max(OUTCOMES, key=probs.get)
        is_correct = predicted == actual
        correct += int(is_correct)
        confidence_correct.append((max(probs.values()), float(is_correct)))
        loss -= math.log(max(probs[actual], 1e-15))
        brier += sum((probs[k] - int(k == actual)) ** 2 for k in OUTCOMES)
    n = len(rows)
    ece = 0.0
    for lower in (i / 10 for i in range(10)):
        bucket = [(confidence, outcome) for confidence, outcome in confidence_correct
                  if lower <= confidence < lower + 0.1 or (lower == 0.9 and confidence == 1.0)]
        if bucket:
            avg_confidence = sum(x[0] for x in bucket) / len(bucket)
            avg_accuracy = sum(x[1] for x in bucket) / len(bucket)
            ece += len(bucket) / n * abs(avg_confidence - avg_accuracy)
    return {"n": n, "accuracy": correct / n, "log_loss": loss / n,
            "brier": brier / n, "ece": ece}


def mix(base: dict[str, float], jev: dict[str, float], weight: float) -> dict[str, float]:
    return {key: (1 - weight) * base[key] + weight * jev[key] for key in OUTCOMES}


def probs_from_ratings(home_rating, away_rating, home_advantage, form_delta, alpha, gamma_draw):
    x = float(home_rating) - float(away_rating) + float(home_advantage or 0) + float(form_delta or 0)
    logits = {"local": alpha * x, "empate": gamma_draw, "visita": -alpha * x}
    high = max(logits.values())
    exp = {key: math.exp(value - high) for key, value in logits.items()}
    total = sum(exp.values())
    probs = {key: value / total for key, value in exp.items()}
    probs = {key: max(0.01, min(0.90, value)) for key, value in probs.items()}
    total = sum(probs.values())
    return {key: value / total for key, value in probs.items()}


def pct(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{100 * value:.1f}%"


def num(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{value:.4f}"


def main() -> None:
    params = {}
    try:
        params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    alpha = float(params.get("alpha", 5.0))
    gamma_draw = float(params.get("gamma_d", -0.405))
    con = duckdb.connect(str(DB_PATH), read_only=True)
    run_rows = con.execute(
        "SELECT run_id, started_at, summary FROM output.runs "
        "WHERE run_type IN ('daily_portfolio', 'weekly_portfolio') ORDER BY started_at"
    ).fetchall()
    # For reruns of a fixture, retain the latest saved prediction.
    predictions: dict[tuple[str, str, str], dict] = {}
    run_ids = set()
    for run_id, started_at, summary in run_rows:
        summary = json.loads(summary) if isinstance(summary, str) else summary
        # Weekly runs nest one daily result per date.
        day_blocks = summary.get("daily_results") or [summary]
        flat = [(str(d.get("date") or summary.get("date") or "")[:10], m)
                for d in day_blocks for m in d.get("processed_matches", [])]
        for day_date, match in flat:
            base = probabilities(match.get("P_base"))
            jev = probabilities(match.get("P_jev"))
            final = probabilities({
                "local": match.get("P_H"),
                "empate": match.get("P_D"),
                "visita": match.get("P_A"),
            })
            if not base:
                continue
            event_date = day_date
            key = (event_date, normalize(match.get("home_team")), normalize(match.get("away_team")))
            record = {
                "date": event_date, "home": match.get("home_team", ""),
                "away": match.get("away_team", ""), "base": base, "jev": jev,
                "final": final,
                "no_form": probs_from_ratings(
                    match.get("home_rating", 0.68), match.get("away_rating", 0.68),
                    match.get("home_advantage_used", 0.05), 0.0,
                    float(match.get("model_alpha") if match.get("model_alpha") is not None else alpha),
                    float(match.get("model_gamma_d") if match.get("model_gamma_d") is not None else gamma_draw),
                ),
                "form_delta": float(match.get("form_delta") or 0),
                "jev_weight": float(match.get("jev_weight") or 0),
                "jev_relevance_score": (float(match["jev_relevance_score"])
                                         if match.get("jev_relevance_score") is not None else None),
                "jev_confidence": (float(match["jev_confidence"])
                                   if match.get("jev_confidence") is not None else None),
                "news_relevant": bool(match.get("news_relevant")),
                "structured_context_relevant": bool(match.get("structured_context_relevant")),
                "jev_context_relevant": (bool(match.get("jev_context_relevant"))
                                          if match.get("jev_context_relevant") is not None else None),
                "jev_weight_config": (float(match["jev_weight_config"])
                                      if match.get("jev_weight_config") is not None else None),
                "jev_source": match.get("jev_source"), "run_id": run_id,
                "news_categories": match.get("news_categories") or [],
                "league_name": match.get("league_name", "sin liga"),
                "started_at": started_at,
            }
            if key not in predictions or started_at > predictions[key]["started_at"]:
                predictions[key] = record
            run_ids.add(run_id)

    actual_rows = con.execute(
        "SELECT r.event_date, r.home_team, r.away_team, x.result_outcome, x.source "
        "FROM output.recommendations r JOIN output.results x USING (run_id, row_number) "
        "WHERE x.result_outcome IN ('local', 'empate', 'visita')"
    ).fetchall()
    actuals = {}
    actual_sources = {}
    sources = Counter()
    for event_date, home, away, outcome, source in actual_rows:
        key = (str(event_date)[:10], normalize(home), normalize(away))
        actuals.setdefault(key, outcome)
        actual_sources.setdefault(key, source or "desconocido")
        sources[source or "desconocido"] += 1

    # Some completed Impect results have no matching recommendation row, so
    # they are not represented in output.results. Use Impect's score table as
    # a fixture-level fallback to score every stored prediction we can match.
    impect_rows = con.execute(
        "SELECT CAST(scheduled_at AS DATE), home_team, away_team, result_outcome "
        "FROM output.impect_results WHERE result_outcome IN ('local','empate','visita')"
    ).fetchall()
    impect_fallbacks = 0
    for event_date, home, away, outcome in impect_rows:
        key = (str(event_date)[:10], normalize(home), normalize(away))
        if key not in actuals and key in predictions:
            actuals[key] = outcome
            actual_sources[key] = "impect_results"
            impect_fallbacks += 1

    rows = []
    for key, pred in predictions.items():
        if key not in actuals:
            continue
        pred["actual"] = actuals[key]
        if pred["jev_context_relevant"] is None:
            # Legacy snapshots did not persist the gate or configured weight.
            pred["eligible"] = pred["jev"] is not None and pred["jev_weight"] > 0
        else:
            pred["eligible"] = (
                pred["jev"] is not None and pred["jev_context_relevant"]
                and pred["jev_relevance_score"] is not None
                and pred["jev_relevance_score"] >= NEWS_RELEVANCE_THRESHOLD
            )
        # Production is the saved blend on eligible fixtures; otherwise core.
        pred["production"] = pred["final"] or (
            mix(pred["base"], pred["jev"], pred["jev_weight"])
            if pred["eligible"] else pred["base"]
        )
        rows.append(pred)
    rows.sort(key=lambda row: (row["date"], row["home"], row["away"]))

    core = metric(rows, "base")
    production = metric(rows, "production")
    no_form = metric(rows, "no_form")
    eligible = [r for r in rows if r["eligible"]]
    jev_active = [r for r in rows if r["jev"] is not None and r["jev_weight"] > 0]
    core_eligible = metric(jev_active, "base")
    prod_eligible = metric(jev_active, "production")
    # Split on whole match dates so same-day results never cross partitions.
    dates = sorted({row["date"] for row in rows})
    date_cut = max(1, min(len(dates) - 1, math.floor(len(dates) * 0.60))) if len(dates) > 1 else len(dates)
    train_dates, holdout_dates = set(dates[:date_cut]), set(dates[date_cut:])
    train = [row for row in rows if row["date"] in train_dates]
    holdout = [row for row in rows if row["date"] in holdout_dates]
    cut = len(train)
    train_eligible = [r for r in train if r["eligible"]]
    holdout_eligible = [r for r in holdout if r["eligible"]]
    grid = [i / 20 for i in range(21)]
    candidates = []
    for weight in grid:
        for subset_name, subset in (("train", train_eligible), ("holdout", holdout_eligible), ("all_eligible", eligible)):
            simulated = []
            for row in subset:
                row_copy = dict(row)
                row_copy["simulated"] = mix(row["base"], row["jev"], weight)
                simulated.append(row_copy)
            candidates.append((weight, subset_name, metric(simulated, "simulated")))
    train_grid = [x for x in candidates if x[1] == "train" and x[2]["n"]]
    best = min(train_grid, key=lambda x: (x[2]["log_loss"], x[2]["brier"])) if train_grid else None
    best_weight = best[0] if best else None
    best_holdout = None
    if best_weight is not None:
        sim_holdout = []
        for row in holdout_eligible:
            rc = dict(row)
            rc["simulated"] = mix(row["base"], row["jev"], best_weight)
            sim_holdout.append(rc)
        best_holdout = metric(sim_holdout, "simulated")

    # Expanding walk-forward: every test date uses only earlier eligible games.
    walk_forward_rows = []
    weight_history = Counter()
    for test_date in dates[date_cut:]:
        earlier = [row for row in eligible if row["date"] < test_date]
        current = [row for row in eligible if row["date"] == test_date]
        if len(earlier) < 20 or not current:
            continue
        scored = []
        for weight in grid:
            simulated = []
            for row in earlier:
                clone = dict(row)
                clone["simulated"] = mix(row["base"], row["jev"], weight)
                simulated.append(clone)
            scored.append((weight, metric(simulated, "simulated")))
        selected_weight, _ = min(scored, key=lambda item: (item[1]["log_loss"], item[1]["brier"]))
        weight_history[selected_weight] += len(current)
        for row in current:
            clone = dict(row)
            clone["simulated"] = mix(row["base"], row["jev"], selected_weight)
            walk_forward_rows.append(clone)
    walk_forward_metric = metric(walk_forward_rows, "simulated")
    walk_forward_core = metric(walk_forward_rows, "base")

    # Probability sharpness is measurable on older snapshots even where the
    # JEV relevance/confidence signals needed for a true dynamic policy were
    # not persisted. Report this diagnostic without tuning a new rule on it.
    jev_candidates = [row for row in rows if row["jev"] is not None]
    jev_probability_buckets = []
    for label, subset in (
        ("máxima P_JEV ≤ 0.70", [r for r in jev_candidates if max(r["jev"].values()) <= 0.70]),
        ("máxima P_JEV > 0.70", [r for r in jev_candidates if max(r["jev"].values()) > 0.70]),
    ):
        jev_probability_buckets.append((label, subset))
    relevance_snapshots = sum(row["jev_relevance_score"] is not None for row in jev_candidates)
    confidence_snapshots = sum(row["jev_confidence"] is not None for row in jev_candidates)
    # Report a full-grid evaluation as descriptive only; never select on it.
    grid_full = {w: m for w, scope, m in candidates if scope == "all_eligible"}
    best_full = min(( (w, m) for w, m in grid_full.items() if m["n"] ), key=lambda x: (x[1]["log_loss"], x[1]["brier"])) if eligible else None

    lines = [
        "# Backtesting de JEV y simulación de pesos",
        "",
        f"Generado: {date.today().isoformat()} · fuente: `{DB_PATH.relative_to(ROOT)}`.",
        "",
        "## Alcance y método",
        "",
        f"Se tomaron las predicciones guardadas en ejecuciones `daily_portfolio`, conservando la última predicción por partido y fecha. Se enlazaron por fecha/equipos con `output.results` y, para partidos sin fila de recomendación, se usó `output.impect_results` como fallback ({impect_fallbacks} resultados añadidos). La muestra enlazada contiene **{len(rows)} partidos** ({len(predictions)} predicciones únicas con probabilidades base); período: **{rows[0]['date'] if rows else '—'} a {rows[-1]['date'] if rows else '—'}**.",
        "",
        f"Resultados disponibles por fuente en toda la tabla: {', '.join(f'{name}: {count}' for name, count in sorted(sources.items())) or 'ninguno'}. Las métricas de este informe usan solo partidos enlazados a una predicción guardada.",
        "",
        f"Se comparan probabilidades `P_base` (core) contra las probabilidades finales persistidas (`P_H/P_D/P_A`). El gate de relevancia vigente es ≥{NEWS_RELEVANCE_THRESHOLD:.2f}. Para simular pesos se usa `P=(1−w)·P_base+w·P_jev`, donde `w` recorre valores candidatos. La cohorte de simulación exige vector JEV válido y gate contextual aprobado; las filas históricas antiguas usan como aproximación el peso efectivo guardado porque no conservaban el gate. Métricas: exactitud, log loss, Brier multiclase (menores son mejores) y ECE top-label de diez bins (menor indica mejor calibración).",
        "",
        "## Core frente a producción con JEV",
        "",
        "| Cohorte | Partidos | Exactitud | Log loss | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Impect rating + localía, sin forma | {no_form['n']} | {pct(no_form['accuracy'])} | {num(no_form['log_loss'])} | {num(no_form['brier'])} | {num(no_form['ece'])} |",
        f"| Todos: core | {core['n']} | {pct(core['accuracy'])} | {num(core['log_loss'])} | {num(core['brier'])} | {num(core['ece'])} |",
        f"| Todos: mezcla guardada | {production['n']} | {pct(production['accuracy'])} | {num(production['log_loss'])} | {num(production['brier'])} | {num(production['ece'])} |",
        f"| JEV activado: core pareado | {core_eligible['n']} | {pct(core_eligible['accuracy'])} | {num(core_eligible['log_loss'])} | {num(core_eligible['brier'])} | {num(core_eligible['ece'])} |",
        f"| JEV activado: mezcla guardada | {prod_eligible['n']} | {pct(prod_eligible['accuracy'])} | {num(prod_eligible['log_loss'])} | {num(prod_eligible['brier'])} | {num(prod_eligible['ece'])} |",
        "",
        f"JEV tuvo peso efectivo mayor que cero en **{len(jev_active)} de {len(rows)}** partidos enlazados. La comparación pareada usa esa cohorte; el simulador de pesos puede incluir además predicciones nuevas con gate aprobado y peso configurado en cero.",
        "",
        "## Simulador de peso JEV",
        "",
        f"Se recorrieron pesos de **0.00 a 1.00 en pasos de 0.05**. La partición se hizo por días completos: {len(train_dates)} días ({len(train)} partidos) para calibración y {len(holdout_dates)} días ({len(holdout)} partidos) como holdout. La selección minimiza log loss en calibración; el holdout se consulta una sola vez para estimar generalización.",
        "",
        "| Peso JEV | N calibración | Log loss calibración | Brier calibración | Exactitud calibración | N holdout | Log loss holdout | Brier holdout |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weight in grid:
        tr = next((m for w, scope, m in candidates if w == weight and scope == "train"), metric([], "base"))
        ho = next((m for w, scope, m in candidates if w == weight and scope == "holdout"), metric([], "base"))
        lines.append(f"| {weight:.2f} | {tr['n']} | {num(tr['log_loss'])} | {num(tr['brier'])} | {pct(tr['accuracy'])} | {ho['n']} | {num(ho['log_loss'])} | {num(ho['brier'])} |")
    lines += ["", f"Peso elegido solo con calibración: **{best_weight:.2f}** (si existe muestra elegible); su holdout: n={best_holdout['n']}, log loss={num(best_holdout['log_loss'])}, Brier={num(best_holdout['brier'])}, exactitud={pct(best_holdout['accuracy'])}." if best_weight is not None else "No hay suficientes partidos elegibles para elegir un peso."]
    if best_full:
        lines.append(f"Como referencia retrospectiva, el mejor peso al evaluar toda la cohorte elegible es **{best_full[0]:.2f}** (log loss {num(best_full[1]['log_loss'])}, Brier {num(best_full[1]['brier'])}); esta cifra está optimizada sobre la muestra evaluada y **no** debe interpretarse como rendimiento futuro ni usarse como estimación imparcial.")
    lines += [
        "",
        "### Validación walk-forward",
        "",
        f"Para cada fecha de holdout se volvió a elegir el peso usando solo los partidos elegibles anteriores (mínimo 20); luego se evaluó esa fecha. Resultado acumulado: n={walk_forward_metric['n']}, exactitud={pct(walk_forward_metric['accuracy'])}, log loss={num(walk_forward_metric['log_loss'])}, Brier={num(walk_forward_metric['brier'])}, ECE={num(walk_forward_metric['ece'])}. Core en esos mismos partidos: exactitud={pct(walk_forward_core['accuracy'])}, log loss={num(walk_forward_core['log_loss'])}, Brier={num(walk_forward_core['brier'])}, ECE={num(walk_forward_core['ece'])}.",
        "",
        "Pesos seleccionados por fecha (ponderados por cantidad de partidos): " + (", ".join(f"{weight:.2f}: {count}" for weight, count in sorted(weight_history.items())) or "sin folds con muestra previa suficiente") + ".",
        "",
        "### Viabilidad de un peso dinámico por relevancia y probabilidades",
        "",
        f"Hay {len(jev_candidates)} partidos enlazados con una respuesta JEV válida ({len(eligible)} se fusionaron históricamente). Solo {relevance_snapshots} tienen score de relevancia y {confidence_snapshots} confianza JEV guardados. Por tanto, el histórico no permite simular honestamente una regla dinámica que dependa de ambas señales. Las corridas nuevas guardan score/gates y, desde este cambio, también persistirán la confianza JEV.",
        "",
        "Como diagnóstico parcial, se agrupan todas las respuestas JEV válidas (incluidas las 23 que el gate histórico no mezcló) por su probabilidad máxima. La concentración no debe tratarse como confianza calibrada: una distribución extrema puede ser sobreconfianza.",
        "",
        f"| Grupo por máxima P_JEV | N | Log loss core | Log loss JEV | Log loss mezcla {JEV_PRIORITY_WEIGHT:.0%} | Brier core | Brier JEV | Brier mezcla {JEV_PRIORITY_WEIGHT:.0%} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, subset in jev_probability_buckets:
        blend_rows = [{**row, "blend_current": mix(row["base"], row["jev"], JEV_PRIORITY_WEIGHT)} for row in subset]
        core_metric, jev_metric, blend_metric = metric(subset, "base"), metric(subset, "jev"), metric(blend_rows, "blend_current")
        lines.append(f"| {label} | {core_metric['n']} | {num(core_metric['log_loss'])} | {num(jev_metric['log_loss'])} | {num(blend_metric['log_loss'])} | {num(core_metric['brier'])} | {num(jev_metric['brier'])} | {num(blend_metric['brier'])} |")
    lines += [
        "",
        "Lectura: probar una regla basada solo en lo extrema que sea P_JEV sería arriesgado. Para producción, primero se debe calibrar confianza/relevancia fuera de muestra; una opción de investigación es `w = w_max × g(relevancia, certeza/frescura de fuentes, calibración de confianza)`, con `w_max` conservador. La divergencia entre core y JEV puede activar revisión o limitar el peso, pero no debe anular automáticamente evidencia fuerte y verificable.",
        "",
        "### Cobertura por categoría y liga",
        "",
        "| Categoría JEV | N | Log loss core | Log loss final | Brier core | Brier final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    topics = sorted({category for row in eligible for category in row["news_categories"]})
    for topic in topics:
        subset = [row for row in eligible if topic in row["news_categories"]]
        lines.append(f"| {topic} | {len(subset)} | {num(metric(subset, 'base')['log_loss'])} | {num(metric(subset, 'production')['log_loss'])} | {num(metric(subset, 'base')['brier'])} | {num(metric(subset, 'production')['brier'])} |")
    if not topics:
        lines.append("| Sin snapshots históricos de categorías | 0 | — | — | — | — |")
    lines += ["", "| Liga | N | Log loss core | Log loss final | Brier core | Brier final |", "|---|---:|---:|---:|---:|---:|"]
    for league in sorted({row["league_name"] for row in rows}):
        subset = [row for row in rows if row["league_name"] == league]
        lines.append(f"| {league} | {len(subset)} | {num(metric(subset, 'base')['log_loss'])} | {num(metric(subset, 'production')['log_loss'])} | {num(metric(subset, 'base')['brier'])} | {num(metric(subset, 'production')['brier'])} |")
    lines += [
        "",
        "## Lectura y límites",
        "",
        "- La diferencia entre core y mezcla guardada describe el cambio observado tras activar JEV dentro de las predicciones que quedaron almacenadas; por sí sola no demuestra causalidad ni mejora estable.",
        "- El peso simulado mezcla las probabilidades JEV ya generadas. No vuelve a ejecutar el algoritmo JEV ni cambia qué partidos superaron su filtro de relevancia.",
        "- La muestra es corta y reciente, y los partidos sin resultado enlazado se excluyen. La selección del peso puede variar bastante con pocos casos; conviene acumular varias jornadas antes de adoptar un peso distinto.",
        "- No se calcula ROI: faltan cuotas históricas completas y homogéneas para comparar las tres salidas en todos los encuentros; reconstruir apuestas solo desde picks seleccionados introduce sesgo de selección.",
        "- La comparación sin forma reconstruye el mismo softmax con rating Impect y ventaja local, anulando `form_delta`; permite estimar la contribución de forma en predicciones guardadas. No hay contrafactuales históricos que aíslen cada KPI Impect ni cada categoría de noticias, así que esas contribuciones no se atribuyen por separado.",
        "- Los parámetros `alpha`/`gamma_d` quedan capturados para ejecuciones nuevas. Las filas antiguas que no los guardaron se reconstruyen con el archivo de parámetros actual, así que la comparación sin forma de esas filas es aproximada.",
        "- El walk-forward es exploratorio y la muestra corta puede producir pesos inestables. No se cambia el peso de producción hasta acumular más fechas y validar en un holdout posterior.",
        "",
        "### Partidos usados",
        "",
        "| Fecha | Partido | Resultado | Fuente | Peso JEV guardado |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        key = (row["date"], normalize(row["home"]), normalize(row["away"]))
        lines.append(f"| {row['date']} | {row['home']} vs {row['away']} | {row['actual']} | {actual_sources.get(key, '—')} | {row['jev_weight']:.2f} |")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe: {REPORT_PATH}")
    print(f"Muestra: {len(rows)}; JEV activo: {len(eligible)}; peso calibrado: {best_weight}")


if __name__ == "__main__":
    main()
