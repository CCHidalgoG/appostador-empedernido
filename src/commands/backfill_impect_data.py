"""Descarga y backfill integral de datos Impect (ratings, scores, averages, predictions, player_scores, matches).

Persiste en raw.artifacts de DuckDB y sincroniza output.impect_results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.apostador.config import IMPECT_RAW_DIR
from src.apostador.duckdb_store import get_store, write_json

# Iteraciones principales 2026 y temporadas recientes
CURRENT_ITERATIONS = [
    1752,  # Colombia Liga Dimayor 2026
    2176,  # Mexico Liga MX 26/27
    1799,  # Brazil Série A 2026
    1864,  # MLS 2026
    1930,  # Argentina Liga Profesional 2026
    2110,  # Spain LaLiga 26/27
    2111,  # Italy Serie A 26/27
    2143,  # Belgium Jupiler Pro 26/27
    2155,  # Russia Premier Liga 26/27
    1929,  # Brazil Série B 2026
    1802,  # Chile Primera 2026
    1801,  # Peru Liga 1 2026
    1863,  # Ecuador LigaPro 2026
    1813,  # Uruguay Liga AUF 2026
    1794,  # Paraguay Primera 2026
    1973,  # Bolivia División Prof 2026
    1800,  # Venezuela Liga FUTVE 2026
]

HISTORICAL_ITERATIONS = [
    1387,  # Spain LaLiga 25/26
    1009,  # Spain LaLiga 24/25
    736,   # Spain LaLiga 23/24
    1402,  # Italy Serie A 25/26
    1002,  # Italy Serie A 24/25
    734,   # Italy Serie A 23/24
    1396,  # Mexico Liga MX 25/26
    975,   # Mexico Liga MX 24/25
    1277,  # Brazil Série A 2025
    835,   # Brazil Série A 2024
    1210,  # Argentina Liga Profesional 2025
    836,   # Argentina Liga Profesional 2024
    1217,  # MLS 2025
    833,   # MLS 2024
    1207,  # Colombia 2025
    866,   # Colombia 2024
]


def _is_forbidden_error(exc: Exception) -> bool:
    """Detecta respuestas HTTP 403 cuando la iteración queda fuera de la licencia Impect activa."""
    msg = str(exc).lower()
    return "403" in msg or "do not have access to this resource" in msg


def _save_versioned_impect_artifact(
    prefix: str,
    iteration: int,
    records: list[dict],
    snap_date: str,
) -> list[dict]:
    """Inyecta as_of_date y guarda clave actual + snapshot histórico point-in-time."""
    stamped: list[dict] = []
    for row in records:
        item = dict(row)
        item.setdefault("as_of_date", snap_date)
        stamped.append(item)
    write_json(IMPECT_RAW_DIR / f"{prefix}_{iteration}.json", stamped)
    write_json(IMPECT_RAW_DIR / f"{prefix}_{iteration}_{snap_date}.json", stamped)
    return stamped


def _ensure_cached_snapshot_on_403(iteration: int, store=None, fallback_date: str | None = None) -> list[str]:
    """Cuando Impect API devuelve 403, crea snapshots fechados desde la caché previa en DuckDB."""
    store = store or get_store()
    ratings_cached = store.get("raw", f"impect/ratings_{iteration}.json", [])
    dates = [
        str(r.get("date") or r.get("as_of_date") or "")[:10]
        for r in (ratings_cached or [])
        if isinstance(r, dict) and (r.get("date") or r.get("as_of_date"))
    ]
    inferred_date = max(dates) if dates else (fallback_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10]

    preserved: list[str] = []
    for prefix in ("ratings", "scores", "averages", "player_scores"):
        key = f"impect/{prefix}_{iteration}.json"
        cached = store.get("raw", key, None)
        if not isinstance(cached, list) or not cached:
            continue
        snap_key = f"impect/{prefix}_{iteration}_{inferred_date}.json"
        if store.get("raw", snap_key, None) is None:
            stamped = []
            for row in cached:
                item = dict(row) if isinstance(row, dict) else row
                if isinstance(item, dict):
                    item.setdefault("as_of_date", inferred_date)
                stamped.append(item)
            write_json(IMPECT_RAW_DIR / f"{prefix}_{iteration}_{inferred_date}.json", stamped)
        preserved.append(prefix)
    return preserved


def backfill_iteration_data(
    iterations: list[int],
    fetch_predictions: bool = True,
    fetch_ratings: bool = True,
    fetch_averages: bool = True,
    fetch_scores: bool = True,
    fetch_player_scores: bool = True,
    fetch_matches: bool = True,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    load_dotenv()
    username = os.getenv("USERNAME_IMPECT")
    password = os.getenv("PASSWORD_IMPECT")
    if not username or not password:
        raise RuntimeError("Faltan USERNAME_IMPECT o PASSWORD_IMPECT en .env")

    try:
        import impectPy as ip
    except ImportError as exc:
        raise RuntimeError("impectPy no está instalado") from exc

    snap_date = (as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10]
    token = ip.getAccessToken(username=username, password=password)
    store = get_store()
    stats = {"processed": 0, "skipped_forbidden": 0, "errors": 0, "details": [], "as_of_date": snap_date}

    for iteration in iterations:
        it_stat = {"iteration": iteration, "saved": [], "forbidden": False}
        print(f"\n>>> Procesando Iteración {iteration}...")

        def _handle_forbidden(err: Exception) -> bool:
            if not _is_forbidden_error(err):
                return False
            preserved = _ensure_cached_snapshot_on_403(iteration, store=store, fallback_date=snap_date)
            it_stat["forbidden"] = True
            it_stat["saved"].extend(f"{p} (cache)" for p in preserved)
            stats["skipped_forbidden"] += 1
            print(
                f"  [SKIP] Iteración {iteration}: HTTP 403 (fuera de licencia activa en API; "
                f"caché DuckDB preservada: {', '.join(preserved) or 'sin caché previa'})"
            )
            return True

        # 1. Match Predictions (xG modelo, xG experto, xG mercado)
        if fetch_predictions:
            try:
                preds = ip.getMatchPredictions(iteration=iteration, token=token)
                if preds is not None and not preds.empty:
                    records = preds.to_dict(orient="records")
                    write_json(IMPECT_RAW_DIR / f"predictions_{iteration}.json", records)
                    it_stat["saved"].append(f"predictions ({len(records)})")
                    print(f"  [OK] Predictions: {len(records)} partidos")
            except Exception as e:
                if _handle_forbidden(e):
                    stats["details"].append(it_stat)
                    continue
                print(f"  [WARN] Predictions error: {e}")

        # 2. Squad Ratings
        if fetch_ratings:
            try:
                ratings = ip.getSquadRatings(iteration=iteration, token=token)
                if ratings is not None and not ratings.empty:
                    records = _save_versioned_impect_artifact(
                        "ratings", iteration, ratings.to_dict(orient="records"), snap_date
                    )
                    it_stat["saved"].append(f"ratings ({len(records)})")
                    print(f"  [OK] Ratings: {len(records)} filas")
            except Exception as e:
                if _handle_forbidden(e):
                    stats["details"].append(it_stat)
                    continue
                print(f"  [WARN] Ratings error: {e}")

        # 3. Squad Iteration Scores (xG probabilities, PPDA, Field tilt, duels)
        if fetch_scores:
            try:
                scores = ip.getSquadIterationScores(iteration=iteration, token=token)
                if scores is not None and not scores.empty:
                    records = _save_versioned_impect_artifact(
                        "scores", iteration, scores.to_dict(orient="records"), snap_date
                    )
                    it_stat["saved"].append(f"scores ({len(records)})")
                    print(f"  [OK] Squad Scores: {len(records)} equipos")
            except Exception as e:
                if _handle_forbidden(e):
                    stats["details"].append(it_stat)
                    continue
                print(f"  [WARN] Squad Scores error: {e}")

        # 4. Squad Iteration Averages (Packing, Shot xG, Conceded xG, presses, PXT, PSxG)
        if fetch_averages:
            try:
                averages = ip.getSquadIterationAverages(iteration=iteration, token=token)
                if averages is not None and not averages.empty:
                    records = _save_versioned_impect_artifact(
                        "averages", iteration, averages.to_dict(orient="records"), snap_date
                    )
                    it_stat["saved"].append(f"averages ({len(records)})")
                    print(f"  [OK] Squad Averages: {len(records)} equipos")
            except Exception as e:
                if _handle_forbidden(e):
                    stats["details"].append(it_stat)
                    continue
                print(f"  [WARN] Squad Averages error: {e}")

        # 5. Player Iteration Scores (IMPECT_SCORE_PACKING, PXT, Scorer, Def Intervention)
        if fetch_player_scores:
            try:
                player_scores = ip.getPlayerIterationScores(iteration=iteration, token=token)
                if player_scores is not None and not player_scores.empty:
                    records = _save_versioned_impect_artifact(
                        "player_scores", iteration, player_scores.to_dict(orient="records"), snap_date
                    )
                    it_stat["saved"].append(f"player_scores ({len(records)})")
                    print(f"  [OK] Player Scores: {len(records)} jugadores")
            except Exception as e:
                if _handle_forbidden(e):
                    stats["details"].append(it_stat)
                    continue
                print(f"  [WARN] Player Scores error: {e}")

        # 6. Matches
        if fetch_matches:
            try:
                matches = ip.getMatches(iteration=iteration, token=token)
                if matches is not None and not matches.empty:
                    from src.apostador.impect_results_fetcher import _finished_match_records
                    records = _finished_match_records(iteration, matches)
                    store.save_impect_results(records)
                    write_json(IMPECT_RAW_DIR / f"matches_{iteration}.json", records)
                    it_stat["saved"].append(f"matches ({len(records)})")
                    print(f"  [OK] Matches: {len(records)} partidos finalizados guardados")
            except Exception as e:
                if _handle_forbidden(e):
                    stats["details"].append(it_stat)
                    continue
                print(f"  [WARN] Matches error: {e}")

        stats["processed"] += 1
        stats["details"].append(it_stat)
        time.sleep(0.5)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-historical", action="store_true", help="Incluir también temporadas anteriores (2024, 2025)")
    parser.add_argument("--iterations", nargs="*", type=int, default=None, help="Iteraciones específicas")
    parser.add_argument("--skip-players", action="store_true", help="Omitir player scores para mayor velocidad")
    parser.add_argument("--as-of-date", type=str, default=None, help="Fecha as-of (YYYY-MM-DD) para snapshots")
    args = parser.parse_args()

    iters = args.iterations or (CURRENT_ITERATIONS + (HISTORICAL_ITERATIONS if args.all_historical else []))
    print(f"Iniciando backfill para {len(iters)} iteraciones de Impect...")
    stats = backfill_iteration_data(
        iters,
        fetch_player_scores=not args.skip_players,
        as_of_date=args.as_of_date,
    )
    print("\n=== Backfill Finalizado ===")
    print(f"Iteraciones procesadas: {stats['processed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
