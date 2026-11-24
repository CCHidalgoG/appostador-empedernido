"""Fetch finished Impect matches, persist API rows and reconcile pending bets."""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

from dotenv import load_dotenv

from .aligner import TeamAligner, normalize_name
from .duckdb_store import get_store

load_dotenv()

ITERATIONS = (1752, 2176, 1799, 1864, 1930, 2110, 2111, 2143, 2155,
              1929, 1802, 1801, 1863, 1813, 1794, 1973, 1800)
RESULT_TYPE_STATUS = {"REGULAR": "FT", "EXTRA_TIME": "AET", "PENALTIES": "PEN"}


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or str(value).lower() in {"nan", "<na>", "none"}:
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _finished_match_records(iteration_id: int, matches) -> list[dict]:
    records = []
    for _, row in matches.iterrows():
        home_goals = _optional_int(row.get("homeSquadGoals"))
        away_goals = _optional_int(row.get("awaySquadGoals"))
        home, away = row.get("homeSquadName"), row.get("awaySquadName")
        match_id = _optional_int(row.get("id"))
        if home_goals is None or away_goals is None or not home or not away or match_id is None:
            continue
        result = "local" if home_goals > away_goals else ("empate" if home_goals == away_goals else "visita")
        scheduled_at = row.get("scheduledDate")
        if hasattr(scheduled_at, "isoformat"):
            scheduled_at = scheduled_at.isoformat()
        if scheduled_at is not None and str(scheduled_at).lower() in {"nan", "nat", "<na>"}:
            scheduled_at = None
        day_index = _optional_int(row.get("matchDayIndex"))
        home_id, away_id = _optional_int(row.get("homeSquadId")), _optional_int(row.get("awaySquadId"))
        result_type = row.get("resultType")
        if result_type is not None and str(result_type).lower() in {"nan", "<na>", "none"}:
            result_type = None
        api_result = row.get("result")
        if api_result is not None and str(api_result).lower() in {"nan", "<na>", "none"}:
            api_result = None
        record = {
            "iteration_id": int(iteration_id), "match_id": match_id,
            "scheduled_at": scheduled_at, "match_day_index": day_index,
            "home_squad_id": home_id, "home_team": str(home),
            "away_squad_id": away_id, "away_team": str(away),
            "home_goals": home_goals, "away_goals": away_goals,
            "result_outcome": result, "result_type": str(result_type) if result_type is not None else None,
            "api_result": str(api_result) if api_result is not None else None,
        }
        records.append(record)
    return records


def _event_day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _scheduled_day(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return _event_day(value)


def _country_iterations(payload: Any) -> set[int] | None:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    country = normalize_name((payload or {}).get("country", ""))
    country_map = {
        "colombia": {1752}, "mexico": {2176}, "brasil": {1799, 1929}, "brazil": {1799, 1929},
        "usa": {1864}, "spain": {2110}, "italy": {2111}, "belgium": {2143},
        "russia": {2155}, "argentina": {1930}, "chile": {1802},
        "peru": {1801}, "ecuador": {1863}, "uruguay": {1813},
        "paraguay": {1794}, "bolivia": {1973}, "venezuela": {1800},
    }
    return country_map.get(country)


def _match_record(home: str, away: str, event_day: date | None,
                  candidates: list[dict], expected_iterations: set[int] | None,
                  aligner: TeamAligner) -> dict | None:
    if not home or not away or not event_day:
        return None
    candidates = [r for r in candidates if _scheduled_day(r.get("scheduled_at")) is not None
                  and abs((_scheduled_day(r.get("scheduled_at")) - event_day).days) <= 1]
    if expected_iterations:
        scoped = [r for r in candidates if r["iteration_id"] in expected_iterations]
        if scoped:
            candidates = scoped
    if not candidates:
        return None

    home_names = list(dict.fromkeys(r["home_team"] for r in candidates))
    home_name, home_score = aligner.find_best_match(home, home_names, threshold=0.85)
    if not home_name or home_score < 0.85:
        return None
    home_candidates = [r for r in candidates if r["home_team"] == home_name]
    away_names = list(dict.fromkeys(r["away_team"] for r in home_candidates))
    away_name, away_score = aligner.find_best_match(away, away_names, threshold=0.85)
    if not away_name or away_score < 0.85:
        return None
    pair_matches = [r for r in home_candidates if r["away_team"] == away_name]
    if not pair_matches:
        return None

    pair_matches.sort(key=lambda r: (
        abs((_scheduled_day(r["scheduled_at"]) - event_day).days),
        r["iteration_id"], r["match_id"],
    ))
    best = pair_matches[0]
    best_day_diff = abs((_scheduled_day(best["scheduled_at"]) - event_day).days)
    ambiguous = [r for r in pair_matches if
                 abs((_scheduled_day(r["scheduled_at"]) - event_day).days) == best_day_diff]
    if len(ambiguous) > 1:
        return None
    return best


def reconcile_pending_impect_results(store=None, verbose: bool = True) -> dict:
    """Link previously stored Impect scores to pending recommendation rows."""
    store = store or get_store()
    all_records = [
        {"iteration_id": row[0], "match_id": row[1], "scheduled_at": row[2],
         "match_day_index": row[3], "home_squad_id": row[4], "home_team": row[5],
         "away_squad_id": row[6], "away_team": row[7], "home_goals": row[8],
         "away_goals": row[9], "result_outcome": row[10], "result_type": row[11]}
        for row in store.list_impect_results()
    ]
    pending = store.pending_result_recommendations()
    aligner = TeamAligner()
    # Common provider differences where both opponents and date still have to agree.
    aligner.static_mappings.update({
        "bahia": "EC Bahia",
        "atletico san luis": "Atlético de San Luis",
        "gimnasia la plata": "Club de Gimnasia y Esgrima La Plata",
        "tigre ba": "Club Atlético Tigre",
    })
    matched_rows, unmatched_rows = 0, []
    for run_id, row_number, event_date, home, away, selection, payload in pending:
        expected_iterations = _country_iterations(payload)
        record = _match_record(home, away, _event_day(event_date), all_records,
                               expected_iterations, aligner)
        if record is None:
            unmatched_rows.append((event_date, home, away))
            continue
        from .outcome_utils import compute_bet_won

        bet_won = compute_bet_won(selection, record["result_outcome"], home, away)
        status = RESULT_TYPE_STATUS.get(record["result_type"], "FT")
        store.save_result(run_id, row_number, record["match_id"], status,
                          record["home_goals"], record["away_goals"], bet_won,
                          source="impect")
        matched_rows += 1
    if verbose:
        print(f"Recomendaciones enlazadas con Impect: {matched_rows}; sin emparejar: {len(unmatched_rows)}")
    return {"recommendations_updated": matched_rows,
            "recommendations_unmatched": len(unmatched_rows), "unmatched": unmatched_rows}


def fetch_and_reconcile_impect_results(iterations: tuple[int, ...] = ITERATIONS,
                                       verbose: bool = True) -> dict:
    """Fetch all configured iterations, save scored API rows and update pending bets."""
    username, password = os.getenv("USERNAME_IMPECT"), os.getenv("PASSWORD_IMPECT")
    if not username or not password:
        raise RuntimeError("Faltan USERNAME_IMPECT o PASSWORD_IMPECT en el entorno")
    try:
        import impectPy as ip
    except ImportError as exc:
        raise RuntimeError("impectPy no está instalado") from exc

    store = get_store()
    token = ip.getAccessToken(username=username, password=password)
    all_records: list[dict] = []
    fetched_iterations, failed_iterations = [], []
    for iteration_id in iterations:
        try:
            matches = ip.getMatches(iteration=int(iteration_id), token=token)
            records = _finished_match_records(int(iteration_id), matches)
            all_records.extend(records)
            fetched_iterations.append(int(iteration_id))
            store.save_impect_results(records)
            store.put("raw", f"impect/matches_iteration_{iteration_id}.json", records)
            if verbose:
                print(f"Iteración {iteration_id}: {len(records)} partidos con marcador guardados")
        except Exception as exc:  # noqa: BLE001
            failed_iterations.append({"iteration_id": int(iteration_id), "error": str(exc)[:250]})
            if verbose:
                print(f"Iteración {iteration_id}: error consultando Impect ({type(exc).__name__})")

    reconciliation = reconcile_pending_impect_results(store, verbose=verbose)

    summary = {
        "iterations_fetched": fetched_iterations,
        "iterations_failed": failed_iterations,
        "api_results_saved": len(all_records),
        **reconciliation,
    }
    return summary
