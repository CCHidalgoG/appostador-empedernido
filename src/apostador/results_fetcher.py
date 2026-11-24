"""Fetch real match results from API-Football and reconcile them against saved bets.

Brave Search is a general web/news search API (no live-scores endpoint), so it is not
suitable for this. API-Football already exposes finished-match scores via its
`/fixtures?date=` endpoint (the same one used to fetch pre-match fixtures), so we reuse
that key here instead.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv

from .aligner import TeamAligner
from .config import APIFOOTBALL_RAW_DIR
from .duckdb_store import get_store, write_json

load_dotenv()

APIFOOTBALL_KEY = os.getenv("APIFOOTBALL")
FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
ESPN_SOCCER_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
FINISHED_STATUSES = {"FT", "AET", "PEN"}


def fetch_fixtures_for_date(date: str) -> list[dict]:
    """Always fetches fresh fixture data (results don't exist yet at prediction time)."""
    if not APIFOOTBALL_KEY:
        raise RuntimeError("Falta APIFOOTBALL en el entorno (.env)")
    resp = requests.get(FIXTURES_URL, headers={"x-apisports-key": APIFOOTBALL_KEY},
                        params={"date": date}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    write_json(APIFOOTBALL_RAW_DIR / f"fixtures_result_{date}.json", data)
    errors = data.get("errors")
    if errors:
        raise RuntimeError(f"Aviso API-Football para {date}: {errors}")
    return data.get("response", [])


def fetch_espn_soccer_for_date(date: str, store=None, use_cache: bool = True) -> list[dict]:
    """Descarga partidos y marcadores reales desde ESPN Open Soccer Scoreboard (sin API key)."""
    clean_date = str(date)[:10]
    ymd = clean_date.replace("-", "")
    cache_key = f"open_soccer/espn_scoreboard_{clean_date}"
    store = store or get_store()

    if use_cache:
        cached = store.get("raw", cache_key)
        if cached and isinstance(cached, dict) and cached.get("fixtures"):
            fxs = cached["fixtures"]
            for f in fxs:
                fid = (f.get("fixture") or {}).get("id")
                if isinstance(fid, str):
                    digits = "".join(ch for ch in fid if ch.isdigit())
                    f["fixture"]["id"] = int(digits) if digits else None
            if fxs and all((f.get("fixture") or {}).get("status", {}).get("short") in FINISHED_STATUSES for f in fxs):
                return fxs

    resp = requests.get(
        ESPN_SOCCER_SCOREBOARD_URL,
        params={"dates": ymd, "limit": 300},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json() or {}
    parsed: list[dict] = []
    for ev in payload.get("events") or []:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_comp or not away_comp:
            continue

        home_name = (home_comp.get("team") or {}).get("displayName")
        away_name = (away_comp.get("team") or {}).get("displayName")
        if not home_name or not away_name:
            continue

        status_type = (comp.get("status") or {}).get("type") or {}
        completed = bool(status_type.get("completed", False))
        status_short = "FT" if completed else ("LIVE" if status_type.get("state") == "in" else "NS")

        try:
            hs = int(home_comp.get("score")) if home_comp.get("score") is not None else None
            aws = int(away_comp.get("score")) if away_comp.get("score") is not None else None
        except (TypeError, ValueError):
            hs, aws = None, None

        raw_id = str(ev.get("id") or "")
        digits_id = "".join(ch for ch in raw_id if ch.isdigit())
        fixture_id = int(digits_id) if digits_id else None

        parsed.append({
            "fixture": {"id": fixture_id, "status": {"short": status_short}},
            "teams": {"home": {"name": home_name}, "away": {"name": away_name}},
            "goals": {"home": hs, "away": aws},
            "source": "espn_open",
        })

    if parsed:
        from datetime import datetime, timezone
        store.put("raw", cache_key, {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": clean_date,
            "fixtures": parsed,
        })
    return parsed


def _match_outcome(outcome: str, home_goals, away_goals, home_team=None, away_team=None):
    from .outcome_utils import normalize_selection_to_side

    side = normalize_selection_to_side(outcome, home_team, away_team)
    if side is None or home_goals is None or away_goals is None:
        return None
    if side == "local":
        return home_goals > away_goals
    if side == "visita":
        return away_goals > home_goals
    return home_goals == away_goals


def _find_fixture(aligner: TeamAligner, fixtures: list[dict], home_team: str, away_team: str):
    valid = [
        f for f in fixtures
        if (f.get("teams") or {}).get("home", {}).get("name")
        and (f.get("teams") or {}).get("away", {}).get("name")
    ]
    home_names = list(dict.fromkeys(f["teams"]["home"]["name"] for f in valid))
    home_match, _ = aligner.find_best_match(home_team, home_names)
    if not home_match:
        return None
    home_cands = [f for f in valid if f["teams"]["home"]["name"] == home_match]
    away_names = list(dict.fromkeys(f["teams"]["away"]["name"] for f in home_cands))
    away_match, _ = aligner.find_best_match(away_team, away_names)
    if not away_match:
        return None
    matched = [f for f in home_cands if f["teams"]["away"]["name"] == away_match]
    return matched[0] if matched else None


def resolve_pending_results(dates: list[str] | None = None, verbose: bool = True) -> dict:
    """Fetches results for pending recommendations and stores them in DuckDB.

    Returns a summary dict with won/lost/pending/unmatched leg counts and a
    per-run_id breakdown (a parlay wins only if all of its legs win).
    """
    store = get_store()
    # Primero conciliar contra resultados Impect ya guardados en DuckDB
    try:
        from .impect_results_fetcher import reconcile_pending_impect_results
        reconcile_pending_impect_results(store=store, verbose=verbose)
    except Exception:
        pass

    pending = store.pending_recommendations()
    if dates:
        pending = [row for row in pending if row[2] in dates]

    summary = {"won": 0, "lost": 0, "pending": 0, "unmatched": 0, "runs": defaultdict(list)}
    if not pending:
        if verbose:
            print("No hay recomendaciones pendientes de resultado.")
        return summary

    aligner = TeamAligner()
    fixtures_cache: dict[str, list[dict]] = {}

    for run_id, row_number, event_date, home_team, away_team, outcome in pending:
        if not event_date:
            summary["unmatched"] += 1
            continue
        if event_date not in fixtures_cache:
            combined_fixtures: list[dict] = []
            try:
                combined_fixtures.extend(fetch_fixtures_for_date(event_date))
            except (requests.RequestException, RuntimeError) as exc:
                if verbose:
                    print(f"{exc}")
            # Complementar con ESPN Open Soccer Scoreboard (incluyendo día previo para horarios nocturnos LATAM/USA en UTC)
            adj_dates = [event_date]
            try:
                from datetime import datetime, timedelta
                prev_day = (datetime.strptime(str(event_date)[:10], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                adj_dates.append(prev_day)
            except Exception:
                pass
            for d_cand in adj_dates:
                try:
                    combined_fixtures.extend(fetch_espn_soccer_for_date(d_cand, store=store))
                except Exception:
                    pass
            fixtures_cache[event_date] = combined_fixtures
            time.sleep(0.3)

        fixture = _find_fixture(aligner, fixtures_cache[event_date], home_team, away_team)
        if fixture is None:
            summary["unmatched"] += 1
            if verbose:
                print(f"No se pudo emparejar: {home_team} vs {away_team} ({event_date})")
            continue

        status = fixture["fixture"]["status"]["short"]
        home_goals, away_goals = fixture["goals"]["home"], fixture["goals"]["away"]
        fixture_id = fixture["fixture"]["id"]
        source = fixture.get("source") or "api"

        if status not in FINISHED_STATUSES:
            store.save_result(run_id, row_number, fixture_id, status, home_goals, away_goals, None, source=source)
            summary["pending"] += 1
            continue

        bet_won = _match_outcome(outcome, home_goals, away_goals, home_team, away_team)
        store.save_result(run_id, row_number, fixture_id, status, home_goals, away_goals, bet_won, source=source)
        summary["runs"][run_id].append(bet_won)
        summary["won" if bet_won else "lost"] += 1

    return summary
