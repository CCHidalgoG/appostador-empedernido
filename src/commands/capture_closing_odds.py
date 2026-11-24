"""Captura de cuotas de cierre propias (T-5 min pre-kickoff) en raw.odds_snapshots (TASK-09)."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from src.apostador.aligner import TeamAligner
from src.apostador.duckdb_store import get_store
from src.commands.main import (
    SUPPORTED_LEAGUES,
    load_or_fetch_apifootball_odds,
    load_or_fetch_odds,
    snapshot_rows_from_apifootball,
    snapshot_rows_from_theoddsapi,
)

load_dotenv()
BOGOTA_TZ = ZoneInfo("America/Bogota")


def capture_closing_snapshots(
    target_date: str | None = None,
    window_minutes: int = 15,
    capture_all: bool = False,
    store=None,
) -> int:
    """Captura snapshots de cierre (is_closing=True) para partidos cercanos al kickoff."""
    active_store = store or get_store()
    date_str = target_date or datetime.now(BOGOTA_TZ).date().isoformat()
    now_utc = datetime.now(timezone.utc)
    aligner = TeamAligner()

    with active_store._connect() as con:
        rows = con.execute(
            """SELECT DISTINCT home_team, away_team, event_date, payload
               FROM output.recommendations
               WHERE event_date LIKE ?""",
            [f"{date_str}%"],
        ).fetchall()

    if not rows:
        print(f"No hay recomendaciones registradas para {date_str}.")
        return 0

    saved_total = 0
    odds_cache: dict[str, list[dict]] = {}
    apifootball_key = os.getenv("APIFOOTBALL", "")

    for league_id, info in SUPPORTED_LEAGUES.items():
        skey = info.get("odds_sport_key")
        if skey and skey not in odds_cache:
            events = load_or_fetch_odds(skey, force_refresh=True, min_date=date_str)
            odds_cache[skey] = events
            snap_rows = snapshot_rows_from_theoddsapi(
                events,
                sport_key=skey,
                league_name=info["name"],
                is_closing=True,
            )
            if not capture_all:
                filtered = []
                for r in snap_rows:
                    ev_t = r.get("event_time")
                    if not ev_t:
                        continue
                    try:
                        kick_dt = datetime.fromisoformat(str(ev_t).replace("Z", "+00:00"))
                        mins = (kick_dt - now_utc).total_seconds() / 60.0
                        if -5.0 <= mins <= float(window_minutes):
                            filtered.append(r)
                    except ValueError:
                        continue
                snap_rows = filtered
            if snap_rows:
                saved_total += active_store.save_odds_snapshots(snap_rows)

    # Captura API-Football para ligas LATAM sin sport_key en The Odds API
    if apifootball_key:
        for home_team, away_team, _, _ in rows:
            try:
                resp = requests.get(
                    "https://v3.football.api-sports.io/fixtures",
                    headers={"x-apisports-key": apifootball_key},
                    params={"date": date_str, "timezone": "America/Bogota"},
                    timeout=15,
                )
                fixtures = (resp.json().get("response") or []) if resp.status_code == 200 else []
            except Exception:
                fixtures = []
            for fix in fixtures:
                lid = (fix.get("league") or {}).get("id")
                l_info = SUPPORTED_LEAGUES.get(lid)
                if not l_info or l_info.get("odds_sport_key"):
                    continue
                h_af = (fix.get("teams") or {}).get("home", {}).get("name", "")
                a_af = (fix.get("teams") or {}).get("away", {}).get("name", "")
                hm, _ = aligner.find_best_match(home_team, [h_af], threshold=0.75)
                am, _ = aligner.find_best_match(away_team, [a_af], threshold=0.75)
                if not (hm and am):
                    continue
                fid = (fix.get("fixture") or {}).get("id")
                ev_time = (fix.get("fixture") or {}).get("date")
                if not capture_all and ev_time:
                    try:
                        kick_dt = datetime.fromisoformat(str(ev_time).replace("Z", "+00:00"))
                        mins = (kick_dt - now_utc).total_seconds() / 60.0
                        if not (-5.0 <= mins <= float(window_minutes)):
                            continue
                    except ValueError:
                        pass
                if fid:
                    af_odds = load_or_fetch_apifootball_odds([fid], force_refresh=True)
                    tup = af_odds.get(int(fid))
                    if tup:
                        snap = snapshot_rows_from_apifootball(
                            fixture_id=fid,
                            home_team=h_af,
                            away_team=a_af,
                            odds_tuple=tup,
                            event_time=ev_time,
                            league_name=l_info["name"],
                            is_closing=True,
                        )
                        saved_total += active_store.save_odds_snapshots(snap)

    print(f"Closing snapshots guardados en raw.odds_snapshots: {saved_total}")
    return saved_total


def main():
    parser = argparse.ArgumentParser(description="Captura de cuotas de cierre propias (T-5m)")
    parser.add_argument("--date", type=str, default=None, help="Fecha YYYY-MM-DD (default: hoy Bogotá)")
    parser.add_argument("--window-minutes", type=int, default=15, help="Ventana previa al kickoff en minutos")
    parser.add_argument("--all", action="store_true", help="Capturar cierre para todos los partidos disponibles del día")
    args = parser.parse_args()
    capture_closing_snapshots(target_date=args.date, window_minutes=args.window_minutes, capture_all=args.all)


if __name__ == "__main__":
    main()
