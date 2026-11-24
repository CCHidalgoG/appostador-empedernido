"""Point-in-time injury and official lineup context from API-Football."""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

from .duckdb_store import get_store

load_dotenv()

API_BASE = "https://v3.football.api-sports.io"
INJURY_TTL = timedelta(hours=4)
LINEUP_TTL = timedelta(minutes=15)
ERROR_TTL = timedelta(minutes=15)


class ApiFootballContextClient:
    """Fetch pre-match injuries daily and lineups only near kickoff; cache in DuckDB."""

    def __init__(self, api_key=None, timeout=15, store=None):
        self.api_key = api_key or os.getenv("APIFOOTBALL")
        self.timeout = timeout
        self.store = store or get_store()

    @property
    def enabled(self):
        return bool(self.api_key) and os.getenv("APIFOOTBALL_PREMATCH_CONTEXT", "1").lower() not in {"0", "false", "no"}

    @staticmethod
    def _fresh(record, ttl):
        if not record:
            return False
        try:
            fetched = datetime.fromisoformat(record["fetched_at"])
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - fetched.astimezone(timezone.utc) < ttl
        except (KeyError, TypeError, ValueError):
            return False

    def _get(self, endpoint, params):
        response = requests.get(
            f"{API_BASE}/{endpoint}", params=params,
            headers={"x-apisports-key": self.api_key}, timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(str(payload["errors"]))
        return payload

    def injuries_for_date(self, match_date):
        """One date-scoped call per cache window; API-Football has no injury history."""
        if not self.enabled:
            return {"records": [], "fetched_at": None, "error": "desactivado"}
        key = f"apifootball/prematch/injuries_{str(match_date)[:10]}"
        cached = self.store.get("raw", key)
        if self._fresh(cached, ERROR_TTL if cached and cached.get("error") else INJURY_TTL):
            return {**cached, "from_cache": True}
        try:
            payload = self._get("injuries", {"date": str(match_date)[:10]})
            result = {"records": payload.get("response") or [],
                      "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "provider": "api_football", "error": None}
            self.store.put("raw", key, result)
            return result
        except Exception as exc:  # provider coverage/quota must not break predictions
            result = {"records": [], "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "provider": "api_football", "error": str(exc)}
            self.store.put("raw", key, result)
            return result

    def lineup_for_fixture(self, fixture_id, kickoff):
        """Only request lineups in the provider's pre-kickoff availability window."""
        if not self.enabled or not fixture_id:
            return None
        try:
            kickoff_dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
            if kickoff_dt.tzinfo is None:
                kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
            minutes_to_kickoff = (kickoff_dt.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 60
        except (TypeError, ValueError):
            return None
        if not -10 <= minutes_to_kickoff <= 55:
            return None

        key = f"apifootball/prematch/lineup_{int(fixture_id)}"
        cached = self.store.get("raw", key)
        if self._fresh(cached, ERROR_TTL if cached and cached.get("error") else LINEUP_TTL):
            return {**cached, "from_cache": True}
        try:
            payload = self._get("fixtures/lineups", {"fixture": int(fixture_id)})
            result = {"records": payload.get("response") or [],
                      "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "provider": "api_football", "error": None}
            self.store.put("raw", key, result)
            return result
        except Exception as exc:  # lineup coverage varies by competition
            result = {"records": [], "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "provider": "api_football", "error": str(exc)}
            self.store.put("raw", key, result)
            return result

    def fixture_context(self, fixture, date_injuries):
        """Return only records linked to this fixture/teams, with clear coverage flags."""
        fixture_id = (fixture.get("fixture") or {}).get("id")
        teams = fixture.get("teams") or {}
        team_ids = {team.get("id") for team in teams.values() if team.get("id") is not None}
        injuries = []
        for record in date_injuries.get("records") or []:
            injury_fixture = (record.get("fixture") or {}).get("id")
            team_id = (record.get("team") or {}).get("id")
            if injury_fixture == fixture_id or (team_id in team_ids and injury_fixture in (None, fixture_id)):
                player = record.get("player") or {}
                team = record.get("team") or {}
                injuries.append({
                    "player": player.get("name"), "team": team.get("name"),
                    "status": player.get("type"), "reason": player.get("reason"),
                    "fixture_id": injury_fixture,
                })
        kickoff = (fixture.get("fixture") or {}).get("date")
        lineup = self.lineup_for_fixture(fixture_id, kickoff)
        lineups = []
        for record in (lineup or {}).get("records", []):
            coach = record.get("coach") or {}
            lineups.append({
                "team": (record.get("team") or {}).get("name"),
                "formation": record.get("formation"), "coach": coach.get("name"),
                "starting_xi": [((p.get("player") or {}).get("name")) for p in record.get("startXI", [])],
                "substitutes": [((p.get("player") or {}).get("name")) for p in record.get("substitutes", [])],
            })
        return {
            "provider": "api_football",
            "injuries": injuries,
            "injuries_as_of": date_injuries.get("fetched_at"),
            "injuries_error": date_injuries.get("error"),
            "injuries_checked": date_injuries.get("fetched_at") is not None and not date_injuries.get("error"),
            "lineups": lineups,
            "lineups_as_of": (lineup or {}).get("fetched_at"),
            "lineups_available": bool(lineups),
            "lineups_error": (lineup or {}).get("error"),
        }
