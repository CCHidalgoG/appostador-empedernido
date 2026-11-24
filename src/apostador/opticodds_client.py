"""Adaptador modular para OpticOdds API v3 y streaming SSE (TASK-12)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import requests

OPTICODDS_BASE_URL = "https://api.opticodds.com/api/v3"


class OpticOddsClient:
    """Cliente modular OpticOdds v3 con control de límite de 5 casas y parser SSE."""

    SPORT_IDENTIFIER = "soccer"
    MAX_BOOKS_PER_SNAPSHOT = 5
    RATE_LIMITS = {
        "standard": "2500 req / 15s",
        "sse_new_connections": "250 / 15s",
        "historical": "10 req / 15s",
    }

    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("OPTICODDS_API_KEY", "")
        self.timeout = int(timeout)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("Falta OPTICODDS_API_KEY en el entorno.")
        return {"X-Api-Key": self.api_key, "Accept": "application/json"}

    def get_fixtures_odds(
        self,
        fixture_ids: list[str] | None = None,
        sportsbooks: list[str] | None = None,
        sport: str = SPORT_IDENTIFIER,
    ) -> list[dict[str, Any]]:
        """Consulta snapshots de cuotas chunking a máximo 5 sportsbooks por petición."""
        books = list(sportsbooks or ["Pinnacle", "Bet365"])
        chunks = [
            books[i : i + self.MAX_BOOKS_PER_SNAPSHOT]
            for i in range(0, len(books), self.MAX_BOOKS_PER_SNAPSHOT)
        ]
        all_items: list[dict[str, Any]] = []
        for book_chunk in chunks:
            params: list[tuple[str, str]] = [("sport", sport)]
            for fid in fixture_ids or []:
                params.append(("fixture_id", str(fid)))
            for sb in book_chunk:
                params.append(("sportsbook", str(sb)))
            resp = requests.get(
                f"{OPTICODDS_BASE_URL}/fixtures/odds",
                headers=self._headers(),
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            all_items.extend(data)
        return all_items

    def get_historical_odds(
        self,
        fixture_id: str,
        sportsbooks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Consulta historial granular de cambios/bloqueos (límite 10 req / 15s)."""
        params: list[tuple[str, str]] = [("fixture_id", str(fixture_id))]
        for sb in (sportsbooks or [])[: self.MAX_BOOKS_PER_SNAPSHOT]:
            params.append(("sportsbook", str(sb)))
        resp = requests.get(
            f"{OPTICODDS_BASE_URL}/fixtures/odds/historical",
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def parse_sse_event(raw_event_text: str) -> dict[str, Any] | None:
        """Parsea un bloque de evento SSE de /stream/odds/soccer (soporta locked-odds)."""
        event_type = "message"
        entry_id = None
        data_lines = []
        for line in str(raw_event_text).splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("id:"):
                entry_id = line[3:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return None
        payload = json.loads("\n".join(data_lines))
        return {
            "event_type": event_type,
            "last_entry_id": entry_id or payload.get("entry_id"),
            "is_locked": event_type == "locked-odds",
            "data": payload.get("data", payload),
        }

    @classmethod
    def normalize_odds_to_snapshots(
        cls,
        items: list[dict[str, Any]],
        sport: str = SPORT_IDENTIFIER,
        is_closing: bool = False,
        is_locked: bool = False,
    ) -> list[dict[str, Any]]:
        """Normaliza respuestas REST o SSE de OpticOdds hacia raw.odds_snapshots."""
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for item in items:
            fixture = item.get("fixture") or item
            fid = str(fixture.get("id") or item.get("fixture_id") or "")
            home = str(fixture.get("home_team_display") or fixture.get("home_team") or item.get("home_team") or "").strip()
            away = str(fixture.get("away_team_display") or fixture.get("away_team") or item.get("away_team") or "").strip()
            if not home or not away:
                continue
            ev_time = fixture.get("start_date") or item.get("start_date")
            league = str((fixture.get("league") or {}).get("name") or item.get("league") or "") or None
            odds_list = item.get("odds") if isinstance(item.get("odds"), list) else [item]
            for o in odds_list:
                price_raw = o.get("price")
                try:
                    price = float(price_raw)
                except (TypeError, ValueError):
                    continue
                if price <= 1.0:
                    continue
                sel_raw = str(o.get("selection") or o.get("name") or "").strip()
                if sel_raw.lower() == home.lower():
                    sel = "Home"
                elif sel_raw.lower() == away.lower():
                    sel = "Away"
                elif sel_raw.lower() in {"draw", "tie", "x"}:
                    sel = "Draw"
                else:
                    sel = sel_raw
                market = str(o.get("market") or "h2h").lower()
                if "moneyline" in market or "3-way" in market or market == "1x2":
                    market = "h2h"
                rows.append({
                    "sport": sport,
                    "league": league,
                    "fixture_id": fid or f"{str(ev_time or '')[:10]}_{home}_{away}",
                    "home_team": home,
                    "away_team": away,
                    "event_time": ev_time,
                    "odds_timestamp": o.get("timestamp") or now_iso,
                    "received_at": now_iso,
                    "provider": "opticodds_v3",
                    "bookmaker": str(o.get("sportsbook") or "OpticOdds"),
                    "market": market,
                    "line": float(o["points"]) if o.get("points") is not None else 0.0,
                    "selection": sel,
                    "price": price,
                    "is_suspended": bool(is_locked or o.get("is_locked") or o.get("locked", False)),
                    "is_closing": is_closing,
                })
        return rows
