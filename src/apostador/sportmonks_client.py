"""Adaptador modular para Sportmonks Football API v3 (TASK-12)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3/football"

SPORTMONKS_PLANS: dict[str, dict[str, Any]] = {
    "Free": {"price_eur": 0, "leagues": 2, "note": "Superliga danesa y Premiership escocesa"},
    "Starter": {"price_eur": 29, "leagues": 5, "rate_limit": "2000 req/entidad/hora"},
    "Growth": {"price_eur": 99, "leagues": 30, "rate_limit": "2500 req/entidad/hora"},
    "Pro": {"price_eur": 249, "leagues": 120, "rate_limit": "3000 req/entidad/hora"},
    "Enterprise": {"price_eur": None, "leagues": 2200, "rate_limit": ">=5000 req/entidad/hora"},
}


class SportmonksFootballClient:
    """Cliente modular Sportmonks v3 (pre-match, inplay y TXOdds premium)."""

    def __init__(self, api_token: str | None = None, timeout: int = 20):
        self.api_token = api_token or os.getenv("SPORTMONKS_API_TOKEN", "")
        self.timeout = int(timeout)

    @property
    def enabled(self) -> bool:
        return bool(self.api_token)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Falta SPORTMONKS_API_TOKEN en el entorno.")
        q = dict(params or {})
        q["api_token"] = self.api_token
        resp = requests.get(f"{SPORTMONKS_BASE_URL}/{path.lstrip('/')}", params=q, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_prematch_odds(self, fixture_id: int) -> dict[str, Any]:
        return self._get(f"odds/pre-match/fixtures/{int(fixture_id)}")

    def get_inplay_odds(self, fixture_id: int) -> dict[str, Any]:
        return self._get(f"odds/inplay/fixtures/{int(fixture_id)}")

    def get_premium_odds(self, fixture_id: int, bookmaker_id: int) -> dict[str, Any]:
        """Endpoint Premium Odds (TXOdds; retención máxima de 7 días post-inicio)."""
        return self._get(f"odds/premium/fixtures/{int(fixture_id)}/bookmakers/{int(bookmaker_id)}")

    @staticmethod
    def _normalize_selection(label: str, home_team: str, away_team: str) -> str:
        low = str(label or "").strip().lower()
        if low in {"1", "home", home_team.lower()}:
            return "Home"
        if low in {"x", "draw", "empate"}:
            return "Draw"
        if low in {"2", "away", away_team.lower()}:
            return "Away"
        return str(label).strip()

    @classmethod
    def normalize_odds_to_snapshots(
        cls,
        payload: dict[str, Any],
        home_team: str,
        away_team: str,
        fixture_id: int | str,
        event_time: str | None = None,
        league: str | None = None,
        is_closing: bool = False,
    ) -> list[dict[str, Any]]:
        """Normaliza la respuesta /v3/football/odds/* hacia el esquema raw.odds_snapshots."""
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for item in payload.get("data") or []:
            price = item.get("value") or item.get("dp3")
            try:
                fprice = float(price)
            except (TypeError, ValueError):
                continue
            if fprice <= 1.0:
                continue
            bm = item.get("bookmaker")
            bm_name = (
                bm.get("name")
                if isinstance(bm, dict)
                else (item.get("bookmaker_name") or f"bookmaker_{item.get('bookmaker_id', 'unknown')}")
            )
            sel = cls._normalize_selection(str(item.get("label") or item.get("name") or ""), home_team, away_team)
            market = str(item.get("market_description") or item.get("market_id") or "h2h").lower()
            if "fulltime result" in market or "match winner" in market or market == "1":
                market = "h2h"
            rows.append({
                "sport": "soccer",
                "league": league,
                "fixture_id": str(fixture_id),
                "home_team": home_team,
                "away_team": away_team,
                "event_time": event_time,
                "odds_timestamp": item.get("latest_bookmaker_update") or now_iso,
                "received_at": now_iso,
                "provider": "sportmonks_v3",
                "bookmaker": str(bm_name),
                "market": market,
                "line": float(item["handicap"]) if item.get("handicap") is not None else 0.0,
                "selection": sel,
                "price": fprice,
                "is_suspended": bool(item.get("stopped", False)),
                "is_closing": is_closing,
            })
        return rows

    @staticmethod
    def compare_plan_coverage_and_cost(target_leagues: int = 27) -> dict[str, Any]:
        """Compara planes Sportmonks frente al stack actual (API-Football + The Odds API)."""
        recommended_plan = "Starter" if target_leagues <= 5 else ("Growth" if target_leagues <= 30 else "Pro")
        base_eur = SPORTMONKS_PLANS[recommended_plan]["price_eur"]
        return {
            "target_leagues": int(target_leagues),
            "recommended_sportmonks_plan": recommended_plan,
            "base_price_eur_month": base_eur,
            "standard_odds_addon_eur_range": "14-69",
            "txodds_retention_days": 7,
            "current_stack_reference_usd": "19-39 (API-Football) + 30-59 (The Odds API)",
            "note": "Confirmar en contrato si el add-on de cuotas y xG están incluidos en el tier elegido.",
        }
