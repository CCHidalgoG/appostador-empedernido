"""Conector de microestructura y liquidez para Betfair Exchange API (TASK-11).

Respeta los límites y políticas de Betfair Exchange:
- Delayed App Key gratuita para desarrollo (retraso 1-180s) o Live App Key (£499).
- La Live App Key no autoriza uso meramente read-only sin propósito operativo/licencia.
- Máximo 5 llamadas/seg por marketId en listMarketBook.
- Peso total de proyecciones * cantidad de mercados <= 200 puntos por petición.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

BETFAIR_API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
MAX_REQS_PER_SEC_PER_MARKET = 5.0
MAX_PROJECTION_WEIGHT_POINTS = 200

# Pesos publicados por Betfair Exchange para priceData / marketProjection
PROJECTION_WEIGHTS = {
    "EX_BEST_OFFERS": 5,
    "EX_ALL_OFFERS": 17,
    "EX_TRADED": 17,
    "SP_AVAILABLE": 3,
    "SP_TRADED": 7,
}


class BetfairExchangeClient:
    """Cliente REST/JSON-RPC para lectura de libro de órdenes y liquidez en Betfair Exchange."""

    def __init__(
        self,
        app_key: str | None = None,
        session_token: str | None = None,
        is_live_key: bool = False,
        read_only_only: bool = False,
        timeout: int = 15,
    ):
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY", "")
        self.session_token = session_token or os.getenv("BETFAIR_SESSION_TOKEN", "")
        self.is_live_key = bool(is_live_key)
        self.read_only_only = bool(read_only_only)
        self.timeout = int(timeout)
        self._last_market_call: dict[str, float] = {}
        self.validate_key_policy()

    def validate_key_policy(self):
        """Previene uso indebido de Live App Key como feed meramente read-only."""
        if self.is_live_key and self.read_only_only:
            raise PermissionError(
                "Betfair Live App Key (£499) no permite uso meramente read-only sin licencia comercial aparte."
            )

    @property
    def enabled(self) -> bool:
        return bool(self.app_key and self.session_token)

    @staticmethod
    def calculate_weight_per_market(price_data: list[str] | None = None) -> int:
        """Calcula los puntos de peso por mercado para no superar el límite de 200 puntos."""
        items = price_data or ["EX_BEST_OFFERS", "EX_TRADED"]
        weight = sum(PROJECTION_WEIGHTS.get(str(p), 5) for p in items)
        return max(1, weight)

    @classmethod
    def chunk_market_ids(cls, market_ids: list[str], price_data: list[str] | None = None) -> list[list[str]]:
        """Divide market_ids en lotes cuyo peso total <= 200 puntos."""
        weight_per_mkt = cls.calculate_weight_per_market(price_data)
        max_markets = max(1, MAX_PROJECTION_WEIGHT_POINTS // weight_per_mkt)
        return [market_ids[i : i + max_markets] for i in range(0, len(market_ids), max_markets)]

    def _respect_market_rate_limit(self, market_ids: list[str]):
        min_interval = 1.0 / MAX_REQS_PER_SEC_PER_MARKET
        now = time.monotonic()
        for mid in market_ids:
            last = self._last_market_call.get(mid, 0.0)
            elapsed = now - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
                now = time.monotonic()
            self._last_market_call[mid] = now

    def _rpc_call(self, method: str, params: dict[str, Any]) -> Any:
        if not self.enabled:
            raise RuntimeError("Faltan BETFAIR_APP_KEY o BETFAIR_SESSION_TOKEN.")
        headers = {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
        resp = requests.post(BETFAIR_API_URL, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Betfair RPC error: {data['error']}")
        return data.get("result")

    def list_market_catalogue(
        self,
        event_type_ids: list[str] | None = None,
        market_type_codes: list[str] | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Lista mercados activos (ej. eventTypeIds=['1'] fútbol, ['7522'] básquet)."""
        params = {
            "filter": {
                "eventTypeIds": event_type_ids or ["1"],
                "marketTypeCodes": market_type_codes or ["MATCH_ODDS"],
            },
            "marketProjection": ["EVENT", "COMPETITION", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
            "maxResults": str(max_results),
        }
        return self._rpc_call("listMarketCatalogue", params) or []

    def list_market_book(
        self,
        market_ids: list[str],
        price_data: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Consulta precios back/lay y liquidez respetando lote <=200 pts y <=5 req/s."""
        if not market_ids:
            return []
        pdata = price_data or ["EX_BEST_OFFERS", "EX_TRADED"]
        results: list[dict[str, Any]] = []
        for batch in self.chunk_market_ids(market_ids, price_data=pdata):
            self._respect_market_rate_limit(batch)
            res = self._rpc_call(
                "listMarketBook",
                {"marketIds": batch, "priceProjection": {"priceData": pdata}},
            )
            if isinstance(res, list):
                results.extend(res)
        return results

    @staticmethod
    def parse_market_book_microstructure(
        market_book: dict[str, Any],
        runner_names: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Extrae best back/lay, spread, liquidez top-3 y probabilidad implícita media."""
        runners_out: dict[str, dict[str, Any]] = {}
        mid_probs_raw: dict[str, float] = {}
        name_map = runner_names or {}

        for r in market_book.get("runners") or []:
            sel_id = int(r.get("selectionId", 0))
            label = str(name_map.get(sel_id) or r.get("runnerName") or sel_id)
            ex = r.get("ex") or {}
            backs = ex.get("availableToBack") or []
            lays = ex.get("availableToLay") or []

            best_back = float(backs[0]["price"]) if backs else None
            best_lay = float(lays[0]["price"]) if lays else None
            top3_liq = float(sum(float(lvl.get("size") or 0.0) for lvl in backs[:3]))
            spread = (best_lay - best_back) if (best_back and best_lay) else None
            if best_back and best_lay:
                mid_price = 0.5 * (best_back + best_lay)
            else:
                mid_price = best_back or best_lay

            if mid_price and mid_price > 1.0:
                mid_probs_raw[label] = 1.0 / mid_price

            runners_out[label] = {
                "selection_id": sel_id,
                "status": r.get("status", "ACTIVE"),
                "best_back_price": best_back,
                "best_lay_price": best_lay,
                "bid_ask_spread": round(spread, 4) if spread is not None else None,
                "mid_price": round(mid_price, 4) if mid_price is not None else None,
                "top3_back_liquidity": round(top3_liq, 2),
                "total_matched": float(r.get("totalMatched") or 0.0),
            }

        tot_raw = sum(mid_probs_raw.values())
        mid_novig = (
            {k: round(v / tot_raw, 6) for k, v in mid_probs_raw.items()}
            if tot_raw > 0
            else {}
        )
        return {
            "market_id": str(market_book.get("marketId") or ""),
            "status": str(market_book.get("status") or ""),
            "inplay": bool(market_book.get("inplay", False)),
            "total_matched": float(market_book.get("totalMatched") or 0.0),
            "runners": runners_out,
            "mid_novig_probs": mid_novig,
        }

    @staticmethod
    def cap_stake_by_liquidity(
        desired_stake: float,
        available_liquidity: float | None,
        max_book_share: float = 0.25,
    ) -> dict[str, Any]:
        """Limita el stake recomendado a una fracción segura de la liquidez disponible."""
        stake = float(desired_stake)
        if available_liquidity is None or float(available_liquidity) <= 0:
            return {"stake": stake, "liquidity_capped": False, "max_allowed": None}
        max_allowed = round(float(available_liquidity) * float(max_book_share), 2)
        if stake > max_allowed:
            return {"stake": max_allowed, "liquidity_capped": True, "max_allowed": max_allowed}
        return {"stake": stake, "liquidity_capped": False, "max_allowed": max_allowed}

    @classmethod
    def to_odds_snapshots(
        cls,
        market_micro: dict[str, Any],
        home_team: str,
        away_team: str,
        event_time: str | None = None,
        sport: str = "soccer",
        league: str | None = None,
        is_closing: bool = False,
    ) -> list[dict[str, Any]]:
        """Convierte microestructura Betfair Exchange al esquema raw.odds_snapshots."""
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        is_susp = str(market_micro.get("status", "")).upper() == "SUSPENDED"
        rows = []
        for sel, info in (market_micro.get("runners") or {}).items():
            price = info.get("best_back_price") or info.get("mid_price")
            if not price or float(price) <= 1.0:
                continue
            rows.append({
                "sport": sport,
                "league": league,
                "fixture_id": market_micro.get("market_id") or f"{home_team}_{away_team}",
                "home_team": home_team,
                "away_team": away_team,
                "event_time": event_time,
                "odds_timestamp": now_iso,
                "received_at": now_iso,
                "provider": "betfair_exchange",
                "bookmaker": "Betfair Exchange",
                "market": "h2h",
                "line": 0.0,
                "selection": sel,
                "price": float(price),
                "is_suspended": is_susp,
                "is_closing": is_closing,
                "microstructure": info,
            })
        return rows
