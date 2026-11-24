"""Ponderación cuantitativa de bajas y XI titular mediante Transfermarkt / SoFIFA.

Nota metodológica: el valor de mercado de Transfermarkt (market_value_eur) y el
rating de SoFIFA son proxies editoriales/relativos de calidad deportiva, nunca
precios confirmados de traspaso ni salarios reales.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import requests

from .aligner import TeamAligner, clean_name
from .config import SOCCERDATA_RAW_DIR
from .duckdb_store import DuckDBStore, get_store

logger = logging.getLogger(__name__)


def parse_market_value_eur(raw_val: Any) -> float | None:
    """Normaliza valores como 45000000, '€45.00m' o '€500k' a euros (float)."""
    if raw_val is None:
        return None
    if isinstance(raw_val, (int, float)):
        val = float(raw_val)
        return val if val >= 0 else None
    text = str(raw_val).strip().lower().replace(",", "").replace("€", "").replace("eur", "").strip()
    if not text or text in {"-", "na", "n/a", "none"}:
        return None
    mult = 1.0
    if text.endswith("m"):
        mult = 1_000_000.0
        text = text[:-1].strip()
    elif text.endswith("k"):
        mult = 1_000.0
        text = text[:-1].strip()
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0)) * mult


class PlayerValuationLoader:
    """Cargador y ponderador de bajas/XI usando raw.player_valuations (Transfermarkt/SoFIFA)."""

    def __init__(self, store: DuckDBStore | None = None, min_request_interval_sec: float = 1.5):
        self.store = store or get_store()
        self.aligner = TeamAligner()
        self.min_request_interval_sec = float(min_request_interval_sec)
        self._last_request_ts = 0.0

    def _respect_rate_limit(self):
        """Aplica espera mínima (<= 2 req / 3s por defecto) antes de peticiones externas."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)
        self._last_request_ts = time.monotonic()

    def fetch_transfermarkt_squad(
        self,
        club_id: str | int,
        team_name: str,
        base_url: str | None = None,
        as_of_date: str | None = None,
        timeout: int = 15,
    ) -> list[dict[str, Any]]:
        """Consulta un servicio transfermarkt-api compatible y persiste en DuckDB."""
        api_base = (base_url or os.getenv("TRANSFERMARKT_API_URL", "")).rstrip("/")
        if not api_base:
            return []

        snap_date = (as_of_date or datetime.now().strftime("%Y-%m-%d"))[:10]
        self._respect_rate_limit()
        url = f"{api_base}/clubs/{club_id}/players"
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo obtener plantilla Transfermarkt para %s (%s): %s", team_name, club_id, exc)
            return []

        players_raw = payload.get("players") if isinstance(payload, dict) else payload
        if not isinstance(players_raw, list):
            return []

        records = []
        for p in players_raw:
            name = str(p.get("name") or p.get("playerName") or "").strip()
            if not name:
                continue
            mv = parse_market_value_eur(p.get("marketValue") or p.get("market_value_eur"))
            records.append(
                {
                    "provider": "transfermarkt",
                    "team_name": team_name,
                    "player_name": name,
                    "as_of_date": snap_date,
                    "market_value_eur": mv,
                    "sofifa_overall": None,
                    "minutes_played": float(p["minutesPlayed"]) if p.get("minutesPlayed") is not None else None,
                    "position": p.get("position"),
                }
            )
        if records:
            self.store.save_player_valuations(records)
        return records

    def sync_sofifa_league(
        self,
        league: str,
        as_of_date: str | None = None,
    ) -> int:
        """Sincroniza valoraciones y overalls de SoFIFA en raw.player_valuations."""
        snap_date = (as_of_date or datetime.now().strftime("%Y-%m-%d"))[:10]
        try:
            import soccerdata as sd

            sf = sd.SoFIFA(leagues=league, data_dir=SOCCERDATA_RAW_DIR / "data" / "SoFIFA")
            df = sf.read_players()
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudieron descargar ratings SoFIFA para %s: %s", league, exc)
            return 0

        if df is None or df.empty:
            return 0
        df = df.reset_index()
        records = []
        for _, row in df.iterrows():
            team = str(row.get("team") or "").strip()
            player = str(row.get("player") or row.get("short_name") or "").strip()
            if not team or not player:
                continue
            overall = row.get("overall") or row.get("overallrating")
            val_eur = parse_market_value_eur(row.get("value_eur") or row.get("value"))
            records.append(
                {
                    "provider": "sofifa",
                    "team_name": team,
                    "player_name": player,
                    "as_of_date": snap_date,
                    "market_value_eur": val_eur,
                    "sofifa_overall": int(overall) if overall is not None and str(overall).isdigit() else None,
                    "minutes_played": None,
                    "position": str(row.get("position") or "") or None,
                }
            )
        if records:
            self.store.save_player_valuations(records)
        return len(records)

    def load_team_valuations(
        self,
        team_name: str,
        as_of_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recupera las valoraciones point-in-time de un equipo con alineación fuzzy."""
        if not team_name:
            return []
        rows = self.store.get_team_player_valuations(team_name, as_of_date=as_of_date)
        if rows:
            return rows

        known_teams = self.store.list_valuation_teams()
        if not known_teams:
            return []
        best_team, score = self.aligner.find_best_match(team_name, known_teams, threshold=0.75)
        if best_team and score >= 0.75:
            return self.store.get_team_player_valuations(best_team, as_of_date=as_of_date)
        return []

    @staticmethod
    def _player_matches_query(player_name: str, queries: list[str]) -> bool:
        full = clean_name(str(player_name).replace(".", " "))
        if not full:
            return False
        full_tokens = full.split()
        first_tok = full_tokens[0] if full_tokens else ""
        last_tok = full_tokens[-1] if full_tokens else ""

        for q in queries:
            q_clean = clean_name(str(q).replace(".", " "))
            if not q_clean:
                continue
            if q_clean == full:
                return True
            q_tokens = q_clean.split()
            if len(q_tokens) >= 2 and full_tokens:
                q_first, q_last = q_tokens[0], q_tokens[-1]
                last_ok = q_last == last_tok or q_last in full_tokens
                first_ok = (
                    q_first == first_tok
                    or (len(q_first) == 1 and first_tok.startswith(q_first))
                    or (len(first_tok) == 1 and q_first.startswith(first_tok))
                )
                if last_ok and first_ok:
                    return True
            if SequenceMatcher(None, q_clean, full).ratio() >= 0.84:
                return True
        return False

    @staticmethod
    def _player_weight(row: dict[str, Any]) -> float:
        mv = row.get("market_value_eur")
        if mv is not None and float(mv) > 0:
            return float(mv)
        ovr = row.get("sofifa_overall")
        if ovr is not None and float(ovr) > 55.0:
            return float((float(ovr) - 55.0) ** 2)
        return 0.0

    def compute_missing_valuation_impact(
        self,
        team_name: str,
        injured_players: list[str] | None = None,
        starting_xi: list[str] | None = None,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        """Calcula el porcentaje de valor/talento ausente respecto al Top 11 habitual."""
        empty = {
            "missing_share_impact": 0.0,
            "missing_market_value_eur": 0.0,
            "missing_value_pct": 0.0,
            "valuation_source": None,
            "matched_missing_players": [],
        }
        if not team_name or (not injured_players and not starting_xi):
            return empty

        squad = self.load_team_valuations(team_name, as_of_date=as_of_date)
        if not squad:
            return empty

        weighted = []
        for r in squad:
            w = self._player_weight(r)
            if w > 0:
                weighted.append({**r, "_weight": w, "_minutes": float(r.get("minutes_played") or 0.0)})
        if not weighted:
            return empty

        weighted.sort(key=lambda x: (x["_minutes"], x["_weight"]), reverse=True)
        base_top11 = weighted[:11]
        base_total_weight = sum(p["_weight"] for p in base_top11)
        if base_total_weight <= 0:
            return empty

        missing_rows: list[dict[str, Any]] = []
        if starting_xi:
            in_xi = [p for p in weighted if self._player_matches_query(p["player_name"], starting_xi)]
            if len(in_xi) >= 7:
                in_xi_names = {p["player_name"] for p in in_xi}
                missing_rows = [p for p in base_top11 if p["player_name"] not in in_xi_names]

        if not missing_rows and injured_players:
            missing_rows = [
                p for p in weighted if self._player_matches_query(p["player_name"], injured_players)
            ]

        missing_weight = sum(p["_weight"] for p in missing_rows)
        missing_share = max(0.0, min(1.0, missing_weight / base_total_weight))
        missing_eur = sum(float(p.get("market_value_eur") or 0.0) for p in missing_rows)

        return {
            "missing_share_impact": float(missing_share),
            "missing_market_value_eur": float(missing_eur),
            "missing_value_pct": round(float(missing_share) * 100.0, 2),
            "valuation_source": str(squad[0].get("provider") or "transfermarkt"),
            "matched_missing_players": [p["player_name"] for p in missing_rows],
        }
