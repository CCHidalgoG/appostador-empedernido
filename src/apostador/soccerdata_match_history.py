"""Ingestor de cuotas y resultados históricos usando soccerdata / football-data.co.uk.

Proporciona soporte resiliente para entornos con restricciones DNS,
normalización de cuotas de apertura/previa y de cierre (*C: PSCH, B365CH, AvgCH, MaxCH),
detección de obsolescencia de Pinnacle post-2025-07-23, cumplimiento de licencia
(uso particular) y persistencia relacional en DuckDB (raw.match_history y raw.artifacts).
"""

from __future__ import annotations

import io
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from .aligner import TeamAligner
from .config import SOCCERDATA_RAW_DIR
from .duckdb_store import DuckDBStore, get_store, write_json

logger = logging.getLogger(__name__)

PINNACLE_STALE_CUTOFF_DATE = "2025-07-23"

# Fallback IPs for football-data.co.uk in case local ISP DNS blocks gambling domains
DNS_OVERRIDES = {
    "www.football-data.co.uk": "217.160.0.118",
    "football-data.co.uk": "217.160.0.246",
}

_orig_getaddrinfo = socket.getaddrinfo


def enable_dns_fallback():
    """Patches socket.getaddrinfo to resolve football-data domains even under ISP blocks."""
    def _safe_getaddrinfo(host, port, *args, **kwargs):
        if host in DNS_OVERRIDES:
            return _orig_getaddrinfo(DNS_OVERRIDES[host], port, *args, **kwargs)
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = _safe_getaddrinfo


# Enable DNS fallback globally on import
enable_dns_fallback()


def patch_soccerdata_session():
    """Ensure soccerdata.MatchHistory uses a DNS-resilient requests session."""
    import soccerdata as sd

    def _safe_download_and_save(self, url: str, filepath: Path | None = None, var: str | Iterable[str] | None = None):
        enable_dns_fallback()
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        last_error = None
        for attempt in range(1, 4):
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                payload = resp.content
                if not getattr(self, "no_store", False) and filepath is not None:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    with filepath.open(mode="wb") as fh:
                        fh.write(payload)
                return io.BytesIO(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Attempt %d/3 failed downloading %s: %s", attempt, url, exc)
                import time
                time.sleep(1.5 * attempt)

        if filepath is not None and filepath.exists():
            logger.info("Falling back to existing cached file: %s", filepath)
            return filepath.open(mode="rb")

        raise ConnectionError(f"Failed to download {url} after 3 attempts: {last_error}") from last_error

    sd.MatchHistory._download_and_save = _safe_download_and_save


# Patch soccerdata on module load
try:
    patch_soccerdata_session()
except Exception as exc:  # noqa: BLE001
    logger.warning("Could not patch soccerdata MatchHistory download handler: %s", exc)


DEFAULT_LEAGUES = [
    "ESP-La Liga",
    "ENG-Premier League",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

DEFAULT_SEASONS = [
    "2223",
    "2324",
    "2425",
    "2526",
    "2627",
]


def is_pinnacle_stale_date(match_date: Any) -> bool:
    """Indica si la fecha corresponde al periodo post-cierre de Pinnacle API (2025-07-23)."""
    if match_date is None or pd.isna(match_date):
        return False
    day_str = str(match_date)[:10]
    return len(day_str) == 10 and PINNACLE_STALE_CUTOFF_DATE <= day_str < "9000-01-01"


def validate_usage_compliance(
    records: list[dict[str, Any]],
    commercial_mode: bool | None = None,
    raise_on_violation: bool = True,
) -> list[dict[str, Any]]:
    """Valida que los registros de football-data.co.uk no se expongan en modo comercial."""
    is_commercial = (
        commercial_mode
        if commercial_mode is not None
        else os.getenv("APOSTADOR_COMMERCIAL_MODE", "0").strip().lower() in {"1", "true", "yes"}
    )
    if not is_commercial:
        return records

    allowed = [
        rec for rec in records
        if (rec.get("usage_scope") or "personal_research_only") != "personal_research_only"
    ]
    if raise_on_violation and len(allowed) < len(records):
        raise PermissionError(
            "Registros con usage_scope='personal_research_only' (football-data.co.uk) "
            "no pueden utilizarse ni redistribuirse cuando APOSTADOR_COMMERCIAL_MODE está activo."
        )
    return allowed


class SoccerDataHistoryLoader:
    """Loader y sincronizador de partidos y cuotas históricas."""

    validate_usage_compliance = staticmethod(validate_usage_compliance)

    def __init__(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        store: DuckDBStore | None = None,
    ):
        self.leagues = leagues or DEFAULT_LEAGUES
        self.seasons = seasons or DEFAULT_SEASONS
        self.store = store or get_store()
        self.aligner = TeamAligner()
        self.data_dir = SOCCERDATA_RAW_DIR / "data" / "MatchHistory"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_season(self, league: str, season: str, force: bool = False) -> Path:
        """Descarga el archivo CSV de una temporada y liga específica."""
        import soccerdata as sd

        mh = sd.MatchHistory(leagues=league, seasons=season, data_dir=SOCCERDATA_RAW_DIR / "data" / "MatchHistory")
        league_code = mh._selected_leagues.get(league, league)
        target_path = self.data_dir / f"{league_code}_{season}.csv"

        if target_path.exists() and not force:
            logger.debug("Season file already exists: %s", target_path)
            return target_path

        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
        logger.info("Downloading %s season %s from %s", league, season, url)

        enable_dns_fallback()
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        resp = session.get(url, timeout=20)
        resp.raise_for_status()

        target_path.write_bytes(resp.content)
        return target_path

    def load_games(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
    ) -> pd.DataFrame:
        """Carga los partidos usando soccerdata y normaliza cuotas previas y de cierre (*C)."""
        import soccerdata as sd

        selected_leagues = leagues or self.leagues
        selected_seasons = seasons or self.seasons

        patch_soccerdata_session()
        mh = sd.MatchHistory(leagues=selected_leagues, seasons=selected_seasons, data_dir=SOCCERDATA_RAW_DIR / "data" / "MatchHistory")
        df = mh.read_games()
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()

        # Cuotas de apertura/previa + cuotas reales de cierre (*C) de football-data.co.uk
        numeric_cols = {
            "B365H": "b365_home",
            "B365D": "b365_draw",
            "B365A": "b365_away",
            "PSH": "pinnacle_home",
            "PSD": "pinnacle_draw",
            "PSA": "pinnacle_away",
            "AvgH": "avg_home",
            "AvgD": "avg_draw",
            "AvgA": "avg_away",
            "MaxH": "max_home",
            "MaxD": "max_draw",
            "MaxA": "max_away",
            "B365CH": "b365_close_home",
            "B365CD": "b365_close_draw",
            "B365CA": "b365_close_away",
            "PSCH": "pinnacle_close_home",
            "PSCD": "pinnacle_close_draw",
            "PSCA": "pinnacle_close_away",
            "AvgCH": "avg_close_home",
            "AvgCD": "avg_close_draw",
            "AvgCA": "avg_close_away",
            "MaxCH": "max_close_home",
            "MaxCD": "max_close_draw",
            "MaxCA": "max_close_away",
            "FTHG": "fthg",
            "FTAG": "ftag",
        }

        new_cols: dict[str, Any] = {}
        for orig_col, target_col in numeric_cols.items():
            if orig_col in df.columns:
                new_cols[target_col] = pd.to_numeric(df[orig_col], errors="coerce")
            else:
                new_cols[target_col] = pd.Series([None] * len(df))

        new_cols["ftr"] = df["FTR"].astype(str) if "FTR" in df.columns else pd.Series([None] * len(df))
        if "game" in df.columns:
            new_cols["game_id"] = df["game"]
        else:
            new_cols["game_id"] = [
                f"{pd.to_datetime(d).strftime('%Y-%m-%d')}-{h}-{a}"
                for d, h, a in zip(df["date"], df["home_team"], df["away_team"])
            ]

        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        new_cols["match_date"] = parsed_dates
        new_cols["pinnacle_is_stale"] = [is_pinnacle_stale_date(dt) for dt in parsed_dates]
        new_cols["data_source"] = "football_data_co_uk"
        new_cols["usage_scope"] = "personal_research_only"

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def to_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convierte el DataFrame en una lista de diccionarios para DuckDB."""
        if df.empty:
            return []

        float_fields = (
            "b365_home", "b365_draw", "b365_away",
            "pinnacle_home", "pinnacle_draw", "pinnacle_away",
            "avg_home", "avg_draw", "avg_away",
            "max_home", "max_draw", "max_away",
            "b365_close_home", "b365_close_draw", "b365_close_away",
            "pinnacle_close_home", "pinnacle_close_draw", "pinnacle_close_away",
            "avg_close_home", "avg_close_draw", "avg_close_away",
            "max_close_home", "max_close_draw", "max_close_away",
        )

        records = []
        for _, row in df.iterrows():
            match_date = row.get("match_date")
            match_date_str = match_date.isoformat() if pd.notna(match_date) else None

            rec: dict[str, Any] = {
                "league": str(row.get("league", "")),
                "season": str(row.get("season", "")),
                "game_id": str(row.get("game_id", "")),
                "match_date": match_date_str,
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
                "fthg": int(row["fthg"]) if pd.notna(row.get("fthg")) else None,
                "ftag": int(row["ftag"]) if pd.notna(row.get("ftag")) else None,
                "ftr": str(row["ftr"]) if pd.notna(row.get("ftr")) else None,
                "pinnacle_is_stale": bool(row["pinnacle_is_stale"])
                if pd.notna(row.get("pinnacle_is_stale"))
                else is_pinnacle_stale_date(match_date_str),
                "data_source": str(row.get("data_source") or "football_data_co_uk"),
                "usage_scope": str(row.get("usage_scope") or "personal_research_only"),
            }
            for field in float_fields:
                rec[field] = float(row[field]) if pd.notna(row.get(field)) else None
            records.append(rec)
        return records

    def sync_to_duckdb(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
    ) -> int:
        """Descarga, procesa y persiste los partidos en DuckDB (raw.match_history y raw.artifacts)."""
        df = self.load_games(leagues=leagues, seasons=seasons)
        if df.empty:
            logger.warning("No games found to sync.")
            return 0

        records = validate_usage_compliance(self.to_records(df))

        # 1. Guardar en tabla relacional raw.match_history
        self.store.save_match_history(records)

        # 2. Guardar en raw.artifacts por (league, season)
        for (league, season), group in df.groupby(["league", "season"]):
            group_records = self.to_records(group)
            clean_league = str(league).replace(" ", "_").replace("-", "_").lower()
            artifact_path = f"soccerdata/match_history/{clean_league}_{season}.json"
            write_json(artifact_path, group_records)

        logger.info("Successfully synced %d matches into raw.match_history and raw.artifacts", len(records))
        return len(records)

    def find_closing_odds(
        self,
        home_team: str,
        away_team: str,
        match_date: str | datetime,
    ) -> dict[str, Any] | None:
        """Busca cuotas de cierre reales en DuckDB para un enfrentamiento y fecha específica."""
        if isinstance(match_date, str):
            dt_prefix = match_date[:10]
        elif isinstance(match_date, (datetime, pd.Timestamp)):
            dt_prefix = match_date.strftime("%Y-%m-%d")
        else:
            dt_prefix = str(match_date)[:10]

        all_matches = self.store.list_match_history()
        candidates = [
            m for m in all_matches
            if m.get("match_date") and str(m["match_date"])[:10] == dt_prefix
        ]

        if not candidates:
            candidates = [
                m for m in all_matches
                if m.get("match_date") and abs((pd.to_datetime(m["match_date"]) - pd.to_datetime(dt_prefix)).days) <= 1
            ]

        for cand in candidates:
            cand_home = cand["home_team"]
            cand_away = cand["away_team"]

            home_match, home_score = self.aligner.find_best_match(home_team, [cand_home], threshold=0.7)
            away_match, away_score = self.aligner.find_best_match(away_team, [cand_away], threshold=0.7)

            if home_match and away_match and home_score >= 0.7 and away_score >= 0.7:
                return cand

        return None
