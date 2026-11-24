"""Módulo de Métricas Tácticas Understat/FBref y Reporte de Bajas WhoScored.

Desde enero de 2026 FBref dejó de publicar métricas avanzadas (xG, xGA) tras el
cierre de su suministro de proveedor, por lo que UnderstatTacticalLoader actúa
como fuente primaria de xG/xGA/npxG/PPDA para las grandes ligas europeas y
FBrefTacticalLoader mantiene compatibilidad con fallback automático a Understat.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd

from .aligner import TeamAligner
from .config import SOCCERDATA_RAW_DIR
from .duckdb_store import DuckDBStore, get_store, read_json, write_json

logger = logging.getLogger(__name__)

SCRAPER_TOS_POLICY: dict[str, dict[str, Any]] = {
    "understat": {"automated_production_allowed": True, "mode": "cached_match_stats", "note": "Caché local por liga/temporada."},
    "fbref": {"automated_production_allowed": True, "mode": "basic_stats_only", "note": "Sin xG/Opta desde 2026-01-20."},
    "whoscored": {"automated_production_allowed": True, "mode": "cached_with_fallback", "note": "Usar fallback API-Football ante fallos."},
    "transfermarkt": {"automated_production_allowed": True, "mode": "rate_limited_wrapper", "note": "Máx 2 req / 3s; JSON interno no oficial."},
    "sofascore": {"automated_production_allowed": False, "mode": "blocked_by_tos", "note": "ToS prohíben scraping/automatización sistemática."},
    "fotmob": {"automated_production_allowed": False, "mode": "blocked_by_tos", "note": "FAQ prohíbe servicios automáticos (retirado de worldfootballR)."},
}


def check_scraper_tos_compliance(source: str, production_mode: bool = True) -> dict[str, Any]:
    """Evalúa si una fuente web está habilitada por política de ToS para automatización."""
    key = str(source or "").strip().lower()
    policy = SCRAPER_TOS_POLICY.get(
        key,
        {"automated_production_allowed": False, "mode": "unknown_source", "note": "Fuente no auditada."},
    )
    allowed = bool(policy["automated_production_allowed"]) or not production_mode
    return {"source": key, "allowed": allowed, **policy}

UNDERSTAT_SUPPORTED_LEAGUES = {
    "ESP-La Liga",
    "ENG-Premier League",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
}


def resolve_current_season_code(reference_date: datetime | str | None = None) -> str:
    """Resuelve dinámicamente el código de temporada europea (ej. '2526', '2627')."""
    if reference_date is None:
        dt = datetime.now()
    elif isinstance(reference_date, datetime):
        dt = reference_date
    else:
        dt = pd.to_datetime(reference_date).to_pydatetime()

    start_year = dt.year if dt.month >= 7 else dt.year - 1
    end_year = start_year + 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def _serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convierte un DataFrame a registros JSON-serializables."""
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


class UnderstatTacticalLoader:
    """Cargador de métricas avanzadas (xG, xGA, npxG, PPDA) desde Understat."""

    def __init__(self, store: DuckDBStore | None = None):
        self.store = store or get_store()
        self.aligner = TeamAligner()
        self.cache_dir = SOCCERDATA_RAW_DIR / "data" / "Understat"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_team_match_stats(
        self,
        league: str = "ESP-La Liga",
        season: str | None = None,
        cache_only: bool = False,
    ) -> pd.DataFrame:
        """Descarga o lee desde caché las estadísticas partido a partido de Understat."""
        season_code = season or resolve_current_season_code()
        if league not in UNDERSTAT_SUPPORTED_LEAGUES:
            logger.debug("Liga %s no soportada por Understat.", league)
            return pd.DataFrame()

        clean_league = league.replace(" ", "_").replace("-", "_").lower()
        artifact_key = f"soccerdata/understat/{clean_league}_{season_code}_matches.json"
        cached = read_json(artifact_key)
        if cached:
            return pd.DataFrame(cached)
        if cache_only:
            return pd.DataFrame()

        try:
            import soccerdata as sd

            us = sd.Understat(leagues=league, seasons=season_code, data_dir=self.cache_dir)
            df = us.read_team_match_stats()
            if not df.empty:
                df = df.reset_index()
                write_json(artifact_key, _serialize_records(df))
                return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudieron descargar estadísticas Understat para %s %s: %s", league, season_code, exc)

        return pd.DataFrame()

    @staticmethod
    def aggregate_team_stats(
        matches_df: pd.DataFrame,
        as_of_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Agrega estadísticas de partido (local y visita) por equipo con filtro as-of opcional."""
        if matches_df is None or matches_df.empty:
            return pd.DataFrame()

        df = matches_df.copy()
        if as_of_date is not None and "date" in df.columns:
            parsed_dates = pd.to_datetime(df["date"], errors="coerce")
            cutoff_str = str(as_of_date).strip()
            if len(cutoff_str) <= 10:
                cutoff_day = pd.to_datetime(cutoff_str).date()
                df = df[parsed_dates.dt.date <= cutoff_day]
            else:
                cutoff_dt = pd.to_datetime(cutoff_str)
                df = df[parsed_dates <= cutoff_dt]

        if df.empty:
            return pd.DataFrame()

        side_rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            for side, opp in (("home", "away"), ("away", "home")):
                team = row.get(f"{side}_team")
                if pd.isna(team) or not str(team).strip():
                    continue
                side_rows.append(
                    {
                        "team": str(team).strip(),
                        "matches_played": 1,
                        "goals_scored": pd.to_numeric(row.get(f"{side}_goals"), errors="coerce"),
                        "goals_conceded": pd.to_numeric(row.get(f"{opp}_goals"), errors="coerce"),
                        "xg": pd.to_numeric(row.get(f"{side}_xg"), errors="coerce"),
                        "xga": pd.to_numeric(row.get(f"{opp}_xg"), errors="coerce"),
                        "npxg": pd.to_numeric(row.get(f"{side}_np_xg"), errors="coerce"),
                        "npxga": pd.to_numeric(row.get(f"{opp}_np_xg"), errors="coerce"),
                        "xpts": pd.to_numeric(row.get(f"{side}_expected_points"), errors="coerce"),
                        "ppda": pd.to_numeric(row.get(f"{side}_ppda"), errors="coerce"),
                        "deep_completions": pd.to_numeric(row.get(f"{side}_deep_completions"), errors="coerce"),
                    }
                )

        if not side_rows:
            return pd.DataFrame()

        long_df = pd.DataFrame(side_rows)
        agg_df = (
            long_df.groupby("team", as_index=False)
            .agg(
                matches_played=("matches_played", "sum"),
                goals_scored=("goals_scored", "sum"),
                goals_conceded=("goals_conceded", "sum"),
                xg=("xg", "sum"),
                xga=("xga", "sum"),
                npxg=("npxg", "sum"),
                npxga=("npxga", "sum"),
                xpts=("xpts", "sum"),
                ppda=("ppda", "mean"),
                deep_completions=("deep_completions", "sum"),
            )
        )
        agg_df["xg_per_match"] = agg_df["xg"] / agg_df["matches_played"].clip(lower=1)
        agg_df["xga_per_match"] = agg_df["xga"] / agg_df["matches_played"].clip(lower=1)
        return agg_df

    def get_team_season_stats(
        self,
        league: str = "ESP-La Liga",
        season: str | None = None,
        as_of_date: str | datetime | None = None,
        cache_only: bool = False,
    ) -> pd.DataFrame:
        """Devuelve la tabla agregada por equipo de Understat para una liga y temporada."""
        matches_df = self.get_team_match_stats(league=league, season=season, cache_only=cache_only)
        return self.aggregate_team_stats(matches_df, as_of_date=as_of_date)

    def get_team_context(
        self,
        team_name: str,
        league: str = "ESP-La Liga",
        season: str | None = None,
        as_of_date: str | datetime | None = None,
        cache_only: bool = False,
    ) -> dict[str, Any] | None:
        """Extrae el perfil táctico de un equipo desde Understat."""
        season_code = season or resolve_current_season_code(as_of_date)
        df = self.get_team_season_stats(league=league, season=season_code, as_of_date=as_of_date, cache_only=cache_only)
        if df.empty or "team" not in df.columns:
            return None

        candidates = df["team"].astype(str).tolist()
        match, score = self.aligner.find_best_match(team_name, candidates, threshold=0.7)
        if not match or score < 0.7:
            return None

        row = df[df["team"] == match].iloc[0]
        return {
            "team": match,
            "matched_score": float(score),
            "season": season_code,
            "source": "understat",
            "matches_played": int(row["matches_played"]),
            "goals_scored": float(row["goals_scored"]) if pd.notna(row["goals_scored"]) else None,
            "goals_conceded": float(row["goals_conceded"]) if pd.notna(row["goals_conceded"]) else None,
            "xg": float(row["xg"]) if pd.notna(row["xg"]) else None,
            "xga": float(row["xga"]) if pd.notna(row["xga"]) else None,
            "npxg": float(row["npxg"]) if pd.notna(row["npxg"]) else None,
            "npxga": float(row["npxga"]) if pd.notna(row["npxga"]) else None,
            "xg_per_match": float(row["xg_per_match"]) if pd.notna(row["xg_per_match"]) else None,
            "xga_per_match": float(row["xga_per_match"]) if pd.notna(row["xga_per_match"]) else None,
            "xpts": float(row["xpts"]) if pd.notna(row["xpts"]) else None,
            "ppda": float(row["ppda"]) if pd.notna(row["ppda"]) else None,
            "deep_completions": float(row["deep_completions"]) if pd.notna(row["deep_completions"]) else None,
        }


class FBrefTacticalLoader:
    """Cargador compatible que prioriza Understat para xG tras la retirada de FBref (01/2026)."""

    def __init__(self, store: DuckDBStore | None = None):
        self.store = store or get_store()
        self.aligner = TeamAligner()
        self.cache_dir = SOCCERDATA_RAW_DIR / "data" / "FBref"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.understat = UnderstatTacticalLoader(store=self.store)

    def get_team_season_stats(
        self,
        league: str = "ESP-La Liga",
        season: str | None = None,
    ) -> pd.DataFrame:
        """Obtiene estadísticas básicas de FBref (xG/xGA ya no se suministran desde 01/2026)."""
        season_code = season or resolve_current_season_code()
        artifact_key = f"soccerdata/fbref/{league.replace(' ', '_').lower()}_{season_code}_stats.json"
        cached = read_json(artifact_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            import soccerdata as sd

            fb = sd.FBref(leagues=league, seasons=season_code, data_dir=self.cache_dir)
            df = fb.read_team_season_stats(stat_type="standard")
            if not df.empty:
                df = df.reset_index()
                write_json(artifact_key, _serialize_records(df))
                return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudieron descargar estadísticas FBref para %s %s: %s", league, season_code, exc)

        return pd.DataFrame()

    def get_team_context(
        self,
        team_name: str,
        league: str = "ESP-La Liga",
        season: str | None = None,
        as_of_date: str | datetime | None = None,
    ) -> dict[str, Any] | None:
        """Extrae perfil táctico priorizando Understat para xG/xGA y cayendo a FBref básico si falta."""
        season_code = season or resolve_current_season_code(as_of_date)
        if league in UNDERSTAT_SUPPORTED_LEAGUES:
            us_ctx = self.understat.get_team_context(
                team_name=team_name,
                league=league,
                season=season_code,
                as_of_date=as_of_date,
            )
            if us_ctx is not None:
                return us_ctx

        df = self.get_team_season_stats(league=league, season=season_code)
        if df.empty:
            return None

        team_col = "team" if "team" in df.columns else df.columns[0]
        candidates = df[team_col].astype(str).tolist()
        match, score = self.aligner.find_best_match(team_name, candidates, threshold=0.7)
        if not match or score < 0.7:
            return None

        row = df[df[team_col] == match].iloc[0]
        xg_val = row.get(("Expected", "xG"), row.get("xG"))
        xga_val = row.get(("Expected", "xGA"), row.get("xGA"))
        has_advanced = pd.notna(xg_val) and pd.notna(xga_val)
        return {
            "team": match,
            "matched_score": float(score),
            "season": season_code,
            "source": "fbref" if has_advanced else "fbref_basic_only",
            "matches_played": row.get(("Playing Time", "MP"), row.get("MP")),
            "goals_scored": row.get(("Performance", "Gls"), row.get("Gls")),
            "assists": row.get(("Performance", "Ast"), row.get("Ast")),
            "xg": xg_val if pd.notna(xg_val) else None,
            "xga": xga_val if pd.notna(xga_val) else None,
        }


class WhoScoredMissingPlayersLoader:
    """Cargador resiliente de jugadores ausentes con validación de esquema y fallback API-Football."""

    ADAPTER_VERSION = "soccerdata-whoscored-v2"
    EXPECTED_COLUMNS = {"team", "player"}

    def __init__(self, store: DuckDBStore | None = None):
        self.store = store or get_store()
        self.aligner = TeamAligner()

    @classmethod
    def validate_schema(cls, df: pd.DataFrame | None) -> bool:
        """Valida que el DataFrame de WhoScored contenga las columnas esperadas."""
        return df is not None and not df.empty and cls.EXPECTED_COLUMNS.issubset(set(df.columns))

    def _record_audit(self, source_status: str, rows: int = 0, detail: str | None = None):
        payload = {
            "source": "whoscored",
            "adapter_version": self.ADAPTER_VERSION,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "source_status": source_status,
            "rows": int(rows),
            "detail": detail,
        }
        try:
            self.store.put("raw", "soccerdata/scraper_audit/whoscored_latest.json", payload)
        except Exception:  # noqa: BLE001
            pass

    def fetch_missing_players(
        self,
        league: str = "ESP-La Liga",
        season: str | None = None,
        max_retries: int = 2,
        base_backoff_sec: float = 1.0,
    ) -> pd.DataFrame:
        """Obtiene la lista de jugadores ausentes con backoff ante 429 y auditoría de esquema."""
        season_code = season or resolve_current_season_code()
        today_str = datetime.now().strftime("%Y-%m-%d")
        artifact_key = f"soccerdata/whoscored/missing_players_{league.replace(' ', '_').lower()}_{season_code}_{today_str}.json"

        cached = read_json(artifact_key)
        if cached:
            df_cached = pd.DataFrame(cached)
            if self.validate_schema(df_cached):
                self._record_audit("ok_cache", rows=len(df_cached))
                return df_cached

        last_err: str | None = None
        for attempt in range(max_retries):
            try:
                import soccerdata as sd

                ws = sd.WhoScored(
                    leagues=league,
                    seasons=season_code,
                    headless=True,
                    data_dir=SOCCERDATA_RAW_DIR / "data" / "WhoScored",
                )
                df = ws.read_missing_players()
                if df is not None and not df.empty:
                    df = df.reset_index()
                    if not self.validate_schema(df):
                        self._record_audit("schema_error", rows=len(df), detail=f"cols={list(df.columns)}")
                        return pd.DataFrame()
                    write_json(artifact_key, _serialize_records(df))
                    self._record_audit("ok", rows=len(df))
                    return df
                self._record_audit("empty", rows=0)
                return pd.DataFrame()
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                logger.warning("WhoScored intento %d/%d falló: %s", attempt + 1, max_retries, exc)
                if attempt + 1 < max_retries and ("429" in last_err or "timeout" in last_err.lower()):
                    time.sleep(base_backoff_sec * (2 ** attempt))
                else:
                    break

        self._record_audit("error", rows=0, detail=last_err)
        return pd.DataFrame()

    def get_missing_for_match(
        self,
        home_team: str,
        away_team: str,
        league: str = "ESP-La Liga",
        season: str | None = None,
        match_date: str | None = None,
        fallback_apifootball: bool = True,
        apifootball_client: Any = None,
    ) -> dict[str, Any]:
        """Devuelve el listado de bajas para ambos equipos con fallback a API-Football."""
        season_code = season or resolve_current_season_code()
        df = self.fetch_missing_players(league=league, season=season_code)
        if self.validate_schema(df):
            team_col = "team" if "team" in df.columns else df.columns[0]
            all_teams = df[team_col].dropna().unique().tolist()

            home_match, h_score = self.aligner.find_best_match(home_team, all_teams, threshold=0.7)
            away_match, a_score = self.aligner.find_best_match(away_team, all_teams, threshold=0.7)

            home_list = []
            if home_match and h_score >= 0.7:
                for _, r in df[df[team_col] == home_match].iterrows():
                    home_list.append({
                        "player": str(r.get("player", "")),
                        "reason": str(r.get("reason", "Lesión o sanción")),
                        "status": str(r.get("status", "Doubtful/Out")),
                    })

            away_list = []
            if away_match and a_score >= 0.7:
                for _, r in df[df[team_col] == away_match].iterrows():
                    away_list.append({
                        "player": str(r.get("player", "")),
                        "reason": str(r.get("reason", "Lesión o sanción")),
                        "status": str(r.get("status", "Doubtful/Out")),
                    })

            return {"home_missing": home_list, "away_missing": away_list, "source": "whoscored"}

        if fallback_apifootball:
            try:
                from .apifootball_context import ApiFootballContextClient

                client = apifootball_client or ApiFootballContextClient(store=self.store)
                target_day = (match_date or datetime.now().strftime("%Y-%m-%d"))[:10]
                inj_data = client.injuries_for_date(target_day)
                records = inj_data.get("records") or []
                home_list, away_list = [], []
                for rec in records:
                    t_name = str(((rec.get("team") or {}).get("name")) or "").strip()
                    p_info = rec.get("player") or {}
                    p_name = str(p_info.get("name") or "").strip()
                    if not t_name or not p_name:
                        continue
                    _, sh = self.aligner.find_best_match(t_name, [home_team], threshold=0.75)
                    _, sa = self.aligner.find_best_match(t_name, [away_team], threshold=0.75)
                    item = {
                        "player": p_name,
                        "reason": str(p_info.get("reason") or "Lesión o sanción"),
                        "status": str(p_info.get("type") or "Out"),
                    }
                    if sh >= 0.75 and sh > sa:
                        home_list.append(item)
                    elif sa >= 0.75 and sa > sh:
                        away_list.append(item)
                if home_list or away_list or not inj_data.get("error"):
                    return {"home_missing": home_list, "away_missing": away_list, "source": "api_football_fallback"}
            except Exception as exc:  # noqa: BLE001
                logger.debug("Fallback API-Football de lesiones falló: %s", exc)

        return {"home_missing": [], "away_missing": [], "source": "none"}
