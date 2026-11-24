import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from .config import IMPECT_RAW_DIR
from .duckdb_store import artifact_key, get_store, read_json

_DATE_SNAP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.json$")


def _normalize_target_day(target_date) -> str | None:
    if target_date is None:
        return None
    ts = pd.to_datetime(target_date, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def resolve_as_of_impect_records(
    raw_impect_dir: str | Path,
    prefix: str,
    iteration_id: int | str,
    target_date=None,
    reader=read_json,
) -> tuple[list[dict], bool]:
    """Resuelve artefactos Impect con garantía estricta `as-of` (snapshots fechados o campo `as_of_date`).

    Retorna `(records, is_point_in_time_safe)`. Si existen snapshots fechados
    `{prefix}_{iteration_id}_{YYYY-MM-DD}.json` con fecha <= `target_date` o
    registros con `as_of_date`, devuelve únicamente datos vigentes en `target_date`
    y marca `is_point_in_time_safe=True`.
    """
    t_day = _normalize_target_day(target_date)
    base_dir = Path(raw_impect_dir)
    pattern = re.compile(
        rf"^{re.escape(prefix)}_{re.escape(str(iteration_id))}_(\d{{4}}-\d{{2}}-\d{{2}})\.json$"
    )

    candidates: list[tuple[str, Path | None, list[dict] | None]] = []
    if base_dir.is_dir():
        for file in base_dir.glob(f"{prefix}_{iteration_id}_*.json"):
            m = pattern.match(file.name)
            if m:
                candidates.append((m.group(1), file, None))

    layer, sample_key = artifact_key(base_dir / f"{prefix}_{iteration_id}.json")
    if layer == "raw":
        key_prefix = f"{Path(sample_key).parent.as_posix()}/{prefix}_{iteration_id}_".lstrip("./")
        if Path(sample_key).parent.as_posix() == ".":
            key_prefix = f"{prefix}_{iteration_id}_"
        try:
            for k, payload in get_store().list("raw", prefix=key_prefix):
                fname = Path(k).name
                m = pattern.match(fname)
                if m and isinstance(payload, list):
                    candidates.append((m.group(1), None, payload))
        except Exception:
            pass

    eligible = [c for c in candidates if t_day is None or c[0] <= t_day]
    if eligible:
        eligible.sort(key=lambda c: c[0])
        _, path_opt, payload_opt = eligible[-1]
        data = payload_opt if payload_opt is not None else reader(path_opt, default=[])
        if isinstance(data, list):
            if t_day and any(isinstance(r, dict) and r.get("as_of_date") for r in data):
                data = [r for r in data if str(r.get("as_of_date") or "")[:10] <= t_day]
            return data, True

    base_path = base_dir / f"{prefix}_{iteration_id}.json"
    data = reader(base_path, default=[])
    if not isinstance(data, list) or not data:
        return [], False

    has_as_of = any(isinstance(r, dict) and bool(r.get("as_of_date")) for r in data)
    if has_as_of:
        if t_day is None:
            return data, True
        filtered = [r for r in data if str(r.get("as_of_date") or "")[:10] <= t_day]
        return filtered, True

    return data, False


class ImpectTeamMetricsLoader:
    _DEFAULT_DIR = os.path.join(
        IMPECT_RAW_DIR
    )

    def __init__(self, raw_impect_dir=None):
        self.raw_impect_dir = raw_impect_dir or self._DEFAULT_DIR
        self.cache = {}

    def load_team_features(self, iteration_id, target_date=None):
        """
        Loads squad strength ratings and squad iteration averages for a given iteration.
        Returns a pandas DataFrame indexed by squadName with relevant team metrics.
        """
        date_str = str(target_date)[:10] if target_date else "latest"
        cache_key = (iteration_id, date_str)
        if cache_key in self.cache:
            return self.cache[cache_key]
        ratings_path = os.path.join(self.raw_impect_dir, f"ratings_{iteration_id}.json")

        ratings_map = {}
        r_data = read_json(ratings_path, default=[])
        if r_data:
            df_r = pd.DataFrame(r_data)
            if {"date", "squadName", "value"}.issubset(df_r.columns):
                df_r["date"] = pd.to_datetime(df_r["date"], utc=True).dt.tz_localize(None)
                available_dates = df_r["date"].dropna().unique()
                if len(available_dates):
                    if target_date is not None:
                        t_date = pd.to_datetime(target_date, utc=True).tz_localize(None)
                        past_dates = [d for d in available_dates if d <= t_date]
                        closest_date = max(past_dates) if past_dates else None
                    else:
                        closest_date = df_r["date"].max()
                    if closest_date is not None:
                        latest = df_r[df_r["date"] == closest_date]
                        ratings_map = dict(zip(latest["squadName"], latest["value"].astype(float)))

        averages_df = pd.DataFrame()
        a_data, is_as_of_safe = resolve_as_of_impect_records(
            self.raw_impect_dir, "averages", iteration_id, target_date=target_date
        )
        if a_data:
            averages_df = pd.DataFrame(a_data)
            # Only discard unversioned season-to-date aggregates on past/current target dates
            # when no point-in-time snapshot (`as_of_date`) backs them.
            if target_date is not None and not is_as_of_safe:
                target_day = pd.to_datetime(target_date, utc=True).date()
                today_utc = datetime.now(timezone.utc).date()
                if target_day <= today_utc:
                    averages_df = pd.DataFrame()

        if averages_df.empty and not ratings_map:
            return pd.DataFrame()

        # Key KPI columns to extract if available
        desired_kpis = [
            "BYPASSED_OPPONENTS",
            "BYPASSED_DEFENDERS",
            "BALL_WIN_NUMBER",
            "BALL_LOSS_NUMBER",
            "SUFFERED_BYPASSED_OPPONENTS",
            "SUFFERED_BYPASSED_DEFENDERS",
            "GOALS",
            "EXPECTED_GOALS",
            "SHOT_XG",
            "CONCEDED_SHOT_XG",
            "POST_SHOT_XG",
            "PXT",
            "SHOT_AT_GOAL_NUMBER",
            "RECEPTIONS_BEHIND_DEFENSIVE_LINE",
            "BALL_WIN_ADDED_TEAMMATES",
            "BALL_LOSS_ADDED_OPPONENTS",
            "MATCH_SHARE",
            "PLAYDURATION",
            "BYPASSED_OPPONENTS_RECEIVING",
            "NEUTRAL_PLAY_NUMBER"
        ]

        team_rows = []
        all_squads = set(list(ratings_map.keys()))
        if not averages_df.empty and "squadName" in averages_df.columns:
            all_squads.update(averages_df["squadName"].tolist())

        avg_lookup = averages_df.set_index("squadName") if not averages_df.empty and "squadName" in averages_df.columns else pd.DataFrame()

        for squad in all_squads:
            row_dict = {"squadName": squad}
            row_dict["squad_rating"] = ratings_map.get(squad, 0.68)

            if not avg_lookup.empty and squad in avg_lookup.index:
                squad_avg_row = avg_lookup.loc[squad]
                if isinstance(squad_avg_row, pd.DataFrame):
                    squad_avg_row = squad_avg_row.iloc[0]
                for kpi in desired_kpis:
                    row_dict[kpi] = float(squad_avg_row[kpi]) if kpi in squad_avg_row and pd.notnull(squad_avg_row[kpi]) else 0.0
            else:
                for kpi in desired_kpis:
                    row_dict[kpi] = 0.0

            team_rows.append(row_dict)

        df_result = pd.DataFrame(team_rows).set_index("squadName")
        self.cache[cache_key] = df_result
        return df_result

    def get_team_tactical_summary(self, iteration_id, squad_name, target_date=None):
        """
        Retorna un resumen compacto con las métricas tácticas clave de un equipo:
        goles/partido, xG, goles concedidos, xGA, pases que rompen líneas, presiones,
        recuperaciones, pxT y post-shot xG (con resolución as-of si hay target_date).
        """
        if target_date is not None:
            data, is_as_of_safe = resolve_as_of_impect_records(
                self.raw_impect_dir, "averages", iteration_id, target_date=target_date
            )
            if not is_as_of_safe:
                averages_path = os.path.join(self.raw_impect_dir, f"averages_{iteration_id}.json")
                data = read_json(averages_path, default=[])
        else:
            averages_path = os.path.join(self.raw_impect_dir, f"averages_{iteration_id}.json")
            data = read_json(averages_path, default=[])
        if not data:
            return {}

        match_row = None
        squad_lower = str(squad_name).strip().casefold()
        for row in data:
            name = str(row.get("squadName", "")).strip().casefold()
            if name == squad_lower:
                match_row = row
                break
            if squad_lower in name or name in squad_lower:
                match_row = row

        if not match_row:
            return {}

        s = match_row
        passes_completed = (s.get("SUCCESSFUL_PASSES_BY_FOOT_RIGHT") or 0) + (s.get("SUCCESSFUL_PASSES_BY_FOOT_LEFT") or 0)
        return {
            "goles_partido": round(float(s.get("GOALS") or 0), 2),
            "xg_partido": round(float(s.get("SHOT_XG") or s.get("EXPECTED_GOALS") or 0), 2),
            "goles_concedidos": round(float(s.get("CONCEDED_GOALS") or 0), 2),
            "xg_concedido": round(float(s.get("CONCEDED_SHOT_XG") or 0), 2),
            "packing_defensores": round(float(s.get("BYPASSED_DEFENDERS") or 0), 1),
            "packing_total": round(float(s.get("BYPASSED_OPPONENTS") or 0), 1),
            "recuperaciones": round(float(s.get("BALL_WIN_NUMBER") or 0), 1),
            "perdidas": round(float(s.get("BALL_LOSS_NUMBER") or 0), 1),
            "presiones": round(float(s.get("NUMBER_OF_PRESSES") or 0), 1),
            "pases_completados": round(float(passes_completed), 1),
            "pxt_partido": round(float(s.get("PXT") or s.get("OFFENSIVE_PXT") or 0.0), 3),
            "psxg_partido": round(float(s.get("POST_SHOT_XG") or s.get("SHOT_XG") or s.get("EXPECTED_GOALS") or 0.0), 2),
            "remates_partido": round(float(s.get("SHOT_AT_GOAL_NUMBER") or 0.0), 1),
        }
