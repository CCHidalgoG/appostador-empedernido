"""Laboratorio offline de estandarización con kloppy y Open Data.

Soporta ingesta y calibración de priors offline sobre:
1. Hudl StatsBomb Open Data (eventos + 360 freeze-frames; atribución obligatoria).
2. IMPECT Open Data Bundesliga 2023/24 (Packing, pxT, xG).
3. SkillCorner Open Data (10 Hz broadcast tracking con flag is_detected).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class OpenDataCalibrationLab:
    """Cargador y calibrador offline de datasets abiertos de fútbol."""

    ATTRIBUTION_NOTICES = {
        "statsbomb": "Hudl StatsBomb Open Data — requiere atribución y logo según LICENSE.pdf.",
        "impect": "IMPECT Open Data (Bundesliga 2023/24) — requiere atribución según licencia.",
        "skillcorner": "SkillCorner Open Data — requiere atribución; contiene posiciones extrapoladas (~97% ID accuracy).",
    }

    @staticmethod
    def has_kloppy() -> bool:
        """Indica si la librería kloppy está instalada en el entorno."""
        try:
            import kloppy  # noqa: F401
            return True
        except ImportError:
            return False

    def load_statsbomb_match(
        self,
        events_path: str | Path,
        three_sixty_path: str | Path | None = None,
        lineups_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Carga eventos (y muestras 360 opcionales) de StatsBomb Open Data."""
        ev_file = Path(events_path)
        ts_file = Path(three_sixty_path) if three_sixty_path else None

        if self.has_kloppy() and lineups_path and Path(lineups_path).exists():
            try:
                from kloppy import statsbomb

                dataset = statsbomb.load(
                    event_data=str(ev_file),
                    lineup_data=str(lineups_path),
                    three_sixty_data=str(ts_file) if ts_file and ts_file.exists() else None,
                )
                df_ev = dataset.to_df()
                shots = df_ev[df_ev["event_type"] == "SHOT"] if "event_type" in df_ev.columns else pd.DataFrame()
                return {
                    "loader": "kloppy",
                    "total_events": int(len(df_ev)),
                    "total_shots": int(len(shots)),
                    "has_360": bool(ts_file and ts_file.exists()),
                    "attribution": self.ATTRIBUTION_NOTICES["statsbomb"],
                }
            except Exception:
                pass

        events = json.loads(ev_file.read_text(encoding="utf-8"))
        three_sixty = json.loads(ts_file.read_text(encoding="utf-8")) if ts_file and ts_file.exists() else []
        frames_by_id = {f.get("event_uuid"): f for f in three_sixty if isinstance(f, dict) and f.get("event_uuid")}

        team_xg: dict[str, float] = {}
        team_shots: dict[str, int] = {}
        team_passes: dict[str, int] = {}
        pressures = 0
        events_with_360 = 0

        for ev in events:
            ev_type = ((ev.get("type") or {}).get("name") or "").lower()
            team = ((ev.get("team") or {}).get("name") or "Unknown")
            if ev.get("id") in frames_by_id:
                events_with_360 += 1
            if ev_type == "shot":
                shot_xg = float((ev.get("shot") or {}).get("statsbomb_xg") or 0.0)
                team_xg[team] = team_xg.get(team, 0.0) + shot_xg
                team_shots[team] = team_shots.get(team, 0) + 1
            elif ev_type == "pass":
                team_passes[team] = team_passes.get(team, 0) + 1
            elif ev_type == "pressure":
                pressures += 1

        return {
            "loader": "native_json",
            "total_events": len(events),
            "total_shots": sum(team_shots.values()),
            "total_pressures": pressures,
            "events_with_360": events_with_360,
            "has_360": bool(three_sixty),
            "team_xg": {k: round(v, 4) for k, v in team_xg.items()},
            "team_shots": team_shots,
            "team_passes": team_passes,
            "attribution": self.ATTRIBUTION_NOTICES["statsbomb"],
        }

    def load_impect_open_bundesliga(self, json_path: str | Path) -> dict[str, Any]:
        """Procesa un JSON abierto de IMPECT Bundesliga 2023/24 (Packing, pxT, xG)."""
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            return {"teams": {}, "row_count": 0, "attribution": self.ATTRIBUTION_NOTICES["impect"]}

        teams: dict[str, dict[str, float]] = {}
        for r in rows:
            sq = str(r.get("squadName") or r.get("squad") or "Unknown")
            bucket = teams.setdefault(
                sq,
                {"matches": 0.0, "bypassed_defenders": 0.0, "bypassed_opponents": 0.0, "pxt": 0.0, "xg": 0.0},
            )
            bucket["matches"] += 1.0
            bucket["bypassed_defenders"] += float(r.get("BYPASSED_DEFENDERS") or 0.0)
            bucket["bypassed_opponents"] += float(r.get("BYPASSED_OPPONENTS") or 0.0)
            bucket["pxt"] += float(r.get("PXT") or r.get("pxT") or 0.0)
            bucket["xg"] += float(r.get("SHOT_XG") or r.get("EXPECTED_GOALS") or 0.0)

        return {
            "row_count": len(rows),
            "teams": teams,
            "attribution": self.ATTRIBUTION_NOTICES["impect"],
        }

    def load_skillcorner_match(
        self,
        tracking_jsonl_path: str | Path,
        dynamic_events_csv_path: str | Path | None = None,
        fps: float = 10.0,
        smoothing_window: int = 5,
    ) -> dict[str, Any]:
        """Carga tracking 10 Hz de SkillCorner filtrando is_detected y suavizando velocidades."""
        frames_path = Path(tracking_jsonl_path)
        player_series: dict[str, list[tuple[int, float, float]]] = {}
        total_observations = 0
        detected_observations = 0

        with frames_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                frame_num = int(frame.get("frame", idx))
                for p in frame.get("player_data", []):
                    total_observations += 1
                    if p.get("is_detected") is not True:
                        continue
                    x, y = p.get("x"), p.get("y")
                    pid = p.get("player_id")
                    if x is None or y is None or pid is None:
                        continue
                    detected_observations += 1
                    player_series.setdefault(str(pid), []).append((frame_num, float(x), float(y)))

        hsr_m = 0.0  # > 19.8 km/h = 5.5 m/s
        sprint_m = 0.0  # > 25.2 km/h = 7.0 m/s
        total_distance_m = 0.0

        for coords in player_series.values():
            if len(coords) < 2:
                continue
            coords.sort(key=lambda t: t[0])
            xs = pd.Series([c[1] for c in coords]).rolling(window=smoothing_window, min_periods=1, center=True).mean()
            ys = pd.Series([c[2] for c in coords]).rolling(window=smoothing_window, min_periods=1, center=True).mean()
            frames_arr = np.array([c[0] for c in coords], dtype=float)

            for i in range(1, len(coords)):
                df = frames_arr[i] - frames_arr[i - 1]
                if df <= 0 or df > fps * 2.0:
                    continue
                dt = df / fps
                dist = math.hypot(float(xs.iloc[i] - xs.iloc[i - 1]), float(ys.iloc[i] - ys.iloc[i - 1]))
                speed_ms = dist / dt
                if speed_ms > 11.5:  # filtrar saltos residuales de tracking
                    continue
                total_distance_m += dist
                if speed_ms >= 5.5:
                    hsr_m += dist
                if speed_ms >= 7.0:
                    sprint_m += dist

        dyn_events_count = 0
        if dynamic_events_csv_path and Path(dynamic_events_csv_path).exists():
            with Path(dynamic_events_csv_path).open("r", encoding="utf-8") as cf:
                dyn_events_count = sum(1 for _ in csv.DictReader(cf))

        return {
            "tracked_players": len(player_series),
            "total_observations": total_observations,
            "detected_observations": detected_observations,
            "detected_ratio": round(detected_observations / max(1, total_observations), 4),
            "total_distance_m": round(total_distance_m, 2),
            "high_speed_running_m": round(hsr_m, 2),
            "sprint_distance_m": round(sprint_m, 2),
            "dynamic_events_count": dyn_events_count,
            "attribution": self.ATTRIBUTION_NOTICES["skillcorner"],
        }

    def calibrate_open_data_priors(
        self,
        statsbomb_summary: dict[str, Any] | None = None,
        impect_summary: dict[str, Any] | None = None,
        skillcorner_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Genera coeficientes calibrados offline a partir de las muestras Open Data."""
        packing_per_xg = 28.0
        if impect_summary and impect_summary.get("teams"):
            total_xg = sum(t.get("xg", 0.0) for t in impect_summary["teams"].values())
            total_pack = sum(t.get("bypassed_defenders", 0.0) for t in impect_summary["teams"].values())
            if total_xg > 0:
                packing_per_xg = round(total_pack / total_xg, 3)

        hsr_km = (skillcorner_summary or {}).get("high_speed_running_m", 0.0) / 1000.0
        fatigue_penalty_per_hsr_km = round(min(0.05, 0.0025 * max(1.0, hsr_km / 10.0)), 5)

        return {
            "xg_to_lambda_scale": 1.0,
            "packing_per_xg_ratio": packing_per_xg,
            "fatigue_penalty_per_hsr_km": fatigue_penalty_per_hsr_km,
            "statsbomb_events_analyzed": int((statsbomb_summary or {}).get("total_events", 0)),
            "skillcorner_detected_ratio": (skillcorner_summary or {}).get("detected_ratio"),
            "attribution_notices": self.ATTRIBUTION_NOTICES,
        }
