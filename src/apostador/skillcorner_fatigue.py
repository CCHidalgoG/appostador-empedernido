"""Módulo de fatiga física y congestión de calendario para fútbol basado en SkillCorner.

Combina métricas de alta intensidad de SkillCorner Physical Data / Game Intelligence
(`high_speed_running_m` > 19.8 km/h, `sprint_distance_m` > 25.2 km/h, acciones de presión
y carreras sin balón) con los días de descanso pre-partido (`rest_days`), produciendo un
ajuste acotado `fatigue_delta` compatible con `SoccerPredictor`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .open_data_lab import OpenDataCalibrationLab

# Valores de referencia colectivos aproximados por partido de equipo (11 jugadores)
BASELINE_TEAM_HSR_M = 9500.0
BASELINE_TEAM_SPRINT_M = 1800.0
BASELINE_TEAM_PRESS_ACTIONS = 180.0
MAX_FATIGUE_DELTA = 0.06


def _compute_team_fatigue_index(
    physical: dict[str, Any] | None = None,
    rest_days: float | None = None,
) -> tuple[float, str]:
    """Calcula índice de fatiga individual de un equipo [0.0, ~0.08] y fuente usada."""
    rest_penalty = 0.0
    if rest_days is not None:
        r = float(rest_days)
        if r <= 2.0:
            rest_penalty = 0.030
        elif r <= 3.0:
            rest_penalty = 0.020
        elif r <= 4.0:
            rest_penalty = 0.008

    if not physical:
        return rest_penalty, ("rest_days_only" if rest_days is not None else "none")

    recent_matches = max(1, int(physical.get("recent_matches_7d") or 1))
    hsr = float(
        physical.get("high_speed_running_m")
        or physical.get("hsr_distance_m")
        or physical.get("hi_distance_m")
        or 0.0
    )
    sprint = float(physical.get("sprint_distance_m") or physical.get("sprint_m") or 0.0)
    press = float(
        physical.get("pressing_actions")
        or physical.get("pressing_index")
        or physical.get("off_ball_runs")
        or 0.0
    )

    hsr_ratio = hsr / (BASELINE_TEAM_HSR_M * recent_matches) if hsr > 0 else 1.0
    sprint_ratio = sprint / (BASELINE_TEAM_SPRINT_M * recent_matches) if sprint > 0 else 1.0
    press_ratio = press / (BASELINE_TEAM_PRESS_ACTIONS * recent_matches) if press > 0 else 1.0

    has_physical_values = hsr > 0 or sprint > 0 or press > 0
    if not has_physical_values:
        congestion_bonus = 0.006 * max(0, recent_matches - 1) if rest_days is not None else 0.0
        src = "rest_days_only" if rest_days is not None else "none"
        return rest_penalty + congestion_bonus, src

    load_ratio = 0.50 * hsr_ratio + 0.35 * sprint_ratio + 0.15 * press_ratio
    load_excess = max(0.0, load_ratio - 1.0) * 0.025
    congestion_bonus = 0.007 * max(0, recent_matches - 1)

    if rest_penalty > 0:
        # El descanso corto amplifica el desgaste cuando la intensidad reciente fue alta
        multiplier = 1.0 + 0.45 * max(0.0, load_ratio - 0.85)
        fatigue_idx = rest_penalty * multiplier + load_excess + congestion_bonus
    else:
        fatigue_idx = load_excess + congestion_bonus

    return float(fatigue_idx), "skillcorner_physical"


def compute_soccer_fatigue_delta(
    home_physical: dict[str, Any] | None = None,
    away_physical: dict[str, Any] | None = None,
    home_rest_days: float | None = None,
    away_rest_days: float | None = None,
) -> dict[str, Any]:
    """Calcula `fatigue_delta` acotado en [-0.06, +0.06] para fútbol.

    Un valor positivo favorece al equipo local (el visitante acumula mayor fatiga),
    mientras que un valor negativo penaliza al equipo local.
    """
    home_idx, home_src = _compute_team_fatigue_index(home_physical, home_rest_days)
    away_idx, away_src = _compute_team_fatigue_index(away_physical, away_rest_days)

    delta = float(np.clip(away_idx - home_idx, -MAX_FATIGUE_DELTA, MAX_FATIGUE_DELTA))
    if "skillcorner_physical" in (home_src, away_src):
        source = "skillcorner_physical"
    elif "rest_days_only" in (home_src, away_src):
        source = "rest_days_only"
    else:
        source = "none"

    return {
        "fatigue_delta": round(delta, 5),
        "home_fatigue_index": round(home_idx, 5),
        "away_fatigue_index": round(away_idx, 5),
        "source": source,
    }


class SkillCornerFatigueLoader:
    """Adaptador de métricas físicas SkillCorner (API Physical/Game Intelligence o Open Data)."""

    def __init__(self, lab: OpenDataCalibrationLab | None = None):
        self.lab = lab or OpenDataCalibrationLab()

    @staticmethod
    def aggregate_physical_records(
        records: list[dict[str, Any]],
        recent_matches_7d: int = 1,
    ) -> dict[str, float]:
        """Agrega registros por jugador/partido de SkillCorner Physical Data a nivel equipo."""
        if not records:
            return {}
        hsr = 0.0
        sprint = 0.0
        total_dist = 0.0
        press = 0.0
        minutes = 0.0
        for r in records:
            hsr += float(
                r.get("high_speed_running_m")
                or r.get("hi_distance")
                or r.get("hsr_distance")
                or 0.0
            )
            sprint += float(r.get("sprint_distance_m") or r.get("sprint_distance") or 0.0)
            total_dist += float(r.get("total_distance_m") or r.get("distance") or 0.0)
            press += float(r.get("pressing_actions") or r.get("count_pressing") or r.get("off_ball_runs") or 0.0)
            minutes += float(r.get("minutes_played") or r.get("minutes") or 0.0)

        return {
            "high_speed_running_m": round(hsr, 2),
            "sprint_distance_m": round(sprint, 2),
            "total_distance_m": round(total_dist, 2),
            "pressing_actions": round(press, 1),
            "minutes_played": round(minutes, 1),
            "recent_matches_7d": max(1, int(recent_matches_7d)),
        }

    def summarize_from_opendata_match(
        self,
        tracking_jsonl_path: str | Path,
        dynamic_events_csv_path: str | Path | None = None,
        recent_matches_7d: int = 1,
    ) -> dict[str, Any]:
        """Extrae carga física SkillCorner desde tracking 10 Hz (+ dynamic events)."""
        summary = self.lab.load_skillcorner_match(
            tracking_jsonl_path,
            dynamic_events_csv_path=dynamic_events_csv_path,
        )
        return {
            "high_speed_running_m": float(summary.get("high_speed_running_m") or 0.0),
            "sprint_distance_m": float(summary.get("sprint_distance_m") or 0.0),
            "total_distance_m": float(summary.get("total_distance_m") or 0.0),
            "pressing_actions": float(summary.get("dynamic_events_count") or 0.0),
            "detected_ratio": summary.get("detected_ratio"),
            "recent_matches_7d": max(1, int(recent_matches_7d)),
            "attribution": summary.get("attribution"),
        }
