"""
recent_form.py — Cálculo de forma reciente de equipos a partir del dataset histórico.

Optimización #2 (Alta Prioridad): Incorporar los últimos N resultados de cada equipo
como features adicionales al modelo de predicción.

Métricas calculadas (ventana de últimos N partidos antes de la fecha del partido):
  - form_wins_rate: % de victorias en ventana
  - form_goals_scored_avg: promedio de goles anotados
  - form_goals_conceded_avg: promedio de goles recibidos
  - form_points_avg: promedio de puntos (3=V, 1=E, 0=D) por partido

El cargador construye el historial desde los fixtures históricos de `raw.artifacts` en DuckDB
y/o desde el dataset histórico procesado.
"""

import glob
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .aligner import TeamAligner
from .config import APIFOOTBALL_RAW_DIR, PROCESSED_DIR
from .duckdb_store import get_store, read_json


class RecentFormLoader:
    """
    Construye y cachea el historial de resultados de equipos a partir de los
    fixtures importados en `raw.artifacts`.
    """

    def __init__(self, raw_fixtures_dir=None,
                 processed_datasets=None, window=5):
        """
        raw_fixtures_dir: directorio con fixtures_{fecha}.json de API-Football
        processed_datasets: lista de paths a datasets procesados adicionales
        window: número de partidos previos a considerar
        """
        self.raw_fixtures_dir = raw_fixtures_dir or APIFOOTBALL_RAW_DIR
        if processed_datasets is None:
            processed_datasets = [
                PROCESSED_DIR / "historical_fixtures_dataset.json",
                PROCESSED_DIR / "americas_historical_fixtures.json",
            ]
        self.processed_datasets = processed_datasets
        self.window = window
        self.aligner = TeamAligner()
        self._history = None  # {normalized_team_name: [match_dict, ...]}
        self._name_to_id = {}  # normalized team name -> history key

    def _build_history(self):
        """
        Construye el historial de resultados con nombres normalizados y fechas UTC.
        """
        history = defaultdict(list)
        name_to_id = {}

        def team_key(name):
            normalized = unicodedata.normalize("NFKD", str(name or ""))
            normalized = normalized.encode("ascii", "ignore").decode("ascii").casefold()
            return re.sub(r"[^a-z0-9]+", " ", normalized).strip()

        def add_fixture(date, home_name, away_name, hg, ag, competition=None, competition_id=None):
            home_key, away_key = team_key(home_name), team_key(away_name)
            if not home_key or not away_key:
                return
            name_to_id[home_key] = home_key
            name_to_id[away_key] = away_key
            history[home_key].append({
                "date": date, "is_home": True, "opponent": away_key,
                "goals_scored": int(hg), "goals_conceded": int(ag),
                "competition": competition, "competition_id": competition_id,
            })
            history[away_key].append({
                "date": date, "is_home": False, "opponent": home_key,
                "goals_scored": int(ag), "goals_conceded": int(hg),
                "competition": competition, "competition_id": competition_id,
            })

        # ── 1. Fixtures API-Football de raw.artifacts ──────
        if Path(self.raw_fixtures_dir).resolve() == APIFOOTBALL_RAW_DIR.resolve():
            fixture_payloads = [payload for _, payload in get_store().list("raw", "apifootball/fixtures_")]
        else:
            fixture_payloads = []
            pattern = os.path.join(self.raw_fixtures_dir, "fixtures_*.json")
            for fpath in glob.glob(pattern):
                try:
                    fixture_payloads.append(read_json(fpath, default={}))
                except Exception:
                    continue
        for data in fixture_payloads:
            try:
                for fix in data.get("response", []):
                    status = fix.get("fixture", {}).get("status", {}).get("short", "")
                    if status not in ["FT", "AET", "PEN"]:
                        continue
                    date_str = fix.get("fixture", {}).get("date", "")
                    if not date_str:
                        continue
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    date = date.astimezone(timezone.utc)

                    home_name = fix.get("teams", {}).get("home", {}).get("name", "")
                    away_name = fix.get("teams", {}).get("away", {}).get("name", "")
                    hg = fix.get("goals", {}).get("home")
                    ag = fix.get("goals", {}).get("away")

                    if hg is None or ag is None:
                        continue
                    league = fix.get("league") or {}
                    add_fixture(date, home_name, away_name, hg, ag,
                                league.get("name"), league.get("id"))
            except Exception:
                continue

        # ── 2. Datasets procesados (historical_fixtures_dataset.json, etc.) ───
        for dpath in self.processed_datasets:
            fixtures = read_json(dpath)
            if fixtures is None:
                continue
            try:
                for fix in fixtures:
                    status = fix.get("fixture", {}).get("status", {}).get("short", "")
                    if status not in ["FT", "AET", "PEN", "Match Finished"]:
                        long_status = fix.get("fixture", {}).get("status", {}).get("long", "")
                        if long_status != "Match Finished":
                            continue
                    date_str = fix.get("fixture", {}).get("date", "")
                    if not date_str:
                        continue
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    date = date.astimezone(timezone.utc)

                    home_name = fix.get("teams", {}).get("home", {}).get("name", "")
                    away_name = fix.get("teams", {}).get("away", {}).get("name", "")
                    hg = fix.get("goals", {}).get("home")
                    ag = fix.get("goals", {}).get("away")

                    if hg is None or ag is None:
                        continue

                    league = fix.get("league") or {}
                    add_fixture(date, home_name, away_name, hg, ag,
                                league.get("name"), league.get("id"))
            except Exception:
                continue

        # Ordenar por fecha
        for team_id in history:
            unique = {}
            for match in history[team_id]:
                key = (match["date"], match["is_home"], match["opponent"],
                       match["goals_scored"], match["goals_conceded"], match.get("competition_id"))
                unique[key] = match
            history[team_id] = sorted(unique.values(), key=lambda x: x["date"])

        self._history = dict(history)
        self._name_to_id = name_to_id
        print(f"[RecentFormLoader] Historial construido: {len(self._history)} equipos únicos")

    def _ensure_loaded(self):
        if self._history is None:
            self._build_history()

    def get_form(self, team_name, before_date, window=None):
        """
        Retorna métricas de forma reciente del equipo ANTES de la fecha indicada.

        Parámetros:
            team_name: nombre del equipo (como aparece en API-Football)
            before_date: string 'YYYY-MM-DD' o datetime
            window: número de partidos (usa self.window si es None)

        Retorna dict con:
            form_wins_rate: % victorias en ventana [0,1]
            form_draw_rate: % empates [0,1]
            form_goals_scored_avg: promedio goles anotados
            form_goals_conceded_avg: promedio goles recibidos
            form_points_avg: promedio puntos (3/1/0)
            form_n: partidos en ventana (0 si no hay histórico)
        """
        self._ensure_loaded()
        w = window or self.window

        # Resolver ID del equipo
        normalized_name = unicodedata.normalize("NFKD", str(team_name or ""))
        normalized_name = normalized_name.encode("ascii", "ignore").decode("ascii").casefold()
        normalized_name = re.sub(r"[^a-z0-9]+", " ", normalized_name).strip()
        if not normalized_name:
            return self._empty_form()
        team_id = self._name_to_id.get(normalized_name)
        if team_id is None and self._history:
            best_match, score = self.aligner.find_best_match(
                team_name, list(self._history.keys()), threshold=0.75
            )
            if best_match and score >= 0.75:
                team_id = best_match
                self._name_to_id[normalized_name] = best_match
        if team_id is None:
            return self._empty_form()

        if isinstance(before_date, str):
            try:
                before_dt = datetime.fromisoformat(before_date.replace("Z", "+00:00"))
                if len(before_date) == 10:
                    before_dt = before_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return self._empty_form()
        else:
            before_dt = before_date
        if before_dt.tzinfo is None:
            before_dt = before_dt.replace(tzinfo=timezone.utc)
        before_dt = before_dt.astimezone(timezone.utc)

        matches = self._history.get(team_id, [])
        prior = [m for m in matches if m["date"] < before_dt]

        if not prior:
            return self._empty_form()

        recent = prior[-w:]
        n = len(recent)
        wins = sum(1 for m in recent if m["goals_scored"] > m["goals_conceded"])
        draws = sum(1 for m in recent if m["goals_scored"] == m["goals_conceded"])
        goals_scored = sum(m["goals_scored"] for m in recent)
        goals_conceded = sum(m["goals_conceded"] for m in recent)
        points = wins * 3 + draws * 1

        # Ponderación exponencial por recencia y diferencia de goles media
        weighted_pts_sum = 0.0
        weight_total = 0.0
        for m in recent:
            days_ago = max(0, (before_dt - m["date"]).days)
            recency_w = math.exp(-0.04 * days_ago)
            m_pts = 3.0 if m["goals_scored"] > m["goals_conceded"] else (1.0 if m["goals_scored"] == m["goals_conceded"] else 0.0)
            weighted_pts_sum += recency_w * m_pts
            weight_total += recency_w
        weighted_points_avg = weighted_pts_sum / weight_total if weight_total > 0 else (points / n)
        goal_diff_avg = (goals_scored - goals_conceded) / n

        last_match = prior[-1]["date"]
        recent_matches = []
        for match in recent:
            scored, conceded = match["goals_scored"], match["goals_conceded"]
            recent_matches.append({
                "date": match["date"].date().isoformat(),
                "venue": "local" if match["is_home"] else "visitante",
                "opponent": match["opponent"],
                "goals_for": scored,
                "goals_against": conceded,
                "competition": match.get("competition"),
                "outcome": "W" if scored > conceded else "D" if scored == conceded else "L",
            })
        return {
            "form_wins_rate": wins / n,
            "form_draw_rate": draws / n,
            "form_goals_scored_avg": goals_scored / n,
            "form_goals_conceded_avg": goals_conceded / n,
            "form_points_avg": points / n,
            "form_weighted_points_avg": weighted_points_avg,
            "form_goal_diff_avg": goal_diff_avg,
            "form_n": n,
            "last_match_date": last_match.date().isoformat(),
            "rest_days": max(0.0, (before_dt - last_match).total_seconds() / 86400),
            "matches_7d": sum(before_dt - m["date"] <= timedelta(days=7) for m in prior),
            "matches_14d": sum(before_dt - m["date"] <= timedelta(days=14) for m in prior),
            "competitions_14d": sorted({m.get("competition") for m in prior
                                         if m.get("competition") and before_dt - m["date"] <= timedelta(days=14)}),
            "recent_matches": recent_matches,
        }

    @staticmethod
    def _empty_form():
        """Valores neutros cuando no hay histórico disponible."""
        return {
            "form_wins_rate": 0.0,
            "form_draw_rate": 0.0,
            "form_goals_scored_avg": 0.0,
            "form_goals_conceded_avg": 0.0,
            "form_points_avg": 0.0,
            "form_weighted_points_avg": 0.0,
            "form_goal_diff_avg": 0.0,
            "form_n": 0,
            "last_match_date": None,
            "rest_days": None,
            "matches_7d": 0,
            "matches_14d": 0,
            "competitions_14d": [],
            "recent_matches": [],
        }

    def get_form_adjustment(self, home_name, away_name, before_date, window=None):
        """
        Calcula el ajuste de forma combinando puntos ponderados por recencia (70%)
        y diferencia de goles media (30%), normalizado en [-0.10, +0.10].

        Retorna:
            form_delta: float en [-0.10, +0.10] (positivo = ventaja local)
            home_form: dict con métricas del equipo local
            away_form: dict con métricas del equipo visitante
        """
        home_form = self.get_form(home_name, before_date, window)
        away_form = self.get_form(away_name, before_date, window)

        h_w_pts = home_form.get("form_weighted_points_avg", home_form["form_points_avg"])
        a_w_pts = away_form.get("form_weighted_points_avg", away_form["form_points_avg"])
        h_gd = home_form.get("form_goal_diff_avg", 0.0)
        a_gd = away_form.get("form_goal_diff_avg", 0.0)

        min_sample = min(home_form["form_n"], away_form["form_n"])
        if min_sample >= 3:
            effective_pts_diff = 0.70 * (h_w_pts - a_w_pts) + 0.30 * max(-3.0, min(3.0, h_gd - a_gd))
            form_delta = max(-0.10, min(0.10, (effective_pts_diff / 3.0) * 0.10))
        else:
            form_delta = 0.0

        return {
            "form_delta": round(form_delta, 5),
            "home_form": home_form,
            "away_form": away_form
        }
