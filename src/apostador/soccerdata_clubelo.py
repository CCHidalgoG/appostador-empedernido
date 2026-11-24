"""Módulo de Ratings Elo con SoccerData (ClubElo).

Proporciona consultas de ratings Elo con point-in-time safety (sin fuga de datos
hacia el futuro), persistencia en DuckDB (raw.clubelo_ratings), tolerancia
a fallos y conversión de diferenciales Elo a probabilidades 1X2.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .aligner import TeamAligner
from .config import SOCCERDATA_RAW_DIR
from .duckdb_store import DuckDBStore, get_store, write_json

logger = logging.getLogger(__name__)


class ClubEloLoader:
    """Cargador y gestor de ratings Elo con point-in-time safety."""

    _network_down: bool = False
    _failed_dates: set[str] = set()

    def __init__(self, store: DuckDBStore | None = None):
        self.store = store or get_store()
        self.aligner = TeamAligner()
        self.cache_dir = SOCCERDATA_RAW_DIR / "data" / "ClubElo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ratings_by_date(self, date_str: str | None = None) -> pd.DataFrame:
        """Descarga o recupera de caché los ratings Elo para una fecha específica."""
        date_str = (date_str or datetime.now().strftime("%Y-%m-%d"))[:10]
        if ClubEloLoader._network_down or date_str in ClubEloLoader._failed_dates:
            return pd.DataFrame()

        # Fast probe + direct CSV parse to avoid redundant roundtrips when api.clubelo.com responds
        try:
            import requests
            probe = requests.get(f"http://api.clubelo.com/{date_str}", timeout=5.0)
            if probe.status_code != 200:
                logger.info("api.clubelo.com no devolvió HTTP 200 (HTTP %d). Usando caché local.", probe.status_code)
                ClubEloLoader._failed_dates.add(date_str)
                return pd.DataFrame()
            text = (getattr(probe, "text", "") or "").strip()
            if text:
                first_line = text.splitlines()[0]
                if "Club" in first_line and "Elo" in first_line:
                    df_csv = pd.read_csv(io.StringIO(text))
                    df_csv.columns = [str(c).strip().lower() for c in df_csv.columns]
                    if "club" in df_csv.columns and "elo" in df_csv.columns:
                        return df_csv.rename(columns={"club": "team"})
        except Exception:
            if not ClubEloLoader._network_down:
                logger.info("api.clubelo.com no es accesible. Usando caché local.")
                ClubEloLoader._network_down = True
            ClubEloLoader._failed_dates.add(date_str)
            return pd.DataFrame()

        try:
            from .config import SOCCERDATA_RAW_DIR
            import soccerdata as sd
            elo_client = sd.ClubElo(data_dir=SOCCERDATA_RAW_DIR / "data" / "ClubElo")
            df = elo_client.read_by_date(date_str)
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001
            logger.info("No se pudo obtener ClubElo para fecha %s (%s). Usando caché local.", date_str, exc)
            ClubEloLoader._failed_dates.add(date_str)

        return pd.DataFrame()

    def sync_date(self, date_str: str | None = None) -> int:
        """Sincroniza una fecha de ClubElo en DuckDB (por defecto hoy)."""
        date_str = (date_str or datetime.now().strftime("%Y-%m-%d"))[:10]
        df = self.fetch_ratings_by_date(date_str)
        if df.empty:
            return 0

        df = df.reset_index()
        records = []
        for _, row in df.iterrows():
            rec = {
                "rating_date": date_str,
                "team_name": str(row.get("team", row.get("club", ""))),
                "country": str(row.get("country", "")),
                "level": int(row["level"]) if pd.notna(row.get("level")) else 1,
                "elo": float(row["elo"]) if pd.notna(row.get("elo")) else 1500.0,
            }
            if rec["team_name"]:
                records.append(rec)

        self.store.save_clubelo_ratings(records)
        write_json(f"soccerdata/clubelo/ratings_{date_str}.json", records)
        logger.info("Guardados %d ratings Elo para la fecha %s", len(records), date_str)
        return len(records)

    def load_by_date(self, date_str: str | None = None) -> int:
        """Alias amigable de sync_date() para sincronizar ratings ClubElo del día en DuckDB."""
        return self.sync_date(date_str)

    def get_team_elo(
        self,
        team_name: str,
        match_date: str | datetime,
        default_elo: float = 1500.0,
    ) -> float:
        """Obtiene el rating Elo de un equipo a la fecha del partido (point-in-time safe)."""
        dt_str = match_date.strftime("%Y-%m-%d") if isinstance(match_date, datetime) else str(match_date)[:10]

        # 1. Buscar en DuckDB point-in-time
        cached_elo = self.store.get_clubelo_rating(team_name, dt_str)
        if cached_elo is not None:
            return cached_elo

        # 2. Intentar buscar con alineación de nombres
        with self.store._connect() as con:
            known_teams = con.execute("SELECT DISTINCT team_name FROM raw.clubelo_ratings").fetch_df()
            if not known_teams.empty:
                candidates = known_teams["team_name"].tolist()
                best_match, score = self.aligner.find_best_match(team_name, candidates, threshold=0.75)
                if best_match and score >= 0.75:
                    matched_elo = self.store.get_clubelo_rating(best_match, dt_str)
                    if matched_elo is not None:
                        return matched_elo

        # 3. Intentar sincronizar la fecha online si la red está disponible y no ha fallado para esta fecha
        if not ClubEloLoader._network_down and dt_str not in ClubEloLoader._failed_dates:
            synced = self.sync_date(dt_str)
            if synced > 0:
                cached_elo = self.store.get_clubelo_rating(team_name, dt_str)
                if cached_elo is not None:
                    return cached_elo

        # 4. Fallback estándar si no hay datos
        return default_elo

    def find_team_elo(
        self,
        team_name: str,
        match_date: str | datetime | None = None,
        allow_network: bool = False,
    ) -> float | None:
        """Busca rating Elo point-in-time en DuckDB; devuelve None si no existe cobertura real."""
        if not team_name or not str(team_name).strip():
            return None
        if isinstance(match_date, datetime):
            dt_str = match_date.strftime("%Y-%m-%d")
        elif match_date:
            dt_str = str(match_date)[:10]
        else:
            dt_str = datetime.now().strftime("%Y-%m-%d")

        try:
            cached_elo = self.store.get_clubelo_rating(team_name, dt_str)
            if cached_elo is not None:
                return float(cached_elo)

            with self.store._connect() as con:
                known_teams = con.execute("SELECT DISTINCT team_name FROM raw.clubelo_ratings").fetch_df()
            if not known_teams.empty:
                candidates = known_teams["team_name"].tolist()
                best_match, score = self.aligner.find_best_match(team_name, candidates, threshold=0.75)
                if best_match and score >= 0.75:
                    matched_elo = self.store.get_clubelo_rating(best_match, dt_str)
                    if matched_elo is not None:
                        return float(matched_elo)

            if allow_network and not ClubEloLoader._network_down and dt_str not in ClubEloLoader._failed_dates:
                synced = self.sync_date(dt_str)
                if synced > 0:
                    return self.find_team_elo(team_name, dt_str, allow_network=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error consultando ClubElo para %s (%s): %s", team_name, dt_str, exc)

        return None

    def get_match_elos(
        self,
        home_team: str,
        away_team: str,
        match_date: str | datetime,
        home_advantage: float = 65.0,
    ) -> dict[str, Any]:
        """Obtiene los Elos de local y visitante y calcula probabilidades base."""
        home_elo = self.get_team_elo(home_team, match_date)
        away_elo = self.get_team_elo(away_team, match_date)
        elo_diff = home_elo - away_elo

        probs = self.calculate_probabilities(home_elo, away_elo, home_advantage=home_advantage)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": str(match_date)[:10],
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_diff": round(elo_diff, 1),
            "prob_home": round(probs["home_win"], 4),
            "prob_draw": round(probs["draw"], 4),
            "prob_away": round(probs["away_win"], 4),
        }

    @staticmethod
    def calculate_probabilities(
        home_elo: float,
        away_elo: float,
        home_advantage: float = 65.0,
    ) -> dict[str, float]:
        """Calcula probabilidades 1X2 a partir del rating Elo y ventaja de localía.

        Usa el modelo logístico estándar de Elo:
        E_home = 1 / (1 + 10 ** (-(home_elo + home_adv - away_elo) / 400))
        Con estimación paramétrica empírica de empates para fútbol de élite.
        """
        effective_diff = (home_elo + home_advantage) - away_elo
        expected_home_score = 1.0 / (1.0 + 10.0 ** (-effective_diff / 400.0))

        # Modelado empírico de empate en fútbol: máximo ~28% cuando los equipos son parejos
        # y decae cuando la diferencia es muy grande.
        prob_draw = 0.28 * max(0.4, 1.0 - (abs(effective_diff) / 600.0) ** 2)

        # Repartir el remanente en función del expected score
        prob_home = (1.0 - prob_draw) * expected_home_score
        prob_away = (1.0 - prob_draw) * (1.0 - expected_home_score)

        # Normalización estricta sum(p) == 1.0
        total = prob_home + prob_draw + prob_away
        return {
            "home_win": prob_home / total,
            "draw": prob_draw / total,
            "away_win": prob_away / total,
        }
