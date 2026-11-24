"""Descarga ratings de Impect y los guarda en la capa raw de DuckDB."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.apostador.config import IMPECT_RAW_DIR
from src.apostador.duckdb_store import write_json

DEFAULT_ITERATIONS = (1801, 1863, 1813, 1794, 1973, 1800)


def _stamp_as_of(records: list[dict], snap_date: str) -> list[dict]:
    """Asegura marca temporal as_of_date en cada registro para backtesting point-in-time."""
    stamped: list[dict] = []
    for row in records:
        item = dict(row)
        item.setdefault("as_of_date", snap_date)
        stamped.append(item)
    return stamped


def fetch_ratings(iterations: list[int], as_of_date: str | None = None) -> int:
    """Descarga las iteraciones pedidas; devuelve el número de fallos."""
    load_dotenv()
    username = os.getenv("USERNAME_IMPECT")
    password = os.getenv("PASSWORD_IMPECT")
    if not username or not password:
        raise RuntimeError("Faltan USERNAME_IMPECT o PASSWORD_IMPECT en el entorno (.env)")

    try:
        import impectPy as ip
    except ImportError as exc:
        raise RuntimeError("impectPy no está instalado. Ejecuta: pip install -r requirements.txt") from exc

    snap_date = (as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10]
    token = ip.getAccessToken(username=username, password=password)
    failures = 0
    for iteration in iterations:
        try:
            ratings = ip.getSquadRatings(iteration=iteration, token=token)
            records = _stamp_as_of(ratings.to_dict(orient="records"), snap_date)
            write_json(IMPECT_RAW_DIR / f"ratings_{iteration}.json", records)
            write_json(IMPECT_RAW_DIR / f"ratings_{iteration}_{snap_date}.json", records)
            print(f"{iteration}: {len(records)} ratings guardados en DuckDB (as_of={snap_date})")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{iteration}: error descargando ratings: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", nargs="*", type=int, default=DEFAULT_ITERATIONS,
                        help="IDs de iteración Impect (por defecto: iteraciones configuradas)")
    args = parser.parse_args()
    try:
        failures = fetch_ratings(args.iterations)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"Proceso terminado: {failures} iteraciones con error")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
