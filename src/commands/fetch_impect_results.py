"""Importa marcadores de Impect y reconcilia recomendaciones pendientes."""
from __future__ import annotations

import argparse

from src.apostador.impect_results_fetcher import ITERATIONS, fetch_and_reconcile_impect_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", nargs="+", type=int, default=None,
                        help="IDs de iteración Impect; por defecto usa las ligas configuradas")
    args = parser.parse_args()
    summary = fetch_and_reconcile_impect_results(tuple(args.iterations or ITERATIONS))
    print("\n=== Importación de resultados Impect ===")
    print(f"Iteraciones consultadas: {len(summary['iterations_fetched'])}")
    print(f"Resultados guardados en output.impect_results: {summary['api_results_saved']}")
    print(f"Recomendaciones actualizadas: {summary['recommendations_updated']}")
    print(f"Sin emparejar: {summary['recommendations_unmatched']}")
    print(f"Iteraciones con error: {len(summary['iterations_failed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
