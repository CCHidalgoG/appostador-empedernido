"""CLI: actualiza en DuckDB los resultados reales de los partidos para las apuestas guardadas.

Usa API-Football (ya integrada en el proyecto) porque expone marcador final y estado
del partido; Brave Search no tiene un endpoint de resultados deportivos.
"""
import argparse

from src.apostador.results_fetcher import resolve_pending_results


def main():
    parser = argparse.ArgumentParser(
        description="Verifica resultados reales de partidos y actualiza las apuestas guardadas en DuckDB")
    parser.add_argument("--date", action="append", dest="dates",
                         help="Fecha YYYY-MM-DD a verificar; puede repetirse. Por defecto revisa todas las pendientes")
    args = parser.parse_args()

    summary = resolve_pending_results(dates=args.dates)

    print("\n=== Resumen de resultados ===")
    print(f"Ganadas: {summary['won']}  Perdidas: {summary['lost']}  "
          f"Sin terminar: {summary['pending']}  Sin emparejar: {summary['unmatched']}")

    if summary["runs"]:
        print("\n=== Resultado por parlay/run ===")
        for run_id, legs in summary["runs"].items():
            won = all(legs)
            print(f"{run_id}: {'GANADO' if won else 'PERDIDO'} ({sum(1 for l in legs if l)}/{len(legs)} legs ganadas)")


if __name__ == "__main__":
    main()
