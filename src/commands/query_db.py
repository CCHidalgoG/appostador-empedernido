"""Interactive SQL query helper for the project's DuckDB file."""
import argparse
from src.apostador.config import DUCKDB_PATH

def main():
    parser = argparse.ArgumentParser(description="Consulta la base local DuckDB")
    parser.add_argument("sql", nargs="?", help="Consulta SQL; sin argumento abre el prompt interactivo")
    args = parser.parse_args()
    import duckdb
    con = duckdb.connect(str(DUCKDB_PATH), read_only=bool(args.sql))
    try:
        if args.sql:
            con.sql(args.sql).show()
        else:
            print(f"DuckDB: {DUCKDB_PATH} (escribe exit para salir)")
            while True:
                try:
                    sql = input("duckdb> ").strip()
                    if sql.lower() in {"exit", "quit"}: break
                    if sql: con.sql(sql).show()
                except Exception as exc:
                    print(f"Error: {exc}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
