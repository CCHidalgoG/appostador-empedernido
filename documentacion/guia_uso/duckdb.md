# Persistencia local en DuckDB

## Instalación y ubicación

Instala la dependencia Python desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```

La base se crea automáticamente en `data/apostador.duckdb` cuando se usa por primera vez el almacenamiento (`DuckDBStore`). No se configura ninguna clave o secreto dentro de la base: las APIs siguen usando las variables de `.env`.

## Capas y tablas

| Capa | Tabla | Contenido |
|---|---|---|
| Raw | `raw.artifacts` | Payload JSON/cache original, identificado por su fuente y ruta lógica |
| Processed | `processed.artifacts` | Datasets, configuraciones calibradas y documentos intermedios |
| Output | `output.runs` | Tipo, fecha, timestamp y resumen JSON de cada ejecución |
| Output | `output.recommendations` | Una selección por fila, enlazada por `run_id`; columnas para partido, resultado, probabilidad, cuota, EV y stake |
| Output | `output.artifacts` | Reportes, evaluaciones, backtests y otros documentos de salida |

Los artifacts conservan entradas y documentos completos. `output.runs.summary` guarda solo metadatos y configuración de la corrida; sus recomendaciones se guardan una vez en `output.recommendations`, con columnas consultables y el payload específico de cada selección.

## Migración inicial

Al abrir la base, el importador detecta archivos `.json`, `.csv` y `.md` bajo `data/raw/`, `data/processed/` y `outputs/`, y los inserta en la tabla de artifacts de su capa. La clave conserva la ruta relativa. La importación es idempotente por clave y convierte los portfolios históricos en filas de `output.runs` y `output.recommendations`. En la migración inicial se importaron los archivos existentes y, tras verificar la carga, se eliminaron las carpetas legacy. El proyecto ya no necesita esos archivos para operar.

## Uso y consultas

Ejecuta el prompt interactivo:

```bash
python -m src.commands.query_db
```

O pasa una consulta SQL:

```bash
python -m src.commands.query_db "SELECT run_type, run_date FROM output.runs ORDER BY started_at DESC"

# Actualizar ratings Impect (opcionalmente indicar IDs de iteración, sin corchetes)
python -m src.commands.fetch_impect 1801 1863 1813 1794 1973 1800
```

Consultas útiles:

```sql
-- Últimas ejecuciones
SELECT run_id, run_type, run_date, started_at
FROM output.runs ORDER BY started_at DESC;

-- Recomendaciones con EV positivo
SELECT event_date, home_team, away_team, outcome, probability, odds, ev, stake
FROM output.recommendations WHERE ev > 0 ORDER BY event_date, ev DESC;

-- Payload raw de una caché
SELECT payload FROM raw.artifacts
WHERE artifact_key = 'apifootball/fixtures_2026-09-27.json';
```

DuckDB permite varios lectores y escritores coordinados dentro del mismo proceso. Para esta base local, evita iniciar al mismo tiempo varios comandos Python que escriban el mismo archivo. Revisa la [guía oficial de concurrencia](https://duckdb.org/docs/stable/connect/concurrency.html) si cambia el modo de despliegue.

## Código

`src/apostador/duckdb_store.py` centraliza las operaciones de persistencia. `read_json` y `read_text` priorizan DuckDB y cargan a la base un archivo antiguo si no existe el artefacto. `write_json` y `write_text` escriben en la base; no vuelven a generar archivos de datos ni reportes. `write_json(..., run_type=..., run_date=...)` también crea una ejecución con identificador único. El resumen no repite las selecciones, que quedan en `output.recommendations`; volver a ejecutar el mismo día no sobrescribe el historial de corridas.
