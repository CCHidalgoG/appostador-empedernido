# Consultas para explorar DuckDB

Esta carpeta reúne consultas de solo lectura para inspeccionar la base local del proyecto. La base predeterminada está en `data/apostador.duckdb`.

## Cómo ejecutarlas

Desde la raíz del repositorio, abre el prompt interactivo:

```bash
python -m src.commands.query_db
```

Pega el contenido de cualquier archivo `.sql` en el prompt `duckdb>` y termina con `exit`. También puedes pasar una consulta corta directamente:

```bash
python -m src.commands.query_db "SELECT run_type, count(*) FROM output.runs GROUP BY 1"
```

La base se crea al ejecutar por primera vez el almacenamiento del proyecto. Si `duckdb` no está instalado, instala las dependencias con `pip install -r requirements.txt`.

## Archivos

- `01_catalogo.sql`: tablas y columnas disponibles, además de conteos rápidos.
- `02_artifacts.sql`: claves, tipos y fechas de los artefactos guardados.
- `03_corridas.sql`: corridas recientes y resumen.
- `04_recomendaciones.sql`: selecciones recomendadas, filtros EV y resumen por corrida.
- `05_resultados.sql`: recomendaciones junto a resultados y métricas de acierto.
- `06_inspeccionar_json.sql`: ejemplos para leer payloads JSON de artifacts y corridas.

Las tablas de artifacts guardan el contenido en una columna JSON (`payload`). Las tablas de recomendaciones exponen campos frecuentes como partido, probabilidad, cuota, EV y stake. Algunas tablas estarán vacías hasta que corran los procesos correspondientes.
