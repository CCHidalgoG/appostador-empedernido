-- Tablas persistidas en esta base
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
ORDER BY table_schema, table_name;

-- Columnas de las tablas del proyecto
SELECT table_schema, table_name, ordinal_position, column_name, data_type
FROM information_schema.columns
WHERE table_schema IN ('raw', 'processed', 'output')
ORDER BY table_schema, table_name, ordinal_position;

-- Conteos por tabla
SELECT 'raw.artifacts' AS tabla, count(*) AS filas FROM raw.artifacts
UNION ALL SELECT 'processed.artifacts', count(*) FROM processed.artifacts
UNION ALL SELECT 'output.artifacts', count(*) FROM output.artifacts
UNION ALL SELECT 'output.runs', count(*) FROM output.runs
UNION ALL SELECT 'output.recommendations', count(*) FROM output.recommendations
UNION ALL SELECT 'output.results', count(*) FROM output.results
ORDER BY tabla;
