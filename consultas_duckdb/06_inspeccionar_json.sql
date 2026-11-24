-- Inspeccionar un payload de artefacto por su clave (ajusta la clave)
SELECT artifact_key, content_type, json_pretty(payload) AS contenido
FROM raw.artifacts
WHERE artifact_key = 'apifootball/fixtures_YYYY-MM-DD.json';

-- Buscar cualquier artifact cuyo nombre contenga una palabra
SELECT artifact_key, json_pretty(payload) AS contenido
FROM processed.artifacts
WHERE artifact_key ILIKE '%portfolio%'
LIMIT 10;

-- Explorar resumen JSON de las corridas recientes
SELECT run_id, json_pretty(summary) AS resumen
FROM output.runs
ORDER BY started_at DESC
LIMIT 20;

-- Ver el payload original de una recomendación concreta
SELECT run_id, row_number, json_pretty(payload) AS recomendacion
FROM output.recommendations
ORDER BY run_id DESC, row_number
LIMIT 50;
