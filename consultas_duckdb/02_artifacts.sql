-- Inventario de artefactos por capa
SELECT 'raw' AS capa, artifact_key, content_type, updated_at FROM raw.artifacts
UNION ALL
SELECT 'processed', artifact_key, content_type, updated_at FROM processed.artifacts
UNION ALL
SELECT 'output', artifact_key, content_type, updated_at FROM output.artifacts
ORDER BY capa, updated_at DESC;

-- Cambia el prefijo para buscar archivos relacionados, por ejemplo '%fixture%'
SELECT artifact_key, content_type, updated_at
FROM raw.artifacts
WHERE artifact_key ILIKE '%fixture%'
ORDER BY updated_at DESC;
