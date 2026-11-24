-- Corridas más recientes
SELECT run_id, run_type, run_date, started_at, summary
FROM output.runs
ORDER BY started_at DESC
LIMIT 100;

-- Volumen de corridas por tipo y fecha
SELECT run_type, run_date, count(*) AS corridas
FROM output.runs
GROUP BY run_type, run_date
ORDER BY run_date DESC, run_type;

-- Corridas con más recomendaciones
SELECT r.run_id, r.run_type, r.run_date, count(rec.row_number) AS recomendaciones
FROM output.runs r
LEFT JOIN output.recommendations rec USING (run_id)
GROUP BY r.run_id, r.run_type, r.run_date
ORDER BY r.run_date DESC, recomendaciones DESC;
