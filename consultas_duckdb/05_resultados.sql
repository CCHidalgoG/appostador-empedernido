-- Recomendaciones y marcador, incluidas las que siguen pendientes
SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team,
       r.outcome, r.odds, res.status, res.home_goals, res.away_goals,
       res.bet_won, res.checked_at
FROM output.recommendations r
LEFT JOIN output.results res
  ON res.run_id = r.run_id AND res.row_number = r.row_number
ORDER BY r.event_date DESC, r.run_id DESC, r.row_number;

-- Tasa de acierto entre apuestas resueltas
SELECT count(*) AS resueltas,
       count(*) FILTER (WHERE bet_won IS TRUE) AS ganadas,
       count(*) FILTER (WHERE bet_won IS FALSE) AS perdidas,
       round(100.0 * count(*) FILTER (WHERE bet_won IS TRUE) / nullif(count(*), 0), 1) AS acierto_pct
FROM output.results
WHERE status IN ('FT', 'AET', 'PEN') AND bet_won IS NOT NULL;

-- Pendientes: sin resultado final registrado
SELECT r.run_id, r.row_number, r.event_date, r.home_team, r.away_team, r.outcome
FROM output.recommendations r
LEFT JOIN output.results res
  ON res.run_id = r.run_id AND res.row_number = r.row_number
WHERE res.run_id IS NULL OR res.status NOT IN ('FT', 'AET', 'PEN')
ORDER BY r.event_date, r.run_id, r.row_number;
