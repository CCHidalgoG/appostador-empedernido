-- Recomendaciones completas más recientes
SELECT run_id, row_number, event_date, home_team, away_team, outcome,
       probability, odds, ev, stake
FROM output.recommendations
ORDER BY event_date DESC, run_id DESC, row_number
LIMIT 200;

-- Recomendaciones con EV positivo
SELECT event_date, home_team, away_team, outcome, probability, odds, ev, stake, run_id
FROM output.recommendations
WHERE ev > 0
ORDER BY event_date DESC, ev DESC;

-- Resumen por corrida
SELECT run_id, count(*) AS selecciones,
       round(avg(probability), 3) AS probabilidad_media,
       round(avg(odds), 2) AS cuota_media,
       round(avg(ev), 4) AS ev_medio,
       round(sum(stake), 2) AS stake_total
FROM output.recommendations
GROUP BY run_id
ORDER BY run_id DESC;
