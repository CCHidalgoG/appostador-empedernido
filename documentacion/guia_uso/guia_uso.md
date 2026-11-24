# Guía de uso — pipeline y base DuckDB

## Instalar y migrar datos existentes

```bash
pip install -r requirements.txt
```

La base `data/apostador.duckdb` ya contiene los JSON, CSV y Markdown que estaban bajo `data/raw/`, `data/processed/` y `outputs/`; tras comprobar la migración, se eliminaron esas carpetas. Las siguientes ejecuciones leen y escriben directamente en DuckDB.

## Ejecutar predicciones

```bash
# Portafolio diario (defaults vigentes: multi_market, flat 3%, --jev-weight 0.20, exposición <=15%, max 3/liga)
python -m src.commands.main --date 2026-10-01 --days 5 --bankroll 100000 --kelly-factor 0.10 \
  --strategy-mode multi_market --jev-weight 0.20 --max-daily-exposure-pct 0.15 --max-bets-per-league 3

# Black Swan v2 (parlay ajustado con noticias/JEV):
# --input acepta la fecha del portfolio generado por main.py
# --target-odds 700 sólo es alcanzable con muchos partidos de cuota alta;
# para semanas típicas con 4-8 piernas EV+ usa --target-odds 30-100
python -m src.commands.black_swan_v2 --input 2026-10-03 --target-odds 30 --max-legs 6 --bankroll 40000

# Portafolio Full Kelly semanal (acepta argumentos CLI desde la última versión):
python -m src.commands.predict_upcoming_portfolio --start-date 2026-10-01 --days 5 --bankroll 100000 --kelly-factor 0.25
```

Las claves `BRAVEAPI`, `JEVAPI` y demás credenciales siguen en `.env`, fuera de DuckDB. Las salidas ya no se generan en `outputs/reports/`: quedan como filas y artifacts dentro de DuckDB.

## Consultar resultados

```bash
python -m src.commands.query_db
python -m src.commands.query_db "SELECT run_type, run_date, started_at FROM output.runs ORDER BY started_at DESC LIMIT 10"
```

```sql
SELECT event_date, home_team, away_team, outcome, probability, odds, ev, stake
FROM output.recommendations
ORDER BY event_date, ev DESC;

SELECT artifact_key, updated_at
FROM output.artifacts
ORDER BY updated_at DESC;
```

Para esquema completo y recomendaciones de concurrencia, consultar [duckdb.md](duckdb.md).


## Gate de noticias, JEV y controles de riesgo

Brave News busca lesiones/bajas, sanciones, formación/convocatoria, rotación, cambios de técnico, carga de calendario y problemas de plantilla. Cada evidencia conserva URL y, cuando están disponibles, dominio y fecha/edad del artículo; la caché vence a las seis horas.

- **Compuerta de cobertura y cuotas (`main.py`)**: En el portafolio diario, Brave News y JEV solo se ejecutan cuando el partido tiene cobertura completa (`data_coverage_ok`, i.e. ratings Impect y `form_n >= 3`) y cuotas válidas (`odds_details is not None`), evitando consumo innecesario en partidos no apostables.
- **Inyección de contexto y suavizado**: JEV recibe el ensamble Core, métricas tácticas, lesiones/once de API-Football y forma ponderada por recencia exponencial y $\Delta GD$. `smooth_jev_probabilities` suaviza hacia el prior informativo `P_base` (no hacia $1/3$). Si `noticias_relevantes >= 0.50` con evidencia válida, aplica `--jev-weight` (`0.20` por defecto vía `JEV_CALIBRATED_WEIGHT`); sin noticias disruptivas con cobertura OK aplica `0.10` (`JEV_BASE_WEIGHT`).
- **Veto activo JEV y sanidad de cartera**: Si `fused["vetada"] = True` (`riesgo_favorito > 0.85` y `confidence > 0.80`), se excluye el signo favorito base (`favorito_base`) antes de optimizar. `KellyOptimizer` utiliza remoción de margen por Power Method y limita la exposición a `--max-bets-per-league 3` y `--max-daily-exposure-pct 0.15` (reescalando proporcionalmente los stakes diarios).

## Notas operativas

- **Black Swan — cuota objetivo**: `--target-odds 700` requiere muchas piernas con cuotas individuales altas. Con jornadas típicas de 4-8 piernas EV+, el rango práctico es **30–150**. Si el comando termina en "Sin combinación sobre la cuota mínima", baja `--target-odds`.
- **predict_upcoming_portfolio** acepta `--start-date`, `--days`, `--bankroll`, `--kelly-factor` por CLI. Sin argumentos usa los próximos 5 días desde hoy.
- Las salidas de todos los comandos se guardan en `output.artifacts` y `output.recommendations`; no se crean archivos en disco.
