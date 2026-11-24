# Arquitectura — Parlays con Brave + JEV y persistencia DuckDB

## Flujo

```text
API-Football / Impect / The Odds API / Brave
                    │
                    ▼
             raw.artifacts
                    │ normalización y cálculos
                    ▼
     processed.artifacts
                    │ predicción y optimización
                    ▼
 output.runs + output.recommendations + output.artifacts
```

La base persistente es `data/apostador.duckdb`. `raw` conserva payloads y cachés; `processed` conserva datasets y resultados intermedios; `output.runs` guarda metadatos de cada corrida, `output.recommendations` conserva cada selección una sola vez como fila consultable y `output.artifacts` guarda reportes y evaluaciones. Los detalles variables de cada selección se preservan en su payload JSON.

## Pipeline analítico

`P_base (Ensamble multi-capa v2 + forma ponderada + bajas/XI + Power no-vig) → guard (cobertura ∧ cuotas) → Brave News + contexto API-Football → gate de relevancia (≥0.50) → JEV + suavizado hacia P_base → fusión (20% JEV prioritario / 10% base táctica) → veto activo JEV → selección de valor y cartera acotada (≤3/liga, ≤15% diario)`.

- `TeamAligner` y `RecentFormLoader` alinean equipos de forma unificada (`output.team_crosswalk` + respaldo difuso) y calculan `form_delta` combinando puntos ponderados por recencia ($\exp(-0.04 \cdot \text{días})$, 70%) y diferencial de goles (30%) sobre los últimos 5 partidos (`form_n >= 3`).
- `SoccerPredictor` integra el ensamble `v2_tactical_xg_ensemble` con control temporal `as-of` sobre `averages`/`scores`/`player_scores` (>14 días), ajuste dinámico de talento por bajas e XI confirmado (`get_available_player_talent` y `missing_share_impact`) y contracción bayesiana al consenso mediante el método Power (`_power_novig_probs`).
- Brave News y JEV solo se consultan si el partido cumple simultáneamente cobertura de datos (`data_coverage_ok`) y disponibilidad de cuotas (`odds_details is not None`), optimizando latencia y cuota de APIs.
- Brave News busca evidencia sobre bajas/lesiones, suspensiones, convocatoria/formación, rotación, técnico, carga de calendario y problemas de plantilla; localiza país/idioma y mantiene URL, dominio, edad/fecha, certeza léxica y alertas de contradicción si están disponibles.
- La caché de noticias vence a las seis horas. El filtro exige URL, mención no genérica de un equipo y un término de categoría.
- La política `src/apostador/news_source_policy.json` permite revisar dominios por país y liga; todo dominio sin curar queda explícitamente como `no_revisado` y no se excluye.
- API-Football aporta reportes de lesiones por fecha y, solo dentro de 55 minutos antes/10 minutos después del inicio, formación, entrenador y once oficial si la competición los cubre; ambas señales alimentan ahora tanto al modelo `SoccerPredictor` (`get_available_player_talent`) como a JEV.
- El suavizado `smooth_jev_probabilities` contrae las probabilidades de JEV hacia el prior informativo `P_base` (evitando desplazamiento artificial hacia el uniforme 1/3).
- El gate se activa si Brave detecta noticias relevantes o si API-Football entrega lesiones/alineación para el partido. Con contexto y una respuesta JEV válida cuya relevancia sea ≥0.50 se usa `P_adj = 0.80·P_base + 0.20·P_jev` (`JEV_CALIBRATED_WEIGHT = 0.20` por defecto en `--jev-weight`); sin noticias disruptivas pero con cobertura OK aplica la base táctica `w = 0.10` (`JEV_BASE_WEIGHT`).
- Si JEV marca `vetada = True` (`riesgo_favorito > 0.85` y `confidence > 0.80`), `main.py` excluye el signo favorito base (`favorito_base`) de `value_bets`.
- `KellyOptimizer` aplica topes de concentración (`max_bets_per_league = 3`) y de exposición diaria agregada (`max_daily_exposure_pct = 0.15`), reescalando proporcionalmente los stakes (`exposure_scaled = True`) antes de persistir recomendaciones en DuckDB.
- Las predicciones nuevas guardan snapshots de forma, noticias, contexto del proveedor, `P_ensemble`, impacto de bajas (`home_missing_impact`, `away_missing_impact`), veto JEV, probabilidades core/JEV y parámetros del modelo.
- Cada ejecución obtiene un `run_id` único; las filas de `output.recommendations` conservan el historial aunque los artifacts de entrada/resultado mantengan su ruta lógica más reciente.

## Migración

La migración inicial importó los `.json`, `.csv` y `.md` de `data/raw`, `data/processed` y `outputs`. Tras verificarla, se eliminaron esas carpetas legacy. Los comandos usan ahora DuckDB como persistencia activa. `.env` queda fuera de la base y se excluye de Git.

## Estado actual del motor de decisión

El modelo core vigente en producción es el ensamble multi-capa `v2_tactical_xg_ensemble` (`SoccerPredictor`: Poisson Dixon-Coles sobre `predModelHome`/`predModelAway` ajustado por `missing_share_impact` + score táctico multi-KPI y talento disponible con control `as-of` + Softmax MLE con `form_delta` calibrado + contracción bayesiana del 20% vía Power Method al consenso sin margen), operando con fusión dual-tier JEV (`JEV_BASE_WEIGHT = 0.10`, `JEV_CALIBRATED_WEIGHT = 0.20` por defecto en `--jev-weight`, suavizado hacia `P_base`), veto activo `fused["vetada"]`, compuerta pre-API (`data_coverage_ok` + cuotas) y estrategia de portafolio predeterminada `multi_market` en `KellyOptimizer` con topes de liga (`3`) y exposición diaria (`15%`).

Ver el análisis crítico del flujo, brechas auditadas ya corregidas, mejoras implementadas y catálogo de futuras fuentes web en [flujo_decision_y_nuevos_datos.md](flujo_decision_y_nuevos_datos.md). Ver [guía DuckDB](duckdb.md) para instalación, tablas y consultas. Los contratos Brave/JEV están en sus [guías](../bravesearch/Documentation.md) y [Jev](../jev/README.md).
