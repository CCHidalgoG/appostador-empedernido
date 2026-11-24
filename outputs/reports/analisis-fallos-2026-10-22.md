# Análisis de fallos — Apuestas del 2026-10-22

## 1. Fill de resultados (ejecutado)

- `python -m src.commands.fetch_results` (API-Football): 2 ganadas / 7 perdidas.
- `python -m src.commands.fetch_impect_results` (Impect): 43 recomendaciones adicionales conciliadas.
- Run `daily_portfolio:2026-10-22` → **PERDIDO (1/6 piernas)**. Run 2026-10-21 → PERDIDO (2/4 conciliadas).

| # | Partido | Apuesta | P modelo | Cuota | EV | Marcador | Resultado |
|---|---------|---------|----------|-------|-----|----------|-----------|
| 1 | FC Iberia 1999 vs Mjallby AIF | Home Win | 0.649 | 3.13 | +1.03 | 1-1 | ❌ Empate |
| 2 | Pafos vs Riga | Home Win | 0.661 | 1.62 | +0.07 | 1-1 | ❌ Empate |
| 3 | Beşiktaş vs Crystal Palace | Home Win | 0.627 | 2.26 | +0.42 | 2-2 | ❌ Empate |
| 4 | FC Midtjylland vs St. Truiden | Home Win | 0.661 | 1.67 | +0.10 | 0-1 | ❌ Visita |
| 5 | Monaco vs SC Freiburg | Home Win | 0.644 | 2.14 | +0.38 | 3-0 | ✅ Local |
| 6 | FC Thun vs Hajduk Split | Home Win | 0.640 | 3.01 | +0.93 | 0-1 | ❌ Visita |

**P&L del día**: 6 × 4.140 = 24.840 apostado; retorno 4.140 × 2.14 = 8.859,6 → **neto −15.980 (−11,6% del bankroll de 138.000)**.
Distribución real: 1 local (17%), 3 empates (50%), 2 visitas (33%) — frente a ~65% de probabilidad asignada a todos los locales.

## 2. Causa raíz (cadena completa del fallo)

1. **Ligas sin ratings Impect**: UEFA Europa League (id 3) y Conference League (id 848) están en `SUPPORTED_LEAGUES` con `impect_iteration: None` (`src/commands/main.py`). Sin ratings, `SoccerPredictor` usa `default_rating = 0.68` para **ambos** equipos (`src/apostador/predictor.py:140`) → `P_base` idéntica para los 6 partidos: local 0.448 / empate 0.284 / visita 0.268 (solo la ventaja de localía, cero información de equipos). Tampoco hay forma (`form_n` = 0-2).
2. **Sobreconfianza extrema de JEV**: con ese vacío de datos, JEV devolvió `P_jev` raw ≈ 0.97 al local en todos los casos (tras smoothing: 0.81–0.88). Dos partidos distintos (Pafos-Riga y Midtjylland-St. Truiden) recibieron **exactamente la misma** distribución (0.875/0.075/0.05) — señal de respuesta plantillada sin información real.
3. **Peso del tier táctico mal configurado**: `JEV_BASE_WEIGHT = 0.50` en `src/apostador/probability_fusion.py:8`, pero el diseño documentado (docstring de `fuse_probabilities` y `.agents/TASKS/jev-full-context-always-on.md`) establece **w = 0.10** para el fallback táctico sin noticias. Con w = 0.50: `P_adj = 0.5·0.448 + 0.5·0.87 ≈ 0.65` para TODOS los locales.
4. **Falsos EV y nichos disparados**: con P ≈ 0.65 fija, los filtros de nicho (`valor_local_medio`: Home Win, cuota 2.00–4.50, P≥60%, EV≥3% en `src/apostador/optimizer.py`) se activaron con EV absurdos (Iberia: EV = 0.649×3.13−1 = +103%). Un edge de ese tamaño contra el consenso de 15–20 bookmakers (probabilidad implícita 32.9%) es un artefacto del modelo, no valor.
5. **Exposición correlacionada**: 6 apuestas del mismo mercado (local), mismo tipo de competición (copas europeas de jueves), mismo sesgo sistemático y 18% del bankroll en flat stakes de 3% — efectivamente un parlay correlacionado.

## 3. Evidencia histórica (325 piernas conciliadas)

**Calibración por bucket (prob. predicha vs. tasa real):**

| Bucket | n | Ganadas | Tasa real | Prob. media predicha |
|--------|---|---------|-----------|----------------------|
| p ≥ 0.60 | 78 | 38 | 48,7% | 74,3% ⚠️ sobreconfianza |
| 0.50–0.60 | 46 | 25 | 54,3% | 54,9% ✅ |
| 0.40–0.50 | 39 | 11 | 28,2% | 45,4% ⚠️ |
| < 0.40 | 162 | 44 | 27,2% | 28,0% ✅ |

El bucket p ≥ 0.60 — exactamente el que seleccionan los nichos — es el peor calibrado.

**Rendimiento de nichos en vivo (post-backtest):** `valor_local_medio` 1/4 (PnL −1.925), `favorito_alta_certeza` 0/2 (−4.140), `ev_positivo` 1/3 (−417). El ROI +31% del README proviene del backtest, no se sostiene fuera de muestra en la data en vivo.

**Deuda de datos**: 48 piernas siguen sin conciliar (fallos de matching del `TeamAligner`, p.ej. "Independiente Santa Fe vs Jaguares de Córdoba"); toda métrica de ROI está sesgada por supervivencia.

## 4. Recomendaciones de optimización (priorizadas)

1. **Guard de cobertura de datos (crítico)**: excluir (o marcar como no-apostable) partidos donde ambos ratings caen al default 0.68 y/o `form_n < 3`, y ligas con `impect_iteration: None`. Hoy las 6 apuestas violaban esto.
2. **Corregir `JEV_BASE_WEIGHT` 0.50 → 0.10** (alinear código con diseño) y llevar el peso a 0 cuando JEV no tiene señales estructuradas ni noticias relevantes. El bucket p≥0.60 gana 48,7% real, no 74%.
3. **Filtro de divergencia vs. mercado**: veto o cap cuando `P_modelo − P_implícita_consensus > 15–20 pp`. Hoy: Iberia +32 pp, Thun +30 pp, Beşiktaş +18 pp, Monaco +18 pp.
4. **Tope de EV plausible**: descartar edges > 30–40% (son artefactos; el edge real sostenible en fútbol es < 10%).
5. **Límites de exposición correlacionada**: máximo 2–3 apuestas simultáneas por competición/día y tope diario de bankroll (p.ej. 10%) en vez de 6 × 3% = 18% en el mismo sesgo.
6. **Reempate del modelo de empates**: las copas europeas de fase de grupos producen muchos empates (3/6 hoy); la P_D fija de ~0.18 del fallback no la refleja. Usar las probabilidades implícitas del consenso como ancla cuando no hay ratings.
7. **Reparar la conciliación**: mejorar el `TeamAligner`/crosswalk para cerrar las 48 piernas sin resultado y poder medir el ROI real antes de ajustar parámetros.

**Conclusión**: hoy no falló la suerte; falló un camino de datos degradado (copas europeas sin ratings ni forma) amplificado por un peso JEV mal configurado (0.50 en lugar de 0.10) y sin contrapeso del mercado. Con los guards 1–4, las 6 apuestas de hoy habrían sido 0 apuestas.

## 5. Estado de implementación de los fixes (2026-10-22)

Fixes 1–4 implementados y validados (21/21 tests en verde):

1. **Guard de cobertura de datos** (`src/commands/main.py`): constantes `MIN_FORM_MATCHES_FOR_BET = 3`, flags `data_coverage_ok` / `data_coverage_flags` por partido (`liga_sin_ratings_impect`, `equipos_sin_rating_impect`, `forma_insuficiente`); los partidos sin cobertura quedan fuera de `value_bets` y desactivan el tier táctico de JEV (`always_on=data_coverage_ok`). Las `value_bets` ahora llevan `market_implied_prob` (implícita sin margen, consenso preferente) calculada con `implied_probabilities_no_vig()`.
2. **`JEV_BASE_WEIGHT` 0.50 → 0.10** (`src/apostador/probability_fusion.py`), alineado con el diseño documentado.
3. **Filtro de divergencia vs. mercado** (`src/apostador/optimizer.py`): veto si `p − p_mercado > max_divergence` (default 0.15). Auditado en el payload como `market_divergence`.
4. **Tope de EV plausible** (`src/apostador/optimizer.py`): veto si `ev > max_ev` (default 0.35).

**Simulación con los datos reales del día**: 25 partidos con cuotas → 24 descartados por el guard de cobertura (jornada completa de copas europeas); 1 candidato con cobertura real (Atlético Nacional vs Pasto, ratings + forma 5/5) fue filtrado por nicho → **0 apuestas finales** (vs. 6 apuestas y −15.980 en la realidad).

Nota: ligas sin ratings Impect (copas UEFA, Eredivisie, Primeira Liga, Championship, Saudi Pro League) no generarán apuestas hasta que tengan ratings/forma; es intencional y configurable (`MIN_FORM_MATCHES_FOR_BET`, `max_ev`, `max_divergence` en `KellyOptimizer`). Pendientes: límites de exposición correlacionada (máx. apuestas por competición/día) y reparación del aligner de resultados (48 piernas sin conciliar).

## 6. Validación de cobertura Impect para ligas europeas (2026-10-22)

Consulta a `impectPy.getIterations()` (103 iteraciones disponibles): **Impect NO ofrece datos de ninguna liga ni copa europea**. Su catálogo cubre únicamente: Argentina (Primera/Nacional), Bélgica, Bolivia, Brasil (Série A/B), Chile, Colombia, Ecuador, España (LaLiga), Italia (Serie A), México, Paraguay, Perú, Rusia, USA (MLS), Uruguay y Venezuela.

**No existen iteraciones para**: UEFA Champions/Europa/Conference League, Premier League, Bundesliga, Ligue 1, Eredivisie, Primeira Liga, Championship ni Saudi Pro League. Por tanto las competiciones europeas quedan permanentemente fuera del universo apostable vía guard de cobertura (no es un problema temporal).

**Corrida normal post-fix (2026-10-23, `python -m src.commands.main --date 2026-10-23`)**:
- 464 fixtures → 432 de ligas no soportadas, 10 sin alineación Impect, 20 procesados.
- Guard de cobertura vetó 10 partidos con cuotas: 7 por `liga_sin_ratings_impect` (Saudi×3, Eredivisie, Bundesliga, Premier League, Primeira Liga/Championship) y 10 por `forma_insuficiente` (incluye Bélgica, LaLiga, Chile, Série B y Argentina parcial).
- 5 value bets candidatos (EV>0) → **0 apuestas finales**, cada rechazo trazado:
  - FC Turin vs Monza (p=0.672, cuota 2.15): vetado por tope de EV (+44.5% > 35%) y divergencia vs. mercado (0.223 > 0.15) — el modelo sobreestimaba al local frente a un consenso de 44.9%.
  - Boyacá Chicó vs Bucaramanga (p=0.431, cuota 3.75): vetado por EV +61.8% (artefacto) y divergencia 0.17.
  - Gimnasia vs Sarmiento (Draw), Boyacá Chicó (Draw), Tigre vs Tucumán (Home Win, p=0.478 < 0.60): no califican en nichos.
- **Conclusión: el pipeline corre normalmente y hoy no existe señal de calidad apostable; 0 apuestas es el resultado correcto** (viernes con calendario dominado por ligas europeas sin datos Impect). El volumen llegará con las jornadas de fin de semana de las ligas americanas/LaLiga/Serie A.


