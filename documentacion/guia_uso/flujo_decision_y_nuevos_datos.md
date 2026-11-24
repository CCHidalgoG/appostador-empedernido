# Auditoría del Flujo de Decisión, Mejoras de Estimación y Catálogo de Nuevas Fuentes Web

Este documento analiza de principio a fin el flujo de toma de decisiones de **Apostador Empedernido** (`v2_tactical_xg_ensemble` + fusión `Brave`/`JEV` + `KellyOptimizer`), valida su solidez matemática y arquitectónica frente al código en producción, registra las correcciones y mejoras ya implementadas sobre los hallazgos de auditoría, y detalla nuevas fuentes de datos disponibles en la web junto con su diseño de integración en DuckDB y el pipeline.

---

## 1. Diagrama y Validación del Flujo de Decisión Vigente (End-to-End)

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 1: Ingesta y Calendario (src/commands/main.py)                                    │
│  API-Football /fixtures (America/Bogota) ──(si falla / plan)──► Fallback The Odds API   │
│  Fallback: fecha convertida UTC→Bogotá + _select_bookmaker (sharp) + consenso evento    │
│  Filtros: status ∈ {NS, TBD} ∧ match_dt > now_utc ∧ league_id ∈ SUPPORTED_LEAGUES (27)  │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 2: Alineación de Identidades y Forma (aligner.py / recent_form.py)                │
│  output.team_crosswalk (provider_team_id) ─► static_mappings ─► fuzzy (_has_conflicting)│
│  RecentFormLoader (con fallback TeamAligner th=0.75): ventana últimos 5 partidos        │
│    • 70% puntos ponderados exp(-0.04·días) + 30% ΔGD acotado [-3,3] ─► [-0.10, +0.10]   │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 3: Guard de Cobertura de Datos y Cuotas (src/commands/main.py L885–L905, L1031)   │
│  Exige: impect_iter válido ∧ ambos equipos en ratings_map ∧ form_n ≥ 3 en ambos lados   │
│  Si falla u odds_details is None ─► salta Brave/JEV y bloquea generación de apuestas    │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 4: Modelo Predictivo Core Multi-Capa (src/apostador/predictor.py L553–L723)       │
│  Control as-of (>14d antes de max_rating_date): atenúa averages/scores/player_scores    │
│  Talento disponible (get_available_player_talent): XI confirmado (≥7) o excluye bajas   │
│    ─► missing_share_impact ajusta λ_H/λ_A (1 - 0.12·impact) y tactical_score            │
│  1. p_ratings: Softmax MLE (α=4.322, γ_d y ha por liga, decay exp(-0.05*(d-7)) si d>7)  │
│  2. p_poisson: Dixon-Coles bivariado (ρ=-0.06) con predModelHome/Away ajustados         │
│  3. p_tactical: Score multi-KPI + ΔTalento disponible + ha + form_delta + Δmissing      │
│  4. p_ensemble: 50% Poisson + 35% Táctico + 15% Ratings (sin λ: 65% Táct. + 35% Rat.)   │
│  5. p_final: Contracción Bayesiana 20% (w_shrink=0.20) vía Power Method (_power_novig)  │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 5: Contexto Pre-Partido y Fusión con JEV (solo si cobertura OK ∧ hay cuotas)      │
│  API-Football: /injuries (diario) + /fixtures/lineups ([-10m, +55m]) ─► predictor + JEV │
│  Brave News: 2 queries bilingües ─► 8 categorías, certeza léxica, tier de dominio       │
│  JEV (SystemOne): choice_probs + noticias_relevantes + riesgo_favorito + confidence     │
│  smooth_jev_probabilities (λ=0.15 hacia prior base P_core) ─► fuse_probabilities:        │
│    • Tier táctico (always_on=data_coverage_ok, sin noticias): w = 0.10                  │
│    • Tier prioritario (evidencia ∧ noticias ≥ 0.50): w = JEV_CALIBRATED_WEIGHT (0.20)   │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 6: Detección de EV, Veto JEV y Cartera (src/apostador/optimizer.py / main.py)     │
│  Si fused["vetada"]=True ─► excluye el signo favorito base (favorito_base) de value_bets│
│  EV = (P_adj × Cuota) - 1 | P_mkt_novig vía Power Method (implied_probabilities_no_vig) │
│  Vetos sanidad: EV ∈ [0.03, 0.35] ∧ (P_adj - P_mkt_novig ≤ 0.15)                        │
│  Nichos (multi_market default): Local [3%,35%], Empate (EV≥4%, ≥2.80), Visita (EV≥4%, ≥2.10)│
│  Cartera: dedup 1X2 ─► orden desc. EV ─► max 3/liga ─► tope 15% diario (reescala stake) │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detalle Técnico por Etapa

1. **Etapa 1 — Ingesta y Calendario (`src/commands/main.py`)**:
   - Solicita fixtures a API-Football usando `timezone="America/Bogota"` (`load_or_fetch_fixtures`, L247–L273) para alinear las fechas locales de la CLI con los partidos nocturnos en UTC.
   - Si API-Football no devuelve partidos o reporta error de plan gratuito (`"plan" in fixtures_errors`, L762–L800), activa el fallback sobre `The Odds API`, convirtiendo `commence_time` de UTC a `America/Bogota` antes de filtrar `commence_date == target_date`.
   - Tanto `get_match_odds` como el modo fallback seleccionan la casa mediante `_select_bookmaker` (`src/commands/main.py`, L373–L387), priorizando `_SHARP_BOOKMAKER_PRIORITY` (`pinnacle`, `betfair`, `matchbook`, `marathon`, `bet365`, `1xbet`, `william hill`) sobre el orden arbitrario de la respuesta, y calculan `consensus_details` vía `_extract_event_consensus` (L407–L435).
   - Filtra partidos en estado previo al arranque (`status in ["NS", "TBD"]`, `match_dt > now_utc`) y restringe el universo a `SUPPORTED_LEAGUES`: **27 ligas configuradas**, de las cuales **17** poseen `impect_iteration` definido (principalmente América, LaLiga, Serie A, Bélgica y Rusia) y **10** tienen `impect_iteration: None`.

2. **Etapa 2 — Alineación de Identidades y Forma Reciente (`src/apostador/aligner.py`, `src/apostador/recent_form.py`)**:
   - `resolve_team` (`src/commands/main.py`, L845–L865) consulta primero identidades persistidas en `output.team_crosswalk` usando `provider_team_id`. Si no existe mapeo, invoca `TeamAligner.find_best_match` (`src/apostador/aligner.py`, L219–L298), persistiendo matches con score $\ge 0.90$.
   - `RecentFormLoader.get_form` (`src/apostador/recent_form.py`, L175–L281) incorpora `self.aligner = TeamAligner()` como respaldo difuso (`threshold=0.75`) cuando el nombre normalizado no coincide exactamente en `_history` (cacheando en `self._name_to_id`), y calcula `form_weighted_points_avg` (ponderación exponencial por recencia $\exp(-0.04 \cdot \text{days\_ago})$) junto con `form_goal_diff_avg`.
   - `RecentFormLoader.get_form_adjustment` (L302–L331) combina el diferencial de puntos ponderados (70%) y el diferencial de goles acotado a $[-3, 3]$ (30%) dentro del rango `[-0.10, +0.10]` cuando `min_sample >= 3` (`MIN_FORM_MATCHES_FOR_BET = 3`).

3. **Etapa 3 — Guard de Cobertura de Datos (`src/commands/main.py`, L885–L905, L1031)**:
   - Exige simultáneamente `ratings_covered` (`impect_iter` presente y ambos equipos en `ratings_map`) y `form_covered` (`form_n >= 3` para local y visitante).
   - Cualquier deficiencia agrega un motivo en `data_coverage_flags` (`liga_sin_ratings_impect`, `equipos_sin_rating_impect`, `forma_insuficiente`), pone `data_coverage_ok = False`, desactiva el tier táctico permanente de JEV (`always_on=data_coverage_ok`) y excluye el partido de `value_bets` (L1128–L1130).
   - Adicionalmente, las consultas externas a `BraveNewsClient` y `JevDecisionClient` se ejecutan solo si `data_coverage_ok and odds_details is not None`.

4. **Etapa 4 — Modelo Predictivo Core Multi-Capa (`src/apostador/predictor.py`, L553–L723)**:
   - **Control temporal *as-of* (`_tactical_decay_factor`, `_attenuate_lookup`, L220–L301)**: Si `target_date` antecede por más de 14 días a la última fecha conocida en `ratings_{it}.json`, `load_tactical_averages`, `load_tactical_scores` y `load_player_talent` contraen sus métricas numéricas hacia la media de liga según $\exp(-0.05 \cdot (\text{days\_ahead} - 14))$ para evitar fuga futura en simulaciones históricas sin alterar predicciones vigentes.
   - **Talento ajustado por disponibilidad (`get_available_player_talent`, L414–L508)**: Empareja nombres de `player_scores_{it}.json` con `starting_xi` y `injured_players` (`_match_player_mask` con limpieza léxica, inicial+apellido y `SequenceMatcher >= 0.86`). Si hay alineación confirmada ($\ge 7$ titulares emparejados), recalcula el talento sobre ese XI; si no, excluye a los lesionados antes de seleccionar el Top-11 disponible por `matchShare`, y calcula `missing_share_impact` $\in [0, 1]$ ponderando `IMPECT_SCORE_PACKING * matchShare` perdido.
   - **`p_ratings` (L572–L584)**: Softmax paramétrico calibrado por MLE (`alpha = 4.322`, `gamma_d` y `home_advantage_per_league` en `calibrated_params.json`), aplicando atenuación exponencial $e^{-0.05 (d - 7)}$ si el rating excede 7 días. Por su parte, `src/commands/calibrate.py` ya incluye `form_delta` de `RecentFormLoader` dentro de `softmax_predict` y `_eval_rows_nll`.
   - **`p_poisson` (L605–L616)**: Distribución Poisson bivariada ajustada por Dixon-Coles (`_dixon_coles_probs`, `rho = -0.06`, `max_goals = 10`) sobre `predModelHome`/`predModelAway` atenuados suavemente por bajas (`lh * (1 - 0.12 * home_missing_impact)`).
   - **`p_tactical` (L618–L656)**: Combina rating, `SHOT_XG`/`EXPECTED_GOALS`, `CONCEDED_SHOT_XG`, `BYPASSED_DEFENDERS`, `FIELD_TILT`, talento disponible ajustado, localía, `form_delta` y `+ 1.5 * (away_missing_impact - home_missing_impact)`.
   - **`p_ensemble` y `p_final` (L658–L723)**: Mezcla `0.50 * P_poisson + 0.35 * P_tactical + 0.15 * P_ratings` (o `0.65/0.35` sin $\lambda$) y, cuando `apply_market_shrinkage=True`, aplica contracción del 20% (`w_shrink = 0.20`) hacia la distribución sin margen calculada mediante el **Power Method** (`_power_novig_probs`, L18–L40). Devuelve además `"P_ensemble"`, `"home_missing_impact"` y `"away_missing_impact"`.

5. **Etapa 5 — Contexto Pre-Partido y Fusión con JEV (`src/commands/main.py`, L982–L1092; `probability_fusion.py`)**:
   - `_extract_side_squad_context` (`src/commands/main.py`, L437–L487) separa por equipo local/visitante las lesiones y los `starting_xi` de `fixture_context` (`ApiFootballContextClient`) para inyectarlos tanto en `SoccerPredictor.predict_match` como en `build_match_state`.
   - Solo si `data_coverage_ok and odds_details is not None`, consulta `BraveNewsClient` (2 queries bilingües, caché 6h) y ejecuta `JevDecisionClient.decide`. En caso contrario registra fallback `"sin_cobertura_o_cuotas"` con costo cero de red/LLM.
   - `fuse_probabilities` (`src/apostador/probability_fusion.py`, L91–L171) suaviza primero `P_jev` mediante `smooth_jev_probabilities` (`JEV_SHRINKAGE_LAMBDA = 0.15`) **hacia `prior_probs=base`** (prior informativo Core Base, con fallback a uniforme $1/3$ si no se provee prior), y mezcla en dos tiers:
     - `w = 0.10` (`JEV_BASE_WEIGHT`) si no hay noticias disruptivas y `always_on=True`.
     - `w = jev_weight` cuyo default en CLI `--jev-weight` ahora es `JEV_CALIBRATED_WEIGHT = 0.20` (manteniendo `JEV_PRIORITY_WEIGHT = 0.50` exportado por compatibilidad) cuando hay evidencia y `noticias_relevantes >= 0.50`.

6. **Etapa 6 — Detección de EV, Veto JEV y Optimización de Cartera (`src/apostador/optimizer.py`, `src/commands/main.py`)**:
   - `main.py` almacena `"vetada"` y `"favorito_base"` en cada `match_entry`. Si `vetada=True`, el bucle de `value_bets` (L1142–L1157) descarta automáticamente el signo favorito base (`favorito_base`), contabilizando `vetadas por JEV`.
   - `implied_probabilities_no_vig` (`src/commands/main.py`, L223–L244) reutiliza `_power_novig_probs` para calcular el ancla de mercado libre de sesgo favorito-longshot.
   - `KellyOptimizer.evaluate_bet_niche` (`src/apostador/optimizer.py`, L75–L130) impone `min_ev <= ev <= max_ev` (`0.03`–`0.35`), `p - market_implied_prob <= 0.15` y las reglas del modo predeterminado `multi_market`.
   - `KellyOptimizer.optimize_portfolio` (`src/apostador/optimizer.py`, L132–L238) deduplica por partido, ordena descendentemente por `ev`, limita a `max_bets_per_league = 3` (`--max-bets-per-league`) cuando se especifica liga, y si la exposición total excede `max_daily_exposure_pct = 0.15` (`--max-daily-exposure-pct`), reescala proporcionalmente todos los stakes (`"exposure_scaled": True`, descartando importes `< min_stake_amount`).

---

## 2. Diagnóstico Crítico y Estado de Correcciones

### A. Fortalezas estructurales
* **Defensa en profundidad contra modelos rotos**: La combinación de *guard de cobertura + gating pre-API + contracción Power Method al mercado + veto activo JEV + topes de EV (35%), divergencia (15 pp), liga (3) y exposición diaria (15%)* acota los modos de falla cuantitativos y operativos.
* **Arquitectura modular y trazable en DuckDB**: Cada predicción guarda su linaje (`P_base`, `P_ensemble`, `P_jev`, `jev_weight`, `vetada`, `favorito_base`, `home_missing_impact`, `away_missing_impact`, `data_coverage_flags`, `market_divergence`, `news_evidence`), permitiendo auditoría post-mortem y backtests temporales sin filtrar datos futuros.
* **Resiliencia ante límites de API**: El fallback horario de The Odds API con consenso multi-bookmaker, el rango de casas sharp en `_select_bookmaker` y el caché persistente en `raw.artifacts` evitan bloqueos ante límites diarios de API-Football o caídas externas (como ClubElo HTTP 502).

### B. Inconsistencias y Bugs Técnicos Detectados en la Auditoría (Todos Corregidos)

1. **`[✅ Corregido]` El veto de riesgo de JEV (`fused["vetada"]`) se calculaba pero se ignoraba en `main.py`**:
   - *Solución aplicada*: [`src/commands/main.py`](../../src/commands/main.py#L1079-L1157) ahora guarda `"vetada"` y `"favorito_base"` en `match_entry` y excluye de `value_bets` el signo favorito base cuando `vetada=True`.
2. **`[✅ Corregido]` Talento Top-11 (`load_player_talent`) ciego a bajas y alineaciones confirmadas**:
   - *Solución aplicada*: [`SoccerPredictor.get_available_player_talent`](../../src/apostador/predictor.py#L414-L508) y [`_extract_side_squad_context`](../../src/commands/main.py#L437-L487) consumen `injuries` y `lineups` de `fixture_context` para recalcular el talento del XI disponible y ajustar $\lambda_H, \lambda_A$ y `tactical_score` vía `missing_share_impact`.
3. **`[✅ Corregido]` Fuga temporal (*Data Leakage*) en `predictor.py` al cargar promedios tácticos**:
   - *Solución aplicada*: [`_tactical_decay_factor`](../../src/apostador/predictor.py#L236-L252) y [`_attenuate_lookup`](../../src/apostador/predictor.py#L254-L276) atenúan `averages`, `scores` y `player_scores` hacia la media de liga cuando `target_date` antecede por más de 14 días a la última fecha de ratings.
4. **`[✅ Corregido]` Desconexión de `RecentFormLoader` respecto a `TeamAligner`**:
   - *Solución aplicada*: [`RecentFormLoader.get_form`](../../src/apostador/recent_form.py#L175-L215) resuelve nombres no mapeados con `TeamAligner.find_best_match(..., threshold=0.75)` y cachea el alias resuelto.
5. **`[✅ Corregido]` Desperdicio de llamadas a Brave y JEV en partidos no apostables**:
   - *Solución aplicada*: [`src/commands/main.py`](../../src/commands/main.py#L1027-L1065) condiciona `BraveNewsClient.search_match_news` y `JevDecisionClient.decide` a `data_coverage_ok and odds_details is not None`.
6. **`[✅ Corregido]` Desalineación del peso JEV prioritario (`0.50` en código vs `0.15–0.20` óptimo en backtest)**:
   - *Solución aplicada*: Se definió `JEV_CALIBRATED_WEIGHT = 0.20` en [`probability_fusion.py`](../../src/apostador/probability_fusion.py#L16) y pasó a ser el valor predeterminado de `--jev-weight` en [`main.py`](../../src/commands/main.py#L630-L632).
7. **`[✅ Corregido]` Suavizado de JEV hacia la distribución uniforme ($1/3$) en lugar del prior base**:
   - *Solución aplicada*: [`smooth_jev_probabilities`](../../src/apostador/probability_fusion.py#L22-L43) contrae `P_jev` hacia `prior_probs=base` en `fuse_probabilities` (conservando el uniforme $1/3$ solo si `prior_probs` es `None`).
8. **`[✅ Corregido]` Ausencia de límite de exposición diaria agregada y concentración por liga en `KellyOptimizer`**:
   - *Solución aplicada*: [`KellyOptimizer.optimize_portfolio`](../../src/apostador/optimizer.py#L132-L238) aplica tope por liga (`max_bets_per_league = 3`) y reescala proporcionalmente los stakes cuando el acumulado supera `max_daily_exposure_pct = 0.15`.
9. **`[✅ Corregido]` Bugs en el modo Fallback de The Odds API y selección de cuota**:
   - *Solución aplicada*: El modo fallback convierte `commence_time` a `America/Bogota` (`main.py`, L775–L783), usa `_select_bookmaker` con prioridad de casas sharp (`L373–L387`) y calcula `consensus_details` con `_extract_event_consensus` (`L912–L918`).

---

## 3. Acciones para Mejorar la Estimación Probabilística (Estado de Avance)

### A1. Ajuste Dinámico de Talento Top-11 por Bajas y Once Confirmado (`✅ Implementado`)
* **Implementación**: `fixture_context["injuries"]` y `fixture_context["lineups"]` se separan por bando (`_extract_side_squad_context`) y alimentan `SoccerPredictor.get_available_player_talent` y `predict_match`.
  - Si existe `lineups` con $\ge 7$ titulares emparejados en Impect, evalúa el promedio sobre ese XI real.
  - En ventana temprana, excluye los jugadores reportados en `injuries` por emparejamiento léxico/inicial/fuzzy y recomputa el Top-11 disponible.
  - Calcula `missing_share_impact` (pérdida ponderada de `IMPECT_SCORE_PACKING * matchShare`), atenuando directamente $\lambda_H, \lambda_A$ (`* (1 - 0.12 * impact)`) y ajustando `tactical_score`.

### A2. Calibración Conjunta y Coherencia Softmax–Forma (`🟡 Parcial: forma integrada; stacking pendiente`)
* **Implementado**: [`src/commands/calibrate.py`](../../src/commands/calibrate.py#L38-L185) integra `RecentFormLoader` en `run_calibration` y suma `form_delta` en `softmax_predict` y `_eval_rows_nll` durante la optimización MLE global y por liga.
* **Pendiente (Fase futura)**: Ampliar `calibrate.py` para ajustar un meta-modelo **Logístico Multinomial con regularización L2 (Ridge)** o **Dirichlet Calibration** sobre `logit(p_ratings)`, `logit(p_poisson)` y `logit(p_tactical)` mediante *expanding-window walk-forward*, guardando los pesos del ensamble en `calibrated_params.json`.

### A3. Remoción de Margen mediante Método de Potencia (`✅ Implementado`)
* **Implementación**: Se implementó [`_power_novig_probs`](../../src/apostador/predictor.py#L18-L40), resolviendo $\sum_{i \in \{H,D,A\}} (1/o_i)^k = 1$ por bisección numérica, reutilizado tanto en la contracción bayesiana de `SoccerPredictor.predict_match` como en `implied_probabilities_no_vig` de `main.py` para el filtro `max_divergence`.

### A4. Suavizado Informado, Peso JEV Calibrado y Veto Obligatorio (`✅ Implementado en lo esencial; pooling log-lineal opcional`)
* **Implementado**:
  1. `smooth_jev_probabilities` contrae `P_jev` hacia `P_base` (`prior_probs=base`).
  2. `--jev-weight` adopta `JEV_CALIBRATED_WEIGHT = 0.20` por defecto; `SoccerPredictor.predict_match` expone `apply_market_shrinkage` (para desacoplar pruebas o futuras experimentaciones post-fusión).
  3. `main.py` aplica obligatoriamente `fused["vetada"]` descartando `favorito_base` de `value_bets`.
* **Opcional futuro**: Evaluar combinación **Log-Linear Pooling** entre `P_ensemble` y `P_jev` previa a la contracción de mercado cuando exista mayor histórico resuelto en DuckDB.

### A5. Forma Reciente Ponderada por Recencia, Diferencia de Goles y Futura Extensión xG (`🟡 Parcial: recencia y ΔGD implementados; xG-Form pendiente`)
* **Implementado**: [`RecentFormLoader`](../../src/apostador/recent_form.py#L239-L331) pondera los partidos recientes por decaimiento exponencial ($\omega_k = e^{-0.04 \cdot \Delta\text{días}_k}$), incorpora diferencia de gol media acotada (`form_goal_diff_avg`, 30% de peso frente al 70% de puntos ponderados) y resuelve nombres difusos vía `TeamAligner`.
* **Pendiente (Fase 3)**: Incorporar rendimiento subyacente por partido (`xG_favor - xG_contra`) y ajuste por fuerza del rival (*Opponent-Adjusted Form*) al habilitar `FBrefTacticalLoader` y `ClubEloLoader`.

---

## 4. Catálogo de Nuevos Datos en la Web y Arquitectura de Integración

Además de aprovechar señales que ya llegan en las respuestas actuales pero se descartan, existen fuentes web abiertas y gratuitas de altísimo valor predictivo. Este bloque resulta doblemente estratégico dado el aviso de próxima desconexión de Impect API registrado en [`src/apostador/api_health.py`](../../src/apostador/api_health.py#L4).

| # | Nueva Fuente Web / Datos | Proveedor / Endpoint | Cobertura / Costo | Impacto en Estimación | Prioridad |
|---|---|---|---|---|---|
| 1 | Altitud de estadios y distancia de viaje | Open-Elevation API + Wikidata / GeoPy + `fixture.venue` (API-Football) | Global / Gratuita | Corrige sesgos de ventaja local en altura (COL, MEX, BOL, ECU, PER) y desgaste por desplazamientos | **P0** |
| 2 | Clima horario exacto del partido | Open-Meteo API (`api.open-meteo.com/v1/forecast`) | Global / Gratuita sin API Key | Ajusta total de goles esperados ($\lambda_H, \lambda_A$) y probabilidad de empate por lluvia/viento | **P0** |
| 3 | Activación de ClubElo + FBref / Understat | `ClubEloLoader`, `FBrefTacticalLoader`, `WhoScoredMissingPlayersLoader` (`soccerdata`) | Ya en repo / Europa + Global / Gratuita | Desbloquea ligas europeas (`impect_iteration: None`) y da respaldo si Impect cae | **P0** |
| 4 | Movimiento de cuotas (Steam Moves) + Totals / Asian Handicap | The Odds API v4 (`markets=h2h,totals,spreads` con snapshots periódicos) | Ligas con `odds_sport_key` / Reusa plan | Detecta dinero institucional (*sharp money*) y calibra $\lambda_H + \lambda_A$ con líneas O/U | **P1** |
| 5 | Valor de mercado de plantilla y minutos ausentes (SVMR) | Transfermarkt datasets / API-Football `/players/squads` | Global / Gratuita / Reusa API-Football | Cuantifica matemáticamente el peso real de cada baja reportada | **P1** |
| 6 | Tabla de posiciones, motivación y árbitros | API-Football `/standings` + `fixture.referee` (ya descargado) | Global / 1 req por liga | Detecta rotación por objetivos cumplidos, presión de descenso y tendencia de penaltis/rojas | **P2** |

---

### Detalle de Integración por Fuente

#### Fuente 1: Altitud Geográfica de Estadios y Distancia de Viaje
* **Origen Web**:
  - `fixture.venue.id`, `fixture.venue.name` y `fixture.venue.city` ya llegan en el payload de `fixtures_{fecha}.json` de API-Football.
  - Coordenadas `(lat, lon)` vía Nominatim / Wikidata SPARQL y elevación exacta vía **Open-Elevation API** (`https://api.open-elevation.com/api/v1/lookup?locations=lat,lon`) o **Open-Meteo Elevation** (`https://api.open-meteo.com/v1/elevation`), ambas gratuitas y consultables una única vez por estadio.
* **Por qué mejora la estimación**: En `calibrated_params.json`, Colombia (`ha=0.081`), México (`0.070`), Ecuador, Perú y Bolivia usan un valor constante de localía por país. Sin embargo, cuando un equipo del llano (ej. Junior en Barranquilla a 18 m o Monterrey a 540 m) visita Bogotá (2,640 m), Pasto (2,527 m) o Toluca (2,660 m), su ventaja real cambia drásticamente respecto a cuando se enfrentan dos clubes de la misma altitud (Millonarios vs. Santa Fe o Toluca vs. América).
* **Arquitectura de Integración**:
  1. **DuckDB**: Crear tabla estática `raw.venues_metadata (venue_id INTEGER PRIMARY KEY, venue_name VARCHAR, city VARCHAR, country VARCHAR, lat DOUBLE, lon DOUBLE, altitude_m DOUBLE, updated_at TIMESTAMP)`.
  2. **`SoccerPredictor.predict_match` (`src/apostador/predictor.py`)**: Calcular `altitude_diff_km = max(0.0, (alt_home - alt_away) / 1000.0)` y `travel_dist_km` (fórmula de Haversine). Añadir términos parametrizados calibrados por MLE en `calibrate.py`:
     $$\text{HA}_{\text{efectivo}} = \text{HA}_{\text{liga}} + \beta_{\text{alt}} \cdot \Delta\text{alt}_{\text{km}} + \beta_{\text{viaje}} \cdot \text{dist}_{1000\text{km}}$$
  3. **`build_match_state` (`src/apostador/jev_decision.py`)**: Enviar `diferencia_altitud_m` y `distancia_viaje_km` a JEV para sustentar ajustes tácticos de desgaste físico.

#### Fuente 2: Clima Horario en el Estadio (Open-Meteo API)
* **Origen Web**: **Open-Meteo Forecast API** (`https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m&timezone=UTC`), gratuita, sin clave de API y con pronóstico horario a 7–16 días vista (e histórico vía `archive-api.open-meteo.com` para calibración retrospectiva).
* **Por qué mejora la estimación**: Lluvia intensa ($\ge 2\text{ mm/h}$) o viento sostenido ($\ge 30\text{ km/h}$) deterioran la circulación rápida de balón de equipos con alto `FIELD_TILT` y pases entre líneas, reducen la tasa esperada de goles ($\lambda_H + \lambda_A$) e incrementan empíricamente la probabilidad de empate.
* **Arquitectura de Integración**:
  1. **Nuevo módulo**: `src/apostador/weather_context.py` con caché en `raw.artifacts` (`weather/fixture_{id}.json`, TTL 3 horas). Extrae las condiciones exactas de la hora de `kickoff_dt` usando `(lat, lon)` de `raw.venues_metadata`.
  2. **`_dixon_coles_probs` (`src/apostador/predictor.py`)**: Aplicar un multiplicador de atenuación de goles en clima adverso ($\lambda'_H = \lambda_H \cdot \gamma_{\text{clima}}, \lambda'_A = \lambda_A \cdot \gamma_{\text{clima}}$, calibrado en histórico con `archive-api.open-meteo.com`).
  3. **`build_match_state` (`src/apostador/jev_decision.py`)**: Inyectar temperatura, precipitación y viento directamente en el bloque de contexto; la pregunta `prob_sorpresa` de JEV ya contempla explícitamente *"clima extremo"*, pero actualmente **no recibe datos meteorológicos**.

#### Fuente 3: Activación de ClubElo + FBref / Understat (Módulos Ya Existentes)
* **Origen Web**:
  - `api.clubelo.com` vía [`ClubEloLoader`](../../src/apostador/soccerdata_clubelo.py) (con fallback local resiliente ante HTTP 502).
  - FBref / Understat vía [`FBrefTacticalLoader`](../../src/apostador/soccerdata_fbref.py) y `soccerdata.Understat` (xG, xGA, npxG, tiros, PPDA, pases progresivos).
  - WhoScored vía [`WhoScoredMissingPlayersLoader`](../../src/apostador/soccerdata_fbref.py#L85-L152).
* **Por qué mejora la estimación**: Actualmente las **10 ligas europeas y saudí** en `SUPPORTED_LEAGUES` con `impect_iteration: None` quedan permanentemente vetadas por `liga_sin_ratings_impect`. Además, ante una eventual desconexión de Impect API, estos módulos proveen un reemplazo inmediato de ratings continuos y métricas xG.
* **Arquitectura de Integración**:
  1. **Fallback de capa Core en `SoccerPredictor.predict_match` (`src/apostador/predictor.py`)**: Si `impect_iteration is None` o `ratings_map` no está disponible, consultar `ClubEloLoader.get_match_elos()` (convirtiendo `elo_diff / 400` a escala compatible con `diff_rating`) y extraer `xg`, `xga` recientes desde `FBrefTacticalLoader` / `Understat`.
  2. **Guard de cobertura en `main.py`**: Distinguir `"cobertura_impect"` vs `"cobertura_clubelo_fbref"` para habilitar partidos europeos de Top-5 y competiciones UEFA con su propia calibración MLE (`calibrate.py`), respaldados adicionalmente con las bajas estructuradas de `WhoScoredMissingPlayersLoader` cuando `/injuries` de API-Football sea incompleto.

#### Fuente 4: Snapshots de Movimiento de Cuotas (Steam Moves) y Mercados Over/Under + Asian Handicap
* **Origen Web**: `The Odds API v4` — ampliar parámetros actuales de `load_or_fetch_odds` (`main.py`, L300–L305) de `markets="h2h"` a `markets="h2h,totals,spreads"`, o consultar el endpoint `/odds` en múltiples ventanas (apertura T-48h, T-6h y T-30m).
* **Por qué mejora la estimación**:
  - **Closing Line Value (CLV) / Steam Moves**: Si Pinnacle baja rápidamente una cuota de 2.20 a 1.92 antes del arranque, es señal de dinero informado (*sharp money*). Apostar contra un movimiento brusco de Pinnacle sin noticia pública suele ser EV negativo oculto.
  - **Líneas Over/Under y Hándicap Asiático**: Las cuotas de `totals` (ej. Over/Under 2.5) revelan la expectativa exacta del mercado sobre $\lambda_H + \lambda_A$, permitiendo anclar el total de goles del componente Poisson Dixon-Coles.
* **Arquitectura de Integración**:
  1. **DuckDB**: Nueva tabla `raw.odds_snapshots (sport_key VARCHAR, event_id VARCHAR, captured_at TIMESTAMP, pinnacle_h DOUBLE, pinnacle_d DOUBLE, pinnacle_a DOUBLE, consensus_h DOUBLE, consensus_d DOUBLE, consensus_a DOUBLE, total_line DOUBLE, over_odds DOUBLE, under_odds DOUBLE)`.
  2. **`KellyOptimizer` (`src/apostador/optimizer.py`)**: Calcular `line_drift_pp = novig_now - novig_open`. Añadir regla de protección: vetar o exigir mayor `min_ev` si el mercado sharp se ha movido fuertemente *en contra* de nuestra selección (`line_drift_pp < -0.04`).
  3. **`SoccerPredictor`**: Usar la mediana de `totals` sin margen para contrastar o acotar $\lambda_H + \lambda_A$ en `_dixon_coles_probs`.

#### Fuente 5: Valor de Mercado por Jugador e Impacto Cuantitativo de Bajas (SVMR)
* **Origen Web**: Datasets públicos de **Transfermarkt** (ej. `davidcariboo/player-scores` actualizado semanalmente o `soccerdata`/Transfermarkt) combinados con `/players/squads` e `/injuries` de **API-Football**.
* **Por qué mejora la estimación**: No todas las ausencias pesan igual. Perder al delantero centro que representa el 22% del valor y minutos del equipo no equivale a perder a un lateral suplente. Hoy JEV y Brave tratan las bajas de forma cualitativa y binaria.
* **Arquitectura de Integración**:
  1. **DuckDB**: Tabla `raw.player_valuations (player_name VARCHAR, team_name VARCHAR, market_value_eur DOUBLE, minutes_pct DOUBLE, position VARCHAR, updated_at DATE)`.
  2. **Cálculo del `Squad Value Missing Ratio` (SVMR)**: Para cada jugador en `fixture_context["injuries"]`, cruzar contra `player_valuations` y `player_scores_{it}.json` para obtener `svmr_home` y `svmr_away` $\in [0, 1]$.
  3. **Uso doble**: (a) Aplicar ajuste proporcional directo sobre el rating efectivo del equipo en `predictor.py`, y (b) reemplazar el flag booleano `structured_relevant` por un peso continuo en JEV cuando el SVMR supere un umbral material (ej. $> 12\%$ de la plantilla titular).

#### Fuente 6: Tabla de Posiciones, Contexto Competitivo y Árbitros
* **Origen Web**:
  - **API-Football `/standings?league={id}&season={year}`** (solo 1 llamada cacheable por 24h por liga; bajo costo de cuota).
  - **`fixture.referee`**: Campo que **ya viene incluido** en la respuesta diaria de `/fixtures` (`fixtures_{fecha}.json`), pero actualmente se ignora en `main.py`.
* **Por qué mejora la estimación**:
  - En el último tercio de liga, equipos matemáticamente salvados o sin aspiración europea tienden a rotar o bajar intensidad frente a equipos que pelean descenso o título (*motivation gap*).
  - Equipos en rondas eliminatorias continentales entre semana suelen rotar el fin de semana aunque la prensa previa no haya filtrado aún el once.
  - Árbitros con tasa histórica alta de penaltis o tarjetas rojas incrementan la varianza de partidos apretados y la probabilidad de goles esperados ($\lambda$).
* **Arquitectura de Integración**:
  1. **Ingesta ligera**: Guardar `standings_{league_id}_{season}.json` en `raw.artifacts` (TTL 24h) y extraer `fixture.referee` directamente en el bucle principal de [`src/commands/main.py`](../../src/commands/main.py).
  2. **Estado enriquecido para JEV (`build_match_state`)**: Añadir el diccionario `"contexto_competitivo"` con posición en tabla, puntos de distancia al descenso/título/copas, y perfil estadístico del árbitro asignado.

---

## 5. Hoja de Ruta Recomendada (Priorización)

| Fase | Estado | Acción | Archivos Afectados | Impacto |
|---|---|---|---|---|
| **Fase 1 (Correcciones inmediatas)** | **✅ Completada** | Conectado veto `fused["vetada"]`, compuerta Brave/JEV sobre `data_coverage_ok and odds_details is not None`, corrección `UTC→Bogotá`, `_select_bookmaker` sharp y `_extract_event_consensus` en fallback Odds API, default `JEV_CALIBRATED_WEIGHT = 0.20`, y topes `max_daily_exposure_pct = 0.15` / `max_bets_per_league = 3`. | `main.py`, `probability_fusion.py`, `optimizer.py` | Ahorro de cuota API, cero fugas de apuestas vetadas y control de concentración y drawdown diario (verificado en suite `pytest`: 61 tests en verde). |
| **Fase 2 (Calibración y Coherencia)** | **✅ Completada** | Unificado `RecentFormLoader` con `TeamAligner` + recencia exponencial y $\Delta GD$, implementado control temporal `as-of` en `predictor.py`, ajuste dinámico de plantilla `get_available_player_talent`, remoción de margen por Power Method (`_power_novig_probs`), shrinkage de JEV hacia `prior_probs=base` y `form_delta` en `calibrate.py`. *(Pendiente opcional para ciclos futuros: stacking L2 walk-forward y xG-Form con FBref/ClubElo).* | `recent_form.py`, `predictor.py`, `probability_fusion.py`, `main.py`, `calibrate.py` | Eliminación de data leakage, menor falso descarte por alias de forma, sensibilidad real a bajas y mejor calibración frente a favoritos/longshots. |
| **Fase 3 (Nuevos Datos P0)** | **⏳ Pendiente** | Integrar altitud/distancia (`raw.venues_metadata`), clima horario (`Open-Meteo`) y activar `ClubEloLoader` + `FBrefTacticalLoader` (más xG-Form) para Europa y contingencia post-Impect. | `predictor.py`, `jev_decision.py`, nuevo `weather_context.py`, `duckdb_store.py` | Ventaja informacional real en altura/clima y cobertura europea + continuidad sin depender exclusivamente de Impect. |
| **Fase 4 (Microestructura P1/P2)** | **⏳ Pendiente** | Snapshots de cuotas para *Steam Moves* y líneas O/U, SVMR por jugador ausente, `/standings` + `referee`, y stacking walk-forward L2 sobre capas. | `main.py`, `optimizer.py`, `calibrate.py`, `duckdb_store.py` | Protección frente a dinero sharp en contra, cuantificación objetiva de lesiones/motivación y calibración conjunta multi-capa. |

