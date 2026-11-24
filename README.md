# 🎯 Apostador Empedernido

Sistema cuantitativo de predicción probabilística de resultados de fútbol, detección de apuestas con valor esperado positivo ($EV > 0$), dimensionamiento óptimo de capital mediante Criterio de Kelly y fusión contextual de noticias en tiempo real con LLMs (JEV).

---

## 🏛️ Arquitectura Actual de la Aplicación

La aplicación opera bajo una arquitectura modular desacoplada en cuatro capas fundamentales: **Ingesta de Fuentes**, **Modelado y Optimización**, **Persistencia Unificada (DuckDB)** e **Interfaces y Auditoría**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       1. FUENTES DE DATOS EXTERNAS                                     │
├──────────────────┬─────────────────┬─────────────────┬──────────────────┬──────────────────────────────┤
│    Impect API    │  API-Football   │  The Odds API   │   Brave Search   │        JEV (TypeSafe)        │
│  Squad Ratings   │ Fixtures diarios│ Cuotas mercado  │ Noticias frescas │ Razonamiento LLM             │
│  KPIs tácticos   │ Lesiones pre-   │ 1X2 (Pinnacle / │ Bajas, rotación, │ Gate relevancia (≥0.50)      │
│  Marcadores ofic.│ partido y once  │ Bet365 / Avg)   │ crisis y DT (6h) │ Probabilidad ajustada P_jev  │
├──────────────────┴─────────────────┴─────────────────┴──────────────────┴──────────────────────────────┤
│                             SoccerData (football-data.co.uk / ClubElo / FBref)                         │
│  Cuotas de cierre históricas (Pinnacle/B365) │ Ratings Elo point-in-time │ Métricas avanzadas xG/tiros │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       2. PIPELINE CORE Y MODELADO                                      │
│                                                                                                        │
│  1. TeamAligner (aligner.py): Mapeo canónico, crosswalk en DuckDB y fuzzy matching anti-derbi          │
│  2. RecentFormLoader (recent_form.py): Forma últimos 5 partidos (70% recencia exp + 30% ΔGD) + Aligner  │
│  3. SoccerPredictor (predictor.py): Ensamble v2 (Ratings MLE + Dixon-Coles + KPIs tácticos +           │
│     Talento XI/bajas + control as-of + contracción Bayesiana al mercado vía Power Method)              │
│  4. Inyección Contextual y Gate de Noticias (news_search.py + jev_decision.py):                        │
│     - Ejecutado solo tras validar cobertura de datos y cuotas disponibles (ahorro de red y LLM)        │
│     - Fusión (probability_fusion.py): P_jev suavizado hacia P_base; w=0.20 (prioridad) / w=0.10 (base) │
│     - Veto activo JEV (main.py): excluye favorito base cuando riesgo_favorito > 0.85 ∧ conf > 0.80    │
│  5. Optimizador de Portafolio y Estrategias (optimizer.py):                                            │
│     - Modo default multi_market (+ hybrid, calibrated_value, value_local, favorites, unconstrained)    │
│     - Capital y Riesgo: Flat 3% o Kelly 0.10 (cap 3%), tope ≤3 apuestas/liga y ≤15% exposición diaria  │
│  6. Motor Cisne Negro (black_swan_v2.py / black_swan_parlay.py):                                       │
│     - Combinatorias óptimas de alta cuota (30x–700x) con micro-stakes y control de asimetría           │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  3. ALMACÉN CENTRAL DUCKDB (apostador.duckdb)                          │
├───────────────────────────────┬────────────────────────────────┬───────────────────────────────────────┤
│  Esquema `raw`                │  Esquema `processed`           │  Esquema `output`                     │
│  - raw.artifacts (JSONs API)  │  - processed.artifacts         │  - output.runs (Metadata ejecuciones) │
│  - raw.match_history (Cuotas) │    (calibrated_params.json,    │  - output.recommendations (Picks EV+) │
│  - raw.clubelo_ratings (Elo)  │     datasets procesados)       │  - output.results (Conciliación real) │
│                               │                                │  - output.impect_results (Marcadores) │
│                               │                                │  - output.team_crosswalk (Mapeos)     │
│                               │                                │  - output.artifacts (Reportes/Backtest│
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       4. INTERFACES Y AUDITORÍA                                        │
├────────────────────────────────────────────────┬───────────────────────────────────────────────────────┤
│  Streamlit Web UI (apps/streamlit_app.py)      │  Línea de Comandos (CLI) & Motores de Backtesting     │
│  - P1: 📊 Pronósticos & Portafolio Recomendado │  - main.py: Pipeline diario operativo                 │
│  - P2: 📈 Backtesting Financiero & ROI Real    │  - predict_upcoming_portfolio.py: Full Kelly semanal  │
│  - P3: 🏟️ Cuotas Históricas & Mercados         │  - black_swan_v2.py: Parlays con validación JEV       │
│  - P4: ⚡ ClubElo & Probabilidades 1X2         │  - backtest_with_historical_odds.py: Auditoría CLV/ROI│
│  - P5: 🎯 Conciliación de Resultados Reales    │  - fetch_results.py / fetch_impect_results.py         │
└────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

---

## 💰 ¿Cómo se Apuesta? (Metodología, Matemáticas y Operativa)

El sistema **no apuesta por corazonada, intuición ni fanatismo**. Aplica una metodología cuantitativa de **Value Betting** inspirada en finanzas algorítmicas, donde solo se arriesga capital cuando existe una **esperanza matemática positiva** comprobada frente a la casa de apuestas.

### 1. Detección de Valor Esperado Positivo ($EV > 0$)

Una apuesta tiene valor cuando la probabilidad real de ocurrencia estimada por nuestro modelo ($P_{\text{adj}}$) es superior a la probabilidad implícita en la cuota ofrecida por el mercado ($P_{\text{mercado}} = 1 / \text{Cuota}$):

$$\text{EV} = (P_{\text{adj}} \times \text{Cuota}) - 1$$

* **Si $\text{EV} \le 0$**: No se apuesta bajo ninguna circunstancia (la ventaja matemática pertenece a la casa).
* **Si $\text{EV} > 0$**: La apuesta califica para el portafolio (ej. un $\text{EV} = +0.08$ representa un rendimiento esperado a largo plazo del $8\%$ sobre el monto apostado).

### 2. Nichos Validados y Dimensionamiento del Capital (StrategyEngine)

El análisis empírico sobre el histórico demostró que maximizar la precisión predictiva sin restricciones de cuota o arriesgando stakes desmedidos (Kelly sin tope) conduce a drawdowns severos debido a probabilidades optimistas de modelos generativos ($P \approx 85\%$). 

Para garantizar márgenes positivos reales y control de riesgo matemático, el sistema incorpora un **StrategyEngine** con 2 nichos validados y gestión estricta de capital:

#### A. Los Dos Nichos Empíricos con Margen Positivo
1. **Nicho 1: Locales en Cuotas Medias (`valor_local_medio`)**:
   - **Criterios**: Mercado Local (`Home Win`), Cuota $[2.00, 4.50]$, $P_{\text{fusionada}} \ge 60\%$, $EV \ge 3\%$.
   - **Rendimiento Comprobado**: **Win Rate 52.6%**, **ROI +31.16%**, PnL neto positivo.
2. **Nicho 2: Favoritos de Alta Certeza (`favorito_alta_certeza`)**:
   - **Criterios**: Mercado Local (`Home Win`), Cuota $[1.20, 1.80]$, $EV \ge 3\%$.
   - **Rendimiento Comprobado**: **Win Rate 75.0%**, **ROI +9.06%**, Max Drawdown $3.9\%$.

#### B. Perfiles de Estrategia (`--strategy-mode`)
* **`multi_market` (Predeterminado en CLI)**: Evalúa los tres signos 1X2 con calibración balanceada (Local $EV \in [3\%, 35\%]$, Empate con $EV \ge 4\%$, cuota $\ge 2.80$ y divergencia $\le 8\text{ pp}$, Visita con $EV \ge 4\%$, cuota $\ge 2.10$ y divergencia $\le 10\text{ pp}$).
* **`calibrated_value`**: Acepta cualquier signo 1X2 que supere los vetos de sanidad ($EV \in [3\%, 35\%]$ y divergencia $\le 15\text{ pp}$).
* **`hybrid`**: Explota simultáneamente los dos nichos históricos (`valor_local_medio` + `favorito_alta_certeza`).
* **`value_local`**: Filtrado estricto exclusivo para locales en cuotas medias.
* **`favorites`**: Filtrado estricto exclusivo para favoritos de alta certeza.
* **`unconstrained`**: Modo general sin filtro de nicho (cualquier selección con $EV > 0$).

#### C. Modalidades de Gestión de Capital y Riesgo Agregado (`Staking`)
* **Flat Stake (`--stake-mode flat`, Recomendado)**:
  - Fija un porcentaje constante del bankroll por apuesta (por defecto `3%`).
  - Elimina el riesgo de quiebra ante rachas normales de fútbol, reduciendo el drawdown máximo de $>80\%$ a $\le 11\%$.
* **Kelly Fraccionario con Hard Cap (`--stake-mode kelly`)**:
  - Aplica la fórmula de Kelly fraccionaria ($f_{\text{frac}} = f^* \times 0.10$), acotando cada apuesta al **Hard Cap** (`max_stake_pct = 3%`).
* **Topes de Concentración y Exposición Diaria (`KellyOptimizer`)**:
  - **`--max-bets-per-league` (default `3`)**: Conserva únicamente las selecciones de mayor EV dentro de cada liga informada.
  - **`--max-daily-exposure-pct` (default `0.15`)**: Si el agregado diario de stakes supera el 15% del bankroll, reescala proporcionalmente cada stake (`exposure_scaled = True`).

### 3. Modalidades de Apuesta Soportadas

| Modalidad | Módulo / Comando | Perfil de Riesgo | Objetivo | Gestión de Stake |
|---|---|---|---|---|
| **Estrategia Híbrida Optimizada** | `main.py` / Streamlit Tab 1 | Equilibrado / Alto Margen | Locales cuotas medias + Favoritos certeros | Flat Stake (3%) o Kelly Hard Cap (3%) |
| **Portafolio Diario de Sencillas** | `main.py` / Streamlit Tab 1 | Conservador / Crecimiento Sostenible | Explotar ineficiencias 1X2 partido a partido | Kelly Fraccionario (0.10) con Hard Cap (3%) |
| **Portafolio Semanal / Multi-Jornada** | `main.py --target-dates` | Controlado / Multi-jornada | Capturar las mejores oportunidades de la semana | Flat Stake (3%) con normalización de exposición |
| **Parlays Cisne Negro (Black Swan)** | `black_swan_v2.py` / Tab 1 | Asimétrico / Alta Recompensa | Parlays de cuota objetivo 30x–700x con gate JEV | Micro-stake estricto (0.01%–0.1% del banco) |

#### Detalle de Apuestas Cisne Negro (Black Swan v2):
Busca capturar eventos con recompensas asimétricas masivas arriesgando una fracción insignificante del banco. Combina de 3 a 6 piernas con $EV > 0$. Antes de incluir una pierna, evalúa:
1. **Gate de noticias Brave**: Detección de bajas clave, rotaciones o problemas internos.
2. **Razonamiento JEV**: Veredicto sobre vulnerabilidad del favorito o probabilidad de sorpresa.
3. Si la relevancia es $\ge 0.50$, ajusta la probabilidad; si no hay certeza o hay contradicciones, mantiene Softmax base o veta la pierna riesgosa.
4. **Stake de $4 a $40 sobre un banco de $40,000**: Si se pierde (en ~99% de los casos), la pérdida es trivial; si se acierta, el retorno multiplica el stake por 30x a 700x.

---

### 4. Flujo Operativo Paso a Paso (Workflow del Apostador)

```
[ 1. Ejecutar Análisis ] ──► [ 2. Revisar Picks y Stakes ] ──► [ 3. Colocar en Bookmaker ]
         │                                                            │
         ▼                                                            ▼
[ 5. Auditoría ROI/CLV ] ◄─── [ 4. Conciliar Marcadores ] ◄───────────┘
```

1. **Paso 1: Generar el Portafolio**:
   - Vía interfaz gráfica: Abrir Streamlit (`streamlit run apps/streamlit_app.py`), seleccionar fecha, banco, factor Kelly y pulsar **"Ejecutar análisis"**.
   - Vía consola: Ejecutar `python -m src.commands.main --date 2026-10-20 --bankroll 100000 --kelly-factor 0.25`.
2. **Paso 2: Inspeccionar Selecciones y Montos**:
   - En la tabla de recomendaciones se presenta: Partido, Selección sugerida (Local, Empate o Visitante), Probabilidad calculada, Cuota de mercado, EV% y **Stake exacto en dinero** (ej. `$2,850`).
   - Se revisan las noticias y justificaciones cualitativas de JEV desplegables en la interfaz.
3. **Paso 3: Colocar la Apuesta Manualmente en la Casa**:
   - Por razones de seguridad financiera y normativas regulatorias de las casas de apuestas, el sistema no ejecuta depósitos ni transacciones automáticas con dinero real.
   - El apostador abre su cuenta en su casa de apuestas preferida (**Pinnacle**, **Bet365** o la que ofrezca la mejor cuota disponible) y coloca exactamente la selección y el monto indicado en `Stake`.
4. **Paso 4: Conciliación Automática o Manual de Resultados**:
   - Al finalizar los partidos, se sincronizan los marcadores oficiales ejecutando `python -m src.commands.fetch_results` o `python -m src.commands.fetch_impect_results`.
   - Alternativamente, en la **Pestaña 5 de Streamlit ("🎯 Resultados & Conciliación")**, se revisan y editan los resultados reales en un click, actualizando `output.results` en DuckDB.
5. **Paso 5: Auditoría de Rendimiento y Actualización de Banco**:
   - En la **Pestaña 2 de Streamlit ("📈 Backtesting & ROI Real")** o con `src.analysis.backtest_with_historical_odds`, se evalúa el rendimiento real: Yield, ROI neto, tasa de acierto y Closing Line Value (CLV).

---

## 📂 Estructura de Archivos del Proyecto

```
apostador-empedernido/
├── .env                              # Credenciales y API Keys (no commitear)
├── requirements.txt                  # Dependencias del proyecto
├── apps/
│   ├── streamlit_app.py              # Interfaz gráfica analítica completa (5 pestañas)
│   └── assets/                       # Estilos e imágenes de la UI
├── src/
│   ├── apostador/                    # Núcleo analítico y modelos
│   │   ├── aligner.py                # Mapeo y fuzzy matching de nombres de equipos
│   │   ├── predictor.py              # Modelo Softmax 1X2 calibrado por MLE
│   │   ├── optimizer.py              # Optimizador de Kelly y análisis comparativo parlay/single
│   │   ├── recent_form.py            # Forma reciente de equipos (últimos 5 partidos)
│   │   ├── news_search.py            # Cliente Brave Search (noticias frescas y caché 6h)
│   │   ├── jev_decision.py           # Cliente JEV (razonamiento cualitativo con LLM)
│   │   ├── probability_fusion.py     # Fusión paramétrica de probabilidades core y JEV
│   │   ├── apifootball_context.py    # Extracción de alineaciones y lesiones pre-partido
│   │   ├── results_fetcher.py        # Reconciliación de resultados vía API-Football
│   │   ├── impect_results_fetcher.py # Reconciliación de marcadores oficiales vía Impect
│   │   ├── impect_team_metrics.py    # Cargador de Squad Ratings y KPIs de Impect
│   │   ├── soccerdata_match_history.py # Ingesta histórica de cuotas (football-data.co.uk)
│   │   ├── soccerdata_clubelo.py     # Ingesta y probabilidades punto en el tiempo ClubElo
│   │   ├── soccerdata_fbref.py       # Ingesta de métricas avanzadas (xG) vía FBref
│   │   ├── duckdb_store.py           # Capa de persistencia relacional y documental DuckDB
│   │   ├── outcome_utils.py          # Utilidades canónicas de resultados (1X2)
│   │   └── config.py                 # Rutas absolutas y configuración global
│   ├── commands/                     # Comandos operativos CLI
│   │   ├── main.py                   # ⭐ Pipeline principal diario (portafolio de valor)
│   │   ├── predict_upcoming_portfolio.py # Portafolio multi-día Full Kelly
│   │   ├── black_swan_v2.py          # Parlays Cisne Negro fusionados con Brave + JEV
│   │   ├── black_swan_parlay.py      # Parlays Cisne Negro v1
│   │   ├── calibrate.py              # Calibración MLE de parámetros Softmax
│   │   ├── fetch_results.py          # Actualizador de resultados vía API-Football
│   │   ├── fetch_impect_results.py   # Actualizador de resultados oficiales vía Impect
│   │   ├── fetch_impect.py           # Descargador masivo de ratings Impect
│   │   ├── query_db.py               # Explorador CLI interactivo de DuckDB
│   │   └── test_apis.py              # Diagnóstico integral de conectividad de APIs
│   └── analysis/                     # Motores de backtesting y evaluación
│       ├── backtest_with_historical_odds.py # ⭐ Backtest financiero con cuotas reales de cierre
│       ├── backtest_news_fusion.py   # Backtest temporal de fusión de noticias JEV
│       ├── backtest_jev_weights.py   # Barrido sistemático de pesos JEV y umbrales
│       ├── backtest.py               # Simulador de estrategias de bankroll (Kelly vs Flat)
│       ├── evaluate_model.py         # Benchmarking de modelos ML (Softmax vs XGBoost)
│       └── evaluate_country_models.py# Benchmarking segmentado por país y liga
├── data/
│   └── apostador.duckdb              # Base de datos central (raw, processed, output)
└── documentacion/                    # Documentación técnica, metodológica y guías
    ├── guia_uso/                     # Arquitectura, guía de uso y manual de DuckDB
    ├── soccerdata/                   # Guía de integración de SoccerData y cuotas de cierre
    ├── backtesting/                  # Informes de rendimiento financiero y ROI real
    ├── bravesearch/                  # Especificación de integración Brave Search
    ├── jev/                          # Especificación y contratos del agente JEV
    ├── impect/                       # Documentación y glosario de Impect
    └── theoddsapi/                   # Guía y endpoints de The Odds API
```

---

## 🗄️ Persistencia Centralizada con DuckDB

Toda la información del sistema se almacena en `data/apostador.duckdb`, eliminando dependencias de carpetas de archivos sueltos:

### Esquemas y Tablas

* **Capa `raw`**:
  * `raw.artifacts`: Documentos y respuestas JSON originales cacheadas (APIs de Impect, The Odds API, API-Football, Brave).
  * `raw.match_history`: Registro tabular de partidos históricos y **cuotas de cierre de Pinnacle, Bet365, Máximas y Medias**.
  * `raw.clubelo_ratings`: Ratings Elo históricos por fecha y equipo con seguridad point-in-time.
* **Capa `processed`**:
  * `processed.artifacts`: Datasets consolidados, parámetros calibrados (`calibrated_params.json`).
* **Capa `output`**:
  * `output.runs`: Registro de cada ejecución (`run_id`, tipo de corrida, fecha, configuración).
  * `output.recommendations`: Registro fila a fila de cada apuesta recomendada (probabilidad, cuota, EV, stake y metadatos).
  * `output.results`: Estado de resolución de cada apuesta (goles local/visita, estado, acierto `bet_won`, origen).
  * `output.impect_results`: Marcadores oficiales de partidos sincronizados desde Impect.
  * `output.team_crosswalk`: Mapeo cruzado de identidades de equipos entre proveedores e Impect.
  * `output.artifacts`: Informes analíticos en Markdown, resúmenes y auditorías.

### Consultas SQL Útiles para el Apostador

```bash
# Entrar a la consola SQL de DuckDB
python -m src.commands.query_db

# O ejecutar una consulta directa
python -m src.commands.query_db "SELECT event_date, home_team, away_team, outcome, probability, odds, ev, stake FROM output.recommendations ORDER BY event_date DESC, ev DESC LIMIT 10"
```

```sql
-- Consultar rendimiento de apuestas resueltas
SELECT 
    r.outcome,
    COUNT(*) as total_apuestas,
    ROUND(AVG(r.ev) * 100, 1) as ev_promedio_pct,
    ROUND(SUM(CASE WHEN res.bet_won THEN (r.odds - 1) * r.stake ELSE -r.stake END), 2) as pnl_neto
FROM output.recommendations r
JOIN output.results res ON r.run_id = res.run_id AND r.row_number = res.row_number
WHERE res.bet_won IS NOT NULL
GROUP BY r.outcome;
```

---

## 💻 Interfaz Gráfica (Streamlit App)

Inicia la aplicación web interactiva con:

```bash
streamlit run apps/streamlit_app.py
```

La aplicación cuenta con **5 pestañas especializadas**:

1. **📊 Pronósticos & Portafolio**: Matriz de probabilidades 1X2 del día, evidencias de noticias y bajas recopiladas por JEV, tabla de apuestas individuales de valor ($EV > 0$) y **sugerencia inferior de Combinada Cisne Negro (Black Swan Parlay)** con optimización dinámica interactiva (cuota objetivo, piernas, micro-stake) y visualización de corridas guardadas en DuckDB. Permite recalcular stakes sobre bancos actualizados sin volver a consultar APIs.
2. **📈 Backtesting Financiero & ROI Real**: Cuadro de mando de rendimiento financiero auditado con cuotas de cierre reales de Pinnacle y Bet365 (ROI / Yield real, PnL neto, Win Rate, Drawdown máximo, y sensibilidad por rango de cuotas y umbral de EV).
3. **🏟️ Cuotas Históricas & Mercados**: Explorador interactivo de cuotas de cierre y marcadores sobre más de 1,500 partidos históricos de `raw.match_history`.
4. **⚡ ClubElo & Probabilidades 1X2**: Calculadora interactiva que estima probabilidades teóricas 1X2 a partir de los ratings Elo de dos equipos y la ventaja de campo.
5. **🎯 Resultados & Conciliación**: Auditoría visual de recomendaciones con botón **"⚡ Fill results"** para ejecutar el subproceso de búsqueda y conciliación automática de marcadores reales (API-Football e Impect), además de edición manual y persistencia directa en DuckDB.

---

## 🧠 Modelo de Predicción Probabilística: Ensamble Multi-Capa (`v2_tactical_xg_ensemble`) + JEV

El modelo central (`SoccerPredictor`) combina tres capas analíticas de Impect, ajuste por bajas/XI confirmado y contracción bayesiana al consenso de mercado antes de fusionar con JEV:

```
Δ = R_H - R_A
x = Δ + home_advantage_liga + form_delta   (form_delta ∈ [-0.10, +0.10]: 70% recencia exp + 30% ΔGD)

Capa 1 (Softmax MLE):     u_H = α·x, u_A = -α·x, u_D = γ_d_liga
Capa 2 (Dixon-Coles):     Poisson bivariado (ρ=-0.06) sobre λ_H, λ_A ajustados por missing_share_impact
Capa 3 (Táctica + XI):    Score multi-KPI (xG, xGA, Packing, Field Tilt, talento disponible XI/bajas)

P_ensemble = 0.50·P_poisson + 0.35·P_tactical + 0.15·P_ratings   (ó 0.65·P_tactical + 0.35·P_ratings sin λ)
P_base     = 0.80·P_ensemble + 0.20·P_mkt_novig                   (remoción de margen Power Method)
```

* **Parámetros calibrados por MLE (`calibrate.py`)**: $\alpha = 4.322$, $\gamma_d = -0.168$ (más `gamma_d_per_league` y `home_advantage_per_league`) calibrados incorporando `form_delta` en `softmax_predict`.
* **Ventaja de Local por Liga**: Brasil ($+0.097$), Argentina ($+0.092$), Colombia ($+0.081$), México ($+0.070$), España ($+0.064$), Italia ($+0.057$), MLS ($+0.052$).
* **Control Temporal `as-of` (Anti-Leakage)**: Los ratings excedidos $> 7$ días decaen exponencialmente ($e^{-0.05(d-7)}$); las tablas agregadas de temporada (`averages`, `scores`, `player_scores`) atenúan hacia la media de liga cuando `target_date` antecede $> 14$ días al último registro conocido (`_tactical_decay_factor`).
* **Talento Disponible (`get_available_player_talent`)**: Recalcula el Top-11 usando alineación confirmada (`starting_xi` si $\ge 7$ titulares coinciden) o excluyendo lesionados de API-Football, cuantificando la pérdida ponderada (`missing_share_impact`) para ajustar $\lambda_H/\lambda_A$ y `tactical_score`.
* **Ponderación y Suavizado JEV (dos tiers, ejecutado solo con cobertura OK + cuotas)**:
  - `smooth_jev_probabilities` contrae `P_jev` ($\lambda = 0.15$) hacia el prior informativo `P_base` (no hacia el uniforme $1/3$).
  - $$\text{Tier táctico (sin noticias disruptivas): } w = 0.10 \implies P_{\text{adj}} = 0.90 \cdot P_{\text{base}} + 0.10 \cdot P_{\text{jev}}$$
  - $$\text{Tier prioridad (noticias relevantes, score} \ge 0.50\text{): } w = 0.20 \text{ (`--jev-weight` default calibrado)}$$
  Si JEV reporta fallback, no está configurado, o falta cobertura/cuotas, el sistema conserva $P_{\text{adj}} = P_{\text{base}}$ ($w=0.0$) sin gastar llamadas de red/LLM.
* **Guards de sanidad y veto activo**:
  - *Cobertura de datos*: solo son apostables partidos con ratings Impect reales de ambos equipos y $\ge 3$ partidos recientes de forma por lado (`data_coverage_flags`).
  - *Veto activo JEV*: si `fused["vetada"] = True` (`riesgo_favorito > 0.85` y `confidence > 0.80`), se excluye el signo favorito base (`favorito_base`) de `value_bets`.
  - *Divergencia vs. mercado (Power Method)*: se veta apostar cuando $P_{\text{modelo}} - P_{\text{implícita consenso Power no-vig}} > 15\text{ pp}$.
  - *Tope de EV plausible y cartera*: edges $> 35\%$ se descartan; el portafolio limita a $\le 3$ apuestas/liga y $\le 15\%$ de exposición diaria agregada (reescalado proporcional).

---

## 🏆 Rendimiento Validado y Backtesting

### Rendimiento Predictivo (740 partidos con Validación Cruzada 5-Fold):
* **Softmax Baseline (Ratings Impect)**: **53.5% Accuracy** | Log Loss: **0.976**
* Regresión Logística (con KPIs Impect): 53.1% Accuracy | Log Loss: 0.981
* XGBoost (con KPIs Impect): 50.7% Accuracy | Log Loss: 1.029

> **Conclusión técnica**: Los modelos ML complejos (XGBoost) tienden a sobreajustar con muestras pequeñas de fútbol. La formulación paramétrica Softmax con ratings calibrados ofrece mayor robustez fuera de muestra.

### Backtest de Estrategias de Banca (500 días de simulación):
| Estrategia de Asignación | Capital Final (Inicio: $1,000) | Comportamiento |
|---|---|---|
| Kelly Puro (1.0) | $723 | Quiebras frecuentes por alta volatilidad |
| **Kelly Fraccionario (0.25)** | **$1,952** | **Crecimiento estable con drawdown controlado ✅** |
| Parlays (2 piernas) | $1,522 | Mayor dispersión |

---

## 🚀 Guía Rápida de Comandos CLI

```bash
# 1. Pipeline diario principal (generar portafolio de valor)
python -m src.commands.main --date 2026-10-20 --bankroll 100000 --kelly-factor 0.25

# 2. Portafolio multi-día Full Kelly (próximos 5 días)
python -m src.commands.predict_upcoming_portfolio --bankroll 100000 --days 5

# 3. Apuestas Cisne Negro v2 (noticias Brave + JEV, cuota objetivo 30 a 100)
python -m src.commands.black_swan_v2 --input 2026-10-20 --target-odds 40 --max-legs 5 --bankroll 40000

# 4. Conciliar resultados reales de partidos finalizados
python -m src.commands.fetch_results           # Vía API-Football
python -m src.commands.fetch_impect_results    # Vía marcadores oficiales Impect

# 5. Ejecutar Backtesting Financiero con cuotas históricas de cierre
python -m src.analysis.backtest_with_historical_odds

# 6. Recalibrar parámetros del modelo MLE (cuando haya nuevos datos históricos)
python -m src.commands.calibrate

# 7. Diagnóstico de conectividad de APIs
python -m src.commands.test_apis
```

---

## 🌍 Ligas Soportadas

| Liga | País | ID API-Football | ID Impect | Clave The Odds API | Cobertura SoccerData |
|---|---|---|---|---|---|
| Primera A | Colombia | 239 | 1752 | *(Sin cuotas The Odds)* | FBref |
| Liga MX | México | 262 | 2176 | `soccer_mexico_ligamx` | MatchHistory / FBref |
| Série A | Brasil | 71 | 1799 | `soccer_brazil_campeonato` | MatchHistory / FBref |
| MLS | USA | 253 | 1864 | `soccer_usa_mls` | MatchHistory / FBref |
| Liga Profesional | Argentina | 128 | 1930 | `soccer_argentina_primera_division` | MatchHistory / FBref |
| LaLiga | España | 140 | 2110 | `soccer_spain_la_liga` | MatchHistory / ClubElo / FBref |
| Serie A | Italia | 135 | 2111 | `soccer_italy_serie_a` | MatchHistory / ClubElo / FBref |
| Jupiler Pro League | Bélgica | 144 | 2143 | `soccer_belgium_first_div` | MatchHistory / ClubElo / FBref |
| Premier Liga | Rusia | 235 | 2155 | `soccer_russia_premier_league` | MatchHistory / ClubElo / FBref |

---

## 🔑 Variables de Entorno (`.env`)

Crea un archivo `.env` en la raíz del proyecto:

```bash
USERNAME_IMPECT=tu_usuario_impect
PASSWORD_IMPECT=tu_contraseña_impect
APIFOOTBALL=tu_api_key_apifootball
ODDSAPI=tu_api_key_theoddsapi
BRAVEAPI=tu_api_key_brave_search
JEVAPI=tu_api_key_typesafe_jev
```
