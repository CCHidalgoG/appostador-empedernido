# SoccerData — Guía de Integración y Referencia Técnica

## 1. Qué es SoccerData

[SoccerData](https://soccerdata.readthedocs.io/en/latest/) es una biblioteca de Python de código abierto desarrollada por Pieter Robberechts (KU Leuven) que unifica el raspado y extracción de datos de múltiples portales estadísticos y de apuestas de fútbol reconocidos a nivel internacional.

Devuelve siempre estructuras tabulares estándar en **Pandas DataFrames** indexadas de forma uniforme por `(league, season, team / player, game)`.

---

## 2. Fuentes de Datos Soportadas y Relevancia para el Proyecto

| Fuente | Clase en SoccerData | Tipo de Datos | Utilidad en `apostador-empedernido` |
|---|---|---|---|
| **Football-Data.co.uk** | `MatchHistory` | Resultados, estadísticas de partido y **cuotas de apuestas históricas** (Bet365, Pinnacle, William Hill, cuotas máximas y medias del mercado). | **Crítica (P0)**: Permite resolver la limitación actual de backtesting calculando **ROI real y drawdown histórico** con cuotas de cierre. |
| **ClubElo** | `ClubElo` | Calificaciones Elo históricas y diarias de clubes europeos. | **Alta (P1)**: Rating alternativo objetivo para calibración MLE y comparación con ratings Impect. |
| **FBref** | `FBref` | Estadísticas avanzadas Opta/StatsPerform: xG, xAG, SCA, pases progresivos, presiones y alineaciones. | **Alta (P1)**: Enriquecimiento del perfil táctico de temporada inyectado a JEV. |
| **WhoScored** | `WhoScored` | Eventos detallados (coordenadas de pases, tiros, regates), calificaciones y **jugadores ausentes/lesionados** (`read_missing_players`). | **Media-Alta (P2)**: Respaldo estructurado de bajas y lesiones antes del partido. |
| **Understat** | `Understat` | Goles esperados (xG) a nivel de disparo, xGChain y xGBuildup para las principales ligas europeas. | **Media (P2)**: Métricas complementarias de peligro ofensivo. |
| **Sofascore** | `Sofascore` | Calendarios, clasificaciones y alineaciones oficiales. | **Media (P3)**: Fuente secundaria de fixtures y tablas. |
| **SoFIFA** | `SoFIFA` | Atributos individuales de jugadores y clubes de EA Sports FC / FIFA. | **Baja (P3)**: Contexto de atributos físicos y técnicos si faltan datos de scouting. |
| **ESPN** | `ESPN` | Hojas de partido, calendarios y alineaciones. | **Baja (P3)**: Redundante con API-Football. |

---

## 3. Análisis de Viabilidad Técnica

### Puntos Fuertes
1. **Extracción unificada a Pandas**: Se integra de forma natural con los pipelines existentes (`impect_team_metrics.py`, `predictor.py`, `calibrate.py`).
2. **Caché en disco configurable**: Permite guardar las descargas en un subdirectorio local (ej. `data/raw/soccerdata`), evitando peticiones repetidas a los servidores externos.
3. **Mapeo de nombres con alias**: Incorpora soporte nativo para `teamname_replacements.json`, facilitando la sincronización con el `TeamAligner` del proyecto.
4. **Gratuito y sin claves de API**: Aporta cuotas históricas y métricas xG sin costo recurrente de suscripción.

### Desafíos y Riesgos Operativos
1. **Fragilidad de Web Scraping**: Fuentes como FBref o WhoScored no son APIs oficiales, sino páginas web protegidas con Cloudflare o Incapsula. FBref impone un límite de **20 solicitudes por minuto** (HTTP 429).
2. **Dependencias de Navegador Headless (WhoScored)**: WhoScored requiere `SeleniumBase` y Chrome/ChromeDriver, lo que añade peso y dependencias en entornos de servidor.
3. **Cobertura Geográfica Heterogénea**:
   - `ClubElo`: Solo clubes de Europa.
   - `Understat`: Solo 6 ligas europeas (Top 5 + Rusia).
   - `MatchHistory`: Excelente en Europa; cubre ligas seleccionadas de América (MLS, Brasil, México).
   - `FBref`: Cobertura global amplia (Europa, MLS, Argentina, Brasil, México, Colombia).

---

## 4. Instalación y Configuración

### Instalación en el entorno virtual
```bash
source .venv/bin/activate
pip install soccerdata
```

### Variables de Entorno Recomendadas
Para integrar la caché de SoccerData dentro de la estructura de datos del proyecto, se deben configurar en `.env`:

```env
# Directorio de caché persistente de SoccerData dentro de data/raw
SOCCERDATA_DIR="data/raw/soccerdata"
SOCCERDATA_NOCACHE="False"
SOCCERDATA_NOSTORE="False"
SOCCERDATA_LOGLEVEL="WARNING"
```

---

## 5. Ejemplos de Uso Práctico

### A. Obtención de Cuotas Históricas para Backtesting (`MatchHistory`)
```python
import soccerdata as sd

# Descargar historial y cuotas de LaLiga española temporada 2023-2024
mh = sd.MatchHistory(leagues="ESP-La Liga", seasons="2324", data_dir="data/raw/soccerdata")
games = mh.read_games()

# Columnas disponibles: B365H, B365D, B365A, PSH, PSD, PSA, MaxH, AvgH, etc.
odds_columns = [col for col in games.columns if "B365" in col or "PS" in col or "Avg" in col]
print(games[["date", "home_team", "away_team", "FTHG", "FTAG", *odds_columns[:6]]].head())
```

### B. Calificaciones Elo de Equipos (`ClubElo`)
```python
import soccerdata as sd

elo = sd.ClubElo(data_dir="data/raw/soccerdata")

# Ratings Elo de todos los clubes europeos para una fecha objetivo (punto en el tiempo)
ratings_snapshot = elo.read_by_date("2024-03-15")
print(ratings_snapshot.loc[["Real Madrid", "Barcelona", "Inter", "Arsenal"]])
```

### C. Métricas Avanzadas y xG (`FBref`)
```python
import soccerdata as sd

fbref = sd.FBref(leagues=["ESP-La Liga"], seasons=["2324"], data_dir="data/raw/soccerdata")

# Estadísticas de tiro y xG por equipo
team_shooting = fbref.read_team_season_stats(stat_type="shooting")
print(team_shooting[["Gls", "Sh", "SoT", "xG", "npxG"]].head())
```

---

## 6. Arquitectura de Integración con DuckDB y el Pipeline

```text
[ Fuentes Web / CSV ] (football-data.co.uk / ClubElo / FBref)
          │
          ▼  vía soccerdata (con caché local)
   data/raw/soccerdata/
          │
          ▼  ETL / Ingesta controlada
   raw.artifacts (DuckDB)
          │
    ┌─────┴────────────────────────┐
    ▼                              ▼
Backtesting con Cuotas         Contexto Táctico JEV
(MatchHistory: B365, Pinnacle)  (FBref: xG, SCA, pases progresivos)
```

1. **Capa Raw**: `soccerdata` descarga en `data/raw/soccerdata`. Un script de ingesta registra los snapshots en la tabla `raw.artifacts` de DuckDB.
2. **Backtesting (`src/analysis/backtest_with_historical_odds.py`)**: Cruza las recomendaciones guardadas con las cuotas históricas reales de `MatchHistory` para calcular **ROI exacto, tasa de acierto y curva de capital**.
## 7. Actualizaciones Regulatorias y de Fuentes (2025–2026)

1. **Retirada de métricas avanzadas en FBref (20 de enero de 2026):** Sports Reference descontinuó el suministro de métricas Opta/xG en FBref. `UnderstatTacticalLoader` (`src/apostador/soccerdata_fbref.py`) actúa ahora como proveedor primario de `xG`, `xGA`, `npxG` y `PPDA` para las grandes ligas europeas, quedando FBref únicamente para estadística básica.
2. **Cierre de Pinnacle API (23 de julio de 2025) y columnas de cierre (`*C`):**
   - `SoccerDataHistoryLoader` extrae las columnas de cierre reales con sufijo `C` (`PSCH/D/A`, `B365CH/D/A`, `AvgCH/D/A`, `MaxCH/D/A`).
   - Los registros posteriores al `2025-07-23` se marcan con `pinnacle_is_stale = TRUE` y el backtester sustituye Pinnacle por `AvgCH` o por cierres capturados localmente.
3. **Licenciamiento de `football-data.co.uk` (`APOSTADOR_COMMERCIAL_MODE`):** Sus CSV gratuitos son exclusivamente para **uso particular** (`usage_scope = 'personal_research_only'`). El guard `validate_usage_compliance()` bloquea su exposición en modo comercial; para uso comercial o entrenamiento abierto, migrar resultados/calendarios hacia **OpenFootball** (dominio público) y el histórico de cuotas hacia `raw.odds_snapshots` (`The Odds API` / `Betfair Exchange` / `OpticOdds`).
