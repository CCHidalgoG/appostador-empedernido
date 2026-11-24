# Guía Técnica: Impect Customer API V5 (impectPy)

Esta documentación detalla el uso de la biblioteca oficial **impectPy** para extraer los datos analíticos avanzados de rendimiento de equipos (KPIs y valoraciones/scores) e incorporarlos en el modelo predictivo.

---

## 1. Autenticación y Acceso

El acceso a la API se realiza mediante las credenciales registradas de Impect, utilizando un flujo OAuth2 administrado internamente por la función `getAccessToken`.

```python
import impectPy as ip

# Obtención del token Bearer (válido para la sesión)
token = ip.getAccessToken(username="ccamilo.hidalgo@gmail.com", password="...")
```

*   **Límite de Tasa**: Impect aplica un límite estricto de **10 peticiones por segundo** por cuenta. La biblioteca `impectPy` gestiona este límite mediante un algoritmo interno de retardo (*token bucket*).

---

## 2. Métodos Clave Utilizados

### A. Listado de Competiciones (`ip.getIterations`)
Obtiene todas las competiciones y temporadas habilitadas para la cuenta. Retorna un DataFrame de Pandas.
*   **Campos de Interés**:
    *   `id`: ID de la iteración (ej. `1207` para Liga Dimayor I 2025).
    *   `competitionName`: Nombre de la liga (ej. "LaLiga", "Liga Dimayor I").
    *   `season`: Nombre de la temporada (ej. "24/25", "2025").
    *   `competitionCountryName`: País del torneo (ej. "Spain", "Colombia").

### B. Valoración de Fuerza de Equipos (`ip.getSquadRatings`)
Obtiene la calificación compuesta de fuerza global (*squad rating*) calculada por Impect de manera histórica. Los datos se calculan quincenalmente (días 1 y 15 de cada mes).
*   **Uso**: `ip.getSquadRatings(iteration=1207, token=token)`
*   **Campos de Interés**:
    *   `squadId` (entero): Identificador del equipo.
    *   `squadName` (cadena): Nombre del equipo.
    *   `date` (cadena): Fecha del cálculo (YYYY-MM-DD).
    *   `value` (flotante, ej. `0.74`): Fuerza relativa del equipo. Los valores más altos denotan escuadras con mayor probabilidad de imponerse en sus encuentros.

### C. KPIs Históricos por Equipo (`ip.getSquadIterationAverages`)
Contiene los promedios acumulados a lo largo de la temporada para más de 1,400 KPIs detallados de rendimiento de cada club.
*   **Uso**: `ip.getSquadIterationAverages(iteration=1207, token=token)`
*   **Métricas predictivas de alto impacto (KPIs)**:
    *   `BYPASSED_OPPONENTS`: Cantidad de rivales superados mediante pases y conducciones (*Packing*).
    *   `BYPASSED_DEFENDERS`: Defensores superados (clave para predecir peligro de gol).
    *   `RECEPTIONS_BEHIND_DEFENSIVE_LINE`: Recepciones de balón a espaldas de la defensa contraria.
    *   `EXPECTED_GOALS`: Goles esperados (xG).
    *   `EXPECTED_GOAL_ASSISTS`: Asistencias esperadas (xA).
    *   `BALL_RECOVERIES`: Recuperaciones de balón.
    *   `PRESSED_PASSES`: Pases bajo presión.

---

## 3. Integración en el Pipeline

1.  **Carga e Ingesta**: Al iniciar el pipeline diario, se realiza la consulta de `ip.getSquadRatings` y `ip.getSquadIterationAverages` para las iteraciones de las ligas del día.
2.  **Extracción de Fuerza**:
    *   Se extraen las últimas valoraciones de fuerza (`value`) de cada equipo desde el set de datos de calificaciones para utilizarlas como el factor principal de fuerza base.
3.  **Extracción de Atributos de Juego**:
    *   Se extraen métricas clave (xG defensivo, xG ofensivo, Packing ofensivo, Packing defensivo) del catálogo de promedios de temporada.
4.  **Uso Predictivo**: Estos parámetros alimentarán el modelo de cálculo de probabilidades de victoria en la Capa 3.
