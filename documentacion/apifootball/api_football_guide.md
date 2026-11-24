# Guía Técnica: API-Football (v3)

Esta documentación describe la integración con **API-Football** para obtener los partidos programados (*fixtures*), información de ligas, y metadatos de los equipos.

---

## 1. Configuración General

*   **URL Base**: `https://v3.football.api-sports.io`
*   **Cabecera de Autenticación**: `x-apisports-key` (Clave provista por API-Sports)
*   **Límites de Tasa**: 100 peticiones al día en el plan gratuito.
*   **Método Recomendado**: Caché de respuestas diarias para la lista completa de partidos para minimizar llamadas.

---

## 2. Endpoints Clave y Esquemas de Respuesta

### A. Endpoint `/status`
Permite verificar la validez del token y el uso diario de la API.
*   **Método**: `GET`
*   **Respuesta**:
    ```json
    {
      "response": {
        "account": {
          "email": "usuario@ejemplo.com"
        },
        "subscription": {
          "plan": "Free",
          "active": true
        },
        "requests": {
          "current": 12,
          "limit_day": 100
        }
      }
    }
    ```

### B. Endpoint `/fixtures?date=YYYY-MM-DD`
Recupera todos los partidos programados para una fecha específica.
*   **Método**: `GET`
*   **Esquema de Respuesta (JSON)**:
    El nodo `response` contiene una lista de objetos con la siguiente estructura:
    *   `fixture`:
        *   `id` (entero): Identificador único del partido (clave principal).
        *   `referee` (cadena/null): Nombre del árbitro.
        *   `date` (cadena ISO): Fecha y hora en UTC.
        *   `timestamp` (entero): Timestamp Unix de inicio.
        *   `venue`: `name` (nombre del estadio) y `city` (ciudad).
        *   `status`: `short` (ej. `FT` para terminado, `NS` para no empezado, `PST` para postergado) y `elapsed` (minutos jugados).
    *   `league`:
        *   `id` (entero): ID de la liga en API-Football.
        *   `name` (cadena): Nombre del torneo (ej. "La Liga").
        *   `country` (cadena): País del torneo (ej. "Spain").
        *   `season` (entero): Año de la temporada (ej. `2026`).
        *   `round` (cadena): Jornada o fase (ej. "Regular Season - 1").
    *   `teams`:
        *   `home`: `id` (entero) y `name` (nombre comercial del equipo local).
        *   `away`: `id` (entero) y `name` (nombre comercial del equipo visitante).
    *   `goals`:
        *   `home` (entero/null) y `away` (entero/null): Marcador en tiempo real o final.

---

## 3. Integración en el Pipeline

Para un día dado ($D$), el pipeline del sistema de predicción realiza las siguientes operaciones con API-Football:
1.  **Consulta**: Envía petición a `GET /fixtures?date=YYYY-MM-DD`.
2.  **Filtrado**: Selecciona únicamente partidos de ligas que estén activas y mapeadas con las ligas de Impect (ej. Liga Dimayor, LaLiga, Serie A, Brasileirão, MLS).
3.  **Persistencia**: Guarda la respuesta cruda en `data/raw/apifootball/fixtures_YYYY-MM-DD.json`.
4.  **Alineación**: Pasa los nombres de los equipos (`teams.home.name` y `teams.away.name`) y el ID del partido al módulo de alineación de entidades.
