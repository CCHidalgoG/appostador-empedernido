# Guía Técnica: The Odds API (v4)

Esta documentación describe la integración con **The Odds API** para obtener las cuotas (odds) decimales en tiempo real para los mercados de fútbol.

---

## 1. Configuración General

*   **URL Base**: `https://api.the-odds-api.com`
*   **Autenticación**: Parámetro de consulta `apiKey` (enviado en la URL).
*   **Límites de Tasa**: Límite basado en créditos de consulta de la suscripción (ej. 500 créditos gratuitos en el plan básico).
*   **Formato de Cuotas**: Decimales (predeterminado).

---

## 2. Endpoints Clave y Esquemas de Respuesta

### A. Endpoint `/v4/sports/`
Lista todas las disciplinas y ligas deportivas disponibles.
*   **Método**: `GET`
*   **Filtro Clave**: Buscar aquellas cuyo atributo `"group"` sea `"Soccer"`.
*   **Respuesta (JSON)**:
    ```json
    [
      {
        "key": "soccer_epl",
        "group": "Soccer",
        "title": "Premier League",
        "description": "English Premier League",
        "active": true,
        "has_outrights": false
      }
    ]
    ```

### B. Endpoint `/v4/sports/{sport_key}/odds/`
Recupera las cuotas vigentes para la liga especificada por `sport_key`.
*   **Método**: `GET`
*   **Parámetros Recomendados**:
    *   `regions=eu`: Filtra casas de apuestas europeas y latinoamericanas (como Bet365, Pinnacle, Betfair, 1xBet, etc.).
    *   `markets=h2h`: Filtra el mercado principal de resultado del partido (1X2: Victoria Local, Empate, Victoria Visitante).
*   **Esquema de Respuesta (JSON)**:
    La respuesta es una lista de objetos que representan encuentros individuales:
    *   `id` (cadena): Identificador único del evento en The Odds API.
    *   `sport_key` (cadena): Identificador de la liga (ej. `"soccer_epl"`).
    *   `commence_time` (cadena ISO): Fecha y hora de inicio en UTC.
    *   `home_team` (cadena): Nombre del equipo local.
    *   `away_team` (cadena): Nombre del equipo visitante.
    *   `bookmakers` (lista): Listado de casas de apuestas que proveen cuotas para este encuentro:
        *   `key` (cadena): ID de la casa de apuestas (ej. `"pinnacle"`, `"bet365"`).
        *   `title` (cadena): Nombre comercial (ej. "Pinnacle").
        *   `markets` (lista): Lista de mercados cotizados:
            *   `key` (cadena): ID del mercado (ej. `"h2h"`).
            *   `outcomes` (lista de 3 elementos):
                *   Cada elemento contiene `name` (nombre del equipo o `"Draw"`) y `price` (cuota decimal, ej. `1.85`).

---

## 3. Integración en el Pipeline

Para un día dado ($D$):
1.  **Identificación**: El pipeline mapea las ligas activas del día en API-Football con las `sport_key` correspondientes en The Odds API.
2.  **Consulta**: Ejecuta una solicitud a `/v4/sports/{sport_key}/odds/` para cada liga.
3.  **Persistencia**: Guarda la respuesta en `data/raw/theoddsapi/odds_{sport_key}.json`.
4.  **Consolidación**: Extrae la cuota de la casa de apuestas seleccionada (ej. **Pinnacle** o el promedio del mercado) y la alinea con el partido coincidente de API-Football usando el módulo de mapeo de nombres de equipos.
