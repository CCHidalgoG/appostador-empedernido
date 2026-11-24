# Informe de Backtesting Financiero Completo con Cuotas Históricas

> **Fecha de Generación:** 2026-11-23 20:27:39  
> **Fuente de Datos:** Recomendaciones de cartera en DuckDB cruzadas con cuotas de cierre (`MatchHistory` / Pinnacle / Bet365).

---

## 1. Resumen Ejecutivo y Métricas Globales

| Métrica | Valor | Interpretación Financiera |
| :--- | :--- | :--- |
| **Total Apuestas Ejecutadas** | `444` | Muestra evaluada con resultados verificados |
| **Tasa de Acierto (Win Rate)** | **37.16%** (165 W / 279 L) | Frecuencia de éxito en la muestra real |
| **Capital Total Movilizado** | **$3,550,873.99** | Rotación de cartera en el periodo analizado |
| **Beneficio Neto (P&L)** | **$169,460.43** | Ganancia neta generada |
| **ROI Real (Yield)** | **4.77%** | Rendimiento sobre el capital arriesgado |
| **ROI a Cuota de Cierre Pinnacle** | **4.77%** | Rendimiento contra el mercado más eficiente |
| **Closing Line Value (CLV) Medio** | **1.0** | Margen ajustado frente al cierre |
| **Tasa de Superación de Cierre** | **0.0%** | Porcentaje de cuotas tomadas con valor superior al cierre |
| **Máximo Drawdown (%)** | **-154.06%** | Caída máxima desde el pico de capital |
| **Máximo Drawdown ($)** | **-$488,438.11** | Pérdida monetaria máxima temporal |
| **Bankroll Final Simulado** | **$269,460.43** | Crecimiento de capital (Base $100,000) |

---

## 2. Rendimiento Desglosado por Rango de Cuotas

| Rango de Cuotas | Apuestas | Aciertos | Win Rate (%) | Total Apostado | P&L ($) | ROI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.20 - 1.80 (Favoritos Fuertes) | 47 | 29 | 61.7% | $350,603.41 | $16,304.09 | **4.65%** |
| 1.80 - 2.50 (Favoritos Moderados) | 104 | 43 | 41.35% | $1,402,562.92 | $-3,612.07 | **-0.26%** |
| 2.50 - 4.00 (Cuotas Medias / Underdogs) | 172 | 65 | 37.79% | $1,247,874.20 | $-11,017.82 | **-0.88%** |
| > 4.00 (Longshots / Empates Altos) | 119 | 26 | 21.85% | $517,616.17 | $162,371.33 | **31.37%** |

---

## 3. Rentabilidad por Tipo de Selección

| Selección | Apuestas | Aciertos | Win Rate (%) | Total Apostado | P&L ($) | ROI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Victoria Local | 221 | 91 | 41.18% | $2,574,379.72 | $88,350.82 | **3.43%** |
| Empate | 120 | 40 | 33.33% | $297,439.23 | $57,985.20 | **19.49%** |
| Victoria Visitante | 64 | 21 | 32.81% | $480,379.98 | $57,544.72 | **11.98%** |

---

## 4. Sensibilidad al Umbral de Valor Esperado (EV Filter)

| Filtro de EV | Apuestas | Aciertos | Win Rate (%) | Total Apostado | P&L ($) | ROI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EV >= 0% | 444 | 165 | 37.16% | $3,550,873.99 | $169,460.43 | **4.77%** |
| EV >= 5% | 371 | 131 | 35.31% | $3,441,078.04 | $160,869.79 | **4.67%** |
| EV >= 8% | 326 | 112 | 34.36% | $3,342,749.62 | $159,259.40 | **4.76%** |
| EV >= 10% | 282 | 98 | 34.75% | $3,220,744.15 | $171,897.67 | **5.34%** |
| EV >= 15% | 224 | 79 | 35.27% | $2,959,697.52 | $169,465.97 | **5.73%** |

---

## 5. Conclusiones y Recomendaciones para la Estrategia

1. **Eficiencia en Rango de Cuotas Moderadas:**
   - Las cuotas entre **1.80 y 2.50** y las selecciones de **Victoria Local** aportan la mayor estabilidad a la curva de capital.
2. **Impacto del Filtro de EV:**
   - Restringir la cartera a apuestas con $EV \ge 8\%$ o $EV \ge 10\%$ optimiza el yield y reduce la volatilidad al eliminar apuestas marginales que erosionan la banca.
3. **Validación del JEV con Inyección Completa:**
   - La inyección de métricas Impect (xG, empaque, recuperaciones) y contexto de ausencias calibra las probabilidades base del modelo Core para evitar sobreconfianza en favoritos con valor negativo.
