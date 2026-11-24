# Backtesting de JEV y simulación de pesos

Generado: 2026-10-19 · fuente: `data/apostador.duckdb`.

## Alcance y método

Se tomaron las predicciones guardadas en ejecuciones `daily_portfolio`, conservando la última predicción por partido y fecha. Se enlazaron por fecha/equipos con `output.results` y, para partidos sin fila de recomendación, se usó `output.impect_results` como fallback (7 resultados añadidos). La muestra enlazada contiene **101 partidos** (127 predicciones únicas con probabilidades base); período: **2026-10-02 a 2026-10-19**.

Resultados disponibles por fuente en toda la tabla: api: 35, impect: 259, manual: 22. Las métricas de este informe usan solo partidos enlazados a una predicción guardada.

Se comparan probabilidades `P_base` (core) contra las probabilidades finales persistidas (`P_H/P_D/P_A`). El gate de relevancia vigente es ≥0.50. Para simular pesos se usa `P=(1−w)·P_base+w·P_jev`, donde `w` recorre valores candidatos. La cohorte de simulación exige vector JEV válido y gate contextual aprobado; las filas históricas antiguas usan como aproximación el peso efectivo guardado porque no conservaban el gate. Métricas: exactitud, log loss, Brier multiclase (menores son mejores) y ECE top-label de diez bins (menor indica mejor calibración).

## Core frente a producción con JEV

| Cohorte | Partidos | Exactitud | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Impect rating + localía, sin forma | 101 | 47.5% | 1.0413 | 0.6259 | 0.0802 |
| Todos: core | 101 | 47.5% | 1.0279 | 0.6169 | 0.0532 |
| Todos: mezcla guardada | 101 | 47.5% | 1.0934 | 0.6505 | 0.1843 |
| JEV activado: core pareado | 62 | 56.5% | 0.9670 | 0.5701 | 0.0936 |
| JEV activado: mezcla guardada | 62 | 56.5% | 1.0738 | 0.6249 | 0.1890 |

JEV tuvo peso efectivo mayor que cero en **62 de 101** partidos enlazados. La comparación pareada usa esa cohorte; el simulador de pesos puede incluir además predicciones nuevas con gate aprobado y peso configurado en cero.

## Simulador de peso JEV

Se recorrieron pesos de **0.00 a 1.00 en pasos de 0.05**. La partición se hizo por días completos: 7 días (73 partidos) para calibración y 6 días (28 partidos) como holdout. La selección minimiza log loss en calibración; el holdout se consulta una sola vez para estimar generalización.

| Peso JEV | N calibración | Log loss calibración | Brier calibración | Exactitud calibración | N holdout | Log loss holdout | Brier holdout |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 51 | 0.9771 | 0.5770 | 54.9% | 11 | 0.9203 | 0.5379 |
| 0.05 | 51 | 0.9710 | 0.5722 | 54.9% | 11 | 0.9077 | 0.5280 |
| 0.10 | 51 | 0.9671 | 0.5689 | 54.9% | 11 | 0.8976 | 0.5201 |
| 0.15 | 51 | 0.9651 | 0.5672 | 54.9% | 11 | 0.8900 | 0.5143 |
| 0.20 | 51 | 0.9653 | 0.5670 | 56.9% | 11 | 0.8849 | 0.5104 |
| 0.25 | 51 | 0.9676 | 0.5684 | 56.9% | 11 | 0.8823 | 0.5086 |
| 0.30 | 51 | 0.9721 | 0.5714 | 54.9% | 11 | 0.8824 | 0.5087 |
| 0.35 | 51 | 0.9790 | 0.5759 | 54.9% | 11 | 0.8852 | 0.5108 |
| 0.40 | 51 | 0.9883 | 0.5819 | 54.9% | 11 | 0.8911 | 0.5150 |
| 0.45 | 51 | 1.0003 | 0.5896 | 54.9% | 11 | 0.9002 | 0.5211 |
| 0.50 | 51 | 1.0154 | 0.5988 | 54.9% | 11 | 0.9130 | 0.5293 |
| 0.55 | 51 | 1.0339 | 0.6095 | 54.9% | 11 | 0.9301 | 0.5394 |
| 0.60 | 51 | 1.0563 | 0.6218 | 54.9% | 11 | 0.9522 | 0.5515 |
| 0.65 | 51 | 1.0835 | 0.6357 | 54.9% | 11 | 0.9804 | 0.5656 |
| 0.70 | 51 | 1.1165 | 0.6511 | 54.9% | 11 | 1.0164 | 0.5818 |
| 0.75 | 51 | 1.1568 | 0.6681 | 54.9% | 11 | 1.0627 | 0.5999 |
| 0.80 | 51 | 1.2071 | 0.6866 | 54.9% | 11 | 1.1234 | 0.6200 |
| 0.85 | 51 | 1.2716 | 0.7067 | 54.9% | 11 | 1.2065 | 0.6421 |
| 0.90 | 51 | 1.3590 | 0.7284 | 54.9% | 11 | 1.3290 | 0.6663 |
| 0.95 | 51 | 1.4925 | 0.7516 | 54.9% | 11 | 1.5439 | 0.6924 |
| 1.00 | 51 | 3.9911 | 0.7764 | 54.9% | 11 | 9.8544 | 0.7205 |

Peso elegido solo con calibración: **0.15** (si existe muestra elegible); su holdout: n=11, log loss=0.8900, Brier=0.5143, exactitud=63.6%.
Como referencia retrospectiva, el mejor peso al evaluar toda la cohorte elegible es **0.20** (log loss 0.9511, Brier 0.5570); esta cifra está optimizada sobre la muestra evaluada y **no** debe interpretarse como rendimiento futuro ni usarse como estimación imparcial.

### Validación walk-forward

Para cada fecha de holdout se volvió a elegir el peso usando solo los partidos elegibles anteriores (mínimo 20); luego se evaluó esa fecha. Resultado acumulado: n=11, exactitud=63.6%, log loss=0.8872, Brier=0.5119, ECE=0.1523. Core en esos mismos partidos: exactitud=63.6%, log loss=0.9203, Brier=0.5379, ECE=0.2309.

Pesos seleccionados por fecha (ponderados por cantidad de partidos): 0.15: 1, 0.20: 10.

### Viabilidad de un peso dinámico por relevancia y probabilidades

Hay 88 partidos enlazados con una respuesta JEV válida (62 se fusionaron históricamente). Solo 8 tienen score de relevancia y 8 confianza JEV guardados. Por tanto, el histórico no permite simular honestamente una regla dinámica que dependa de ambas señales. Las corridas nuevas guardan score/gates y, desde este cambio, también persistirán la confianza JEV.

Como diagnóstico parcial, se agrupan todas las respuestas JEV válidas (incluidas las 23 que el gate histórico no mezcló) por su probabilidad máxima. La concentración no debe tratarse como confianza calibrada: una distribución extrema puede ser sobreconfianza.

| Grupo por máxima P_JEV | N | Log loss core | Log loss JEV | Log loss mezcla 100% | Brier core | Brier JEV | Brier mezcla 100% |
|---|---:|---:|---:|---:|---:|---:|---:|
| máxima P_JEV ≤ 0.70 | 9 | 1.1185 | 1.9132 | 1.9132 | 0.6800 | 0.9653 | 0.9653 |
| máxima P_JEV > 0.70 | 79 | 1.0063 | 7.6679 | 7.6679 | 0.5995 | 0.9179 | 0.9179 |

Lectura: probar una regla basada solo en lo extrema que sea P_JEV sería arriesgado. Para producción, primero se debe calibrar confianza/relevancia fuera de muestra; una opción de investigación es `w = w_max × g(relevancia, certeza/frescura de fuentes, calibración de confianza)`, con `w_max` conservador. La divergencia entre core y JEV puede activar revisión o limitar el peso, pero no debe anular automáticamente evidencia fuerte y verificable.

### Cobertura por categoría y liga

| Categoría JEV | N | Log loss core | Log loss final | Brier core | Brier final |
|---|---:|---:|---:|---:|---:|
| alineacion_nomina | 2 | 1.0043 | 1.0117 | 0.5994 | 0.6046 |
| cambio_tecnico | 1 | 1.2189 | 1.4421 | 0.7511 | 0.9174 |
| lesion_baja | 1 | 0.7897 | 0.5813 | 0.4477 | 0.2919 |
| rotacion | 1 | 0.7897 | 0.5813 | 0.4477 | 0.2919 |
| suspension_roja | 1 | 1.2189 | 1.4421 | 0.7511 | 0.9174 |

| Liga | N | Log loss core | Log loss final | Brier core | Brier final |
|---|---:|---:|---:|---:|---:|
| Jupiler Pro League (Belgium) | 4 | 1.0875 | 1.0875 | 0.6600 | 0.6600 |
| LaLiga (Spain) | 9 | 0.9126 | 0.9626 | 0.5344 | 0.5737 |
| Liga MX (Mexico) | 9 | 1.0860 | 1.1655 | 0.6615 | 0.7074 |
| Liga Profesional (Argentina) | 25 | 1.0287 | 1.1013 | 0.6145 | 0.6355 |
| Major League Soccer (USA) | 22 | 1.0746 | 1.0611 | 0.6475 | 0.6318 |
| Primera A (Colombia) | 4 | 0.9821 | 1.2071 | 0.5962 | 0.7498 |
| Primera División (Chile) | 5 | 0.9624 | 0.8305 | 0.5745 | 0.4932 |
| Serie A (Italy) | 5 | 1.0322 | 1.3249 | 0.6138 | 0.7829 |
| Série A (Brazil) | 4 | 0.8507 | 0.9138 | 0.4896 | 0.5214 |
| Série B (Brazil) | 14 | 1.0581 | 1.1996 | 0.6435 | 0.7344 |

## Lectura y límites

- La diferencia entre core y mezcla guardada describe el cambio observado tras activar JEV dentro de las predicciones que quedaron almacenadas; por sí sola no demuestra causalidad ni mejora estable.
- El peso simulado mezcla las probabilidades JEV ya generadas. No vuelve a ejecutar el algoritmo JEV ni cambia qué partidos superaron su filtro de relevancia.
- La muestra es corta y reciente, y los partidos sin resultado enlazado se excluyen. La selección del peso puede variar bastante con pocos casos; conviene acumular varias jornadas antes de adoptar un peso distinto.
- No se calcula ROI: faltan cuotas históricas completas y homogéneas para comparar las tres salidas en todos los encuentros; reconstruir apuestas solo desde picks seleccionados introduce sesgo de selección.
- La comparación sin forma reconstruye el mismo softmax con rating Impect y ventaja local, anulando `form_delta`; permite estimar la contribución de forma en predicciones guardadas. No hay contrafactuales históricos que aíslen cada KPI Impect ni cada categoría de noticias, así que esas contribuciones no se atribuyen por separado.
- Los parámetros `alpha`/`gamma_d` quedan capturados para ejecuciones nuevas. Las filas antiguas que no los guardaron se reconstruyen con el archivo de parámetros actual, así que la comparación sin forma de esas filas es aproximada.
- El walk-forward es exploratorio y la muestra corta puede producir pesos inestables. No se cambia el peso de producción hasta acumular más fechas y validar en un holdout posterior.

### Partidos usados

| Fecha | Partido | Resultado | Fuente | Peso JEV guardado |
|---|---|---|---|---:|
| 2026-10-02 | CA Independiente vs Instituto AC Córdoba | visita | impect | 0.70 |
| 2026-10-02 | FC São Paulo vs FC Santos | visita | impect | 0.70 |
| 2026-10-02 | Londrina EC (PR) vs Criciúma Esporte Clube | visita | impect | 0.70 |
| 2026-10-02 | São Bernardo FC (SP) vs Clube de Regatas Brasil (AL) | local | impect | 0.00 |
| 2026-10-02 | Universidad de Concepción vs Huachipato FC | visita | impect | 0.00 |
| 2026-10-03 | AC Goianiense vs América FC (MG) | local | impect | 0.70 |
| 2026-10-03 | Avaí FC (SC) vs Ceará SC | visita | impect | 0.70 |
| 2026-10-03 | Botafogo FC (SP) vs Vila Nova FC (GO) | visita | impect | 0.70 |
| 2026-10-03 | CA Boca Juniors vs Club Atlético Unión | local | impect | 0.70 |
| 2026-10-03 | CA Newell's Old Boys vs Club Atlético Lanús | empate | impect | 0.70 |
| 2026-10-03 | Club Atlético Tucumán vs CA Barracas Central | visita | impect | 0.70 |
| 2026-10-03 | Cuiabá EC (MT) vs AA Ponte Preta | local | impect | 0.70 |
| 2026-10-03 | Defensa y Justicia vs CA San Lorenzo de Almagro | local | impect | 0.70 |
| 2026-10-03 | Independiente Rivadavia vs Club de Gimnasia y Esgrima La Plata | empate | impect | 0.70 |
| 2026-10-04 | Argentinos Juniors vs Club Atlético Tigre | local | impect | 0.70 |
| 2026-10-05 | AA Estudiantes (Río Cuarto) vs Racing Club | visita | impect | 0.70 |
| 2026-10-05 | CD Riestra vs CA Central Córdoba (SdE) | empate | impect | 0.00 |
| 2026-10-05 | Vélez Sarsfield vs CA Platense | empate | impect | 0.00 |
| 2026-10-10 | Atlanta United FC vs FC Cincinnati | visita | impect | 0.70 |
| 2026-10-10 | CA Barracas Central vs CA Huracán | local | impect | 0.00 |
| 2026-10-10 | CA Central Córdoba (SdE) vs Club Estudiantes de La Plata | empate | impect | 0.70 |
| 2026-10-10 | Cercle Brügge vs RSC Anderlecht | visita | impect | 0.00 |
| 2026-10-10 | Charlotte FC vs FC Dallas | local | impect | 0.70 |
| 2026-10-10 | Chicago Fire FC vs New York City FC | local | impect | 0.70 |
| 2026-10-10 | Club Atlético Tigre vs CA Banfield | visita | impect | 0.70 |
| 2026-10-10 | Deportivo Pereira vs Deportivo Cali | empate | impect | 0.70 |
| 2026-10-10 | FC Barcelona vs FC Getafe | local | impect | 0.70 |
| 2026-10-10 | FC Juárez vs Club Tijuana | empate | impect | 0.00 |
| 2026-10-10 | Inter Miami CF vs D.C. United | local | impect | 0.00 |
| 2026-10-10 | New England Revolution vs Seattle Sounders FC | visita | impect | 0.70 |
| 2026-10-10 | New York Red Bulls vs San Diego FC | local | impect | 0.70 |
| 2026-10-10 | Orlando City SC vs Columbus Crew | local | impect | 0.70 |
| 2026-10-10 | Philadelphia Union vs Real Salt Lake City | local | impect | 0.70 |
| 2026-10-10 | Querétaro FC vs CF Atlante | local | impect | 0.00 |
| 2026-10-10 | RAAL La Louvière vs FC Brügge | visita | impect | 0.00 |
| 2026-10-10 | Real Madrid vs Villarreal CF | local | impect | 0.70 |
| 2026-10-10 | Tigres UANL vs Deportivo Toluca | local | impect | 0.70 |
| 2026-10-10 | Toronto FC vs CF Montréal | local | impect | 0.70 |
| 2026-10-11 | AC Goianiense vs Sport Recife | empate | impect_results | 0.00 |
| 2026-10-11 | Atlético Nacional vs Deportivo Pasto | local | manual | 0.00 |
| 2026-10-11 | Atlético de San Luis vs Santos Laguna | visita | manual | 0.00 |
| 2026-10-11 | Austin FC vs Nashville SC | visita | manual | 0.70 |
| 2026-10-11 | CA Platense vs Argentinos Juniors | empate | manual | 0.00 |
| 2026-10-11 | CA River Plate vs AA Estudiantes (Río Cuarto) | local | manual | 0.70 |
| 2026-10-11 | CA Rosario Central vs Independiente Rivadavia | local | manual | 0.70 |
| 2026-10-11 | CA San Lorenzo de Almagro vs CD Riestra | local | impect_results | 0.70 |
| 2026-10-11 | CF América vs CF Monterrey | empate | manual | 0.70 |
| 2026-10-11 | CF Pachuca vs Club Necaxa | empate | manual | 0.00 |
| 2026-10-11 | Colorado Rapids vs San Jose Earthquakes | empate | manual | 0.70 |
| 2026-10-11 | Coquimbo Unido vs CSD Colo Colo | local | manual | 0.70 |
| 2026-10-11 | Cuiabá EC (MT) vs Avaí FC (SC) | visita | impect_results | 0.00 |
| 2026-10-11 | Deportes Concepción vs Unión La Calera | empate | manual | 0.00 |
| 2026-10-11 | Deportes Limache vs CD Palestino | empate | manual | 0.00 |
| 2026-10-11 | EC Bahia vs Mirassol FC (SP) | local | impect | 0.70 |
| 2026-10-11 | EC Juventude vs São Bernardo FC (SP) | empate | impect | 0.70 |
| 2026-10-11 | Jaguares de Córdoba vs Deportes Tolima | local | impect | 0.70 |
| 2026-10-11 | KV Mechelen vs VV St. Truiden | local | impect | 0.00 |
| 2026-10-11 | Lazio Rom vs AC Monza | local | impect | 0.70 |
| 2026-10-11 | Los Angeles FC vs Vancouver Whitecaps FC | empate | impect | 0.70 |
| 2026-10-11 | Minnesota United FC vs Houston Dynamo FC | empate | impect | 0.00 |
| 2026-10-11 | Racing Santander vs FC Valencia | empate | impect | 0.70 |
| 2026-10-11 | Real Betis vs CA Osasuna | local | impect | 0.70 |
| 2026-10-11 | Sporting Kansas City vs Portland Timbers | empate | impect | 0.70 |
| 2026-10-11 | St. Louis CITY SC vs Los Angeles Galaxy | local | impect_results | 0.00 |
| 2026-10-11 | Standard Lüttich vs RSC Charleroi | visita | impect | 0.00 |
| 2026-10-11 | US Lecce vs FC Bologna | visita | impect | 0.70 |
| 2026-10-11 | US Sassuolo vs AC Mailand | empate | impect | 0.70 |
| 2026-10-12 | Chapecoense vs CA Paranaense | local | manual | 0.70 |
| 2026-10-12 | Club Atlético Lanús vs Vélez Sarsfield | local | manual | 0.70 |
| 2026-10-12 | Club Universidad de Chile vs CD Ñublense | local | manual | 0.70 |
| 2026-10-12 | Gimnasia y Esgrima de Mendoza vs CA Newell's Old Boys | local | manual | 0.70 |
| 2026-10-12 | Grêmio Novorizontino (SP) vs AA Ponte Preta | local | manual | 0.70 |
| 2026-10-12 | Levante UD vs FC Sevilla | empate | manual | 0.70 |
| 2026-10-13 | Athletic Club (MG) vs Londrina EC (PR) | empate | manual | 0.00 |
| 2026-10-13 | Botafogo FC (SP) vs Ceará SC | local | manual | 0.70 |
| 2026-10-14 | CA Sarmiento (Junin) vs CA River Plate | empate | api | 0.00 |
| 2026-10-14 | CF Montréal vs Atlanta United FC | visita | api | 0.00 |
| 2026-10-14 | Columbus Crew vs Charlotte FC | local | api | 0.70 |
| 2026-10-14 | D.C. United vs New York Red Bulls | local | api | 0.00 |
| 2026-10-14 | FC Cincinnati vs New England Revolution | visita | api | 0.00 |
| 2026-10-14 | Inter Miami CF vs New York City FC | local | api | 0.70 |
| 2026-10-14 | Toronto FC vs Orlando City SC | empate | api | 0.70 |
| 2026-10-15 | CA Huracán vs Club de Gimnasia y Esgrima La Plata | visita | impect | 0.00 |
| 2026-10-16 | CA Boca Juniors vs Club Atlético Talleres | local | impect | 0.00 |
| 2026-10-16 | CF Atlante vs UNAM Pumas | visita | impect_results | 0.00 |
| 2026-10-16 | Deportivo La Coruna vs Levante UD | empate | impect | 0.20 |
| 2026-10-16 | Frosinone Calcio vs US Sassuolo | local | impect | 0.20 |
| 2026-10-16 | Mirassol FC (SP) vs SC Internacional Porto Alegre | local | impect_results | 0.20 |
| 2026-10-18 | Atlético Bucaramanga vs CD América de Cali | empate | api | 0.00 |
| 2026-10-18 | Avaí FC (SC) vs Athletic Club (MG) | empate | api | 0.00 |
| 2026-10-18 | CA Osasuna vs Racing Santander | local | api | 0.20 |
| 2026-10-18 | CF Monterrey vs CF Pachuca | local | impect | 0.00 |
| 2026-10-18 | FC Cincinnati vs Columbus Crew | visita | api | 0.20 |
| 2026-10-18 | Juventus Turin vs Lazio Rom | visita | impect_results | 0.20 |
| 2026-10-18 | Racing Club vs CA Independiente | empate | api | 0.00 |
| 2026-10-18 | Real Madrid vs FC Sevilla | local | api | 0.20 |
| 2026-10-18 | Vila Nova FC (GO) vs Grêmio Novorizontino (SP) | empate | api | 0.00 |
| 2026-10-19 | CA Newell's Old Boys vs CA Central Córdoba (SdE) | local | api | 0.00 |
| 2026-10-19 | CA Sarmiento (Junin) vs CA Barracas Central | local | api | 0.00 |
| 2026-10-19 | CF Monterrey vs CF Pachuca | local | api | 0.00 |
| 2026-10-19 | FC Getafe vs Rayo Vallecano | empate | api | 0.00 |
