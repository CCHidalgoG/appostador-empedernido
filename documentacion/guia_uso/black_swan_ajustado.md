# Black Swan ajustado — de P_base a P_adj (ejemplo 2026-09-03)

## 1. Idea en una línea

El Black Swan v1 multiplica `P_base` (estadística pura). El v2 multiplica
`P_adj` (estadística corregida por noticias vía JEV) y penaliza el riesgo.

## 2. Ejemplo numérico (fusión)

Base Softmax: `{local: 0.50, empate: 0.27, visita: 0.23}` (favorito: local).
Brave encuentra una noticia de baja del local con fuente y mención del equipo; JEV confirma relevancia (`noticias_relevantes=0.82`) y devuelve `choice={local:0.20, empate:0.20, visita:0.60}`. La confianza de Choice se registra para auditoría, pero no se trata como tasa de acierto ni determina el peso.

```
w_jev = 0.20  (gate local + confirmación JEV ≥ 0.50)
P_final = 0.80·P_base + 0.20·P_jev
         = {local: 0.440, empate: 0.256, visita: 0.304}
```

Sin noticia candidata o sin confirmación JEV, `P_final = P_base` (`w=0`).

La pierna "local" pasa de `EV = 0.50·cuota − 1` a `EV_adj = 0.44·cuota − 1`:
si la cuota era 2.10, el EV cae de +5% a −7.6%; el ajuste depende de la cuota y de los parámetros configurados.
Ese es el ajuste buscado: la noticia elimina piernas tramposas antes de combinar.

## 3. Corrida offline de referencia (2026-09-03, sin red)

```
python -m src.commands.black_swan_v2 --input 2026-09-03 --target-odds 30 \
  --max-legs 3 --bankroll 40000 --no-news --no-jev
→ 80 apuestas → 57 piernas EV+ → cuota 28.61x, P 10.44%, stake $4.00
→ report almacenado en DuckDB: `output.artifacts` (`reports/black_swan_v2_2026-09-26.md`)
```

Con red (`Brave + JEV`) el mismo comando sin flags recalcula `P_adj` por pierna:
espera menos piernas elegibles, cuota similar, `P_adj` menor pero más honesta,
algunas piernas vetadas y `w_medio > 0`.

> **Nota sobre `--target-odds`**: Para jornadas típicas con 4-10 piernas EV+, `--target-odds 700` es inalcanzable (requiere que el producto de cuotas individuales supere ~490x, el umbral del 70%). El rango práctico diario es **20-150**. Si el comando termina en `Sin combinación sobre la cuota mínima`, baja el objetivo. Considera `--target-odds 700` solo si el portafolio tiene 10+ piernas de cuotas individuales ≥3.5.

## 4. Cuándo confiar en el ajuste

- `decision_source=jev_priority`: hubo evidencia candidata y JEV la confirmó con relevancia ≥0.50; mezcla predeterminada 20/80.
- `decision_source=softmax`: no hubo evidencia pertinente, JEV no la confirmó o falló una API.
- El peso 0.20 y umbral 0.50 son parámetros operativos iniciales; comparar Log Loss/Brier por periodo antes de tratarlos como política calibrada.
- Muchas vetadas (>20%): jornada conmocionada; considera bajar `--target-odds`
  o no jugar el Black Swan ese día.
