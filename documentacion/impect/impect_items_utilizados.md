# Documentación de Ítems de Impect Utilizados en la Generación de Probabilidades

Este documento describe formalmente las variables y métricas provistas por **Impect** que alimentan el modelo probabilístico multi-capa (`SoccerPredictor` / `v2_tactical_xg_ensemble`), su control temporal *as-of*, el ajuste por disponibilidad de plantilla y la formulación matemática utilizada para estimar probabilidades 1X2.

---

## 1. Fuentes e Ítems de Impect Consumidos

1. **`ratings_{iteration_id}.json` (`value`, `date`, `squadId`, `squadName`)**:
   - **Squad Strength Rating** por equipo (rango típico `0.45`–`0.85`, neutro `0.68`), usado en la capa Softmax MLE y para fijar la fecha máxima observada en el control *as-of*.
2. **`predictions_{iteration_id}.json` (`predModelHome`, `predModelAway`, `scheduledDate`)**:
   - Goles esperados por partido ($\lambda_H, \lambda_A$) estimados por el modelo oficial de Impect, insumo de la capa **Dixon-Coles Poisson**.
3. **`averages_{iteration_id}.json` y `scores_{iteration_id}.json`**:
   - Promedios e índices tácticos por escuadra: `SHOT_XG` (o `EXPECTED_GOALS`), `CONCEDED_SHOT_XG`, `BYPASSED_DEFENDERS` y `FIELD_TILT`.
4. **`player_scores_{iteration_id}.json` (`playerName`, `firstname`, `lastname`, `matchShare`, `IMPECT_SCORE_PACKING`, `OFFENSIVE_IMPECT_SCORE_PACKING`, `DEFENSIVE_IMPECT_SCORE_PACKING`, `PXT`)**:
   - Rendimiento individual acumulado por 90 minutos y participación en plantilla (`matchShare`), consumido dinámicamente por `get_available_player_talent`.

---

## 2. Formulación Matemática Multi-Capa (`SoccerPredictor`)

### A. Capa 1: Softmax MLE de Ratings y Forma Reciente (`p_ratings`)

1. **Diferencial neto ajustado**:
   $$x = (R_H - R_A) + \text{home\_advantage}_{\text{liga}} + \text{form\_delta}$$
   donde `home_advantage_per_league` y `gamma_d_per_league` se calibran por MLE en `src/commands/calibrate.py` (incluyendo `form_delta` histórico en `softmax_predict`). Defaults globales vigentes: $\alpha = 4.322$, $\gamma_d = -0.168$, `home_advantage_global = 0.065`.
2. **Probabilidades Softmax**:
   $$u_H = \alpha \cdot x, \quad u_A = -\alpha \cdot x, \quad u_D = \gamma_{d,\text{liga}}$$
   $$P_i = \frac{e^{u_i}}{e^{u_H} + e^{u_A} + e^{u_D}}, \quad i \in \{H, D, A\}$$

### B. Capa 2: Poisson Bivariado Dixon-Coles Ajustado por Bajas (`p_poisson`)

Si `predModelHome` ($\lambda_H > 0$) y `predModelAway` ($\lambda_A > 0$) están disponibles, se atenúan por el impacto de ausencias calculado desde `player_scores`:
$$\lambda'_H = \max\bigl(0.05,\; \lambda_H \cdot (1 - 0.12 \cdot \text{home\_missing\_impact})\bigr)$$
$$\lambda'_A = \max\bigl(0.05,\; \lambda_A \cdot (1 - 0.12 \cdot \text{away\_missing\_impact})\bigr)$$
Con ellas se evalúa `_dixon_coles_probs(lh, la, rho=-0.06, max_goals=10)` corrigiendo marcadores bajos (`0-0`, `1-0`, `0-1`, `1-1`). Si no existen predicciones $\lambda$ válidas, `p_poisson = p_ratings`.

### C. Capa 3: Score Táctico y Talento Disponible (`p_tactical` + `get_available_player_talent`)

* **`get_available_player_talent`**:
  - Si hay alineación confirmada (`starting_xi`) con $\ge 7$ titulares emparejados contra `player_scores_{id}.json` (matching léxico, inicial+apellido y `SequenceMatcher >= 0.86`), promedia el talento de ese XI.
  - Si solo hay reporte de bajas (`injured_players`), excluye a los jugadores lesionados antes de tomar el Top-11 por `matchShare` y calcula `missing_share_impact` $\in [0, 1]$ como la proporción perdida de `IMPECT_SCORE_PACKING * matchShare` respecto al Top-11 habitual.
* **`tactical_score`**: combina diferencial de rating (`0.35 * ΔR * 10`), `SHOT_XG` (`0.25`), `CONCEDED_SHOT_XG` (`0.20`), `BYPASSED_DEFENDERS / 20` (`0.10`), `FIELD_TILT` (`0.10`), talento Top-11 disponible (`0.10 * Δpacking * 2`), localía (`ha * 5`), `form_delta * 3` y `+ 1.5 * (away_missing_impact - home_missing_impact)`, proyectado vía Softmax (recortado en $[-5, 5]$).

### D. Ensamble y Contracción Bayesiana al Mercado (`_power_novig_probs`)

* **Ensamble**: `0.50 * P_poisson + 0.35 * P_tactical + 0.15 * P_ratings` cuando hay $\lambda$ (o `0.65 * P_tactical + 0.35 * P_ratings` si falta Poisson).
* **Shrinkage al mercado (`apply_market_shrinkage=True`)**: Mezcla `0.80 * P_ensemble + 0.20 * P_mkt_novig`, eliminando el vigorish del consenso mediante `_power_novig_probs` (resolver $\sum_{i \in \{H,D,A\}} (1/o_i)^k = 1$ por bisección para corregir el *favorite-longshot bias*).

---

## 3. Control Temporal (*Decay* de Ratings y Bloqueo Anti-Leakage)

1. **Ratings por fecha (`load_ratings`, $> 7$ días)**:
   Si `days_diff = (t_date - closest_date).days > 7`, el rating se atenúa exponencialmente hacia la media de liga ($\mu_{\text{squad}}$) con factor $e^{-0.05 \cdot (\text{days\_diff} - 7)}$.
2. **Métricas agregadas de temporada (`_tactical_decay_factor`, $> 14$ días *as-of*)**:
   Dado que `averages_{it}.json`, `scores_{it}.json` y `player_scores_{it}.json` son acumulados de temporada sin marca temporal por partido, cuando se evalúa una `target_date` histórica anterior por más de 14 días a la fecha máxima observada en `ratings_{it}.json` (`days_ahead = (max_date - t_date).days > 14`), `_attenuate_lookup` contrae todas las métricas numéricas hacia la media de liga con $e^{-0.05 \cdot (\text{days\_ahead} - 14)}$. En ejecuciones vigentes (`days_ahead <= 14`), el factor es exactamente `1.0`.
