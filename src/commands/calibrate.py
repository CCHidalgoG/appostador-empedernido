"""
calibrate.py — Calibración MLE de parámetros del modelo Softmax y ventaja local por liga.

Optimizaciones implementadas:
  1. Calibración MLE de alpha y gamma_d (global) mediante scipy.optimize
  2. Ventaja local por liga (no un único valor global de 0.05)

Uso:
    python -m src.commands.calibrate
    → Persiste en processed.artifacts (`calibrated_params.json`)
"""

import os
import json
import numpy as np
from scipy.optimize import minimize

from src.apostador.aligner import TeamAligner
from src.apostador.impect_team_metrics import ImpectTeamMetricsLoader
from src.apostador.config import PROCESSED_DIR, ROOT_DIR, IMPECT_RAW_DIR
from src.apostador.duckdb_store import read_json, write_json
from src.apostador.recent_form import RecentFormLoader


# ─── Mapeo: country -> league_id Impect (para ventaja local por liga) ────────
COUNTRY_TO_LEAGUE_ID = {
    "Colombia":  239,
    "England":   39,
    "Italy":     135,
    "Spain":     140,
    "USA":       253,
    "Brazil":    71,
    "Argentina": 128,
    "Mexico":    262,
}


def softmax_predict(r_h, r_a, alpha, gamma_d, home_adv, form_delta=0.0):
    """Calcula P_H, P_D, P_A dados los parámetros del modelo Softmax (incluyendo forma reciente)."""
    x = (r_h - r_a) + home_adv + form_delta
    u_h = alpha * x
    u_a = -alpha * x
    u_d = gamma_d
    e_h = np.exp(u_h)
    e_a = np.exp(u_a)
    e_d = np.exp(u_d)
    s = e_h + e_a + e_d
    return e_h / s, e_d / s, e_a / s


def _eval_rows_nll(rows, alpha, gamma_d, home_adv_override=None):
    """Evalúa el Negative Log-Likelihood acumulado sobre filas alineadas usando form_delta as-of."""
    eps = 1e-9
    total_ll = 0.0
    for r in rows:
        ha = home_adv_override if home_adv_override is not None else r["home_adv"]
        p_h, p_d, p_a = softmax_predict(
            r["r_h"], r["r_a"], alpha, gamma_d, ha, r.get("form_delta", 0.0)
        )
        p_h = max(p_h, eps)
        p_d = max(p_d, eps)
        p_a = max(p_a, eps)
        result = r["result"]
        if result == 0:
            total_ll += np.log(p_h)
        elif result == 1:
            total_ll += np.log(p_d)
        else:
            total_ll += np.log(p_a)
    return -total_ll


def negative_log_likelihood(params, rows):
    """
    Función de coste: Log Loss negativo total sobre el dataset.
    params = [alpha, gamma_d]  (home_adv se calibra por separado por liga)
    """
    alpha, gamma_d = params
    # Restricciones de dominio
    if alpha <= 0 or alpha > 30:
        return 1e9
    return _eval_rows_nll(rows, alpha, gamma_d)


def run_calibration(
    dataset_path=None,
    americas_path=None,
    output_path=None
):
    """
    Pipeline principal de calibración:
    1. Carga datasets históricos
    2. Alinea equipos con Impect y extrae ratings
    3. Calibra alpha y gamma_d con MLE global
    4. Calibra home_advantage y gamma_d por liga
    5. Guarda resultados en JSON
    """
    if dataset_path is None:
        dataset_path = os.path.join(ROOT_DIR, "data", "processed", "historical_fixtures_dataset.json")
    if americas_path is None:
        americas_path = os.path.join(ROOT_DIR, "data", "processed", "americas_historical_fixtures.json")
    if output_path is None:
        output_path = PROCESSED_DIR / "calibrated_params.json"

    loader = ImpectTeamMetricsLoader(raw_impect_dir=str(IMPECT_RAW_DIR))
    aligner = TeamAligner()
    form_loader = RecentFormLoader()

    # ── 1. Cargar ambos datasets ─────────────────────────────────────────────
    all_fixtures = []
    for path in [dataset_path, americas_path]:
        fixtures = read_json(path)
        if fixtures is not None:
            all_fixtures.extend(fixtures)
    print(f"Total fixtures cargados para calibración: {len(all_fixtures)}")

    metrics_cache = {}
    align_cache = {}
    all_rows = []          # todos los partidos (calibración global)
    league_rows = {}       # partidos agrupados por país (calibración local)

    for item in all_fixtures:
        it_id = item.get("impect_iteration")
        if not it_id:
            continue
        match_date = item.get("fixture", {}).get("date", "")
        home_af = item.get("teams", {}).get("home", {}).get("name")
        away_af = item.get("teams", {}).get("away", {}).get("name")
        hg = item.get("goals", {}).get("home")
        ag = item.get("goals", {}).get("away")
        if hg is None or ag is None or not home_af or not away_af:
            continue

        result = 0 if hg > ag else (1 if hg == ag else 2)
        country = item.get("league", {}).get("country", item.get("country_name", "Unknown"))

        cache_key = (it_id, match_date[:10])
        if cache_key not in metrics_cache:
            metrics_cache[cache_key] = loader.load_team_features(it_id, target_date=match_date)
        df_team = metrics_cache[cache_key]
        if df_team.empty:
            continue

        squads = list(df_team.index)

        ak = (home_af, tuple(squads))
        if ak not in align_cache:
            align_cache[ak] = aligner.find_best_match(home_af, squads)
        home_imp, _ = align_cache[ak]

        ak2 = (away_af, tuple(squads))
        if ak2 not in align_cache:
            align_cache[ak2] = aligner.find_best_match(away_af, squads)
        away_imp, _ = align_cache[ak2]

        if not home_imp or not away_imp:
            continue

        home_row = df_team.loc[home_imp]
        away_row = df_team.loc[away_imp]
        if isinstance(home_row, type(df_team)):
            home_row = home_row.iloc[0]
        if isinstance(away_row, type(df_team)):
            away_row = away_row.iloc[0]

        r_h = float(home_row.get("squad_rating", 0.68))
        r_a = float(away_row.get("squad_rating", 0.68))
        form_res = form_loader.get_form_adjustment(
            home_af, away_af, match_date[:10] if match_date else None
        )
        form_delta = float(form_res.get("form_delta", 0.0))

        row = {"r_h": r_h, "r_a": r_a, "result": result, "home_adv": 0.05, "form_delta": form_delta}
        all_rows.append(row)

        if country not in league_rows:
            league_rows[country] = []
        league_rows[country].append(row)

    print(f"Partidos alineados exitosamente: {len(all_rows)}")
    if len(all_rows) < 50:
        print("Insuficientes datos para calibración. Abortando.")
        return

    # ── 3. Calibración MLE global de alpha y gamma_d ─────────────────────────
    print("\n[1/2] Calibrando alpha y gamma_d (MLE global)...")
    x0 = [5.0, -0.405]
    result_mle = minimize(
        negative_log_likelihood,
        x0,
        args=(all_rows,),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 5000}
    )
    alpha_cal = float(result_mle.x[0])
    gamma_d_cal = float(result_mle.x[1])

    # Log loss baseline (parámetros originales)
    ll_orig = negative_log_likelihood([5.0, -0.405], all_rows) / len(all_rows)
    ll_cal = result_mle.fun / len(all_rows)
    print(f"  Parámetros originales: alpha=5.0, gamma_d=-0.405 | Log Loss medio: {ll_orig:.5f}")
    print(f"  Parámetros calibrados: alpha={alpha_cal:.4f}, gamma_d={gamma_d_cal:.4f} | Log Loss medio: {ll_cal:.5f}")
    print(f"  Mejora en Log Loss: {ll_orig - ll_cal:+.5f}")

    # ── 4. Calibración conjunta de home_advantage y gamma_d por liga ─────────
    print("\n[2/2] Calibrando home_advantage y gamma_d por liga (optimización conjunta)...")
    home_adv_per_league = {}
    gamma_d_per_league = {}
    for country, rows in league_rows.items():
        if len(rows) < 50:
            print(f"  {country}: Insuficientes datos ({len(rows)}), usando default 0.05")
            home_adv_per_league[country] = 0.05
            gamma_d_per_league[country] = round(gamma_d_cal, 5)
            continue

        def neg_ll_league(params):
            ha, gd = params
            return _eval_rows_nll(rows, alpha_cal, gd, home_adv_override=ha)

        res_league = minimize(
            neg_ll_league, [0.05, gamma_d_cal],
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000}
        )
        ha_cal = float(res_league.x[0])
        gd_cal = float(res_league.x[1])
        home_adv_per_league[country] = round(ha_cal, 5)
        gamma_d_per_league[country] = round(gd_cal, 5)
        print(f"  {country:15} ({len(rows)} partidos): ha = {ha_cal:.4f}, gamma_d = {gd_cal:.4f}")

    # ── 5. Guardar resultados ─────────────────────────────────────────────────
    calibrated = {
        "alpha": round(alpha_cal, 5),
        "gamma_d": round(gamma_d_cal, 5),
        "gamma_d_per_league": gamma_d_per_league,
        "home_advantage_global": 0.05,
        "home_advantage_per_league": home_adv_per_league,
        "n_fixtures_used": len(all_rows),
        "log_loss_original": round(ll_orig, 6),
        "log_loss_calibrated": round(ll_cal, 6),
        "improvement": round(ll_orig - ll_cal, 6)
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_json(output_path, calibrated)

    print(f"\nCalibración completada. Guardado en: {output_path}")
    print(json.dumps(calibrated, indent=2, ensure_ascii=False))
    return calibrated


if __name__ == "__main__":
    run_calibration()
