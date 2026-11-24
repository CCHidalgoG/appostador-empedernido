import os
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.apostador.aligner import TeamAligner
from src.apostador.impect_team_metrics import ImpectTeamMetricsLoader
from src.apostador.predictor import SoccerPredictor
from src.apostador.config import PROCESSED_DIR
from src.apostador.duckdb_store import read_json, write_json

def evaluate_by_country(dataset_path=PROCESSED_DIR / "americas_historical_fixtures.json"):
    fixtures = read_json(dataset_path)
    if fixtures is None:
        print(f"Dataset {dataset_path} not found in DuckDB.")
        return

    print(f"Loaded {len(fixtures)} total historical fixtures across Americas leagues.")

    loader = ImpectTeamMetricsLoader()
    aligner = TeamAligner()
    predictor = SoccerPredictor()

    metrics_cache = {}
    align_cache = {}

    # Group fixtures by country
    country_fixtures = {}
    for item in fixtures:
        country = item.get("country_name", "Unknown")
        if country not in country_fixtures:
            country_fixtures[country] = []
        country_fixtures[country].append(item)

    country_summary_dict = {}

    print("\n=======================================================")
    print(" EVALUACIÓN COMPARATIVA Y SELECCIÓN DE MODELO POR PAÍS")
    print("=======================================================\n")

    for country, country_items in country_fixtures.items():
        print(f"--- Procesando País: {country} ({len(country_items)} partidos) ---")

        dataset_rows = []
        aligned_count = 0
        skipped_count = 0

        for item in country_items:
            it_id = item.get("impect_iteration")
            if not it_id:
                continue

            match_date = item.get("fixture", {}).get("date")
            home_af = item.get("teams", {}).get("home", {}).get("name")
            away_af = item.get("teams", {}).get("away", {}).get("name")

            home_goals = item.get("goals", {}).get("home")
            away_goals = item.get("goals", {}).get("away")

            if home_goals is None or away_goals is None:
                continue

            if home_goals > away_goals:
                target_class = "H"
                target_num = 0
            elif home_goals == away_goals:
                target_class = "D"
                target_num = 1
            else:
                target_class = "A"
                target_num = 2

            cache_key = (it_id, match_date[:10] if match_date else None)
            if cache_key not in metrics_cache:
                df_team = loader.load_team_features(it_id, target_date=match_date)
                metrics_cache[cache_key] = df_team
            else:
                df_team = metrics_cache[cache_key]

            if df_team.empty:
                skipped_count += 1
                continue

            impect_squads = list(df_team.index)

            def cached_align(name):
                k = (name, tuple(impect_squads))
                if k not in align_cache:
                    align_cache[k] = aligner.find_best_match(name, impect_squads)
                return align_cache[k]

            home_imp, h_score = cached_align(home_af)
            away_imp, a_score = cached_align(away_af)

            if not home_imp or not away_imp:
                skipped_count += 1
                continue

            home_row = df_team.loc[home_imp]
            away_row = df_team.loc[away_imp]

            if isinstance(home_row, pd.DataFrame):
                home_row = home_row.iloc[0]
            if isinstance(away_row, pd.DataFrame):
                away_row = away_row.iloc[0]

            pred_base = predictor.predict_match(home_imp, away_imp, it_id, match_date=match_date)
            aligned_count += 1

            feature_row = {
                "fixture_id": item.get("fixture", {}).get("id"),
                "date": match_date,
                "target_num": target_num,
                "pred_base_H": pred_base["P_H"],
                "pred_base_D": pred_base["P_D"],
                "pred_base_A": pred_base["P_A"],
                "home_goals_real": home_goals,
                "away_goals_real": away_goals,
                "home_squad_rating": home_row["squad_rating"],
                "away_squad_rating": away_row["squad_rating"],
                "home_bypassed_opponents": home_row.get("BYPASSED_OPPONENTS", 0.0),
                "away_bypassed_opponents": away_row.get("BYPASSED_OPPONENTS", 0.0),
                "home_bypassed_defenders": home_row.get("BYPASSED_DEFENDERS", 0.0),
                "away_bypassed_defenders": away_row.get("BYPASSED_DEFENDERS", 0.0),
                "home_ball_win": home_row.get("BALL_WIN_NUMBER", 0.0),
                "away_ball_win": away_row.get("BALL_WIN_NUMBER", 0.0),
                "home_ball_loss": home_row.get("BALL_LOSS_NUMBER", 0.0),
                "away_ball_loss": away_row.get("BALL_LOSS_NUMBER", 0.0),
                "home_match_share": home_row.get("MATCH_SHARE", 50.0),
                "away_match_share": away_row.get("MATCH_SHARE", 50.0),
                "diff_squad_rating": home_row["squad_rating"] - away_row["squad_rating"],
                "diff_bypassed_opponents": home_row.get("BYPASSED_OPPONENTS", 0.0) - away_row.get("BYPASSED_OPPONENTS", 0.0),
                "diff_bypassed_defenders": home_row.get("BYPASSED_DEFENDERS", 0.0) - away_row.get("BYPASSED_DEFENDERS", 0.0),
                "diff_ball_win": home_row.get("BALL_WIN_NUMBER", 0.0) - away_row.get("BALL_WIN_NUMBER", 0.0),
                "diff_ball_loss": home_row.get("BALL_LOSS_NUMBER", 0.0) - away_row.get("BALL_LOSS_NUMBER", 0.0),
                "diff_goals": home_row.get("GOALS", 0.0) - away_row.get("GOALS", 0.0),
            }
            dataset_rows.append(feature_row)

        df = pd.DataFrame(dataset_rows)
        if df.empty or len(df) < 50:
            print(f"Insuficientes partidos alineados para {country} ({len(df)} partidos). Saltando.")
            continue

        print(f"  Partidos alineados con éxito en {country}: {len(df)}")

        # League Tactical Averages
        avg_total_goals = float((df["home_goals_real"] + df["away_goals_real"]).mean())
        avg_ball_wins = float((df["home_ball_win"] + df["away_ball_win"]).mean() / 2.0)
        avg_ball_losses = float((df["home_ball_loss"] + df["away_ball_loss"]).mean() / 2.0)
        avg_match_share = float((df["home_match_share"] + df["away_match_share"]).mean() / 2.0)
        avg_bypassed_defenders = float((df["home_bypassed_defenders"] + df["away_bypassed_defenders"]).mean() / 2.0)

        # Baseline evaluation
        base_preds_num = [np.argmax([r["pred_base_H"], r["pred_base_D"], r["pred_base_A"]]) for _, r in df.iterrows()]
        base_probs = df[["pred_base_H", "pred_base_D", "pred_base_A"]].values
        base_acc = float(accuracy_score(df["target_num"], base_preds_num))
        base_ll = float(log_loss(df["target_num"], base_probs))

        feature_cols = [
            "diff_squad_rating",
            "diff_bypassed_opponents",
            "diff_bypassed_defenders",
            "diff_ball_win",
            "diff_ball_loss",
            "diff_goals"
        ]

        X = df[feature_cols].values
        y = df["target_num"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 1. Logistic Regression
        lr_probs = np.zeros((len(df), 3))
        lr_preds = np.zeros(len(df))
        for tr_idx, va_idx in skf.split(X_scaled, y):
            clf_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf_lr.fit(X_scaled[tr_idx], y[tr_idx])
            lr_probs[va_idx] = clf_lr.predict_proba(X_scaled[va_idx])
            lr_preds[va_idx] = clf_lr.predict(X_scaled[va_idx])
        lr_acc = float(accuracy_score(y, lr_preds))
        lr_ll = float(log_loss(y, lr_probs))

        # 2. Random Forest
        rf_probs = np.zeros((len(df), 3))
        rf_preds = np.zeros(len(df))
        for tr_idx, va_idx in skf.split(X, y):
            clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            clf_rf.fit(X[tr_idx], y[tr_idx])
            rf_probs[va_idx] = clf_rf.predict_proba(X[va_idx])
            rf_preds[va_idx] = clf_rf.predict(X[va_idx])
        rf_acc = float(accuracy_score(y, rf_preds))
        rf_ll = float(log_loss(y, rf_probs))

        # 3. XGBoost
        xgb_probs = np.zeros((len(df), 3))
        xgb_preds = np.zeros(len(df))
        xgb_imp_list = []
        for tr_idx, va_idx in skf.split(X, y):
            clf_xgb = xgb.XGBClassifier(
                n_estimators=80, learning_rate=0.03, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                num_class=3, random_state=42, eval_metric="mlogloss"
            )
            clf_xgb.fit(X[tr_idx], y[tr_idx])
            xgb_probs[va_idx] = clf_xgb.predict_proba(X[va_idx])
            xgb_preds[va_idx] = clf_xgb.predict(X[va_idx])
            xgb_imp_list.append(clf_xgb.feature_importances_)
        xgb_acc = float(accuracy_score(y, xgb_preds))
        xgb_ll = float(log_loss(y, xgb_probs))

        # Model comparison table for country
        candidate_models = {
            "Baseline_Softmax": {"accuracy": float(base_acc), "log_loss": float(base_ll)},
            "Logistic_Regression": {"accuracy": float(lr_acc), "log_loss": float(lr_ll)},
            "Random_Forest": {"accuracy": float(rf_acc), "log_loss": float(rf_ll)},
            "XGBoost": {"accuracy": float(xgb_acc), "log_loss": float(xgb_ll)}
        }

        # Find best model (highest accuracy, lowest log_loss as tie-breaker)
        best_name = max(candidate_models.keys(), key=lambda k: (candidate_models[k]["accuracy"], -candidate_models[k]["log_loss"]))
        best_stats = candidate_models[best_name]
        avg_xgb_imp = np.mean(xgb_imp_list, axis=0)
        xgb_imp_series = pd.Series(avg_xgb_imp, index=feature_cols)
        xgb_imp_dict = json.loads(xgb_imp_series.to_json())

        country_summary_dict[country] = {
            "country_name": country,
            "total_matches_evaluated": len(df),
            "best_model_name": best_name,
            "best_model_accuracy": float(best_stats["accuracy"]),
            "best_model_log_loss": float(best_stats["log_loss"]),
            "candidate_models_metrics": candidate_models,
            "league_tactical_averages": {
                "goals_per_match": round(float(avg_total_goals), 2),
                "avg_match_share_possession": round(float(avg_match_share), 2),
                "avg_ball_recoveries": round(float(avg_ball_wins), 2),
                "avg_ball_losses": round(float(avg_ball_losses), 2),
                "avg_bypassed_defenders": round(float(avg_bypassed_defenders), 2),
                "estimated_fouls_per_match": round(float(avg_ball_wins * 0.28), 1),
                "estimated_yellow_cards_per_match": round(float(avg_total_goals * 1.8 + 1.2), 1)
            },
            "xgb_feature_importance": xgb_imp_dict
        }

        print(f"  --> Mejor Modelo para {country}: {best_name} (Accuracy: {best_stats['accuracy']*100:.2f}%, Log Loss: {best_stats['log_loss']:.4f})\n")

    out_file = PROCESSED_DIR / "country_best_models.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(f"Writing country_summary_dict with {len(country_summary_dict)} countries to {out_file}...")
    write_json(out_file, country_summary_dict)
    print(f"Successfully wrote {os.path.getsize(out_file)} bytes to {out_file}.")

    print("=======================================================")
    print(f"DICCIONARIO DE MODELOS POR PAÍS GENERADO EN {out_file}")
    print("=======================================================")

if __name__ == "__main__":
    evaluate_by_country()
