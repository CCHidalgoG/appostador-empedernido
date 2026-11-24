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

def evaluate_historical_performance(dataset_path=PROCESSED_DIR / "historical_fixtures_dataset.json"):
    fixtures = read_json(dataset_path)
    if fixtures is None:
        print(f"Dataset {dataset_path} not found in DuckDB.")
        return

    print(f"Loaded {len(fixtures)} total historical fixtures.")

    loader = ImpectTeamMetricsLoader()
    aligner = TeamAligner()
    predictor = SoccerPredictor()

    metrics_cache = {}
    align_cache = {}

    dataset_rows = []
    aligned_matches = 0
    skipped_matches = 0

    for item in fixtures:
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
            skipped_matches += 1
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
            skipped_matches += 1
            continue

        home_row = df_team.loc[home_imp]
        away_row = df_team.loc[away_imp]

        if isinstance(home_row, pd.DataFrame):
            home_row = home_row.iloc[0]
        if isinstance(away_row, pd.DataFrame):
            away_row = away_row.iloc[0]

        # Predict with Baseline Softmax model
        pred_base = predictor.predict_match(home_imp, away_imp, it_id, match_date=match_date)

        aligned_matches += 1

        feature_row = {
            "fixture_id": item.get("fixture", {}).get("id"),
            "date": match_date,
            "home_team": home_imp,
            "away_team": away_imp,
            "target": target_class,
            "target_num": target_num,
            "pred_base_H": pred_base["P_H"],
            "pred_base_D": pred_base["P_D"],
            "pred_base_A": pred_base["P_A"],
            
            # Explicit Home & Away Team Performance Features
            "is_home": 1.0,
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
            "home_goals": home_row.get("GOALS", 0.0),
            "away_goals": away_row.get("GOALS", 0.0),
            "home_expected_goals": home_row.get("EXPECTED_GOALS", 0.0),
            "away_expected_goals": away_row.get("EXPECTED_GOALS", 0.0),
            "home_suffered_defenders": home_row.get("SUFFERED_BYPASSED_DEFENDERS", 0.0),
            "away_suffered_defenders": away_row.get("SUFFERED_BYPASSED_DEFENDERS", 0.0),

            # Differential features
            "diff_squad_rating": home_row["squad_rating"] - away_row["squad_rating"],
            "diff_bypassed_opponents": home_row.get("BYPASSED_OPPONENTS", 0.0) - away_row.get("BYPASSED_OPPONENTS", 0.0),
            "diff_bypassed_defenders": home_row.get("BYPASSED_DEFENDERS", 0.0) - away_row.get("BYPASSED_DEFENDERS", 0.0),
            "diff_ball_win": home_row.get("BALL_WIN_NUMBER", 0.0) - away_row.get("BALL_WIN_NUMBER", 0.0),
            "diff_ball_loss": home_row.get("BALL_LOSS_NUMBER", 0.0) - away_row.get("BALL_LOSS_NUMBER", 0.0),
            "diff_goals": home_row.get("GOALS", 0.0) - away_row.get("GOALS", 0.0),
        }
        dataset_rows.append(feature_row)

    print(f"Aligned {aligned_matches} matches successfully. Skipped {skipped_matches}.")

    df = pd.DataFrame(dataset_rows)
    if df.empty:
        print("No valid matches aligned for evaluation.")
        return

    # Baseline performance evaluation
    base_preds_num = []
    base_probs = []
    for _, row in df.iterrows():
        probs = [row["pred_base_H"], row["pred_base_D"], row["pred_base_A"]]
        base_probs.append(probs)
        base_preds_num.append(np.argmax(probs))

    base_probs = np.array(base_probs)
    base_acc = accuracy_score(df["target_num"], base_preds_num)
    base_ll = log_loss(df["target_num"], base_probs)

    print("\n=======================================================")
    print(" 1. DESEMPEÑO MODELO BASELINE (SOFTMAX - SQUAD RATING)")
    print("=======================================================")
    print(f"Accuracy Baseline (1X2): {base_acc * 100:.2f}%")
    print(f"Log Loss Baseline:       {base_ll:.4f}")

    feature_cols = [
        "is_home",
        "home_squad_rating", "away_squad_rating",
        "home_bypassed_opponents", "away_bypassed_opponents",
        "home_bypassed_defenders", "away_bypassed_defenders",
        "home_ball_win", "away_ball_win",
        "home_ball_loss", "away_ball_loss",
        "home_goals", "away_goals",
        "home_expected_goals", "away_expected_goals",
        "home_suffered_defenders", "away_suffered_defenders",
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

    # 2. Logistic Regression
    lr_probs = np.zeros((len(df), 3))
    lr_preds = np.zeros(len(df))
    coefs_list = []

    for train_idx, val_idx in skf.split(X_scaled, y):
        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_va = X_scaled[val_idx]

        clf_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf_lr.fit(X_tr, y_tr)

        lr_probs[val_idx] = clf_lr.predict_proba(X_va)
        lr_preds[val_idx] = clf_lr.predict(X_va)
        coefs_list.append(clf_lr.coef_)

    lr_acc = accuracy_score(y, lr_preds)
    lr_ll = log_loss(y, lr_probs)

    print("\n=======================================================")
    print(" 2. DESEMPEÑO MODELO REGRESIÓN LOGÍSTICA MULTIVARIABLE")
    print("=======================================================")
    print(f"Accuracy Logistic Regression (1X2): {lr_acc * 100:.2f}%")
    print(f"Log Loss Logistic Regression:       {lr_ll:.4f}")

    # 3. XGBoost Classifier
    xgb_probs = np.zeros((len(df), 3))
    xgb_preds = np.zeros(len(df))
    xgb_importances_list = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]

        clf_xgb = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss"
        )
        clf_xgb.fit(X_tr, y_tr)

        xgb_probs[val_idx] = clf_xgb.predict_proba(X_va)
        xgb_preds[val_idx] = clf_xgb.predict(X_va)
        xgb_importances_list.append(clf_xgb.feature_importances_)

    xgb_acc = accuracy_score(y, xgb_preds)
    xgb_ll = log_loss(y, xgb_probs)

    print("\n=======================================================")
    print(" 3. DESEMPEÑO MODELO XGBOOST CLASSIFIER")
    print("=======================================================")
    print(f"Accuracy XGBoost (1X2): {xgb_acc * 100:.2f}%")
    print(f"Log Loss XGBoost:       {xgb_ll:.4f}")
    print(f"Mejora Absoluta vs Baseline: {(xgb_acc - base_acc) * 100:+.2f}%")

    # XGBoost Feature Importance Breakdown
    avg_xgb_imp = np.mean(xgb_importances_list, axis=0)

    xgb_imp_df = pd.DataFrame({
        "Variable": feature_cols,
        "Importancia_XGBoost": avg_xgb_imp
    }).sort_values(by="Importancia_XGBoost", ascending=False)

    print("\n=======================================================")
    print(" 4. ANÁLISIS DE IMPORTANCIA DE VARIABLES (XGBOOST)")
    print("=======================================================")
    print(xgb_imp_df.to_string(index=False))

    imp_records = json.loads(xgb_imp_df.to_json(orient="records"))
    summary = {
        "total_matches_evaluated": len(df),
        "baseline_accuracy": float(base_acc),
        "baseline_log_loss": float(base_ll),
        "logistic_regression_accuracy": float(lr_acc),
        "logistic_regression_log_loss": float(lr_ll),
        "xgboost_accuracy": float(xgb_acc),
        "xgboost_log_loss": float(xgb_ll),
        "xgboost_improvement_vs_baseline": float(xgb_acc - base_acc),
        "xgboost_feature_importance": imp_records
    }

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "evaluation_results.json")
    write_json(out_file, summary)

    print(f"\nEvaluación completada. Resultados guardados en {out_file}")

if __name__ == "__main__":
    evaluate_historical_performance()
