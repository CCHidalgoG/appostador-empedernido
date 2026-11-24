import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.apostador.aligner import TeamAligner
from src.apostador.config import IMPECT_RAW_DIR, PROCESSED_DIR, REPORTS_DIR
from src.apostador.duckdb_store import read_json, report_writer, write_json
from src.apostador.impect_team_metrics import ImpectTeamMetricsLoader
from src.apostador.predictor import SoccerPredictor
from src.apostador.recent_form import RecentFormLoader


def generate_upcoming_portfolio(
    target_dates=None,
    bankroll=82000.0,
    kelly_factor=1.0,  # Full Kelly / Máxima Exposición
    max_stake_pct=0.25, # Tope máximo de 25% por partido individual
    max_total_exposure=1.0 # Exposición máxima total del 100% del banco
):
    """
    Genera el portafolio de apuestas para los partidos en el rango target_dates
    utilizando MÁXIMA EXPOSICIÓN DE KELLY (Full Kelly = 1.0) con normalización al 100% del banco.
    Por defecto analiza los próximos 5 días a partir de hoy.
    """
    if target_dates is None:
        today = datetime.now(timezone.utc).date()
        target_dates = [(today + timedelta(days=i)).isoformat() for i in range(5)]
    start_date = target_dates[0]
    end_date = target_dates[-1]

    print("==========================================================================")
    print(f" PORTAFOLIO DE APUESTAS PREDICTIVAS (MÁXIMA EXPOSICIÓN KELLY = {kelly_factor})")
    print(f" Banco Inicial: ${bankroll:,.2f} USD/COP | Rango: {start_date} al {end_date}")
    print("==========================================================================\n")

    loader = ImpectTeamMetricsLoader()
    aligner = TeamAligner()
    predictor = SoccerPredictor()
    form_loader = RecentFormLoader()

    # Prioridad de países según Accuracy empírico en 5-Fold Cross Validation
    COUNTRY_PRIORITY = {
        "soccer_usa_mls": {"country": "USA", "league": "MLS", "iteration_id": 1864, "accuracy": "53.64%", "model": "Regresión Logística"},
        "soccer_brazil_campeonato": {"country": "Brasil", "league": "Série A", "iteration_id": 1799, "accuracy": "52.89%", "model": "Regresión Logística"},
        "soccer_argentina_primera_division": {"country": "Argentina", "league": "Liga Profesional", "iteration_id": 1930, "accuracy": "51.59%", "model": "Regresión Logística"},
        "soccer_mexico_ligamx": {"country": "México", "league": "Liga MX", "iteration_id": 2176, "accuracy": "51.18%", "model": "Baseline Softmax"},
        "soccer_colombia_liga_dimayor": {"country": "Colombia", "league": "Liga Dimayor", "iteration_id": 1752, "accuracy": "48.84%", "model": "Random Forest"}
    }

    odds_dir = IMPECT_RAW_DIR.parent / "theoddsapi"
    upcoming_matches = []

    for sport_key, meta in COUNTRY_PRIORITY.items():
        odds_file = odds_dir / f"odds_{sport_key}.json"
        data = read_json(odds_file)
        if data is None:
            continue

        df_teams = loader.load_team_features(meta["iteration_id"], target_date=target_dates[0])
        if df_teams.empty:
            continue

        impect_squads = list(df_teams.index)
        align_cache: dict[str, str] = {}

        def get_aligned(name, _cache=align_cache, _squads=impect_squads):
            if name not in _cache:
                m, _ = aligner.find_best_match(name, _squads)
                _cache[name] = m
            return _cache[name]

        for match in data:
            commence_time = match.get("commence_time", "")
            match_date = commence_time[:10] if commence_time else ""
            if match_date not in target_dates:
                continue

            home_raw = match.get("home_team")
            away_raw = match.get("away_team")

            home_imp = get_aligned(home_raw)
            away_imp = get_aligned(away_raw)

            if not home_imp or not away_imp:
                continue

            # Get market odds from Pinnacle / Bet365 / First Available
            bookmakers = match.get("bookmakers", [])
            if not bookmakers:
                continue

            bm_selected = bookmakers[0]
            for bm in bookmakers:
                if bm.get("title", "").lower() in ["pinnacle", "bet365", "unibet"]:
                    bm_selected = bm
                    break

            markets = bm_selected.get("markets", [])
            if not markets:
                continue

            h2h_outcomes = markets[0].get("outcomes", [])
            odds_dict = {}
            for out in h2h_outcomes:
                if out["name"] == home_raw:
                    odds_dict["H"] = float(out["price"])
                elif out["name"] == away_raw:
                    odds_dict["A"] = float(out["price"])
                elif out["name"] == "Draw":
                    odds_dict["D"] = float(out["price"])

            if len(odds_dict) < 3:
                continue

            # Get recent form adjustment for home vs. away team
            form_result = form_loader.get_form_adjustment(home_raw, away_raw, match_date)
            form_delta = form_result["form_delta"]

            # Generate model probabilities (with calibrated params + form)
            pred = predictor.predict_match(
                home_imp, away_imp, meta["iteration_id"],
                match_date=match_date,
                league=meta["country"],
                form_delta=form_delta
            )

            p_h = pred["P_H"]
            p_d = pred["P_D"]
            p_a = pred["P_A"]

            # Compute Expected Value (EV) for each outcome
            ev_h = p_h * odds_dict["H"] - 1.0
            ev_d = p_d * odds_dict["D"] - 1.0
            ev_a = p_a * odds_dict["A"] - 1.0

            outcomes_meta = [
                ("H", "Home Win", p_h, odds_dict["H"], ev_h, home_raw),
                ("D", "Draw", p_d, odds_dict["D"], ev_d, "Empate"),
                ("A", "Away Win", p_a, odds_dict["A"], ev_a, away_raw)
            ]

            # Find best EV outcome
            best_outcome = max(outcomes_meta, key=lambda x: x[4])

            upcoming_matches.append({
                "country": meta["country"],
                "league": meta["league"],
                "country_accuracy": meta["accuracy"],
                "optimal_model": meta["model"],
                "commence_time": commence_time,
                "date": match_date,
                "home_team": home_raw,
                "away_team": away_raw,
                "home_aligned": home_imp,
                "away_aligned": away_imp,
                "bookmaker": bm_selected.get("title", "Market Avg"),
                "odds_H": odds_dict["H"],
                "odds_D": odds_dict["D"],
                "odds_A": odds_dict["A"],
                "prob_H": p_h,
                "prob_D": p_d,
                "prob_A": p_a,
                "best_bet_code": best_outcome[0],
                "best_bet_desc": best_outcome[1],
                "best_bet_team": best_outcome[5],
                "best_prob": best_outcome[2],
                "best_odds": best_outcome[3],
                "best_ev": best_outcome[4]
            })

    df_matches = pd.DataFrame(upcoming_matches)
    if df_matches.empty:
        print(f"No se encontraron partidos para el rango {target_dates}.")
        return

    # Sort matches by Country Predictability Rank and then EV
    rank_order = {"USA": 1, "Brasil": 2, "Argentina": 3, "México": 4, "Colombia": 5}
    df_matches["country_rank"] = df_matches["country"].apply(lambda x: rank_order.get(x, 99))
    df_matches = df_matches.sort_values(by=["country_rank", "best_ev"], ascending=[True, False]).reset_index(drop=True)

    records: list[dict] = df_matches.to_dict(orient="records")

    # 1. Calculate Full Kelly Fraction for each match
    raw_fractions: list[float] = []
    for record in records:
        p = float(record["best_prob"])
        b = float(record["best_odds"]) - 1.0
        ev = float(record["best_ev"])
        q = 1.0 - p

        if b > 0 and ev > 0.015:
            full_kelly = (p * b - q) / b
            stake_pct = min(full_kelly * kelly_factor, max_stake_pct)
        else:
            stake_pct = 0.0
        raw_fractions.append(stake_pct)

    sum_fractions = sum(raw_fractions)

    # 2. Normalize portfolio exposure if total exceeds max_total_exposure
    if sum_fractions > max_total_exposure:
        norm_factor = max_total_exposure / sum_fractions
        final_fractions = [f * norm_factor for f in raw_fractions]
        is_normalized = True
    else:
        final_fractions = raw_fractions
        is_normalized = False

    portfolio_bets = []
    total_allocated = 0.0

    for i, record in enumerate(records):
        stake_pct = final_fractions[i]
        stake_amount = bankroll * stake_pct
        ev = float(record["best_ev"])

        if stake_amount > 0:
            total_allocated += stake_amount

        portfolio_bets.append({
            "rank": i + 1,
            "country": record["country"],
            "country_accuracy": record["country_accuracy"],
            "optimal_model": record["optimal_model"],
            "date": record["date"],
            "home_team": record["home_team"],
            "away_team": record["away_team"],
            "match": f"{record['home_team']} vs {record['away_team']}",
            "bookmaker": record["bookmaker"],
            "recommended_bet": record["best_bet_desc"],
            "outcome": record["best_bet_desc"],
            "selection": record["best_bet_team"],
            "model_probability": f"{record['best_prob']*100:.1f}%",
            "probability": record["best_prob"],
            "market_odds": record["best_odds"],
            "odds": record["best_odds"],
            "ev": round(ev, 6),
            "expected_value_pct": f"{ev*100:+.2f}%",
            "kelly_stake_pct": f"{stake_pct*100:.2f}%",
            "stake": round(stake_amount, 2),
            "stake_amount": round(stake_amount, 2),
            "expected_profit": round(stake_amount * ev, 2),
            "is_value_bet": stake_amount > 0
        })

    payload = {
        "target_dates": target_dates,
        "bankroll": bankroll,
        "kelly_factor": kelly_factor,
        "kelly_mode": "Full Kelly (Máxima Exposición)" if kelly_factor == 1.0 else f"Fractional Kelly ({kelly_factor})",
        "is_portfolio_normalized": is_normalized,
        "total_allocated": round(total_allocated, 2),
        "total_allocated_pct": f"{(total_allocated/bankroll)*100:.2f}%",
        "bets": portfolio_bets
    }

    # 1. Guardar el portfolio histórico en DuckDB
    json_dated_file = PROCESSED_DIR / f"portfolio_bets_{start_date}_{end_date}_max_kelly.json"
    write_json(json_dated_file, payload, run_type="max_kelly", run_date=start_date)

    # 2. Guardar el reporte Markdown en output.artifacts
    md_report_file = REPORTS_DIR / f"report_{start_date}_{end_date}_max_kelly.md"

    md_lines = []
    md_lines.append(f"# Reporte de Apuestas - Máxima Exposición Full Kelly ({start_date} al {end_date})\n")
    md_lines.append("## Parámetros de Máxima Exposición")
    md_lines.append(f"- **Banco Inicial:** ${bankroll:,.2f}")
    md_lines.append("- **Criterio de Kelly:** **Full Kelly 100% (Factor 1.0 - Máxima Exposición)**")
    md_lines.append(f"- **Exposición Total del Banco:** **${total_allocated:,.2f} ({(total_allocated/bankroll)*100:.2f}% del banco)**")
    md_lines.append(f"- **Normalización de Portafolio:** {'Sí (Topado a Máx 100% del Banco)' if is_normalized else 'No'}")
    md_lines.append(f"- **Total de Partidos Evaluados:** {len(portfolio_bets)}")
    md_lines.append(f"- **Apuestas de Valor Identificadas:** {sum(1 for b in portfolio_bets if b['is_value_bet'])}\n")

    md_lines.append("---")
    md_lines.append("## Detalle de Apuestas con Máxima Exposición Priorizadas por País\n")

    country_groups = {}
    for b in portfolio_bets:
        cnt = b["country"]
        if cnt not in country_groups:
            country_groups[cnt] = []
        country_groups[cnt].append(b)

    for cnt, bets_list in country_groups.items():
        meta = bets_list[0]
        md_lines.append(f"### {cnt} (Accuracy Modelo: {meta['country_accuracy']} | Modelo: {meta['optimal_model']})\n")
        md_lines.append("| Fecha | Partido | Apuesta Recomendada | Selección | Cuota Mercado | Prob. Modelo | Valor Esperado ($EV$) | Stake Máxima Exposición |")
        md_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for b in bets_list:
            if b["is_value_bet"]:
                md_lines.append(f"| {b['date']} | **{b['match']}** | **{b['recommended_bet']}** | {b['selection']} | **{b['market_odds']}** | {b['model_probability']} | **{b['expected_value_pct']}** | **${b['stake_amount']:,.2f}** ({b['kelly_stake_pct']}) |")
            else:
                md_lines.append(f"| {b['date']} | {b['match']} | {b['recommended_bet']} | {b['selection']} | {b['market_odds']} | {b['model_probability']} | {b['expected_value_pct']} | $0.00 (Sin valor) |")

        md_lines.append("\n")

    with report_writer(md_report_file) as f:
        f.write("\n".join(md_lines))

    print(f" Portafolio MÁXIMA EXPOSICIÓN KELLY Generado Exitosamente ({len(portfolio_bets)} partidos)")
    print(f" Capital Total Asignado: ${total_allocated:,.2f} ({(total_allocated/bankroll)*100:.2f}% del banco)")
    print(f" Datos guardados en DuckDB: {json_dated_file}")
    print(f" Reporte almacenado en DuckDB: output.artifacts ({md_report_file.name})\n")

def main():
    parser = argparse.ArgumentParser(
        description="Genera portafolio Full Kelly para un rango de fechas usando cuotas almacenadas en DuckDB")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Fecha de inicio YYYY-MM-DD (default: hoy)")
    parser.add_argument("--days", type=int, default=5,
                        help="Número de días a cubrir a partir de --start-date (default: 5)")
    parser.add_argument("--bankroll", type=float, default=82000.0,
                        help="Banco inicial en USD/COP")
    parser.add_argument("--kelly-factor", type=float, default=1.0,
                        help="Factor de Kelly (1.0 = Full Kelly, 0.25 = Fraccionario cuarto)")
    parser.add_argument("--max-stake-pct", type=float, default=0.25,
                        help="Tope máximo de stake por apuesta como fracción del banco (default: 0.25)")
    parser.add_argument("--max-exposure", type=float, default=1.0,
                        help="Exposición total máxima del banco (default: 1.0 = 100%%)")
    args = parser.parse_args()

    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()  # noqa: DTZ007
    else:
        start = datetime.now(timezone.utc).date()
    target_dates = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]

    generate_upcoming_portfolio(
        target_dates=target_dates,
        bankroll=args.bankroll,
        kelly_factor=args.kelly_factor,
        max_stake_pct=args.max_stake_pct,
        max_total_exposure=args.max_exposure,
    )


if __name__ == "__main__":
    main()
