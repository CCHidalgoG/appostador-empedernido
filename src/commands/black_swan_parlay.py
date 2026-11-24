import json
import itertools
from datetime import datetime, timedelta
from src.apostador.optimizer import KellyOptimizer
from src.apostador.config import PROCESSED_DIR, REPORTS_DIR
from src.apostador.duckdb_store import read_json, write_json, write_text, get_store

def load_value_bets_from_report(report_path):
    """
    Carga las apuestas de valor desde el reporte generado por main.py
    """
    data = read_json(report_path)
    if data is None:
        raise FileNotFoundError(report_path)
    
    value_bets = []
    for bet in data['value_bets']:
        value_bets.append({
            'home_team': bet['home_team'],
            'away_team': bet['away_team'],
            'outcome': bet['market_outcome'],
            'probability': bet['probability'],
            'odds': bet['odds'],
            'ev': bet['ev']
        })
    
    return value_bets

def generate_black_swan_parlay(value_bets, target_odds=700, bankroll=40000, max_legs=6):
    """
    Genera una apuesta combinada (parlay) que de aproximadamente la cuota objetivo,
    maximizando la probabilidad de éxito.
    
    Args:
        value_bets: Lista de apuestas de valor disponibles
        target_odds: Cuota objetivo (default 700x)
        bankroll: Banco total
        max_legs: Máximo número de piernas en el parlay
    """
    
    print(f"Generando 'Black Swan Parlay' con cuota objetivo ~{target_odds}x")
    print(f"Apuestas de valor disponibles: {len(value_bets)}")
    
    # Agrupar apuestas por partido para evitar múltiples selecciones del mismo partido
    from collections import defaultdict
    bets_by_match = defaultdict(list)
    for bet in value_bets:
        match_key = f"{bet['home_team']} vs {bet['away_team']}"
        bets_by_match[match_key].append(bet)
    
    print(f"Partidos únicos disponibles: {len(bets_by_match)}")
    
    # Buscar la mejor combinación que maximize probabilidad con cuota >= target_odds * 0.7
    min_acceptable_odds = target_odds * 0.7  # Aceptamos desde 490x para el objetivo de 700x
    best_parlay = None
    best_score = 0  # Score = probabilidad - penalización por distancia al target
    
    # Generar combinaciones válidas (una apuesta por partido máximo)
    match_keys = list(bets_by_match.keys())
    
    for n_legs in range(2, min(max_legs + 1, len(match_keys) + 1)):
        print(f"\n--- Probando combinaciones de {n_legs} piernas ---")
        
        # Combinaciones de partidos
        match_combinations = list(itertools.combinations(match_keys, n_legs))
        print(f"Total combinaciones de partidos: {len(match_combinations)}")
        
        # Para cada combinación de partidos, generar todas las permutaciones de apuestas
        for match_combo in match_combinations:
            # Generar todas las combinaciones posibles de apuestas para estos partidos
            bet_options = [bets_by_match[match] for match in match_combo]
            bet_combinations = list(itertools.product(*bet_options))
            
            for combo in bet_combinations:
                # Calcular cuota y probabilidad del parlay
                parlay_odds = 1.0
                parlay_prob = 1.0
                
                for bet in combo:
                    parlay_odds *= bet['odds']
                    parlay_prob *= bet['probability']
                
                # Solo considerar combinaciones con cuota mínima aceptable
                if parlay_odds >= min_acceptable_odds:
                    # Score: maximizar probabilidad, pero preferir cuotas más cercanas al target
                    distance_penalty = abs(parlay_odds - target_odds) / target_odds
                    score = parlay_prob - (distance_penalty * 0.1)  # Penalización pequeña por distancia
                    
                    if score > best_score:
                        best_score = score
                        best_parlay = {
                            'legs': combo,
                            'total_odds': parlay_odds,
                            'total_probability': parlay_prob,
                            'n_legs': n_legs
                        }
                        print(f"Mejor combinación encontrada: {parlay_odds:.2f}x (prob: {parlay_prob:.4f}, score: {score:.4f})")
    
    if not best_parlay:
        print("No se encontró combinación con cuota mínima aceptable. Buscando la más alta posible...")
        
        # Si no encontramos nada con la cuota mínima, buscar la cuota más alta posible
        best_parlay = None
        max_odds = 0
        
        for n_legs in range(2, min(max_legs + 1, len(match_keys) + 1)):
            match_combinations = list(itertools.combinations(match_keys, n_legs))
            
            for match_combo in match_combinations:
                bet_options = [bets_by_match[match] for match in match_combo]
                bet_combinations = list(itertools.product(*bet_options))
                
                for combo in bet_combinations:
                    parlay_odds = 1.0
                    parlay_prob = 1.0
                    
                    for bet in combo:
                        parlay_odds *= bet['odds']
                        parlay_prob *= bet['probability']
                    
                    if parlay_odds > max_odds:
                        max_odds = parlay_odds
                        best_parlay = {
                            'legs': combo,
                            'total_odds': parlay_odds,
                            'total_probability': parlay_prob,
                            'n_legs': n_legs
                        }
    
    return best_parlay

def calculate_black_swan_stake(parlay, bankroll, kelly_factor=0.01, min_stake_amount=1.0):
    """
    Calcula el stake para una apuesta cisne negro usando Kelly fraccionario muy conservador.
    Para apuestas de alto riesgo, usamos un factor de Kelly muy pequeño (0.01 = 1%).
    """
    optimizer = KellyOptimizer(bankroll=bankroll, fraction_factor=kelly_factor, min_stake_amount=min_stake_amount)
    
    p = parlay['total_probability']
    odds = parlay['total_odds']
    
    f_star = optimizer.calculate_kelly_fraction(p, odds)
    f_frac = f_star * kelly_factor
    
    ev = (p * odds) - 1.0
    
    # Para apuestas cisne negro, establecer un stake mínimo fijo para que sea significativo
    min_stake_pct = 0.01  # 0.01% del banco como mínimo
    stake = max(f_frac * bankroll, bankroll * min_stake_pct / 100)
    stake = max(round(stake, 2), float(min_stake_amount))
    
    return {
        'kelly_pure': f_star,
        'kelly_fractional': f_frac,
        'stake': stake,
        'stake_pct': (stake / bankroll) * 100,
        'ev': ev,
        'payout': round(stake * odds, 2)
    }

def generate_black_swan_report(parlay, stake_info, bankroll, report_path):
    """Render the human-readable summary and persist it in output.artifacts."""
    from io import StringIO
    f = StringIO()
    f.write(f"# 🦢 Apuesta Cisne Negro - {datetime.now().strftime('%Y-%m-%d')}\n\n")
    f.write(f"**Cuota Objetivo**: ~700x | **Cuota Alcanzada**: {parlay['total_odds']:.2f}x\n")
    f.write(f"**Probabilidad de Éxito**: {parlay['total_probability']*100:.4f}%\n")
    f.write(f"**Número de Piernas**: {parlay['n_legs']}\n\n")
    f.write("## 🎯 Detalles de la Apuesta Combinada\n\n")
    f.write("| # | Partido | Apuesta | Cuota | Probabilidad |\n|---|---|---|---|---|\n")
    for i, leg in enumerate(parlay['legs'], 1):
        outcome = leg.get('outcome', leg.get('market_outcome', 'Unknown'))
        f.write(f"| {i} | {leg['home_team']} vs {leg['away_team']} | {outcome} | {leg['odds']:.2f} | {leg['probability']*100:.1f}% |\n")
    f.write(f"\n## 💰 Gestión del Banco\n\n**Banco Total**: ${bankroll:,.2f}\n")
    f.write(f"**Stake Recomendado**: ${stake_info['stake']:,.2f} ({stake_info['stake_pct']:.4f}% del banco)\n")
    f.write(f"**EV de la Apuesta**: {stake_info['ev']*100:.2f}%\n")
    f.write(f"**Ganancia Potencial**: ${stake_info['stake'] * parlay['total_odds']:,.2f}\n")
    f.write("\n## ⚠️ Análisis de Riesgo\n\n")
    prob_lose = (1 - parlay['total_probability']) * 100
    f.write(f"**Probabilidad de Pérdida Total**: {prob_lose:.4f}%\n")
    f.write(f"**Probabilidad de Ganancia Masiva**: {parlay['total_probability']*100:.4f}%\n")
    write_text(report_path, f.getvalue())

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Genera apuestas cisne negro (parlays de alta cuota)")
    parser.add_argument("--date", type=str, default=None, help="Fecha en formato YYYY-MM-DD (default: hoy)")
    parser.add_argument("--target-odds", type=float, default=700, help="Cuota objetivo para el parlay")
    parser.add_argument("--bankroll", type=float, default=40000, help="Banco total")
    parser.add_argument("--min-stake", type=float, default=1.0, help="Stake mínimo por apuesta en unidades monetarias")
    parser.add_argument("--max-legs", type=int, default=6, help="Máximo número de piernas en el parlay")
    
    args = parser.parse_args()
    
    # Determinar la fecha
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Cargar apuestas de valor del reporte más reciente
    
    # Cargar las apuestas de valor del reporte más reciente
    import os
    json_path = PROCESSED_DIR / f"portfolio_bets_{target_date}.json"
    
    # Verificar si existe el JSON de la fecha objetivo
    data = read_json(json_path)
    if data is None:
        print(f"No se encontró portfolio para {target_date}. Usando fecha de ayer...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        json_path = PROCESSED_DIR / f"portfolio_bets_{yesterday}.json"
        data = read_json(json_path)
    
    if data is None:
        print(f"ERROR: No se encontró el archivo JSON procesado para {target_date}")
        print(f"Buscando en: {json_path}")
        print("Por favor, ejecuta primero main.py para generar los datos de apuestas:")
        print(f"python -m src.commands.main --date {target_date} --bankroll {args.bankroll} --kelly-factor 1.0")
        return
    
    print(f"Cargando apuestas desde {json_path}...")

    value_bets = []
    for bet in data.get('value_bets', []):
        value_bets.append({
            'home_team': bet['home_team'],
            'away_team': bet['away_team'],
            'outcome': bet['market_outcome'],
            'probability': bet['probability'],
            'odds': bet['odds'],
            'ev': bet['ev']
        })
    
    # Generar parlay cisne negro
    parlay = generate_black_swan_parlay(value_bets, target_odds=args.target_odds, bankroll=args.bankroll, max_legs=args.max_legs)
    
    if parlay:
        print(f"\n✅ Parlay Cisne Negro encontrado:")
        print(f"Cuota: {parlay['total_odds']:.2f}x")
        print(f"Probabilidad: {parlay['total_probability']*100:.4f}%")
        print(f"Piernas: {parlay['n_legs']}")
        
        for i, leg in enumerate(parlay['legs'], 1):
            outcome_key = leg.get('outcome', leg.get('market_outcome', 'Unknown'))
            print(f"{i}. {leg['home_team']} vs {leg['away_team']} - {outcome_key} ({leg['odds']:.2f}x, {leg['probability']*100:.1f}%)")
        
        # Calcular stake
        stake_info = calculate_black_swan_stake(parlay, bankroll=args.bankroll, kelly_factor=0.001, min_stake_amount=args.min_stake)
        
        print(f"\n💰 Gestión del Banco:")
        print(f"Stake recomendado: ${stake_info['stake']:,.2f} ({stake_info['stake_pct']:.4f}%)")
        print(f"Ganancia potencial: ${stake_info['stake'] * parlay['total_odds']:,.2f}")
        print(f"EV: {stake_info['ev']*100:.2f}%")
        
        # Generar reporte
        report_output = REPORTS_DIR / f"black_swan_parlay_{target_date}.md"
        generate_black_swan_report(parlay, stake_info, args.bankroll, report_output)
        output = {"parlay": parlay, "stake": stake_info, "legs": parlay.get("legs", [])}
        write_json(PROCESSED_DIR / f"black_swan_parlay_{target_date}.json", output,
                   run_type="black_swan_parlay", run_date=target_date)
        print(f"\n📊 Resultado guardado en DuckDB: output.recommendations (run {target_date})")
    else:
        print("No se pudo generar un parlay con las apuestas disponibles")

if __name__ == "__main__":
    main()