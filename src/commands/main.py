from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from src.apostador.jev_decision import JevDecisionClient, build_match_state, fallback_decision
from src.apostador.news_search import BraveNewsClient
from src.apostador.probability_fusion import JEV_CALIBRATED_WEIGHT, fuse_probabilities

from src.apostador.aligner import TeamAligner
from src.apostador.apifootball_context import ApiFootballContextClient
from src.apostador.config import (
    APIFOOTBALL_RAW_DIR,
    IMPECT_RAW_DIR,
    ODDS_RAW_DIR,
    PROCESSED_DIR,
    ensure_runtime_directories,
)
from src.apostador.duckdb_store import get_store, read_json, write_json
from src.apostador.impect_team_metrics import ImpectTeamMetricsLoader
from src.apostador.optimizer import KellyOptimizer
from src.apostador.predictor import SoccerPredictor, _power_novig_probs
from src.apostador.recent_form import RecentFormLoader

# Load environment variables
load_dotenv()

USERNAME_IMPECT = os.getenv("USERNAME_IMPECT")
PASSWORD_IMPECT = os.getenv("PASSWORD_IMPECT")
APIFOOTBALL_KEY = os.getenv("APIFOOTBALL")
ODDSAPI_KEY = os.getenv("ODDSAPI")

# Setup directories
RAW_DIR = APIFOOTBALL_RAW_DIR.parent
ensure_runtime_directories()

# Supported Leagues Mapping
# API Football League ID -> {Impect iteration ID, Odds API sport key, country}
SUPPORTED_LEAGUES = {
    239: {
        "name": "Primera A (Colombia)",
        "country": "Colombia",
        "impect_iteration": 1752,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    262: {
        "name": "Liga MX (Mexico)",
        "country": "Mexico",
        "impect_iteration": 2176,
        "odds_sport_key": "soccer_mexico_ligamx"
    },
    71: {
        "name": "Série A (Brazil)",
        "country": "Brazil",
        "impect_iteration": 1799,
        "odds_sport_key": "soccer_brazil_campeonato"
    },
    253: {
        "name": "Major League Soccer (USA)",
        "country": "USA",
        "impect_iteration": 1864,
        "odds_sport_key": "soccer_usa_mls"
    },
    128: {
        "name": "Liga Profesional (Argentina)",
        "country": "Argentina",
        "impect_iteration": 1930,
        "odds_sport_key": "soccer_argentina_primera_division"
    },
    140: {
        "name": "LaLiga (Spain)",
        "country": "Spain",
        "impect_iteration": 2110,
        "odds_sport_key": "soccer_spain_la_liga"
    },
    135: {
        "name": "Serie A (Italy)",
        "country": "Italy",
        "impect_iteration": 2111,
        "odds_sport_key": "soccer_italy_serie_a"
    },
    144: {
        "name": "Jupiler Pro League (Belgium)",
        "country": "Belgium",
        "impect_iteration": 2143,
        "odds_sport_key": "soccer_belgium_first_div"
    },
    235: {
        "name": "Premier Liga (Russia)",
        "country": "Russia",
        "impect_iteration": 2155,
        "odds_sport_key": "soccer_russia_premier_league"
    },
    72: {
        "name": "Série B (Brazil)",
        "country": "Brazil",
        "impect_iteration": 1929,
        "odds_sport_key": "soccer_brazil_serie_b"
    },
    265: {
        "name": "Primera División (Chile)",
        "country": "Chile",
        "impect_iteration": 1802,
        "odds_sport_key": "soccer_chile_campeonato",
    },
    281: {
        "name": "Liga 1 (Peru)",
        "country": "Peru",
        "impect_iteration": 1801,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    242: {
        "name": "Liga Pro Serie A (Ecuador)",
        "country": "Ecuador",
        "impect_iteration": 1863,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    268: {
        "name": "Primera Division (Uruguay)",
        "country": "Uruguay",
        "impect_iteration": 1813,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    252: {
        "name": "Division Profesional (Paraguay)",
        "country": "Paraguay",
        "impect_iteration": 1794,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    344: {
        "name": "Division Profesional (Bolivia)",
        "country": "Bolivia",
        "impect_iteration": 1973,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    299: {
        "name": "Liga FUTVE (Venezuela)",
        "country": "Venezuela",
        "impect_iteration": 1800,
        "odds_sport_key": None,
        "odds_source": "apifootball",
    },
    39: {
        "name": "Premier League (England)",
        "country": "England",
        "impect_iteration": None,
        "odds_sport_key": "soccer_epl",
    },
    78: {
        "name": "Bundesliga (Germany)",
        "country": "Germany",
        "impect_iteration": None,
        "odds_sport_key": "soccer_germany_bundesliga",
    },
    61: {
        "name": "Ligue 1 (France)",
        "country": "France",
        "impect_iteration": None,
        "odds_sport_key": "soccer_france_ligue_one",
    },
    2: {
        "name": "UEFA Champions League",
        "country": "Europe",
        "impect_iteration": None,
        "odds_sport_key": "soccer_uefa_champs_league",
    },
    3: {
        "name": "UEFA Europa League",
        "country": "Europe",
        "impect_iteration": None,
        "odds_sport_key": "soccer_uefa_europa_league",
    },
    848: {
        "name": "UEFA Europa Conference League",
        "country": "Europe",
        "impect_iteration": None,
        "odds_sport_key": "soccer_uefa_europa_conference_league",
    },
    88: {
        "name": "Eredivisie (Netherlands)",
        "country": "Netherlands",
        "impect_iteration": None,
        "odds_sport_key": "soccer_netherlands_eredivisie",
    },
    94: {
        "name": "Primeira Liga (Portugal)",
        "country": "Portugal",
        "impect_iteration": None,
        "odds_sport_key": "soccer_portugal_primeira_liga",
    },
    40: {
        "name": "Championship (England)",
        "country": "England",
        "impect_iteration": None,
        "odds_sport_key": "soccer_efl_champ",
    },
    307: {
        "name": "Saudi Pro League (Saudi Arabia)",
        "country": "Saudi Arabia",
        "impect_iteration": None,
        "odds_sport_key": "soccer_saudi_arabia_pro_league",
    },
}

# --- Guard de cobertura de datos (incidente 2026-10-22) ---
# Un partido solo es apostable si el modelo tiene señal real: ratings Impect de ambos
# equipos y al menos MIN_FORM_MATCHES_FOR_BET partidos recientes por lado. Las ligas
# sin ratings (copas europeas, Eredivisie, Championship, Saudi Pro League...) caen al
# rating default 0.68 para ambos equipos y el Core Softmax degenera en un prior fijo
# de localía; JEV encima infla al local y los nichos disparan EVs falsos.
MIN_FORM_MATCHES_FOR_BET = 3


def implied_probabilities_no_vig(odds_map) -> dict | None:
    """Probabilidades implícitas 1X2 sin margen usando el método Power (corrigiendo favorite-longshot bias).

    Ancla de mercado para el filtro de divergencia modelo vs. consenso de bookmakers.
    Requiere las tres cuotas; devuelve None si están incompletas o corruptas.
    """
    if not isinstance(odds_map, dict):
        return None
    prices = {}
    inv_sum = 0.0
    for key in ("Home", "Draw", "Away"):
        try:
            price = float(odds_map.get(key))
        except (TypeError, ValueError):
            return None
        if price is None or price <= 1.0:
            return None
        prices[key] = price
        inv_sum += 1.0 / price
    if not 0.95 < inv_sum < 1.60:
        return None
    ph, pd_, pa = _power_novig_probs(prices["Home"], prices["Draw"], prices["Away"])
    return {"Home": ph, "Draw": pd_, "Away": pa}

def load_or_fetch_fixtures(target_date):
    """Load fixtures for the requested Bogotá calendar date."""
    timezone_name = "America/Bogota"
    cache_path = APIFOOTBALL_RAW_DIR / f"fixtures_{target_date}_{timezone_name.replace('/', '-')}.json"
    cached = read_json(cache_path)
    if cached is not None:
        return cached

    print(f"Cache miss: Fetching fixtures from API-Football for {target_date}...")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": APIFOOTBALL_KEY or ""}
    # The CLI dates are interpreted as Bogotá calendar dates. Request the same
    # timezone from API-Football so late-night UTC fixtures are not assigned to
    # the neighboring local day.
    params = {"date": target_date, "timezone": timezone_name}

    try:
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            data = resp.json()
            write_json(cache_path, data)
            return data
        else:
            print(f"Error fetching fixtures: {resp.text}")
    except Exception as e:  # noqa: BLE001
        print(f"Exception fetching fixtures: {e}")
    return {"response": []}

def snapshot_rows_from_theoddsapi(
    events: list[dict],
    sport_key: str,
    league_name: str | None = None,
    received_at: str | None = None,
    is_closing: bool = False,
) -> list[dict]:
    """Normaliza eventos de The Odds API hacia el esquema raw.odds_snapshots."""
    ts_recv = received_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    for ev in events or []:
        home = str(ev.get("home_team") or "").strip()
        away = str(ev.get("away_team") or "").strip()
        if not home or not away:
            continue
        ev_time = ev.get("commence_time")
        fid = str(ev.get("id") or f"{str(ev_time or '')[:10]}_{home}_{away}")
        for bm in ev.get("bookmakers") or []:
            bm_title = str(bm.get("title") or bm.get("key") or "Unknown")
            bm_ts = bm.get("last_update") or ts_recv
            for mkt in bm.get("markets") or []:
                mkt_key = str(mkt.get("key") or "h2h")
                mkt_ts = mkt.get("last_update") or bm_ts
                for o in mkt.get("outcomes") or []:
                    try:
                        price = float(o.get("price") or 0)
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        continue
                    name = str(o.get("name") or "").strip()
                    sel = "Home" if name == home else ("Away" if name == away else ("Draw" if name.lower() == "draw" else name))
                    rows.append({
                        "sport": "soccer",
                        "league": league_name or sport_key,
                        "fixture_id": fid,
                        "home_team": home,
                        "away_team": away,
                        "event_time": ev_time,
                        "odds_timestamp": mkt_ts,
                        "received_at": ts_recv,
                        "provider": "the_odds_api",
                        "bookmaker": bm_title,
                        "market": mkt_key,
                        "line": float(o["point"]) if o.get("point") is not None else 0.0,
                        "selection": sel,
                        "price": price,
                        "is_suspended": False,
                        "is_closing": is_closing,
                    })
    return rows


def snapshot_rows_from_apifootball(
    fixture_id: int | str,
    home_team: str,
    away_team: str,
    odds_tuple: tuple[float, float, float, str],
    event_time: str | None = None,
    league_name: str | None = None,
    received_at: str | None = None,
    is_closing: bool = False,
) -> list[dict]:
    """Normaliza cuotas 1X2 de API-Football hacia raw.odds_snapshots."""
    ts_recv = received_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    oh, od, oa, bm_name = odds_tuple
    rows = []
    for sel, price in (("Home", oh), ("Draw", od), ("Away", oa)):
        if price and float(price) > 1.0:
            rows.append({
                "sport": "soccer",
                "league": league_name,
                "fixture_id": str(fixture_id),
                "home_team": home_team,
                "away_team": away_team,
                "event_time": event_time,
                "odds_timestamp": ts_recv,
                "received_at": ts_recv,
                "provider": "api_football",
                "bookmaker": str(bm_name or "API-Football"),
                "market": "h2h",
                "line": 0.0,
                "selection": sel,
                "price": float(price),
                "is_suspended": False,
                "is_closing": is_closing,
            })
    return rows


def load_or_fetch_odds(sport_key, force_refresh=None, min_date=None):
    """Load The Odds API odds from cache, or fetch and save to cache.

    force_refresh True obliga red; False obliga cache; None auto (refresca si el caché
    no contiene eventos vigentes para min_date).
    """
    if not sport_key:
        return []

    cache_path = ODDS_RAW_DIR / f"odds_{sport_key}.json"
    if force_refresh is not True:
        cached = read_json(cache_path)
        if force_refresh is False:
            return cached if cached is not None else []
        if cached is not None:
            if not min_date:
                return cached
            # En modo auto, verificar que el caché tenga eventos cercanos o posteriores a min_date
            has_current_events = any(
                isinstance(ev, dict) and str(ev.get("commence_time", ""))[:10] >= str(min_date)[:10]
                for ev in cached
            )
            if has_current_events or not ODDSAPI_KEY:
                return cached
            print(f"Cache stale for {sport_key} (no events >= {min_date}); refreshing from The Odds API...")

    print(f"Cache miss: Fetching odds from The Odds API for {sport_key}...")
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDSAPI_KEY,
        "regions": "eu",
        "markets": "h2h"
    }

    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            write_json(cache_path, data)
            try:
                snap_rows = snapshot_rows_from_theoddsapi(data, sport_key=sport_key)
                if snap_rows:
                    get_store().save_odds_snapshots(snap_rows)
            except Exception:  # noqa: BLE001
                pass
            return data
        else:
            print(f"Error fetching odds: {resp.text}")
    except Exception as e:  # noqa: BLE001
        print(f"Exception fetching odds: {e}")
    return read_json(cache_path) or []


def _find_matching_odds_event(odds_list, home_team_af, away_team_af, aligner, target_date=None):
    """Find the best matching event in odds_list validating both teams and date proximity."""
    if not odds_list:
        return None

    target_day = None
    if target_date:
        try:
            target_day = datetime.strptime(str(target_date)[:10], "%Y-%m-%d").date()
        except ValueError:
            target_day = None

    best_event = None
    best_combined_score = 0.0

    for match in odds_list:
        if target_day and match.get("commence_time"):
            try:
                commence_dt = datetime.fromisoformat(str(match["commence_time"]).replace("Z", "+00:00"))
                commence_local_day = commence_dt.astimezone(ZoneInfo("America/Bogota")).date()
                if abs((commence_local_day - target_day).days) > 2:
                    continue
            except ValueError:
                pass

        m_home, score_h = aligner.find_best_match(home_team_af, [match.get("home_team", "")], threshold=0.7)
        if not m_home:
            continue
        m_away, score_a = aligner.find_best_match(away_team_af, [match.get("away_team", "")], threshold=0.7)
        if not m_away:
            continue

        combined = float(score_h) + float(score_a)
        if combined > best_combined_score:
            best_combined_score = combined
            best_event = match

    return best_event


_SHARP_BOOKMAKER_PRIORITY = (
    "pinnacle",
    "betfair",
    "matchbook",
    "marathon",
    "bet365",
    "1xbet",
    "william hill",
)


def _select_bookmaker(bookmakers: list[dict], preferred_name: str | None) -> dict | None:
    """Selecciona la casa de apuestas preferida, priorizando casas sharp/líquidas sobre el resto."""
    if not bookmakers:
        return None
    if preferred_name:
        pref_low = preferred_name.lower()
        for bm in bookmakers:
            if (bm.get("title") or "").lower() == pref_low:
                return bm
    for prio in _SHARP_BOOKMAKER_PRIORITY:
        for bm in bookmakers:
            if prio in (bm.get("title") or "").lower():
                return bm
    return bookmakers[0]


def _extract_event_bm_odds(event: dict, selected_bm: dict) -> dict | None:
    """Extrae cuotas 1X2 desde un evento de Odds API y la casa seleccionada."""
    markets = selected_bm.get("markets", [])
    if not markets:
        return None
    odds = {}
    for outcome in markets[0].get("outcomes", []):
        name = outcome.get("name")
        price = float(outcome["price"])
        if name == event.get("home_team"):
            odds["Home"] = price
        elif name == event.get("away_team"):
            odds["Away"] = price
        elif name == "Draw":
            odds["Draw"] = price
    return {"bookmaker": selected_bm["title"], "odds": odds}


def _extract_event_consensus(event: dict) -> dict | None:
    """Calcula las cuotas promedio de consenso entre todos los bookmakers de un evento."""
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None
    home_prices, draw_prices, away_prices = [], [], []
    for bm in bookmakers:
        for market in bm.get("markets", []):
            for outcome in market.get("outcomes", []):
                name = outcome.get("name")
                price = float(outcome["price"])
                if name == event.get("home_team"):
                    home_prices.append(price)
                elif name == event.get("away_team"):
                    away_prices.append(price)
                elif name == "Draw":
                    draw_prices.append(price)
    if not home_prices or not away_prices:
        return None
    return {
        "bookmaker": "Market Consensus",
        "n_bookmakers": len(bookmakers),
        "odds": {
            "Home": round(sum(home_prices) / len(home_prices), 3),
            "Draw": round(sum(draw_prices) / len(draw_prices), 3) if draw_prices else None,
            "Away": round(sum(away_prices) / len(away_prices), 3),
        },
    }


def _extract_side_squad_context(
    fixture_context: dict,
    home_candidates: list[str],
    away_candidates: list[str],
    aligner: TeamAligner,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Separa lesiones y alineaciones confirmadas de API-Football por equipo local y visitante."""
    home_clean = [c for c in home_candidates if c]
    away_clean = [c for c in away_candidates if c]
    home_lower = {c.lower() for c in home_clean}
    away_lower = {c.lower() for c in away_clean}

    def resolve_side(team_name: str | None) -> str | None:
        if not team_name:
            return None
        tl = team_name.strip().lower()
        if tl in home_lower and tl not in away_lower:
            return "home"
        if tl in away_lower and tl not in home_lower:
            return "away"
        _, sh = aligner.find_best_match(team_name, home_clean, threshold=0.70) if home_clean else (None, 0.0)
        _, sa = aligner.find_best_match(team_name, away_clean, threshold=0.70) if away_clean else (None, 0.0)
        if sh >= 0.70 and sh > sa:
            return "home"
        if sa >= 0.70 and sa > sh:
            return "away"
        return None

    home_injuries: list[str] = []
    away_injuries: list[str] = []
    for inj in fixture_context.get("injuries") or []:
        player = str(inj.get("player") or "").strip()
        if not player:
            continue
        side = resolve_side(inj.get("team"))
        if side == "home" and player not in home_injuries:
            home_injuries.append(player)
        elif side == "away" and player not in away_injuries:
            away_injuries.append(player)

    home_lineup: list[str] = []
    away_lineup: list[str] = []
    for lu in fixture_context.get("lineups") or []:
        side = resolve_side(lu.get("team"))
        xi = [str(p).strip() for p in (lu.get("starting_xi") or []) if str(p).strip()]
        if side == "home" and xi:
            home_lineup = xi
        elif side == "away" and xi:
            away_lineup = xi

    return home_injuries, away_injuries, home_lineup, away_lineup


def get_match_odds(odds_list, home_team_af, away_team_af, bookmaker_name, aligner, target_date=None):
    """
    Finds the odds for a match by aligning team names and extracting from bookmaker.
    """
    event = _find_matching_odds_event(
        odds_list, home_team_af, away_team_af, aligner, target_date=target_date
    )
    if not event:
        return None

    selected_bm = _select_bookmaker(event.get("bookmakers", []), bookmaker_name)
    if not selected_bm:
        return None

    return _extract_event_bm_odds(event, selected_bm)


def get_consensus_odds(odds_list, home_team_af, away_team_af, aligner, target_date=None):
    """
    Computes consensus (market average) odds across ALL available bookmakers for a match.
    This is a stronger signal of the 'true' market probability than a single bookmaker.

    Returns dict with:
        - 'odds': {'Home': avg_home, 'Draw': avg_draw, 'Away': avg_away}
        - 'n_bookmakers': number of bookmakers averaged
        - 'bookmaker': 'Market Consensus'
    """
    event = _find_matching_odds_event(
        odds_list, home_team_af, away_team_af, aligner, target_date=target_date
    )
    if not event:
        return None

    return _extract_event_consensus(event)


def load_or_fetch_apifootball_odds(
    fixture_ids,
    bookmaker=None,
    bet_id=1,
    max_calls=12,
    force_refresh=False,
):
    """Cuotas 1X2 pre-partido via API-Football /odds?fixture= (fallback SA).

    Para ligas sin sport_key en The Odds API (Colombia 239, Peru 281, Ecuador 242,
    Uruguay 268, Paraguay 252, Bolivia 344, Venezuela 299).
    Retorna dict fixture_id -> (Home, Draw, Away, bookmaker).
    Cachea en raw.artifacts apifootball/odds_fixture_<id>.json.
    """
    out = {}
    calls = 0
    for fid in fixture_ids:
        try:
            fid_int = int(fid)
        except (TypeError, ValueError):
            continue
        cache_path = APIFOOTBALL_RAW_DIR / f"odds_fixture_{fid_int}.json"
        payload = None if force_refresh else read_json(cache_path)
        if payload is None and calls < max_calls and APIFOOTBALL_KEY:
            try:
                resp = requests.get(
                    "https://v3.football.api-sports.io/odds",
                    headers={"x-apisports-key": APIFOOTBALL_KEY},
                    params={"fixture": fid_int, "bet": bet_id},
                    timeout=25,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    write_json(cache_path, payload)
                    calls += 1
                else:
                    print(f"API-Football odds {fid_int}: HTTP {resp.status_code}")
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"API-Football odds {fid_int}: {exc}")
                continue
        if not payload:
            continue
        best = _parse_apifootball_1x2(payload, bookmaker=bookmaker)
        if best:
            out[fid_int] = best
            out[str(fid_int)] = best
    if calls:
        print(f"API-Football odds: {calls} llamadas de red (fixture 1X2).")
    return out


def _parse_apifootball_1x2(payload, bookmaker=None):
    """Extrae (Home, Draw, Away, bookmaker) del formato /odds API-Football."""
    try:
        response = payload.get("response", []) if isinstance(payload, dict) else []
    except AttributeError:
        return None
    home_prices, draw_prices, away_prices = [], [], []
    bm_names = []
    want = str(bookmaker).lower() if bookmaker is not None else None
    for entry in response:
        for bm in entry.get("bookmakers", []) or []:
            bmid = str(bm.get("id", ""))
            bmname = str(bm.get("name", ""))
            if want and want not in (bmid.lower(), bmname.lower()):
                continue
            bm_names.append(bmname or bmid)
            for bid in bm.get("bets", bm.get("bids", [])) or []:
                try:
                    if int(bid.get("id", 1)) != 1:
                        continue
                except (TypeError, ValueError):
                    pass
                for v in bid.get("values", []) or []:
                    name = str(v.get("value") or v.get("name") or "").strip().lower()
                    try:
                        price = float(v.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if price is None or price <= 1.0:
                        continue
                    if name == "home":
                        home_prices.append(price)
                    elif name == "draw":
                        draw_prices.append(price)
                    elif name == "away":
                        away_prices.append(price)
    if not (home_prices and draw_prices and away_prices):
        return None
    label = bookmaker if bookmaker is not None else (
        f"API-Football best of {len(set(bm_names))}"
    )
    return (max(home_prices), max(draw_prices), max(away_prices), str(label))


def main():
    parser = argparse.ArgumentParser(description="Football Prediction and Portfolio Betting System")
    default_date = datetime.now(ZoneInfo("America/Bogota")).date().isoformat()
    parser.add_argument("--date", type=str, default=default_date, help="Date in YYYY-MM-DD format (default: today in Bogotá)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to analyze starting from --date")
    parser.add_argument("--target-dates", nargs="+", help="Fechas exactas a analizar (YYYY-MM-DD); si se indica, reemplaza --days")
    parser.add_argument("--bankroll", type=float, default=138000.0, help="Initial betting bankroll")
    parser.add_argument("--kelly-factor", type=float, default=0.10, help="Kelly Criterion scaling factor (default: 0.10 seguro)")
    parser.add_argument("--jev-weight", type=float, default=JEV_CALIBRATED_WEIGHT,
                        help="JEV share of the blended probability when relevance gates pass (0-1, default 0.20)")
    parser.add_argument("--strategy-mode", type=str, default="multi_market",
                        choices=["multi_market", "calibrated_value", "hybrid", "value_local", "favorites", "unconstrained"],
                        help="Modo de estrategia de portafolio: multi_market (1X2 balanceado validado en backtest)|calibrated_value|hybrid|value_local|favorites|unconstrained")
    parser.add_argument("--stake-mode", type=str, default="flat",
                        choices=["flat", "kelly"],
                        help="Modo de dimensionamiento de apuesta: flat (3%% fijo) o kelly (fraccionario seguro)")
    parser.add_argument("--max-stake-pct", type=float, default=0.03,
                        help="Tope máximo de capital por apuesta (fracción 0-1, default 0.03 = 3%%)")
    parser.add_argument("--flat-stake-pct", type=float, default=0.03,
                        help="Porcentaje fijo por apuesta en modo flat (default 0.03 = 3%%)")
    parser.add_argument("--min-ev", type=float, default=0.03,
                        help="Umbral mínimo de valor esperado (default 0.03 = +3%%)")
    parser.add_argument("--min-stake", type=float, default=1.0,
                        help="Stake mínimo por apuesta en unidades monetarias (default: 1.0)")
    parser.add_argument("--max-daily-exposure-pct", type=float, default=0.15,
                        help="Tope máximo de exposición diaria sobre el bankroll (fracción 0-1, default 0.15 = 15%%)")
    parser.add_argument("--max-bets-per-league", type=int, default=3,
                        help="Máximo de apuestas recomendadas por liga en un día (default: 3)")
    parser.add_argument("--bookmaker", type=str, default="Pinnacle", help="Preferred bookmaker name")
    parser.add_argument("--odds-source", type=str, default="auto",
                        help="Fuente de cuotas para ligas SA sin The Odds API: auto|apifootball|off")
    parser.add_argument("--no-refresh-odds", action="store_true",
                        help="No llamar a The Odds API; reutilizar cache (ahorra creditos)")
    parser.add_argument("--refresh-odds", action="store_true",
                        help="Forzar refresco de cuotas The Odds API aunque haya cache")
    parser.add_argument("--no-refresh-af-odds", action="store_true",
                        help="No llamar a API-Football /odds; reutilizar cache odds_fixture_*")
    parser.add_argument("--af-odds-bookmaker", type=str, default=None,
                        help="Filtro bookmaker API-Football (id o nombre); default mejor cuota por signo")
    parser.add_argument("--af-odds-max-calls", type=int, default=12,
                        help="Tope de llamadas a API-Football /odds por corrida (plan free 100/dia)")

    args = parser.parse_args()
    start_date = args.date
    days = args.days
    if args.target_dates:
        try:
            date_list = sorted({datetime.strptime(value, "%Y-%m-%d").date().isoformat()
                                for value in args.target_dates})
        except ValueError:
            parser.error("--target-dates solo acepta fechas con formato YYYY-MM-DD")
        if not date_list:
            parser.error("--target-dates debe incluir al menos una fecha")
        start_date = date_list[0]
        days = len(date_list)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
        date_list = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    bankroll = args.bankroll
    kelly_factor = args.kelly_factor
    jev_weight_config = args.jev_weight
    if not 0.0 <= jev_weight_config <= 1.0:
        parser.error("--jev-weight debe estar entre 0 y 1")
    preferred_bm = args.bookmaker
    _odds_source = str(getattr(args, "odds_source", "auto")).lower()
    _force_refresh = True if getattr(args, "refresh_odds", False) else (False if getattr(args, "no_refresh_odds", False) else None)
    _refresh_af = not bool(getattr(args, "no_refresh_af_odds", False))
    _af_bm = getattr(args, "af_odds_bookmaker", None)
    _af_max = int(getattr(args, "af_odds_max_calls", 12) or 12)
    print(f"Odds: source={_odds_source} refresh={_force_refresh} af_refresh={_refresh_af} af_bm={_af_bm} af_max={_af_max}")

    strategy_mode = getattr(args, "strategy_mode", "hybrid")
    stake_mode = getattr(args, "stake_mode", "flat")
    max_stake_pct = float(getattr(args, "max_stake_pct", 0.03))
    flat_stake_pct = float(getattr(args, "flat_stake_pct", 0.03))
    min_ev = float(getattr(args, "min_ev", 0.03))
    min_stake_amount = float(getattr(args, "min_stake", 1.0))
    max_daily_exposure_pct = float(getattr(args, "max_daily_exposure_pct", 0.15))
    max_bets_per_league = getattr(args, "max_bets_per_league", 3)

    print(f"=== Starting System for Date: {start_date} | Days: {days} ===")
    print(f"Settings: Bankroll: ${bankroll:.2f} | Strategy: {strategy_mode} | Staking: {stake_mode} (Max stake: {max_stake_pct*100:.1f}%) | Min stake: ${min_stake_amount:.2f} | JEV Weight: {jev_weight_config:.2f}")

    # Initialize components
    aligner = TeamAligner()
    store = get_store()
    predictor = SoccerPredictor()
    impect_loader = ImpectTeamMetricsLoader(raw_impect_dir=str(IMPECT_RAW_DIR))
    optimizer = KellyOptimizer(
        bankroll=bankroll,
        fraction_factor=kelly_factor,
        strategy_mode=strategy_mode,
        stake_mode=stake_mode,
        flat_stake_pct=flat_stake_pct,
        max_stake_pct=max_stake_pct,
        min_ev=min_ev,
        min_stake_amount=min_stake_amount,
        max_daily_exposure_pct=max_daily_exposure_pct,
        max_bets_per_league=max_bets_per_league,
    )
    form_loader = RecentFormLoader()
    prematch_context_client = ApiFootballContextClient(api_key=APIFOOTBALL_KEY, store=store)
    # Inicialización independiente de Brave (noticias) y JEV (decisiones estructuradas)
    brave = jev = None
    if os.getenv("JEVAPI") or os.getenv("TYPESAFE_API_KEY"):
        try:
            jev = JevDecisionClient()
        except Exception as exc:  # noqa: BLE001
            print(f"JEV desactivado; se conserva Softmax: {exc}")
    if os.getenv("BRAVEAPI"):
        try:
            brave = BraveNewsClient()
        except Exception as exc:  # noqa: BLE001
            print(f"Brave desactivado: {exc}")

    # Cache Odds data for the day (lazy-loaded per active league or fallback)
    odds_cache = {}
    min_target_date = min(date_list) if date_list else start_date
    today_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    effective_min_date = min_target_date if min_target_date >= today_utc_str else None

    all_days_results = {}

    # Process each date
    for target_date in date_list:
        print(f"\n--- Processing Date: {target_date} ---")

        # Load fixtures
        fixtures_data = load_or_fetch_fixtures(target_date)
        fixtures = fixtures_data.get("response", [])
        target_day = datetime.strptime(target_date, "%Y-%m-%d").date()
        if target_day >= datetime.now(timezone.utc).date():
            date_injuries = prematch_context_client.injuries_for_date(target_date)
        else:
            date_injuries = {"records": [], "fetched_at": None,
                             "error": "fuente sin histórico de lesiones"}
        fixtures_errors = fixtures_data.get("errors", {})
        print(f"Total fixtures loaded for {target_date}: {len(fixtures)}")

        # Fallback to Odds API if no fixtures returned or plan limits error
        if not fixtures or "plan" in fixtures_errors:
            print(f"Fallback: Fetching fixtures from Odds API for {target_date}...")
            fallback_fixtures = []
            for league_id, league_info in SUPPORTED_LEAGUES.items():
                odds_key = league_info["odds_sport_key"]
                if not odds_key:
                    continue
                if odds_key not in odds_cache:
                    odds_cache[odds_key] = load_or_fetch_odds(
                        odds_key, force_refresh=_force_refresh, min_date=effective_min_date
                    )
                odds_list = odds_cache.get(odds_key, [])
                for match in odds_list:
                    commence_raw = str(match.get("commence_time") or "")
                    try:
                        commence_dt = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
                        if commence_dt.tzinfo is None:
                            commence_dt = commence_dt.replace(tzinfo=timezone.utc)
                        commence_date = commence_dt.astimezone(ZoneInfo("America/Bogota")).date().isoformat()
                    except ValueError:
                        commence_date = commence_raw.split("T")[0]
                    if commence_date == target_date:
                        fake_fix = {
                            "fixture": {
                                "status": {"short": "NS"},
                                "date": match["commence_time"]
                            },
                            "league": {
                                "id": league_id
                            },
                            "teams": {
                                "home": {"name": match["home_team"]},
                                "away": {"name": match["away_team"]}
                            },
                            "is_fallback": True,
                            "raw_match": match
                        }
                        fallback_fixtures.append(fake_fix)
            fixtures = fallback_fixtures
            print(f"Fallback loaded {len(fixtures)} fixtures from Odds API.")

        # Processed matches container
        _af_odds_seen = set()
        processed_matches = []
        crosswalk_updates = []
        skipped = {"no_prematch_status": 0, "kickoff_passed": 0,
                   "unsupported_league": 0, "impect_alignment": 0}

        # Process each fixture
        for fix in fixtures:
            status = fix["fixture"]["status"]["short"]
            match_date_str = fix["fixture"]["date"]
            match_dt = datetime.fromisoformat(match_date_str.replace("Z", "+00:00"))
            now_utc = datetime.now(timezone.utc)

            if status not in ["NS", "TBD"]:
                skipped["no_prematch_status"] += 1
                continue
            if match_dt <= now_utc:
                skipped["kickoff_passed"] += 1
                continue

            league_id = fix["league"]["id"]
            if league_id not in SUPPORTED_LEAGUES:
                skipped["unsupported_league"] += 1
                continue

            league_info = SUPPORTED_LEAGUES[league_id]
            impect_iter = league_info.get("impect_iteration")
            odds_key = league_info.get("odds_sport_key")

            home_af = fix["teams"]["home"]["name"]
            away_af = fix["teams"]["away"]["name"]

            # Load Impect squad list to map names if iteration is defined.
            # (hoisted fuera del if para que el guard de cobertura pueda verificar
            #  si ambos equipos tienen ratings reales y no el default 0.68)
            ratings_map = predictor.load_ratings(impect_iter, target_date) if impect_iter else {}
            if impect_iter:
                impect_candidates = list(ratings_map.keys())
                impect_ids = predictor.load_squad_ids(impect_iter, target_date)

                # Reuse stable provider IDs when available; otherwise align names
                # and persist only high-confidence mappings for later runs.
                def resolve_team(side):
                    team = fix.get("teams", {}).get(side, {})
                    provider_id = team.get("id")
                    saved = None
                    if provider_id is not None:
                        try:
                            saved = store.get_team_crosswalk("api_football", int(provider_id), impect_iter)
                        except (TypeError, ValueError):
                            provider_id = None
                    if saved and saved["impect_squad_name"] in impect_candidates:
                        return saved["impect_squad_name"]
                    resolved, score = aligner.find_best_match(team.get("name", ""), impect_candidates)
                    if resolved and score >= 0.90 and provider_id is not None:
                        crosswalk_updates.append({
                            "provider": "api_football", "provider_team_id": int(provider_id),
                            "iteration_id": impect_iter, "provider_team_name": team.get("name"),
                            "impect_squad_id": impect_ids.get(resolved),
                            "impect_squad_name": resolved, "match_score": float(score),
                            "mapping_source": "aligner_high_confidence",
                        })
                    return resolved

                home_impect = resolve_team("home")
                away_impect = resolve_team("away")

                # Check if matched successfully
                if not home_impect or not away_impect:
                    # Skip if we couldn't resolve teams in Impect
                    skipped["impect_alignment"] += 1
                    print(f"Skipping {home_af} vs {away_af}: no Impect alignment")
                    continue
            else:
                home_impect = home_af
                away_impect = away_af

            # Get recent form adjustment (home vs. away momentum delta)
            form_result = form_loader.get_form_adjustment(home_af, away_af, target_date)
            form_delta = form_result["form_delta"]

            # --- Guard de cobertura de datos: Impect o fallback ClubElo + forma reciente ---
            home_form_ctx = form_result.get("home_form") or {}
            away_form_ctx = form_result.get("away_form") or {}
            ratings_covered = (
                bool(impect_iter)
                and home_impect in ratings_map
                and away_impect in ratings_map
            )
            clubelo_covered = False
            if not ratings_covered:
                elo_h, elo_a = predictor.resolve_clubelo_pair(home_impect, away_impect, match_date=target_date)
                clubelo_covered = (elo_h is not None and elo_a is not None)
            form_covered = (
                int(home_form_ctx.get("form_n") or 0) >= MIN_FORM_MATCHES_FOR_BET
                and int(away_form_ctx.get("form_n") or 0) >= MIN_FORM_MATCHES_FOR_BET
            )
            data_coverage_flags = []
            if not (ratings_covered or clubelo_covered):
                if not impect_iter:
                    data_coverage_flags.append("liga_sin_ratings_impect_ni_clubelo")
                else:
                    data_coverage_flags.append("equipos_sin_rating_impect_ni_clubelo")
            if not form_covered:
                data_coverage_flags.append("forma_insuficiente")
            data_coverage_ok = not data_coverage_flags

            fixture_context = prematch_context_client.fixture_context(fix, date_injuries)
            structured_relevant = bool(fixture_context.get("injuries") or fixture_context.get("lineups_available"))

            # Get odds if available
            odds_details = None
            consensus_details = None
            if fix.get("is_fallback"):
                match_data = fix["raw_match"]
                selected_bm = _select_bookmaker(match_data.get("bookmakers", []), preferred_bm)
                if selected_bm:
                    odds_details = _extract_event_bm_odds(match_data, selected_bm)
                consensus_details = _extract_event_consensus(match_data)
            elif odds_key:
                if odds_key not in odds_cache:
                    odds_cache[odds_key] = load_or_fetch_odds(
                        odds_key, force_refresh=_force_refresh, min_date=effective_min_date
                    )
                odds_list = odds_cache.get(odds_key, [])
                odds_details = get_match_odds(
                    odds_list, home_af, away_af, preferred_bm, aligner, target_date=target_date
                )
                # --- Optimización: Cuotas Consenso (promedio de todos los bookmakers) ---
                if odds_list:
                    consensus_details = get_consensus_odds(
                        odds_list, home_af, away_af, aligner, target_date=target_date
                    )
                else:
                    consensus_details = None

            if odds_details is None and _odds_source in ("auto", "apifootball") and data_coverage_ok:
                _fid = None
                _fix_id = (fix.get("fixture") or {}).get("id")
                if _fix_id is not None:
                    try:
                        _fid = int(_fix_id)
                    except (TypeError, ValueError):
                        _fid = None
                if _fid:
                    _af = load_or_fetch_apifootball_odds(
                        [_fid], bookmaker=_af_bm, max_calls=_af_max,
                        force_refresh=(_refresh_af if _fid in _af_odds_seen else bool(_refresh_af)),
                    )
                    _af_odds_seen.add(_fid)
                    _t = _af.get(_fid) or _af.get(str(_fid))
                    if _t:
                        odds_details = {"bookmaker": _t[3], "odds": {"Home": _t[0], "Draw": _t[1], "Away": _t[2]}}
                        consensus_details = {"bookmaker": "API-Football", "n_bookmakers": 1,
                                             "odds": {"Home": _t[0], "Draw": _t[1], "Away": _t[2]}}
                        try:
                            af_snaps = snapshot_rows_from_apifootball(
                                _fid, home_af, away_af, _t,
                                event_time=match_date_str, league_name=league_info.get("name"),
                            )
                            store.save_odds_snapshots(af_snaps)
                        except Exception:  # noqa: BLE001
                            pass

            # Determine market odds for Bayesian shrinkage
            market_odds_for_pred = None
            if consensus_details and isinstance(consensus_details.get("odds"), dict):
                market_odds_for_pred = consensus_details["odds"]
            elif odds_details and isinstance(odds_details.get("odds"), dict):
                market_odds_for_pred = odds_details["odds"]

            # Deriva de línea y edad de cuota desde raw.odds_snapshots (TASK-10)
            opening_odds = None
            odds_age_minutes = None
            try:
                traj = store.get_odds_trajectory(home_af, away_af, event_date=target_date, market="h2h")
                if not traj and home_impect != home_af:
                    traj = store.get_odds_trajectory(home_impect, away_impect, event_date=target_date, market="h2h")
                if traj:
                    first_book = traj[0].get("bookmaker")
                    open_map = {}
                    for tr in traj:
                        if tr.get("bookmaker") == first_book and tr.get("selection") not in open_map:
                            open_map[tr["selection"]] = float(tr["price"])
                    if {"Home", "Draw", "Away"}.issubset(open_map.keys()):
                        opening_odds = open_map
                    latest_ts = traj[-1].get("odds_timestamp")
                    recv_ts = traj[-1].get("received_at")
                    if latest_ts and recv_ts:
                        dt_o = datetime.fromisoformat(str(latest_ts).replace("Z", "+00:00"))
                        dt_r = datetime.fromisoformat(str(recv_ts).replace("Z", "+00:00"))
                        odds_age_minutes = max(0.0, (dt_r - dt_o).total_seconds() / 60.0)
            except Exception:  # noqa: BLE001
                pass

            # Extract squad context (injuries & confirmed lineups) for player talent sensitivity
            home_injuries, away_injuries, home_lineup, away_lineup = _extract_side_squad_context(
                fixture_context,
                [home_af, home_impect],
                [away_af, away_impect],
                aligner,
            )

            # Get predictions (with multi-layer Impect model: Poisson xG + Tactical Multi-KPI + Squad Talent + Shrinkage)
            pred = predictor.predict_match(
                home_impect, away_impect, impect_iter,
                match_date=target_date,
                league=league_info.get("country"),
                form_delta=form_delta,
                market_odds=market_odds_for_pred,
                home_injuries=home_injuries or None,
                away_injuries=away_injuries or None,
                home_lineup=home_lineup or None,
                away_lineup=away_lineup or None,
                opening_odds=opening_odds,
                odds_age_minutes=odds_age_minutes,
            )

            match_entry = {
                "league_name": league_info["name"],
                "home_team": home_impect,
                "away_team": away_impect,
                "P_H": pred["P_H"],
                "P_D": pred["P_D"],
                "P_A": pred["P_A"],
                "home_rating": pred["home_rating"],
                "away_rating": pred["away_rating"],
                "home_advantage_used": pred.get("home_advantage_used", 0.05),
                "form_delta": pred.get("form_delta", 0.0),
                "data_coverage_ok": data_coverage_ok,
                "data_coverage_flags": data_coverage_flags,
                "model_alpha": pred.get("model_alpha"),
                "model_gamma_d": pred.get("model_gamma_d"),
                "model_version": pred.get("model_version"),
                "signal_source": pred.get("signal_source"),
                "clubelo_covered": bool(pred.get("clubelo_covered", False)),
                "poisson_available": pred.get("poisson_available", False),
                "lambda_home": pred.get("lambda_home"),
                "lambda_away": pred.get("lambda_away"),
                "home_missing_impact": pred.get("home_missing_impact", 0.0),
                "away_missing_impact": pred.get("away_missing_impact", 0.0),
                "home_missing_value_pct": pred.get("home_missing_value_pct", 0.0),
                "away_missing_value_pct": pred.get("away_missing_value_pct", 0.0),
                "line_drift_pp": pred.get("line_drift_pp"),
                "steam_move_detected": bool(pred.get("steam_move_detected", False)),
                "odds_age_minutes": pred.get("odds_age_minutes"),
                "P_poisson": pred.get("P_poisson"),
                "P_tactical": pred.get("P_tactical"),
                "P_ratings_base": pred.get("P_ratings_base"),
                "P_ensemble": pred.get("P_ensemble"),
                "form_context": {
                    "window_matches": 5,
                    "home": form_result.get("home_form", {}),
                    "away": form_result.get("away_form", {}),
                },
                "odds_available": False,
                "odds": None,
                "consensus_odds": None,
                "bookmaker": None
            }

            if impect_iter:
                tactical_h = impect_loader.get_team_tactical_summary(impect_iter, home_impect, target_date=target_date)
                tactical_a = impect_loader.get_team_tactical_summary(impect_iter, away_impect, target_date=target_date)
            else:
                tactical_h = predictor.get_understat_tactical_summary(league_info.get("country"), home_impect, target_date=target_date)
                tactical_a = predictor.get_understat_tactical_summary(league_info.get("country"), away_impect, target_date=target_date)
            tactical_context = {"local": tactical_h, "visitante": tactical_a}

            base_probs = {"local": pred["P_H"], "empate": pred["P_D"], "visita": pred["P_A"]}
            jev_reason = "sin_cobertura_o_cuotas" if not (data_coverage_ok and odds_details is not None) else "sin_configuracion_jev"
            jev_result = fallback_decision(jev_reason)
            news = {"news_relevant": False, "digest": "", "sources": [], "news_evidence": []}
            context_relevant = False
            if data_coverage_ok and odds_details is not None:
                if brave:
                    try:
                        news = brave.search_match_news(
                            home_impect, away_impect, target_date,
                            country=league_info.get("country"),
                            league_id=league_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        news["errors"] = [str(exc)]
                context_relevant = bool(news.get("news_relevant") or structured_relevant)
                if jev:
                    state = build_match_state(
                        home_impect, away_impect, league_info["name"], base_probs,
                        pred["home_rating"], pred["away_rating"], form_delta,
                        odds=(odds_details or {}).get("odds", {}),
                        news_digest=news.get("digest"),
                        news_sources=news.get("sources"),
                        news_evidence=news.get("news_evidence"),
                        structured_context={
                            "forma_ultimos_5_y_descanso": {
                                "local": form_result.get("home_form", {}),
                                "visitante": form_result.get("away_form", {}),
                            },
                            "api_football": fixture_context,
                        },
                        tactical_context=tactical_context,
                        home_advantage=pred.get("home_advantage_used", 0.05),
                        adjusted_diff=pred.get("adjusted_difference"),
                        home_missing_impact=pred.get("home_missing_impact", 0.0),
                        away_missing_impact=pred.get("away_missing_impact", 0.0),
                        home_missing_value_pct=pred.get("home_missing_value_pct"),
                        away_missing_value_pct=pred.get("away_missing_value_pct"),
                    )
                    try:
                        jev_result = jev.decide(state)
                    except Exception as exc:  # noqa: BLE001
                        jev_result = fallback_decision(f"jev_error: {exc}")
            # El tier táctico de JEV ("always on") solo mezcla cuando hay cobertura real
            # de datos (ratings + forma). Sin ella, la opinión táctica del LLM no es una
            # señal confiable: el 2026-10-22 infló a todos los locales a P_jev ~0.87.
            fused = fuse_probabilities(
                base_probs, jev_result, news_relevant=context_relevant,
                jev_weight=jev_weight_config,
                always_on=data_coverage_ok,
            )
            fav_base = fused.get("favorito_base") or max(base_probs, key=base_probs.get)
            match_entry.update({
                "P_H": fused["P_adj"]["local"], "P_D": fused["P_adj"]["empate"],
                "P_A": fused["P_adj"]["visita"], "P_base": fused["P_base"],
                "P_jev": fused.get("P_jev"), "jev_weight": fused["w"],
                "jev_source": fused["decision_source"], "jev_reason": fused["motivo"],
                "vetada": bool(fused.get("vetada", False)),
                "favorito_base": fav_base,
                "choque_estilos": fused.get("choque_estilos"),
                "tactical_context": tactical_context,
                "news_evidence": news.get("news_evidence", []),
                "news_categories": news.get("news_categories", []),
                "news_digest": news.get("digest", ""),
                "news_sources": news.get("sources", []),
                "news_fetched_at": news.get("fetched_at"),
                "news_from_cache": news.get("from_cache", False),
                "provider_context": fixture_context,
                "jev_context_relevant": context_relevant,
                "jev_weight_config": jev_weight_config,
                "news_relevant": bool(news.get("news_relevant")),
                "structured_context_relevant": structured_relevant,
                "jev_relevance_score": fused.get("news_relevance_score", 0.0),
                "jev_confidence": jev_result.get("confidence"),
            })

            if odds_details:
                match_entry["odds_available"] = True
                match_entry["odds"] = odds_details["odds"]
                match_entry["bookmaker"] = odds_details["bookmaker"]
                if consensus_details:
                    match_entry["consensus_odds"] = consensus_details

            processed_matches.append(match_entry)

        if crosswalk_updates:
            store.save_team_crosswalk(crosswalk_updates)
        print(f"Matches aligned and predicted for {target_date}: {len(processed_matches)}; skipped={skipped}")

        # Separate value bets
        value_bets = []
        coverage_skipped = 0
        jev_vetoed = 0
        fav_outcome_map = {
            "local": "Home Win",
            "empate": "Draw",
            "visita": "Away Win",
            "Home Win": "Home Win",
            "Draw": "Draw",
            "Away Win": "Away Win",
        }
        for match in processed_matches:
            if not match["odds_available"]:
                continue
            # Guard de cobertura: sin ratings Impect ni forma reciente el modelo no
            # tiene señal real (caso copas europeas 2026-10-22) -> no es apostable.
            if not match.get("data_coverage_ok", True):
                coverage_skipped += 1
                continue

            odds_dict = match["odds"]
            # Ancla de mercado: probabilidad implícita sin margen, preferiendo el
            # consenso de bookmakers cuando está disponible.
            consensus_entry = match.get("consensus_odds") or {}
            market_odds = (
                consensus_entry.get("odds")
                if isinstance(consensus_entry, dict) and consensus_entry.get("odds")
                else odds_dict
            )
            market_probs = implied_probabilities_no_vig(market_odds)
            vetoed_fav_outcome = (
                fav_outcome_map.get(match.get("favorito_base"))
                if match.get("vetada")
                else None
            )
            # Outcomes: 'Home Win', 'Draw', 'Away Win'
            outcomes_mapping = {
                "Home Win": ("P_H", "Home"),
                "Draw": ("P_D", "Draw"),
                "Away Win": ("P_A", "Away")
            }

            for name, (prob_key, odds_key) in outcomes_mapping.items():
                if vetoed_fav_outcome and name == vetoed_fav_outcome:
                    jev_vetoed += 1
                    continue
                prob = match[prob_key]
                price = odds_dict.get(odds_key)

                if price:
                    ev = (prob * price) - 1.0
                    if ev > 0.0:
                        value_bets.append({
                            "date": target_date,
                            "home_team": match["home_team"],
                            "away_team": match["away_team"],
                            "league": match.get("league_name", ""),
                            "market_outcome": name,
                            "probability": prob,
                            "odds": price,
                            "ev": ev,
                            "market_implied_prob": (market_probs or {}).get(odds_key),
                            "line_drift_pp": match.get("line_drift_pp"),
                            "odds_age_minutes": match.get("odds_age_minutes"),
                        })

        print(f"Value bets identified (EV > 0) for {target_date}: {len(value_bets)}"
              f" | descartadas por guard de cobertura: {coverage_skipped}"
              f" | vetadas por JEV: {jev_vetoed}")

        # Portfolio optimization
        portfolio = optimizer.optimize_portfolio(value_bets)

        all_days_results[target_date] = {
            "processed_matches": processed_matches,
            "value_bets": value_bets,
            "portfolio": portfolio,
            "errors": fixtures_data.get("errors", {})
        }

    # Persistencia estructurada: los artefactos JSON se guardan en DuckDB.
    if days == 1:
        day_data = all_days_results[start_date]
        payload = {
            "date": start_date,
            "bankroll": bankroll,
            "kelly_factor": kelly_factor,
            "jev_weight_config": jev_weight_config,
            "value_bets": day_data["value_bets"],
            "portfolio": day_data["portfolio"],
            "processed_matches": day_data["processed_matches"],
            "errors": day_data["errors"],
        }
        write_json(
            PROCESSED_DIR / f"portfolio_bets_{start_date}.json",
            payload,
            run_type="daily_portfolio",
            run_date=start_date,
        )
        print(f"Resultados estructurados guardados en DuckDB: processed/portfolio_bets_{start_date}.json")
        print("\n--- Summary recommendations ---")
        if day_data["portfolio"]:
            for bet in day_data["portfolio"]:
                niche_tag = f" [{bet['niche']}]" if bet.get("niche") else ""
                print(f"Apuesta{niche_tag}: {bet['home_team']} vs {bet['away_team']} -> {bet['outcome']} | Cuota: {bet['odds']:.2f} | Stake: ${bet['stake']:.2f} ({bet['kelly_scaled']:.2%})")
        else:
            print("No bets recommended.")
    else:
        end_date = date_list[-1]
        daily_results = []
        consolidated_portfolio = []
        total_value_bets = 0
        for target_date in date_list:
            day_data = all_days_results[target_date]
            day_payload = {
                "date": target_date,
                "bankroll": bankroll,
                "kelly_factor": kelly_factor,
                "jev_weight_config": jev_weight_config,
                "value_bets": day_data["value_bets"],
                "portfolio": day_data["portfolio"],
                "processed_matches": day_data["processed_matches"],
                "errors": day_data["errors"],
            }
            write_json(
                PROCESSED_DIR / f"portfolio_bets_{target_date}.json",
                day_payload,
                run_type="daily_portfolio",
                run_date=target_date,
            )
            daily_results.append(day_payload)
            total_value_bets += len(day_data["value_bets"])
            consolidated_portfolio.extend(
                {**bet, "date": target_date} for bet in day_data["portfolio"]
            )

        weekly_payload = {
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "bankroll": bankroll,
            "kelly_factor": kelly_factor,
            "jev_weight_config": jev_weight_config,
            "total_value_bets": total_value_bets,
            "portfolio": consolidated_portfolio,
            "daily_results": daily_results,
        }
        write_json(
            PROCESSED_DIR / f"portfolio_bets_{start_date}_{end_date}_weekly.json",
            weekly_payload,
            run_type="weekly_portfolio",
            run_date=start_date,
        )
        print(f"Resultados diarios y resumen semanal guardados en DuckDB: processed/portfolio_bets_{start_date}_{end_date}_weekly.json")
        print("\n--- Consolidated weekly recommendations ---")
        if consolidated_portfolio:
            for bet in consolidated_portfolio:
                niche_tag = f" [{bet.get('niche')}]" if bet.get("niche") else ""
                print(f"[{bet['date']}] Apuesta{niche_tag}: {bet['home_team']} vs {bet['away_team']} -> {bet['outcome']} | Cuota: {bet['odds']:.2f} | Stake: ${bet['stake']:.2f} ({bet['kelly_scaled']:.2%})")
        else:
            print("No weekly bets recommended.")

if __name__ == "__main__":
    main()
