"""Normalización de pronósticos a 'local' / 'empate' / 'visita'.

El histórico mezcla vocabularios en `output.recommendations.outcome`:
- Canónico inglés del pipeline diario: 'Home Win' / 'Draw' / 'Away Win'.
- Español de `predict_upcoming_portfolio.py`: 'Victoria Local' / 'Empate' / 'Victoria Visitante'.
- Nombres de equipo (legado de `selection`/`best_bet_team`): p.ej. 'Santos', 'Platense'.

Los conciliadores (`results_fetcher`, `impect_results_fetcher`) solo entendían
el vocabulario inglés, por lo que `bet_won` quedaba NULL aunque
`result_outcome` sí existía, y la columna "Acierto" del Streamlit mostraba "—".
"""
from __future__ import annotations


def _clean_label(value: str) -> str:
    n = (value or "").strip().lower()
    for src, dst in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ü", "u"), ("ñ", "n")):
        n = n.replace(src, dst)
    n = " ".join(n.split())
    return n


_CANONICAL = {
    # local
    "home win": "local", "home": "local", "h": "local", "1": "local",
    "local": "local", "victoria local": "local", "gana local": "local",
    "triunfo local": "local", "ganador local": "local", "win home": "local",
    # empate
    "draw": "empate", "d": "empate", "x": "empate",
    "empate": "empate", "tie": "empate", "ties": "empate",
    # visita
    "away win": "visita", "away": "visita", "a": "visita", "2": "visita",
    "visita": "visita", "victoria visitante": "visita", "gana visitante": "visita",
    "triunfo visitante": "visita", "ganador visitante": "visita", "win away": "visita",
}


def normalize_selection_to_side(selection: str | None, home_team: str | None = None,
                                away_team: str | None = None) -> str | None:
    """Convierte cualquier vocabulario de pronóstico a local/empate/visita."""
    if selection is None:
        return None
    cleaned = _clean_label(str(selection))
    if not cleaned:
        return None
    if cleaned in _CANONICAL:
        return _CANONICAL[cleaned]
    # Fallback: la selección es el nombre del equipo (legado).
    if home_team or away_team:
        try:
            from .aligner import normalize_name
        except ImportError:  # pragma: no cover - import relativo alternativo
            from src.apostador.aligner import normalize_name  # type: ignore
        norm_sel = normalize_name(str(selection))
        if norm_sel:
            if home_team and norm_sel == normalize_name(str(home_team)):
                return "local"
            if away_team and norm_sel == normalize_name(str(away_team)):
                return "visita"
    return None


def compute_bet_won(selection: str | None, result_outcome: str | None,
                    home_team: str | None = None, away_team: str | None = None) -> bool | None:
    """Compara el pronóstico normalizado contra el resultado real.

    Devuelve None si alguno de los dos lados no se puede normalizar.
    """
    if result_outcome is None:
        return None
    side = normalize_selection_to_side(selection, home_team, away_team)
    if side is None:
        return None
    norm_result = _clean_label(str(result_outcome))
    # result_outcome ya viene como local/empate/visita, pero toleramos variantes.
    norm_result = _CANONICAL.get(norm_result, norm_result)
    if norm_result not in ("local", "empate", "visita"):
        return None
    return side == norm_result
