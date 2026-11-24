"""Decision-gated probability fusion: Core Softmax baseline, dual-tier JEV (tactical base + news priority)."""

import math

from .jev_decision import NEWS_RELEVANCE_THRESHOLD

# Default JEV weighting (dos tiers, ver .agents/TASKS/jev-full-context-always-on.md):
# - Tactical base tier (JEV evaluó el contexto pero NO hay noticias disruptivas):
#   peso conservador 0.10. El 0.50 anterior infló los locales a ~0.65 en partidos sin
#   ratings Impect ni forma (copas europeas, incidente 2026-10-22: 5/6 apuestas perdidas).
# - Priority tier (noticias relevantes por encima del umbral):
#   JEV_CALIBRATED_WEIGHT = 0.20 calibrado en backtest walk-forward (jev_backtesting.md).
#   JEV_PRIORITY_WEIGHT = 0.50 se mantiene exportado por compatibilidad.
JEV_BASE_WEIGHT = 0.10
JEV_PRIORITY_WEIGHT = 0.50
JEV_CALIBRATED_WEIGHT = 0.20
P_MIN, P_MAX = 0.01, 0.90
OUTCOME_KEYS = ("local", "empate", "visita")
JEV_SHRINKAGE_LAMBDA = 0.15


def smooth_jev_probabilities(probs, shrink_factor=JEV_SHRINKAGE_LAMBDA, prior_probs=None):
    """
    Shrink raw JEV probabilities toward an informative prior (`prior_probs`, e.g. Core Base)
    or toward uniform distribution (1/3) when no prior is given, countering LLM overconfidence.
    """
    if not probs:
        return None
    if prior_probs and all(k in prior_probs for k in OUTCOME_KEYS):
        prior_vals = {k: max(0.0, float(prior_probs[k])) for k in OUTCOME_KEYS}
        prior_sum = sum(prior_vals.values())
        if prior_sum > 0:
            prior = {k: v / prior_sum for k, v in prior_vals.items()}
        else:
            uniform = 1.0 / len(OUTCOME_KEYS)
            prior = {k: uniform for k in OUTCOME_KEYS}
    else:
        uniform = 1.0 / len(OUTCOME_KEYS)
        prior = {k: uniform for k in OUTCOME_KEYS}

    smoothed = {k: (1.0 - shrink_factor) * float(probs[k]) + shrink_factor * prior[k] for k in OUTCOME_KEYS}
    total = sum(smoothed.values()) or 1.0
    return {k: v / total for k, v in smoothed.items()}


def blend_weight(
    news_relevant,
    relevance_score=0.0,
    jev_weight=JEV_PRIORITY_WEIGHT,
    always_on=True,
    base_weight=JEV_BASE_WEIGHT,
):
    """
    Return blend weight for JEV:
    - Default is 1.0 (100% JEV, 0% Core, as Core is already an input to JEV).
    - If news/injuries are relevant (>= threshold): use full configured jev_weight.
    - If always_on is True (default): use min(base_weight, jev_weight).
    - If jev_weight == 0 or JEV disabled: return 0.0.
    """
    jev_weight = float(jev_weight)
    if not 0.0 <= jev_weight <= 1.0:
        raise ValueError("jev_weight debe estar entre 0 y 1")
    if jev_weight == 0.0:
        return 0.0
    if news_relevant and float(relevance_score or 0.0) >= NEWS_RELEVANCE_THRESHOLD:
        return jev_weight
    if always_on:
        return min(base_weight, jev_weight)
    return 0.0


def _clamp_renorm(probs):
    clamped = {k: max(P_MIN, min(P_MAX, float(probs.get(k, 1 / 3)))) for k in OUTCOME_KEYS}
    total = sum(clamped.values()) or 1.0
    return {k: v / total for k, v in clamped.items()}


def _normalize_jev(probs):
    try:
        values = {k: float(probs[k]) for k in OUTCOME_KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        return None
    total = sum(values.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in values.items()}


def fuse_probabilities(
    base_probs,
    jev,
    news_relevant=False,
    jev_weight=JEV_PRIORITY_WEIGHT,
    always_on=True,
    base_weight=JEV_BASE_WEIGHT,
):
    """
    Fuse Core Softmax probabilities with JEV decisions.
    - Uses Core as fundamental anchor.
    - If news_relevant and threshold passes: applies disruptive news tier (w=0.20).
    - If no disruptive news, but JEV evaluated tactical context: applies tactical base tier (w=0.10).
    - Raw JEV probabilities are smoothed to control for LLM overconfidence.
    """
    base = _clamp_renorm({k: float(base_probs.get(k, 1 / 3)) for k in OUTCOME_KEYS})
    raw_jev_p = _normalize_jev((jev or {}).get("choice_probs"))
    jev_p = smooth_jev_probabilities(raw_jev_p, prior_probs=base) if raw_jev_p else None

    relevance_score = float((jev or {}).get("noticias_relevantes", 0.0) or 0.0)
    jev_ok = bool(jev and not jev.get("fallback") and jev_p)

    if not jev_ok:
        reason = (jev or {}).get("motivo", "sin_jev")
        return {
            "P_adj": base, "P_base": base, "P_jev": None, "P_jev_raw": raw_jev_p,
            "w": 0.0, "vetada": False, "motivo": reason,
            "decision_source": "softmax", "news_relevant": False,
            "news_relevance_score": relevance_score,
            "choque_estilos": (jev or {}).get("choque_estilos"),
        }

    is_disruptive = bool(news_relevant and relevance_score >= NEWS_RELEVANCE_THRESHOLD)
    w = blend_weight(
        news_relevant=is_disruptive,
        relevance_score=relevance_score,
        jev_weight=jev_weight,
        always_on=always_on,
        base_weight=base_weight,
    )

    if w <= 0:
        reason = "peso_jev_configurado_en_cero" if jev_weight == 0 else "sin_peso_fused"
        return {
            "P_adj": base, "P_base": base, "P_jev": jev_p, "P_jev_raw": raw_jev_p,
            "w": 0.0, "vetada": False, "motivo": reason,
            "decision_source": "softmax", "news_relevant": False,
            "news_relevance_score": relevance_score,
            "choque_estilos": (jev or {}).get("choque_estilos"),
        }

    mixed = {k: (1.0 - w) * base[k] + w * jev_p[k] for k in OUTCOME_KEYS}
    fav = max(base, key=base.get)
    confidence = float(jev.get("confidence", 0.0) or 0.0)
    risk = float(jev.get("riesgo_favorito", 0.0) or 0.0)
    veto = bool(risk > 0.85 and confidence > 0.8)

    if is_disruptive:
        reason = "jev_prioridad_noticias_pertinentes"
        source = "jev_priority"
    else:
        reason = "jev_tactico_always_on"
        source = "jev_tactical_base"

    return {
        "P_adj": _clamp_renorm(mixed),
        "P_base": base,
        "P_jev": jev_p,
        "P_jev_raw": raw_jev_p,
        "w": round(w, 4),
        "vetada": veto,
        "favorito_base": fav,
        "motivo": reason,
        "decision_source": source,
        "news_relevant": is_disruptive,
        "news_relevance_score": relevance_score,
        "volatilidad": jev.get("volatilidad"),
        "confidence": confidence,
        "riesgo_favorito": risk,
        "choque_estilos": jev.get("choque_estilos"),
    }
