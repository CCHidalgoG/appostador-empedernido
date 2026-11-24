"""Tests for JEV probability fusion weights and gating tiers."""

from src.apostador.probability_fusion import (
    JEV_BASE_WEIGHT,
    JEV_PRIORITY_WEIGHT,
    blend_weight,
    fuse_probabilities,
)

BASE = {"local": 0.448, "empate": 0.284, "visita": 0.268}  # prior fijo sin ratings Impect
JEV_OVERCONFIDENT = {
    # Patrón real del incidente 2026-10-22: JEV sin datos devuelve ~97% al local
    "choice_probs": {"local": 0.97, "empate": 0.02, "visita": 0.01},
    "noticias_relevantes": 0.0,
    "fallback": False,
}


def test_tactical_base_weight_is_conservative():
    """El tier táctico usa 0.10 (diseño documentado), no el 0.50 del incidente."""
    assert JEV_BASE_WEIGHT == 0.10
    assert JEV_PRIORITY_WEIGHT == 0.50


def test_tactical_tier_does_not_inflate_uncovered_matches():
    """Sin noticias, JEV sobreconfiado solo mueve ~5 pp la probabilidad base."""
    fused = fuse_probabilities(
        BASE, JEV_OVERCONFIDENT, news_relevant=False, jev_weight=0.50, always_on=True,
    )
    assert fused["w"] == 0.10
    assert fused["P_adj"]["local"] < 0.55  # antes: ~0.65 con w=0.50
    assert fused["decision_source"] == "jev_tactical_base"


def test_always_on_disabled_returns_pure_base():
    """Con cobertura de datos insuficiente el tier táctico no mezcla en absoluto."""
    fused = fuse_probabilities(
        BASE, JEV_OVERCONFIDENT, news_relevant=False, jev_weight=0.50, always_on=False,
    )
    assert fused["w"] == 0.0
    assert fused["P_adj"] == fused["P_base"]
    assert fused["decision_source"] == "softmax"


def test_priority_tier_uses_full_configured_weight():
    """Con noticias relevantes por encima del umbral se usa el peso configurado."""
    jev = dict(JEV_OVERCONFIDENT, noticias_relevantes=0.90)
    fused = fuse_probabilities(
        BASE, jev, news_relevant=True, jev_weight=0.50, always_on=False,
    )
    assert fused["w"] == 0.50
    assert fused["decision_source"] == "jev_priority"


def test_blend_weight_caps_base_tier_at_configured_weight():
    assert blend_weight(False, 0.0, jev_weight=0.05, always_on=True) == 0.05
    assert blend_weight(False, 0.0, jev_weight=0.50, always_on=True) == JEV_BASE_WEIGHT
    assert blend_weight(True, 0.90, jev_weight=0.50, always_on=False) == 0.50
    assert blend_weight(False, 0.0, jev_weight=0.0, always_on=True) == 0.0


def test_calibrated_weight_and_prior_anchored_shrinkage():
    from src.apostador.probability_fusion import (
        JEV_CALIBRATED_WEIGHT,
        smooth_jev_probabilities,
    )

    assert JEV_CALIBRATED_WEIGHT == 0.20
    assert blend_weight(True, 0.90, jev_weight=JEV_CALIBRATED_WEIGHT, always_on=False) == 0.20

    raw = {"local": 1.0, "empate": 0.0, "visita": 0.0}
    prior = {"local": 0.60, "empate": 0.25, "visita": 0.15}
    shrunk_prior = smooth_jev_probabilities(raw, shrink_factor=0.35, prior_probs=prior)
    shrunk_uniform = smooth_jev_probabilities(raw, shrink_factor=0.35)

    assert abs(sum(shrunk_prior.values()) - 1.0) < 1e-6
    # El anclaje hacia prior conserva asimetría (local > uniforme y empate > visita)
    assert shrunk_prior["local"] > shrunk_uniform["local"]
    assert shrunk_prior["empate"] > shrunk_prior["visita"]


