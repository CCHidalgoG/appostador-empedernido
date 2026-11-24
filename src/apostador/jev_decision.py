"""jev_decision.py — Modelo de decision JEV (TypeSafe System One).

POST https://api.typesafe.ai/v1/systemone con
Header Authorization: Bearer <JEVAPI|TYPESAFE_API_KEY>.
Ver documentacion/jev/README.md. Ante fallos devuelve fallback.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

VOLATILITY_LEVELS = ["estable", "tensa", "critica"]
VOLATILITY_WEIGHT = {"estable": 0.35, "tensa": 0.7, "critica": 1.0}
NEWS_RELEVANCE_THRESHOLD = 0.50


def build_match_state(home_team, away_team, league, base_probs, home_rating,
                      away_rating, form_delta, odds=None, news_digest=None, news_sources=None,
                      news_evidence=None, structured_context=None, tactical_context=None,
                      home_advantage=0.05, adjusted_diff=None,
                      home_missing_impact=0.0, away_missing_impact=0.0,
                      home_missing_value_pct=None, away_missing_value_pct=None):
    fav = max(base_probs, key=base_probs.get)
    diff = float(home_rating) - float(away_rating)
    adj_x = adjusted_diff if adjusted_diff is not None else (diff + float(home_advantage or 0) + float(form_delta or 0))

    structured = structured_context or {}
    forma_context = structured.get("forma_ultimos_5_y_descanso") or {}
    provider_ctx = structured.get("api_football") or {}

    h_miss = float(home_missing_impact or 0.0)
    a_miss = float(away_missing_impact or 0.0)
    h_val_pct = float(home_missing_value_pct) if home_missing_value_pct is not None else round(h_miss * 100.0, 2)
    a_val_pct = float(away_missing_value_pct) if away_missing_value_pct is not None else round(a_miss * 100.0, 2)

    state = {
        "partido": f"{home_team} vs {away_team}",
        "liga": league,
        "modelo_core": {
            "probabilidades_base": {k: round(float(v), 4) for k, v in base_probs.items()},
            "favorito_core": fav,
            "rating_local": round(float(home_rating), 4),
            "rating_visitante": round(float(away_rating), 4),
            "diferencia_ratings": round(diff, 4),
            "ventaja_localia": round(float(home_advantage), 4),
            "ajuste_forma_delta": round(float(form_delta), 4),
            "diferencia_neta_ajustada_x": round(float(adj_x), 4),
        },
        "perfil_tactico_impect": tactical_context or {},
        "forma_reciente_y_calendario": forma_context,
        "impacto_bajas_valoracion": {
            "local_missing_share_impact": round(h_miss, 4),
            "visitante_missing_share_impact": round(a_miss, 4),
            "local_missing_value_pct": round(h_val_pct, 2),
            "visitante_missing_value_pct": round(a_val_pct, 2),
        },
        "proveedor_deportivo": {
            "lesiones_reportadas": provider_ctx.get("injuries") or [],
            "alineaciones": provider_ctx.get("lineups") or [],
            "alineacion_disponible": provider_ctx.get("lineups_available", False),
        },
        "cuotas_mercado": odds or {},
        "noticias_prensa": {
            "resumen": (news_digest or "")[:2500] if news_digest else "Sin noticias relevantes de última hora reportadas.",
            "fuentes": (news_sources or [])[:8],
            "evidencias": (news_evidence or [])[:8],
        },
        # Campos de compatibilidad con versiones anteriores
        "prob_base": {k: round(float(v), 4) for k, v in base_probs.items()},
        "favorito_base": fav,
        "ratings": {"local": round(float(home_rating), 4), "visita": round(float(away_rating), 4)},
        "form_delta": round(float(form_delta), 4),
        "cuotas": odds or {},
        "noticias": (news_digest or "")[:2500],
        "contexto_deportivo": structured,
        "consigna": ("Usa el modelo core como ancla probabilística fundamental. "
                     "Analiza conjuntamente las métricas tácticas avanzadas (xG, solidez, pases entre líneas, presiones), "
                     "el descanso/calendario y las novedades o bajas de plantilla. "
                     "No trates rumores como hechos y mantén rigor: si el choque táctico es parejo y no hay bajas "
                     "críticas, conserva una calibración consistente con el core."),
    }
    return state


def build_questions():
    return {
        "resultado": {
            "type": "choice",
            "instructions": ("Considerando el modelo core base, el perfil táctico de ambos equipos (xG a favor/contra, "
                             "pases que rompen líneas, presiones), el descanso y las novedades de plantilla/noticias, "
                             "¿cuál es la distribución de probabilidad más adecuada para el resultado del partido?"),
            "criteria": {
                "local": "Victoria del equipo local",
                "empate": "El partido termina en empate",
                "visita": "Victoria del equipo visitante",
            },
        },
        "choque_estilos": {
            "type": "choice",
            "instructions": "¿Qué equipo presenta una ventaja en el balance táctico y estilo de juego?",
            "criteria": {
                "local": "Ventaja táctica y de métricas para el local",
                "parejo": "Duelo táctico equilibrado o sin ventaja clara",
                "visita": "Ventaja táctica y de métricas para el visitante",
            },
        },
        "noticias_relevantes": {
            "type": "noul",
            "instructions": ("¿Existen noticias, bajas de jugadores clave, sanciones o rotaciones confirmadas "
                             "que alteren significativamente el potencial de alguno de los dos equipos?"),
            "criteria": {
                "true": "Hay una baja clave, sanción, cambio de técnico o rotación masiva confirmada con impacto material.",
                "false": "No hay noticias relevantes ni bajas disruptivas reportadas; el partido se rige por su perfil táctico normal.",
            },
        },
        "riesgo_favorito": {
            "type": "noul",
            "instructions": "¿Las métricas tácticas, fatiga o novedades elevan el riesgo de que falle el favorito del core?",
        },
        "prob_sorpresa": {
            "type": "noul",
            "instructions": "¿Hay señal concreta de sorpresa táctica o circunstancial (baja clave, sanción, clima extremo)?",
        },
        "volatilidad": {
            "type": "score",
            "instructions": "¿Cuánta incertidumbre táctica o de contexto presenta el pronóstico?",
            "criteria": VOLATILITY_LEVELS,
        },
    }


def fallback_decision(reason="sin_noticias"):
    return {
        "choice_probs": None, "choice": None, "confidence": 0.0,
        "noticias_relevantes": 0.0,
        "riesgo_favorito": 0.0, "prob_sorpresa": 0.0,
        "volatilidad": "estable", "fallback": True,
        "motivo": reason, "raw": None, "usage": {},
    }

class JevDecisionClient:
    """Llamada HTTP a JEV con reintentos y parseo tipado."""

    def __init__(self, api_key=None, model=JEV_MODEL, timeout=30, max_retries=2):
        self.api_key = api_key or os.getenv("JEVAPI") or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Falta JEVAPI (o TYPESAFE_API_KEY) en .env")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def decide(self, state, questions=None):
        questions = questions or build_questions()
        body = {"state": state, "model": self.model, "questions": questions}
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    JEV_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return self.parse(resp.json())
            except Exception as exc:  # noqa: BLE001 - JEV nunca rompe el pipeline
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        return fallback_decision(reason=f"jev_error: {last_err}")

    @staticmethod
    def parse(payload):
        answers = (payload or {}).get("answers", {})
        res = answers.get("resultado", {})
        out = fallback_decision(reason="ok")
        out["fallback"] = False
        out["motivo"] = "ok"
        probs = res.get("probabilities") or {}
        if probs:
            total = sum(float(v) for v in probs.values()) or 1.0
            out["choice_probs"] = {k: float(v) / total for k, v in probs.items()}
        out["choice"] = res.get("choice")
        out["confidence"] = float(res.get("confidence", 0.0) or 0.0)
        out["choque_estilos"] = answers.get("choque_estilos", {}).get("choice")
        for key in ("noticias_relevantes", "riesgo_favorito", "prob_sorpresa"):
            out[key] = float((answers.get(key, {}).get("noul", 0.0)) or 0.0)
        vol = answers.get("volatilidad", {})
        score = vol.get("score", "estable")
        if isinstance(score, (int, float)):
            score = VOLATILITY_LEVELS[max(0, min(2, int(round(score))))]
        elif isinstance(score, dict):
            score = max(score, key=score.get)
        if score not in VOLATILITY_LEVELS:
            score = "estable"
        out["volatilidad"] = score
        out["raw"] = payload
        out["usage"] = (payload or {}).get("usage", {})
        return out

