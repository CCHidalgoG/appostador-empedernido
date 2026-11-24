"""Diagnóstico de vigencia, conectividad, cuotas y respaldo local de las APIs del sistema.

Permite validar en tiempo real si las credenciales y suscripciones siguen vigentes
(especialmente relevante ante la próxima desconexión de Impect API), además de
auditar la cobertura de datos respaldados localmente en DuckDB para operar en
modo offline/caché cuando una API deje de estar disponible.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv

from .config import DUCKDB_PATH
from .duckdb_store import get_store


def _mask_secret(secret: str | None) -> str:
    """Enmascara una clave o contraseña mostrando solo los primeros y últimos caracteres."""
    if not secret:
        return "No configurada"
    s = str(secret).strip()
    if len(s) <= 6:
        return "***"
    return f"{s[:3]}...{s[-3:]}"


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decodifica sin verificar firma el payload de un JWT si tiene formato estándar."""
    try:
        parts = str(token).split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def check_impect_api(verify_data_access: bool = True) -> dict[str, Any]:
    """Valida el estado y vigencia de la conexión a Impect API V5 (OAuth2 + acceso a iteraciones)."""
    load_dotenv(override=True)
    username = os.getenv("USERNAME_IMPECT")
    password = os.getenv("PASSWORD_IMPECT")

    result: dict[str, Any] = {
        "api": "Impect Customer API V5",
        "key": "impect",
        "configured": bool(username and password),
        "credential_hint": f"{username} / {_mask_secret(password)}" if username else "Sin credenciales",
        "status": "unknown",
        "status_label": "Sin verificar",
        "is_active": False,
        "latency_ms": None,
        "quota_info": "Límite: 10 req/s (OAuth2)",
        "expiration_info": "Próxima desconexión programada",
        "message": "",
        "details": {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if not username or not password:
        result["status"] = "missing_credentials"
        result["status_label"] = "⚠️ Sin Credenciales"
        result["message"] = "Faltan USERNAME_IMPECT o PASSWORD_IMPECT en el archivo .env."
        return result

    try:
        import impectPy as ip
    except ImportError as exc:
        result["status"] = "error"
        result["status_label"] = "❌ Librería Faltante"
        result["message"] = f"impectPy no está instalado: {exc}"
        return result

    t0 = time.perf_counter()
    try:
        token = ip.getAccessToken(username=username, password=password)
        auth_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status"] = "disconnected"
        result["status_label"] = "🔴 Desconectada / Expirada"
        result["message"] = f"Fallo al autenticar con Impect OAuth2 (posible desconexión de cuenta): {exc}"
        return result

    if not token or not isinstance(token, str):
        result["latency_ms"] = auth_ms
        result["status"] = "disconnected"
        result["status_label"] = "🔴 Token Inválido"
        result["message"] = "El servidor de Impect no devolvió un token Bearer válido."
        return result

    jwt_info = _decode_jwt_payload(token)
    exp_ts = jwt_info.get("exp")
    token_exp_str = None
    if isinstance(exp_ts, (int, float)):
        exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
        mins_left = max(0, int((exp_dt - datetime.now(timezone.utc)).total_seconds() // 60))
        token_exp_str = f"Token sesión válido ({mins_left} min restantes)"

    details: dict[str, Any] = {
        "usuario": jwt_info.get("preferred_username") or username,
        "auth_latency_ms": auth_ms,
        "token_sesion": token_exp_str or "Token OAuth2 obtenido",
    }

    if verify_data_access:
        t1 = time.perf_counter()
        try:
            iterations_df = ip.getIterations(token=token)
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            iter_count = len(iterations_df) if iterations_df is not None else 0
            recent_iters = 0
            sample_leagues: list[str] = []
            if iterations_df is not None and not iterations_df.empty:
                if "season" in iterations_df.columns:
                    recent_mask = iterations_df["season"].astype(str).str.contains("2026|26/27|2025|25/26", regex=True)
                    recent_iters = int(recent_mask.sum())
                if "competitionName" in iterations_df.columns:
                    sample_leagues = iterations_df["competitionName"].dropna().unique().tolist()[:8]

            result["latency_ms"] = total_ms
            result["is_active"] = iter_count > 0
            result["status"] = "online" if iter_count > 0 else "restricted"
            result["status_label"] = "🟢 Vigente y Activa" if iter_count > 0 else "🟡 Token OK / Sin Iteraciones"
            result["quota_info"] = f"{iter_count} iteraciones accesibles ({recent_iters} recientes)"
            result["expiration_info"] = token_exp_str or "Suscripción aún vigente"
            result["message"] = (
                f"Conexión Impect V5 vigente. Token OAuth2 activo y acceso verificado a {iter_count} "
                f"competiciones/temporadas ({recent_iters} de ciclos 2025-2027)."
            )
            details["iteraciones_totales"] = iter_count
            details["iteraciones_recientes"] = recent_iters
            details["ligas_muestra"] = sample_leagues
        except Exception as exc:
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            result["status"] = "restricted"
            result["status_label"] = "🟡 Auth OK / Datos Restringidos"
            result["message"] = f"Autenticación exitosa, pero falló la descarga de iteraciones (acceso revocado o en corte): {exc}"
            details["error_datos"] = str(exc)
    else:
        result["latency_ms"] = auth_ms
        result["is_active"] = True
        result["status"] = "online"
        result["status_label"] = "🟢 Token Vigente"
        result["message"] = "Token OAuth2 de Impect obtenido correctamente."

    result["details"] = details
    return result


def check_apifootball() -> dict[str, Any]:
    """Valida el estado, plan, vigencia y cuota diaria restante de API-Football (v3)."""
    load_dotenv(override=True)
    api_key = os.getenv("APIFOOTBALL")

    result: dict[str, Any] = {
        "api": "API-Football (v3)",
        "key": "apifootball",
        "configured": bool(api_key),
        "credential_hint": _mask_secret(api_key),
        "status": "unknown",
        "status_label": "Sin verificar",
        "is_active": False,
        "latency_ms": None,
        "quota_info": "—",
        "expiration_info": "—",
        "message": "",
        "details": {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if not api_key:
        result["status"] = "missing_credentials"
        result["status_label"] = "⚠️ Sin API Key"
        result["message"] = "Falta la variable APIFOOTBALL en .env."
        return result

    t0 = time.perf_counter()
    try:
        resp = requests.get(
            "https://v3.football.api-sports.io/status",
            headers={"x-apisports-key": api_key},
            timeout=15,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        result["latency_ms"] = latency

        if resp.status_code != 200:
            result["status"] = "error"
            result["status_label"] = f"🔴 HTTP {resp.status_code}"
            result["message"] = f"Respuesta inesperada del servidor: {resp.text[:200]}"
            return result

        payload = resp.json() or {}
        errors = payload.get("errors")
        if errors:
            result["status"] = "disconnected"
            result["status_label"] = "🔴 Clave Inválida / Error"
            result["message"] = f"Error reportado por API-Football: {errors}"
            result["details"] = {"errors": errors}
            return result

        response_obj = payload.get("response") or {}
        if isinstance(response_obj, list):
            response_obj = response_obj[0] if response_obj else {}

        account = response_obj.get("account") or {}
        subscription = response_obj.get("subscription") or {}
        reqs = response_obj.get("requests") or {}

        active = bool(subscription.get("active", False))
        plan = subscription.get("plan", "Desconocido")
        sub_end = subscription.get("end", "—")
        used = int(reqs.get("current", 0) or 0)
        limit_day = int(reqs.get("limit_day", 0) or 0)
        remaining = max(0, limit_day - used) if limit_day else 0

        result["is_active"] = active and (remaining > 0 or limit_day == 0)
        if not active:
            result["status"] = "disconnected"
            result["status_label"] = "🔴 Suscripción Inactiva"
        elif limit_day > 0 and remaining == 0:
            result["status"] = "quota_exhausted"
            result["status_label"] = "🟡 Cuota Diaria Agotada"
        else:
            result["status"] = "online"
            result["status_label"] = "🟢 Vigente y Activa"

        result["quota_info"] = f"{used:,} / {limit_day:,} usadas hoy ({remaining:,} libres)"
        result["expiration_info"] = f"Plan {plan} (vence: {str(sub_end)[:10]})"
        result["message"] = (
            f"Cuenta {account.get('email', '—')} | Plan {plan} ({'Activo' if active else 'Inactivo'}) | "
            f"Peticiones hoy: {used}/{limit_day}."
        )
        result["details"] = {
            "email": account.get("email"),
            "nombre": f"{account.get('firstname', '')} {account.get('lastname', '')}".strip(),
            "plan": plan,
            "activa": active,
            "vencimiento": sub_end,
            "peticiones_usadas_hoy": used,
            "limite_diario": limit_day,
            "peticiones_restantes": remaining,
        }
        return result
    except Exception as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status"] = "error"
        result["status_label"] = "🔴 Error de Red"
        result["message"] = f"No se pudo conectar a API-Football: {exc}"
        return result


def check_odds_api() -> dict[str, Any]:
    """Valida el estado y créditos disponibles en The Odds API (v4)."""
    load_dotenv(override=True)
    api_key = os.getenv("ODDSAPI")

    result: dict[str, Any] = {
        "api": "The Odds API (v4)",
        "key": "theoddsapi",
        "configured": bool(api_key),
        "credential_hint": _mask_secret(api_key),
        "status": "unknown",
        "status_label": "Sin verificar",
        "is_active": False,
        "latency_ms": None,
        "quota_info": "—",
        "expiration_info": "Créditos mensuales",
        "message": "",
        "details": {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if not api_key:
        result["status"] = "missing_credentials"
        result["status_label"] = "⚠️ Sin API Key"
        result["message"] = "Falta la variable ODDSAPI en .env."
        return result

    t0 = time.perf_counter()
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/",
            params={"apiKey": api_key},
            timeout=15,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        result["latency_ms"] = latency

        rem_hdr = resp.headers.get("x-requests-remaining")
        used_hdr = resp.headers.get("x-requests-used")
        last_hdr = resp.headers.get("x-requests-last")

        if resp.status_code != 200:
            result["status"] = "disconnected" if resp.status_code in (401, 403) else "error"
            result["status_label"] = f"🔴 HTTP {resp.status_code}"
            result["message"] = f"Error en The Odds API: {resp.text[:200]}"
            return result

        sports = resp.json() if isinstance(resp.json(), list) else []
        soccer_leagues = [s for s in sports if "soccer" in str(s.get("key", "")) and s.get("active")]

        remaining_val = float(rem_hdr) if rem_hdr is not None and str(rem_hdr).replace(".", "", 1).isdigit() else None
        used_val = float(used_hdr) if used_hdr is not None and str(used_hdr).replace(".", "", 1).isdigit() else None

        result["is_active"] = remaining_val is None or remaining_val > 0
        if remaining_val is not None and remaining_val <= 0:
            result["status"] = "quota_exhausted"
            result["status_label"] = "🟡 Sin Créditos"
        else:
            result["status"] = "online"
            result["status_label"] = "🟢 Vigente y Activa"

        if remaining_val is not None and used_val is not None:
            result["quota_info"] = f"{int(remaining_val):,} créditos libres ({int(used_val):,} usados)"
        else:
            result["quota_info"] = f"{len(soccer_leagues)} ligas de fútbol activas"

        result["expiration_info"] = f"{len(soccer_leagues)} torneos de fútbol activos"
        result["message"] = (
            f"Conectada correctamente. {len(soccer_leagues)} ligas de fútbol disponibles. "
            f"Créditos restantes: {rem_hdr if rem_hdr is not None else 'N/D'} (esta verificación cuesta 0 créditos)."
        )
        result["details"] = {
            "creditos_restantes": remaining_val,
            "creditos_usados": used_val,
            "costo_ultima_llamada": last_hdr,
            "deportes_totales": len(sports),
            "ligas_futbol_activas": len(soccer_leagues),
        }
        return result
    except Exception as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status"] = "error"
        result["status_label"] = "🔴 Error de Red"
        result["message"] = f"No se pudo conectar a The Odds API: {exc}"
        return result


def check_brave_api() -> dict[str, Any]:
    """Valida el estado y vigencia de Brave Search News API."""
    load_dotenv(override=True)
    api_key = os.getenv("BRAVEAPI")

    result: dict[str, Any] = {
        "api": "Brave Search News API",
        "key": "brave",
        "configured": bool(api_key),
        "credential_hint": _mask_secret(api_key),
        "status": "unknown",
        "status_label": "Sin verificar",
        "is_active": False,
        "latency_ms": None,
        "quota_info": "—",
        "expiration_info": "Suscripción API",
        "message": "",
        "details": {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if not api_key:
        result["status"] = "missing_credentials"
        result["status_label"] = "⚠️ Sin API Key"
        result["message"] = "Falta la variable BRAVEAPI en .env."
        return result

    t0 = time.perf_counter()
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/news/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            params={"q": "football match", "count": 1, "freshness": "pw"},
            timeout=15,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        result["latency_ms"] = latency

        limit_hdr = resp.headers.get("x-ratelimit-limit")
        rem_hdr = resp.headers.get("x-ratelimit-remaining")
        reset_hdr = resp.headers.get("x-ratelimit-reset")

        if resp.status_code != 200:
            result["status"] = "disconnected" if resp.status_code in (401, 403) else "error"
            result["status_label"] = f"🔴 HTTP {resp.status_code}"
            result["message"] = f"Error en Brave Search API: {resp.text[:200]}"
            return result

        payload = resp.json() or {}
        news_results = payload.get("results") or []

        result["is_active"] = True
        result["status"] = "online"
        result["status_label"] = "🟢 Vigente y Activa"
        result["quota_info"] = f"Rate-limit restante: {rem_hdr or 'OK'} (límite: {limit_hdr or 'estándar'})"
        result["expiration_info"] = f"Reset ventana: {reset_hdr or 'activo'}"
        result["message"] = f"Token de suscripción Brave válido. Búsqueda de noticias operativa ({len(news_results)} resultado de prueba)."
        result["details"] = {
            "ratelimit_limit": limit_hdr,
            "ratelimit_remaining": rem_hdr,
            "ratelimit_reset": reset_hdr,
            "resultados_prueba": len(news_results),
        }
        return result
    except Exception as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status"] = "error"
        result["status_label"] = "🔴 Error de Red"
        result["message"] = f"No se pudo conectar a Brave Search API: {exc}"
        return result


def check_jev_api() -> dict[str, Any]:
    """Valida el estado y vigencia de JEV (TypeSafe System One API)."""
    load_dotenv(override=True)
    api_key = os.getenv("JEVAPI") or os.getenv("TYPESAFE_API_KEY")

    result: dict[str, Any] = {
        "api": "JEV (TypeSafe System One)",
        "key": "jev",
        "configured": bool(api_key),
        "credential_hint": _mask_secret(api_key),
        "status": "unknown",
        "status_label": "Sin verificar",
        "is_active": False,
        "latency_ms": None,
        "quota_info": "Modelo: jev-latest",
        "expiration_info": "Bearer API Key",
        "message": "",
        "details": {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    if not api_key:
        result["status"] = "missing_credentials"
        result["status_label"] = "⚠️ Sin API Key"
        result["message"] = "Falta JEVAPI (o TYPESAFE_API_KEY) en .env."
        return result

    t0 = time.perf_counter()
    try:
        body = {
            "state": {"ping": "Verificación de salud del servicio desde App-ostador"},
            "model": "jev-latest",
            "questions": {
                "activo": {
                    "type": "noul",
                    "instructions": "¿El estado corresponde a una verificación de salud?",
                }
            },
        }
        resp = requests.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=20,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        result["latency_ms"] = latency

        if resp.status_code != 200:
            result["status"] = "disconnected" if resp.status_code in (401, 402, 403) else "error"
            result["status_label"] = f"🔴 HTTP {resp.status_code}"
            result["message"] = f"Error en JEV TypeSafe API: {resp.text[:200]}"
            return result

        payload = resp.json() or {}
        usage = payload.get("usage") or {}
        noul_val = ((payload.get("answers") or {}).get("activo") or {}).get("noul")

        result["is_active"] = True
        result["status"] = "online"
        result["status_label"] = "🟢 Vigente y Activa"
        result["quota_info"] = f"Tokens ping: {usage.get('total_tokens', 'OK')}"
        result["expiration_info"] = "Suscripción TypeSafe activa"
        result["message"] = f"Motor de decisión JEV operativo (respuesta tipada noul={noul_val})."
        result["details"] = {
            "modelo": "jev-latest",
            "respuesta_noul": noul_val,
            "usage": usage,
        }
        return result
    except Exception as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status"] = "error"
        result["status_label"] = "🔴 Error de Red"
        result["message"] = f"No se pudo conectar a JEV API: {exc}"
        return result


def check_clubelo_api() -> dict[str, Any]:
    """Valida la conectividad con la API pública de ClubElo (api.clubelo.com)."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result: dict[str, Any] = {
        "api": "ClubElo API (SoccerData)",
        "key": "clubelo",
        "configured": True,
        "credential_hint": "Pública (Sin clave)",
        "status": "unknown",
        "status_label": "Sin verificar",
        "is_active": False,
        "latency_ms": None,
        "quota_info": "API Abierta",
        "expiration_info": "Permanente",
        "message": "",
        "details": {},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    t0 = time.perf_counter()
    try:
        resp = requests.get(f"http://api.clubelo.com/{today_str}", timeout=12)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        result["latency_ms"] = latency

        if resp.status_code == 200 and len(resp.text.splitlines()) > 10:
            lines_count = len(resp.text.splitlines()) - 1
            result["is_active"] = True
            result["status"] = "online"
            result["status_label"] = "🟢 Vigente y Activa"
            result["quota_info"] = f"{lines_count:,} clubes rankeados hoy"
            result["message"] = f"Feed diario de ClubElo operativo ({lines_count:,} equipos recibidos para {today_str})."
            result["details"] = {"fecha": today_str, "equipos_rankeados": lines_count}
        elif resp.status_code == 502:
            result["status"] = "error"
            result["status_label"] = "🔴 HTTP 502 (Bad Gateway)"
            result["message"] = (
                "El servidor upstream de api.clubelo.com no responde (502 Bad Gateway). "
                "Causa externa en la infraestructura de ClubElo. La app opera en modo contingencia "
                "con ratings locales/default sin afectar las predicciones principales."
            )
            result["details"] = {
                "http_status": 502,
                "server": resp.headers.get("Server", "Microsoft-IIS/10.0"),
                "fecha_consultada": today_str,
                "impacto": "Afecta únicamente la calculadora ClubElo secundaria (Tab 5). El ensamble Impect V5 + JEV opera normalmente.",
            }
        else:
            result["status"] = "error"
            result["status_label"] = f"🔴 HTTP {resp.status_code}"
            result["message"] = f"Respuesta vacía o inesperada de api.clubelo.com (HTTP {resp.status_code})."
            result["details"] = {"http_status": resp.status_code}
        return result
    except Exception as exc:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status"] = "error"
        result["status_label"] = "🔴 Error de Red"
        result["message"] = f"No se pudo conectar a ClubElo API: {exc}"
        return result


def get_duckdb_backup_inventory() -> dict[str, Any]:
    """Audita el respaldo local en DuckDB (especialmente crítico para operar tras la desconexión de Impect)."""
    inventory: dict[str, Any] = {
        "db_exists": DUCKDB_PATH.exists(),
        "impect_artifacts_total": 0,
        "impect_by_type": {},
        "impect_results_count": 0,
        "impect_iterations_cached": 0,
        "impect_last_sync": None,
        "apifootball_artifacts": 0,
        "odds_artifacts": 0,
        "brave_artifacts": 0,
        "historical_matches": 0,
    }
    if not DUCKDB_PATH.exists():
        return inventory

    store = get_store()
    con = store.connect()
    try:
        rows = con.execute("""
            SELECT
                CASE
                    WHEN artifact_key LIKE 'data/raw/impect/ratings_%' THEN 'Squad Ratings'
                    WHEN artifact_key LIKE 'data/raw/impect/averages_%' THEN 'KPI Averages (xG / Packing)'
                    WHEN artifact_key LIKE 'data/raw/impect/scores_%' THEN 'Squad Scores (Field Tilt)'
                    WHEN artifact_key LIKE 'data/raw/impect/player_scores_%' THEN 'Player Talent (Top 11)'
                    WHEN artifact_key LIKE 'data/raw/impect/predictions_%' THEN 'Match Predictions (xG)'
                    WHEN artifact_key LIKE 'data/raw/impect/matches_%' THEN 'Match Schedules'
                    WHEN artifact_key LIKE 'data/raw/impect/%' THEN 'Catálogo / Otros Impect'
                    ELSE 'Otros'
                END AS tipo,
                COUNT(*) AS cantidad,
                MAX(updated_at) AS ultima_actualizacion
            FROM raw.artifacts
            WHERE artifact_key LIKE 'data/raw/impect/%'
            GROUP BY 1
            ORDER BY cantidad DESC
        """).fetchall()

        by_type = {}
        total_imp = 0
        latest_ts = None
        for tipo, cant, max_ts in rows:
            by_type[tipo] = {
                "count": int(cant),
                "last_updated": str(max_ts)[:19] if max_ts else "—",
            }
            total_imp += int(cant)
            if max_ts and (latest_ts is None or str(max_ts) > str(latest_ts)):
                latest_ts = str(max_ts)[:19]

        inventory["impect_artifacts_total"] = total_imp
        inventory["impect_by_type"] = by_type
        inventory["impect_last_sync"] = latest_ts

        imp_res = con.execute("""
            SELECT COUNT(*), COUNT(DISTINCT iteration_id), MAX(fetched_at)
            FROM output.impect_results
        """).fetchone()
        if imp_res:
            inventory["impect_results_count"] = int(imp_res[0] or 0)
            inventory["impect_iterations_cached"] = int(imp_res[1] or 0)

        provider_counts = con.execute("""
            SELECT
                SUM(CASE WHEN artifact_key LIKE 'data/raw/apifootball/%' THEN 1 ELSE 0 END) AS af,
                SUM(CASE WHEN artifact_key LIKE 'data/raw/theoddsapi/%' THEN 1 ELSE 0 END) AS odds,
                SUM(CASE WHEN artifact_key LIKE 'data/raw/brave/%' THEN 1 ELSE 0 END) AS brave
            FROM raw.artifacts
        """).fetchone()
        if provider_counts:
            inventory["apifootball_artifacts"] = int(provider_counts[0] or 0)
            inventory["odds_artifacts"] = int(provider_counts[1] or 0)
            inventory["brave_artifacts"] = int(provider_counts[2] or 0)

        hist_row = con.execute(
            "SELECT COUNT(*) FROM raw.match_history WHERE league != 'TEST-League'"
        ).fetchone()
        if hist_row:
            inventory["historical_matches"] = int(hist_row[0] or 0)

    except Exception:
        pass
    finally:
        con.close()

    return inventory


def check_scrapers_resilience(store: DuckDBStore | None = None) -> dict[str, Any]:
    """Audita resiliencia técnica y cumplimiento ToS de scrapers configurados."""
    from .soccerdata_fbref import SCRAPER_TOS_POLICY

    active_store = store or get_store()
    whoscored_audit = active_store.get("raw", "soccerdata/scraper_audit/whoscored_latest.json")
    blocked_sources = [
        src for src, pol in SCRAPER_TOS_POLICY.items()
        if not pol.get("automated_production_allowed", False)
    ]
    ws_status = (whoscored_audit or {}).get("source_status", "untested")
    overall = "warning" if ws_status in {"schema_error", "error"} else "ok"
    return {
        "service": "Scrapers & Open Data Policy",
        "status": overall,
        "status_label": "⚠️ Degradado (Fallback activo)" if overall == "warning" else "🟢 Conforme",
        "whoscored_audit": whoscored_audit or {"source_status": "untested"},
        "tos_blocked_for_automated_production": blocked_sources,
        "policy": SCRAPER_TOS_POLICY,
    }


def run_all_api_checks(include_jev: bool = True, include_brave: bool = True) -> list[dict[str, Any]]:
    """Ejecuta la validación en vivo sobre todas las APIs configuradas y retorna sus diagnósticos."""
    results = [
        check_impect_api(verify_data_access=True),
        check_apifootball(),
        check_odds_api(),
    ]
    if include_brave:
        results.append(check_brave_api())
    if include_jev:
        results.append(check_jev_api())
    results.append(check_clubelo_api())
    return results

