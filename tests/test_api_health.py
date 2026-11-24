"""Pruebas unitarias para el módulo de salud y vigencia de APIs (api_health.py)."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from apostador.api_health import (
    _decode_jwt_payload,
    _mask_secret,
    check_apifootball,
    check_clubelo_api,
    check_impect_api,
    check_odds_api,
    get_duckdb_backup_inventory,
)


def test_mask_secret():
    assert _mask_secret(None) == "No configurada"
    assert _mask_secret("abc") == "***"
    assert _mask_secret("1234567890") == "123...890"


def test_decode_jwt_payload():
    payload = {"preferred_username": "user@test.com", "exp": 1999999999}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    token = f"eyJhbGciOiJIUzI1NiJ9.{encoded}.signature"
    decoded = _decode_jwt_payload(token)
    assert decoded["preferred_username"] == "user@test.com"
    assert decoded["exp"] == 1999999999


@patch("apostador.api_health.os.getenv")
def test_check_impect_missing_credentials(mock_getenv):
    mock_getenv.return_value = None
    res = check_impect_api(verify_data_access=False)
    assert res["status"] == "missing_credentials"
    assert res["is_active"] is False


@patch("apostador.api_health.requests.get")
@patch("apostador.api_health.os.getenv")
def test_check_apifootball_online(mock_getenv, mock_get):
    mock_getenv.return_value = "abcdef123456"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "errors": [],
        "response": {
            "account": {"email": "test@example.com", "firstname": "Test", "lastname": "User"},
            "subscription": {"plan": "Pro", "active": True, "end": "2027-01-01T00:00:00+00:00"},
            "requests": {"current": 15, "limit_day": 100},
        },
    }
    mock_get.return_value = mock_resp

    res = check_apifootball()
    assert res["is_active"] is True
    assert res["status"] == "online"
    assert res["details"]["peticiones_restantes"] == 85


@patch("apostador.api_health.requests.get")
@patch("apostador.api_health.os.getenv")
def test_check_odds_api_online(mock_getenv, mock_get):
    mock_getenv.return_value = "oddskey123456"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"x-requests-remaining": "420", "x-requests-used": "80", "x-requests-last": "0"}
    mock_resp.json.return_value = [
        {"key": "soccer_epl", "active": True},
        {"key": "soccer_spain_la_liga", "active": True},
    ]
    mock_get.return_value = mock_resp

    res = check_odds_api()
    assert res["is_active"] is True
    assert res["status"] == "online"
    assert res["details"]["ligas_futbol_activas"] == 2


def test_duckdb_backup_inventory():
    inv = get_duckdb_backup_inventory()
    assert "impect_artifacts_total" in inv
    assert "impect_results_count" in inv


@patch("apostador.api_health.requests.get")
def test_check_clubelo_api_502(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.headers = {"Server": "Microsoft-IIS/10.0"}
    mock_resp.text = ""
    mock_get.return_value = mock_resp

    res = check_clubelo_api()
    assert res["status"] == "error"
    assert "502" in res["status_label"]
    assert res["is_active"] is False
    assert res["details"]["http_status"] == 502


@patch("apostador.api_health.requests.get")
def test_check_clubelo_api_online(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "Rank/Club/Country/Level/Elo\n" + "\n".join([f"{i}/Club{i}/ESP/1/1600" for i in range(20)])
    mock_get.return_value = mock_resp

    res = check_clubelo_api()
    assert res["is_active"] is True
    assert res["status"] == "online"
    assert res["details"]["equipos_rankeados"] == 20

