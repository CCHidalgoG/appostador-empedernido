import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.evaluate_b2b_samples import B2BSampleEvaluator
from src.apostador.duckdb_store import DuckDBStore
from src.apostador.impect_team_metrics import (
    ImpectTeamMetricsLoader,
    resolve_as_of_impect_records,
)
from src.apostador.predictor import SoccerPredictor
from src.apostador.skillcorner_fatigue import (
    SkillCornerFatigueLoader,
    compute_soccer_fatigue_delta,
)


def test_impect_as_of_snapshots_and_pxt_psxg_loading(tmp_path, monkeypatch):
    """TASK-13: Snapshots as-of evitan descartar averages históricos e integran PXT y POST_SHOT_XG."""
    def _local_read_json(p, default=None):
        fp = Path(p)
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else default

    monkeypatch.setattr("src.apostador.impect_team_metrics.read_json", _local_read_json)
    monkeypatch.setattr("src.apostador.predictor.read_json", _local_read_json)

    (tmp_path / "ratings_900.json").write_text(
        json.dumps([
            {"date": "2026-09-01T00:00:00Z", "squadName": "Equipo Uno", "squadId": 10, "value": 0.72},
            {"date": "2026-09-01T00:00:00Z", "squadName": "Equipo Dos", "squadId": 20, "value": 0.66},
        ]),
        encoding="utf-8",
    )
    (tmp_path / "averages_900_2026-09-01.json").write_text(
        json.dumps([
            {
                "squadName": "Equipo Uno",
                "as_of_date": "2026-09-01",
                "BYPASSED_DEFENDERS": 45.0,
                "SHOT_XG": 1.75,
                "CONCEDED_SHOT_XG": 0.90,
                "POST_SHOT_XG": 1.95,
                "PXT": 0.42,
                "SHOT_AT_GOAL_NUMBER": 14.0,
            },
            {
                "squadName": "Equipo Dos",
                "as_of_date": "2026-09-01",
                "BYPASSED_DEFENDERS": 32.0,
                "SHOT_XG": 1.05,
                "CONCEDED_SHOT_XG": 1.45,
                "POST_SHOT_XG": 0.85,
                "PXT": 0.14,
                "SHOT_AT_GOAL_NUMBER": 9.0,
            },
        ]),
        encoding="utf-8",
    )
    (tmp_path / "averages_900_2026-10-01.json").write_text(
        json.dumps([
            {
                "squadName": "Equipo Uno",
                "as_of_date": "2026-10-01",
                "BYPASSED_DEFENDERS": 52.0,
                "SHOT_XG": 2.10,
                "CONCEDED_SHOT_XG": 0.75,
                "POST_SHOT_XG": 2.30,
                "PXT": 0.55,
                "SHOT_AT_GOAL_NUMBER": 16.0,
            },
            {
                "squadName": "Equipo Dos",
                "as_of_date": "2026-10-01",
                "BYPASSED_DEFENDERS": 30.0,
                "SHOT_XG": 0.95,
                "CONCEDED_SHOT_XG": 1.50,
                "POST_SHOT_XG": 0.80,
                "PXT": 0.12,
                "SHOT_AT_GOAL_NUMBER": 8.0,
            },
        ]),
        encoding="utf-8",
    )

    rows_sep, is_safe = resolve_as_of_impect_records(
        tmp_path, "averages", 900, target_date="2026-09-15", reader=_local_read_json
    )
    assert is_safe is True
    assert len(rows_sep) == 2
    assert rows_sep[0]["PXT"] == pytest.approx(0.42)

    loader = ImpectTeamMetricsLoader(raw_impect_dir=str(tmp_path))
    df_sep = loader.load_team_features(900, target_date="2026-09-15")
    assert not df_sep.empty
    assert df_sep.loc["Equipo Uno", "PXT"] == pytest.approx(0.42)
    assert df_sep.loc["Equipo Uno", "POST_SHOT_XG"] == pytest.approx(1.95)
    assert df_sep.loc["Equipo Uno", "SHOT_AT_GOAL_NUMBER"] == pytest.approx(14.0)

    summary_sep = loader.get_team_tactical_summary(900, "Equipo Uno", target_date="2026-09-15")
    assert summary_sep["pxt_partido"] == pytest.approx(0.42)
    assert summary_sep["psxg_partido"] == pytest.approx(1.95)
    assert summary_sep["remates_partido"] == pytest.approx(14.0)

    summary_oct = loader.get_team_tactical_summary(900, "Equipo Uno", target_date="2026-10-15")
    assert summary_oct["pxt_partido"] == pytest.approx(0.55)

    pred = SoccerPredictor(raw_impect_dir=str(tmp_path))
    out = pred.predict_match("Equipo Uno", "Equipo Dos", iteration_id=900, match_date="2026-09-15")
    assert out["signal_source"] == "impect"
    assert out["P_H"] > out["P_A"]


def test_team_crosswalk_b2b_ids_and_sample_evaluator(tmp_path):
    """TASK-14: Extiende output.team_crosswalk y evalúa muestras Wyscout v3, StatsBomb 360 y Sportradar."""
    store = DuckDBStore(tmp_path / "b2b_crosswalk.duckdb")

    n_ingested = store.ingest_hudl_id_mapping(
        [
            {
                "team_name": "Atlético Nacional",
                "impect_squad_id": 501,
                "impect_squad_name": "Atlético Nacional",
                "wyscout_team_id": 8801,
                "statsbomb_team_id": 9901,
                "sportradar_id": "sr:competitor:7001",
                "skillcorner_id": 4401,
            }
        ],
        iteration_id=1752,
    )
    assert n_ingested == 1

    saved = store.get_team_crosswalk("hudl_mapping", 8801, 1752)
    assert saved is not None
    assert saved["wyscout_id"] == 8801
    assert saved["statsbomb_id"] == 9901
    assert saved["sportradar_id"] == "sr:competitor:7001"
    assert saved["skillcorner_id"] == 4401

    evaluator = B2BSampleEvaluator()
    wy_sample = evaluator.normalize_wyscout_v3_sample({
        "matchId": 555001,
        "teams": {
            "8801": {"name": "Atlético Nacional"},
            "8802": {"name": "Millonarios"},
        },
        "events": [
            {"teamId": 8801, "type": {"primary": "pass"}},
            {"teamId": 8801, "type": {"primary": "shot"}, "shot": {"xg": 0.31}},
            {"teamId": 8802, "type": {"primary": "shot"}, "shot": {"xg": 0.12}},
        ],
    })
    assert wy_sample["provider"] == "hudl_wyscout_v3"
    assert wy_sample["total_events"] == 3

    sb_sample = evaluator.normalize_statsbomb_360_sample(
        events=[
            {"id": "ev-1", "team": {"id": 9901, "name": "Atletico Nacional"}, "type": {"name": "Shot"}, "shot": {"statsbomb_xg": 0.34}, "obv_total_net": 0.18},
            {"id": "ev-2", "team": {"id": 9902, "name": "Millonarios FC"}, "type": {"name": "Pass"}, "obv_total_net": 0.02},
        ],
        frames_360=[{"event_uuid": "ev-1", "visible_area": [0.0, 0.0], "freeze_frame": []}],
        id_mapping=[{"statsbomb_team_id": 9901, "wyscout_team_id": 8801}],
    )
    assert sb_sample["has_360_context"] is True
    assert sb_sample["has_obv_or_pxt"] is True
    assert sb_sample["teams"][0]["wyscout_id"] == 8801

    sr_sample = evaluator.normalize_sportradar_trial_sample({
        "sport_event": {
            "id": "sr:sport_event:12345",
            "competitors": [
                {"id": "sr:competitor:7001", "name": "Atl. Nacional", "qualifier": "home"},
                {"id": "sr:competitor:7002", "name": "Millonarios FC", "qualifier": "away"},
            ],
        },
        "sport_event_status": {"status": "closed"},
        "statistics": {
            "totals": {
                "competitors": [
                    {"id": "sr:competitor:7001", "statistics": {"shots_total": 12, "shots_on_target": 5, "xg": 1.65}},
                    {"id": "sr:competitor:7002", "statistics": {"shots_total": 8, "shots_on_target": 3, "xg": 0.88}},
                ]
            }
        },
    })
    assert sr_sample["provider"] == "sportradar_extended_v4"
    assert len(sr_sample["teams"]) == 2

    known_impect = ["Atlético Nacional", "Millonarios FC"]
    report = evaluator.evaluate_crosswalk_alignment(
        [wy_sample, sb_sample, sr_sample],
        known_impect_teams=known_impect,
        store=store,
        iteration_id=1752,
        impect_ids_by_name={"Atlético Nacional": 501, "Millonarios FC": 502},
    )
    assert report["overall_crosswalk_match_rate"] == pytest.approx(1.0)
    assert report["persisted_crosswalk_rows"] == 6


def test_skillcorner_fatigue_delta_and_predictor_integration(tmp_path):
    """TASK-15: Calcula fatigue_delta desde SkillCorner Physical/Tracking y lo integra en SoccerPredictor."""
    tracking_file = tmp_path / "sc_tracking.jsonl"
    frames = []
    for f in range(30):
        # Dos jugadores simulados moviéndose a ~6 m/s (HSR) y ~8 m/s (Sprint) con is_detected=True
        frames.append(json.dumps({
            "frame": f,
            "player_data": [
                {"player_id": 1, "x": f * 0.62, "y": 10.0, "is_detected": True},
                {"player_id": 2, "x": f * 0.80, "y": 25.0, "is_detected": True},
                {"player_id": 3, "x": 50.0, "y": 50.0, "is_detected": False},
            ],
        }))
    tracking_file.write_text("\n".join(frames), encoding="utf-8")

    loader = SkillCornerFatigueLoader()
    phys_summary = loader.summarize_from_opendata_match(tracking_file, recent_matches_7d=2)
    assert phys_summary["high_speed_running_m"] > 0.0
    assert phys_summary["sprint_distance_m"] > 0.0

    fat = compute_soccer_fatigue_delta(
        home_physical={"high_speed_running_m": 8800.0, "sprint_distance_m": 1500.0, "recent_matches_7d": 1},
        away_physical={"high_speed_running_m": 22500.0, "sprint_distance_m": 4400.0, "pressing_actions": 410.0, "recent_matches_7d": 2},
        home_rest_days=6.0,
        away_rest_days=2.5,
    )
    assert fat["source"] == "skillcorner_physical"
    assert 0.0 < fat["fatigue_delta"] <= 0.06
    assert fat["away_fatigue_index"] > fat["home_fatigue_index"]

    pred = SoccerPredictor()
    res_neutral = pred.predict_match("Team A", "Team B", iteration_id=None)
    res_away_tired = pred.predict_match(
        "Team A",
        "Team B",
        iteration_id=None,
        home_rest_days=6.0,
        away_rest_days=2.0,
        away_physical={"high_speed_running_m": 22000.0, "sprint_distance_m": 4200.0, "recent_matches_7d": 2},
    )
    assert res_neutral["fatigue_delta"] == 0.0
    assert res_away_tired["fatigue_delta"] > 0.0
    assert res_away_tired["P_H"] > res_neutral["P_H"]
    assert np.isclose(res_away_tired["P_H"] + res_away_tired["P_D"] + res_away_tired["P_A"], 1.0, atol=1e-4)


def test_rfp_document_contains_12_blocks_and_vendors():
    """TASK-16: Valida el Pliego RFP Corporativo de 12 Bloques y sus plantillas por proveedor."""
    rfp_path = Path(__file__).resolve().parents[2] / "research" / "rfp_proveedores_futbol_b2b.md"
    assert rfp_path.exists()
    content = rfp_path.read_text(encoding="utf-8")
    for vendor in ("Hudl", "Wyscout", "StatsBomb", "Stats Perform", "SkillCorner", "IMPECT", "Sportradar", "Genius Sports"):
        assert vendor in content
    for block_num in range(1, 13):
        assert f"| **{block_num}** |" in content
    for variant in ("Variante A", "Variante B", "Variante C"):
        assert variant in content


def test_clubelo_load_by_date_alias_and_impect_403_cache_snapshot(tmp_path, monkeypatch):
    """Valida ClubEloLoader.load_by_date() con CSV de respuesta y preservación as-of ante HTTP 403."""
    from types import SimpleNamespace
    from src.apostador.soccerdata_clubelo import ClubEloLoader
    from src.commands import backfill_impect_data as bfi

    store = DuckDBStore(tmp_path / "ops_fix.duckdb")
    ClubEloLoader._network_down = False
    ClubEloLoader._failed_dates = set()

    csv_body = (
        "Rank,Club,Country,Level,Elo,From,To\n"
        "1,Arsenal,ENG,1,1965.4,2026-11-01,2026-11-20\n"
        "2,Liverpool,ENG,1,1952.1,2026-11-01,2026-11-20\n"
    )
    monkeypatch.setattr(
        "requests.get",
        lambda url, timeout=5.0: SimpleNamespace(status_code=200, text=csv_body),
    )
    monkeypatch.setattr("src.apostador.soccerdata_clubelo.write_json", lambda *a, **kw: None)

    loader = ClubEloLoader(store=store)
    n_saved = loader.load_by_date("2026-11-20")
    assert n_saved == 2
    assert store.get_clubelo_rating("Arsenal", "2026-11-20") == pytest.approx(1965.4)

    # Simular caché previa de Impect sin snapshot fechado tras un 403 en API
    store.put("raw", "impect/ratings_2110.json", [{"date": "2026-10-18", "squadName": "Real Madrid", "value": 0.81}])
    store.put("raw", "impect/averages_2110.json", [{"squadName": "Real Madrid", "SHOT_XG": 2.05}])

    written_keys: list[str] = []
    monkeypatch.setattr(
        bfi,
        "write_json",
        lambda p, payload: written_keys.append(Path(p).name),
    )
    assert bfi._is_forbidden_error(RuntimeError("Received status code 403 (You do not have access to this resource.)"))
    preserved = bfi._ensure_cached_snapshot_on_403(2110, store=store, fallback_date="2026-11-21")
    assert preserved == ["ratings", "averages"]
    assert "ratings_2110_2026-10-18.json" in written_keys
    assert "averages_2110_2026-10-18.json" in written_keys

