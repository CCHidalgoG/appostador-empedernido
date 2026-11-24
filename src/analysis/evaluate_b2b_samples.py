"""Evaluador técnico de muestras B2B (Hudl Wyscout v3, StatsBomb 360 y Sportradar Trial).

Permite normalizar payloads de prueba de proveedores institucionales sobre un mismo
partido benchmark, medir completitud de eventos/xG/OBV/360 y evaluar la tasa de
alineación automática contra `output.team_crosswalk` en DuckDB.
"""
from __future__ import annotations

import hashlib
from typing import Any

from src.apostador.aligner import TeamAligner
from src.apostador.duckdb_store import DuckDBStore


class B2BSampleEvaluator:
    """Normalizador y evaluador comparativo de feeds B2B para TASK-14."""

    def __init__(self, aligner: TeamAligner | None = None):
        self.aligner = aligner or TeamAligner()

    @staticmethod
    def _stable_provider_int_id(raw_id: Any) -> int:
        if raw_id is None:
            return 0
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            digest = hashlib.sha1(str(raw_id).encode("utf-8")).hexdigest()[:12]
            return int(digest, 16)

    def normalize_wyscout_v3_sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normaliza una muestra de Hudl Wyscout API v3 (Events Pack / Stats Pack)."""
        events = payload.get("events") or []
        teams_meta = payload.get("teams") or {}

        team_stats: dict[str, dict[str, Any]] = {}
        for ev in events:
            tid = ev.get("teamId")
            t_info = teams_meta.get(str(tid), {}) if isinstance(teams_meta, dict) else {}
            t_name = str(t_info.get("name") or ev.get("teamName") or f"wyscout_{tid}")
            bucket = team_stats.setdefault(
                t_name,
                {"wyscout_id": int(tid) if tid is not None else None, "events": 0, "passes": 0, "shots": 0, "xg": 0.0},
            )
            bucket["events"] += 1
            ev_type = str(((ev.get("type") or {}).get("primary")) or ev.get("eventName") or "").lower()
            if "pass" in ev_type:
                bucket["passes"] += 1
            if "shot" in ev_type:
                bucket["shots"] += 1
                shot_obj = ev.get("shot") or {}
                bucket["xg"] += float(shot_obj.get("xg") or 0.0)

        if not team_stats and isinstance(teams_meta, dict):
            for tid, t_info in teams_meta.items():
                t_name = str(t_info.get("name") or f"wyscout_{tid}")
                team_stats[t_name] = {
                    "wyscout_id": int(t_info.get("wyId") or tid),
                    "events": int(t_info.get("events") or 0),
                    "passes": int(t_info.get("passes") or 0),
                    "shots": int(t_info.get("shots") or 0),
                    "xg": float(t_info.get("xg") or 0.0),
                }

        return {
            "provider": "hudl_wyscout_v3",
            "match_id": payload.get("matchId"),
            "total_events": len(events) if events else sum(v["events"] for v in team_stats.values()),
            "has_360_context": False,
            "has_obv_or_pxt": False,
            "teams": [
                {
                    "team_name": name,
                    "wyscout_id": vals["wyscout_id"],
                    "events": vals["events"],
                    "passes": vals["passes"],
                    "shots": vals["shots"],
                    "xg": round(vals["xg"], 4),
                }
                for name, vals in team_stats.items()
            ],
        }

    def normalize_statsbomb_360_sample(
        self,
        events: list[dict[str, Any]],
        frames_360: list[dict[str, Any]] | None = None,
        id_mapping: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Normaliza eventos StatsBomb Pro + 360 freeze-frames y cruce StatsBomb↔Wyscout."""
        frames_by_id = {
            f.get("event_uuid"): f
            for f in (frames_360 or [])
            if isinstance(f, dict) and f.get("event_uuid")
        }
        sb_to_wy = {}
        for m in (id_mapping or []):
            sb_id = m.get("statsbomb_team_id") or m.get("statsbomb_id")
            wy_id = m.get("wyscout_team_id") or m.get("wyscout_id")
            if sb_id is not None and wy_id is not None:
                sb_to_wy[int(sb_id)] = int(wy_id)

        team_stats: dict[str, dict[str, Any]] = {}
        events_with_360 = 0
        has_obv = False

        for ev in events:
            team_obj = ev.get("team") or {}
            t_name = str(team_obj.get("name") or "Unknown")
            sb_tid = team_obj.get("id")
            bucket = team_stats.setdefault(
                t_name,
                {
                    "statsbomb_id": int(sb_tid) if sb_tid is not None else None,
                    "wyscout_id": sb_to_wy.get(int(sb_tid)) if sb_tid is not None else None,
                    "events": 0,
                    "shots": 0,
                    "xg": 0.0,
                    "obv_net": 0.0,
                },
            )
            bucket["events"] += 1
            if ev.get("id") in frames_by_id:
                events_with_360 += 1
            obv = ev.get("obv_total_net")
            if obv is not None:
                has_obv = True
                bucket["obv_net"] += float(obv)
            ev_type = str(((ev.get("type") or {}).get("name")) or "").lower()
            if ev_type == "shot":
                bucket["shots"] += 1
                bucket["xg"] += float((ev.get("shot") or {}).get("statsbomb_xg") or 0.0)

        return {
            "provider": "hudl_statsbomb_360",
            "total_events": len(events),
            "events_with_360": events_with_360,
            "has_360_context": events_with_360 > 0 or bool(frames_360),
            "has_obv_or_pxt": has_obv,
            "teams": [
                {
                    "team_name": name,
                    "statsbomb_id": vals["statsbomb_id"],
                    "wyscout_id": vals["wyscout_id"],
                    "events": vals["events"],
                    "shots": vals["shots"],
                    "xg": round(vals["xg"], 4),
                    "obv_net": round(vals["obv_net"], 4),
                }
                for name, vals in team_stats.items()
            ],
        }

    def normalize_sportradar_trial_sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normaliza muestra de Sportradar Soccer Extended API v4 (trial 30 días / 1.000 req)."""
        sport_event = payload.get("sport_event") or payload
        competitors = sport_event.get("competitors") or []
        statistics = ((payload.get("statistics") or {}).get("totals") or {}).get("competitors") or []
        stats_by_id = {str(c.get("id")): (c.get("statistics") or {}) for c in statistics}
        timeline = payload.get("timeline") or []

        teams_out: list[dict[str, Any]] = []
        for comp in competitors:
            sr_id = str(comp.get("id") or "")
            t_stats = stats_by_id.get(sr_id, {})
            xg_val = t_stats.get("xg") or t_stats.get("expected_goals")
            teams_out.append({
                "team_name": str(comp.get("name") or sr_id),
                "sportradar_id": sr_id,
                "qualifier": comp.get("qualifier"),
                "shots_total": int(t_stats.get("shots_total") or 0),
                "shots_on_target": int(t_stats.get("shots_on_target") or 0),
                "xg": float(xg_val) if xg_val is not None else None,
            })

        return {
            "provider": "sportradar_extended_v4",
            "event_id": sport_event.get("id"),
            "status": (payload.get("sport_event_status") or {}).get("status"),
            "total_events": len(timeline) if timeline else len(competitors),
            "has_360_context": False,
            "has_obv_or_pxt": False,
            "teams": teams_out,
        }

    def evaluate_crosswalk_alignment(
        self,
        samples: list[dict[str, Any]],
        known_impect_teams: list[str],
        store: DuckDBStore | None = None,
        iteration_id: int = 0,
        impect_ids_by_name: dict[str, int] | None = None,
        threshold: float = 0.78,
    ) -> dict[str, Any]:
        """Evalúa tasa de cruce de nombres contra Impect y persiste IDs en output.team_crosswalk."""
        impect_ids_by_name = impect_ids_by_name or {}
        provider_reports: list[dict[str, Any]] = []
        crosswalk_rows: list[dict[str, Any]] = []
        total_teams = 0
        total_matched = 0

        for sample in samples:
            provider = str(sample.get("provider") or "unknown")
            teams = sample.get("teams") or []
            matched = 0
            unmatched: list[str] = []

            for t in teams:
                total_teams += 1
                t_name = str(t.get("team_name") or "")
                best_match, score = self.aligner.find_best_match(t_name, known_impect_teams, threshold=threshold)
                if best_match:
                    matched += 1
                    total_matched += 1
                    raw_pid = (
                        t.get("wyscout_id")
                        or t.get("statsbomb_id")
                        or t.get("sportradar_id")
                        or t_name
                    )
                    crosswalk_rows.append({
                        "provider": provider,
                        "provider_team_id": self._stable_provider_int_id(raw_pid),
                        "iteration_id": int(iteration_id),
                        "provider_team_name": t_name,
                        "impect_squad_id": impect_ids_by_name.get(best_match),
                        "impect_squad_name": best_match,
                        "match_score": float(score),
                        "mapping_source": "b2b_sample_evaluator",
                        "wyscout_id": t.get("wyscout_id"),
                        "statsbomb_id": t.get("statsbomb_id"),
                        "sportradar_id": t.get("sportradar_id"),
                        "understat_id": t.get("understat_id"),
                        "skillcorner_id": t.get("skillcorner_id"),
                    })
                else:
                    unmatched.append(t_name)

            n_teams = len(teams)
            provider_reports.append({
                "provider": provider,
                "teams_evaluated": n_teams,
                "teams_matched": matched,
                "crosswalk_match_rate": round(matched / max(1, n_teams), 4),
                "total_events": int(sample.get("total_events") or 0),
                "has_360_context": bool(sample.get("has_360_context")),
                "has_obv_or_pxt": bool(sample.get("has_obv_or_pxt")),
                "unmatched_teams": unmatched,
            })

        if store is not None and crosswalk_rows:
            store.save_team_crosswalk(crosswalk_rows)

        return {
            "total_teams_evaluated": total_teams,
            "total_teams_matched": total_matched,
            "overall_crosswalk_match_rate": round(total_matched / max(1, total_teams), 4),
            "persisted_crosswalk_rows": len(crosswalk_rows) if store is not None else 0,
            "providers": provider_reports,
        }
