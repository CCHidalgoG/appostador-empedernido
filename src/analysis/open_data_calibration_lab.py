"""CLI del laboratorio offline de calibración con Open Data (StatsBomb, IMPECT, SkillCorner)."""

from __future__ import annotations

import argparse
import json

from apostador.open_data_lab import OpenDataCalibrationLab


def main():
    parser = argparse.ArgumentParser(description="Laboratorio offline Open Data + kloppy (TASK-07)")
    parser.add_argument("--statsbomb-events", type=str, default=None, help="Ruta a events/{match_id}.json")
    parser.add_argument("--statsbomb-360", type=str, default=None, help="Ruta a three-sixty/{match_id}.json")
    parser.add_argument("--impect-json", type=str, default=None, help="Ruta a JSON IMPECT Open Data Bundesliga")
    parser.add_argument("--skillcorner-tracking", type=str, default=None, help="Ruta a tracking JSONL SkillCorner")
    parser.add_argument("--skillcorner-events", type=str, default=None, help="Ruta a dynamic events CSV SkillCorner")
    args = parser.parse_args()

    lab = OpenDataCalibrationLab()
    sb = lab.load_statsbomb_match(args.statsbomb_events, args.statsbomb_360) if args.statsbomb_events else None
    imp = lab.load_impect_open_bundesliga(args.impect_json) if args.impect_json else None
    sc = lab.load_skillcorner_match(args.skillcorner_tracking, args.skillcorner_events) if args.skillcorner_tracking else None

    priors = lab.calibrate_open_data_priors(sb, imp, sc)
    print(json.dumps({"statsbomb": sb, "impect": imp, "skillcorner": sc, "priors": priors}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
