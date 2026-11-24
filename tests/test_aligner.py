import pytest
from apostador.aligner import TeamAligner, clean_name, normalize_name, get_similarity


def test_clean_and_normalize_name():
    # Accents removal including Portuguese
    assert clean_name("São Paulo") == "sao paulo"
    assert clean_name("Grêmio") == "gremio"
    assert clean_name("Atlético-MG") == "atletico mg"
    
    # normalize_name removes club prefixes/suffixes
    assert normalize_name("Corinthians São Paulo") == "corinthians sao paulo"
    assert normalize_name("Coritiba FC") == "coritiba"
    assert normalize_name("SE Palmeiras São Paulo") == "palmeiras sao paulo"
    assert normalize_name("EC Bahia") == "bahia"
    assert normalize_name("Mirassol FC (SP)") == "mirassol sp"


def test_corinthians_vs_coritiba_disambiguation():
    aligner = TeamAligner()
    candidates = [
        "Corinthians São Paulo",
        "Coritiba FC",
        "Mirassol FC (SP)",
        "FC Santos",
    ]
    
    # 1. Corinthians should match Corinthians São Paulo (not Coritiba FC)
    match, score = aligner.find_best_match("Corinthians", candidates)
    assert match == "Corinthians São Paulo"
    assert score >= 0.90
    
    # 2. Coritiba should match Coritiba FC
    match_cor, score_cor = aligner.find_best_match("Coritiba", candidates)
    assert match_cor == "Coritiba FC"
    assert score_cor >= 0.90
    
    # 3. Mirassol should match Mirassol FC (SP)
    match_mira, score_mira = aligner.find_best_match("Mirassol", candidates)
    assert match_mira == "Mirassol FC (SP)"
    assert score_mira >= 0.80


def test_token_containment_without_static_mapping():
    # An aligner with no static overrides should still pick Corinthians São Paulo
    # over Coritiba FC thanks to token containment logic.
    raw_aligner = TeamAligner(static_mappings={})
    candidates = ["Corinthians São Paulo", "Coritiba FC"]
    
    match, score = raw_aligner.find_best_match("Corinthians", candidates)
    assert match == "Corinthians São Paulo"
    assert score > 0.85


def test_brazil_serie_a_and_b_teams():
    aligner = TeamAligner()
    serie_a_candidates = [
        "Clube Atlético Mineiro",
        "Fluminense Rio de Janeiro",
        "Esporte Clube Vitória",
        "FC Santos",
        "EC Cruzeiro Belo Horizonte",
        "CA Paranaense",
        "SE Palmeiras São Paulo",
        "SC Internacional Porto Alegre",
        "FC São Paulo",
        "Corinthians São Paulo",
        "Coritiba FC",
        "Chapecoense",
        "Flamengo Rio de Janeiro",
        "Botafogo Rio de Janeiro",
        "Grêmio Porto Alegre",
        "Vasco da Gama Rio de Janeiro",
        "EC Bahia",
        "Red Bull Bragantino",
        "Clube do Remo (PA)",
        "Mirassol FC (SP)",
    ]
    
    expected_matches = {
        "Corinthians": "Corinthians São Paulo",
        "Coritiba": "Coritiba FC",
        "Palmeiras": "SE Palmeiras São Paulo",
        "Sao Paulo": "FC São Paulo",
        "Flamengo": "Flamengo Rio de Janeiro",
        "Fluminense": "Fluminense Rio de Janeiro",
        "Gremio": "Grêmio Porto Alegre",
        "Internacional": "SC Internacional Porto Alegre",
        "Santos": "FC Santos",
        "Atletico-MG": "Clube Atlético Mineiro",
        "Cruzeiro": "EC Cruzeiro Belo Horizonte",
        "Botafogo": "Botafogo Rio de Janeiro",
        "Bahia": "EC Bahia",
        "Vitoria": "Esporte Clube Vitória",
        "Bragantino": "Red Bull Bragantino",
        "Athletico Paranaense": "CA Paranaense",
        "Remo": "Clube do Remo (PA)",
        "Chapecoense": "Chapecoense",
    }
    
    for query, expected in expected_matches.items():
        res, score = aligner.find_best_match(query, serie_a_candidates)
        assert res == expected, f"Failed for {query}: got {res}, expected {expected}"
        assert score >= 0.70


def test_empty_or_low_similarity_fallbacks():
    aligner = TeamAligner()
    candidates = ["Real Madrid", "FC Barcelona"]
    
    assert aligner.find_best_match("", candidates) == (None, 0.0)
    assert aligner.find_best_match(None, candidates) == (None, 0.0)
    
    match, score = aligner.find_best_match("Completely Unrelated Team", candidates, threshold=0.7)
    assert match is None
    assert score < 0.7


def test_same_city_derby_disambiguation():
    aligner = TeamAligner()
    # New York Red Bulls must NOT match New York City FC
    assert aligner.find_best_match("New York Red Bulls", ["New York City FC"], threshold=0.7) == (None, 0.0)
    assert aligner.find_best_match("New York City FC", ["New York Red Bulls"], threshold=0.7) == (None, 0.0)
    # Los Angeles Galaxy must NOT match Los Angeles FC
    assert aligner.find_best_match("Los Angeles Galaxy", ["Los Angeles FC"], threshold=0.7) == (None, 0.0)
    # Manchester United must NOT match Manchester City
    assert aligner.find_best_match("Manchester United", ["Manchester City"], threshold=0.7) == (None, 0.0)
    # São Paulo must NOT match Corinthians São Paulo or SE Palmeiras São Paulo
    assert aligner.find_best_match("São Paulo", ["Corinthians São Paulo", "SE Palmeiras São Paulo"], threshold=0.7) == (None, 0.0)


def test_recent_form_loader_fuzzy_match_and_weighted_form(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from src.apostador.recent_form import RecentFormLoader

    def _local_read_json(p, default=None):
        fp = Path(p)
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else default

    monkeypatch.setattr("src.apostador.recent_form.read_json", _local_read_json)

    # Deportivo Pereira: pierde hace 45d (0-2), empata hace 25d (1-1), gana recientemente (3-0) -> racha ascendente
    # Millonarios: gana hace 45d (2-0), empata hace 25d (1-1), pierde recientemente (0-2) -> racha descendente
    # Ambos tienen idénticos 4 puntos en los 3 partidos (points_avg = 1.333), pero diferente recencia y GD.
    payload = {
        "response": [
            {
                "fixture": {"status": {"short": "FT"}, "date": "2026-10-01T20:00:00Z"},
                "teams": {"home": {"name": "Deportivo Pereira"}, "away": {"name": "Rival A"}},
                "goals": {"home": 0, "away": 2},
                "league": {"name": "Primera A", "id": 239},
            },
            {
                "fixture": {"status": {"short": "FT"}, "date": "2026-10-20T20:00:00Z"},
                "teams": {"home": {"name": "Deportivo Pereira"}, "away": {"name": "Rival B"}},
                "goals": {"home": 1, "away": 1},
                "league": {"name": "Primera A", "id": 239},
            },
            {
                "fixture": {"status": {"short": "FT"}, "date": "2026-11-12T20:00:00Z"},
                "teams": {"home": {"name": "Deportivo Pereira"}, "away": {"name": "Rival C"}},
                "goals": {"home": 3, "away": 0},
                "league": {"name": "Primera A", "id": 239},
            },
            {
                "fixture": {"status": {"short": "FT"}, "date": "2026-10-01T20:00:00Z"},
                "teams": {"home": {"name": "Millonarios"}, "away": {"name": "Rival D"}},
                "goals": {"home": 2, "away": 0},
                "league": {"name": "Primera A", "id": 239},
            },
            {
                "fixture": {"status": {"short": "FT"}, "date": "2026-10-20T20:00:00Z"},
                "teams": {"home": {"name": "Millonarios"}, "away": {"name": "Rival E"}},
                "goals": {"home": 1, "away": 1},
                "league": {"name": "Primera A", "id": 239},
            },
            {
                "fixture": {"status": {"short": "FT"}, "date": "2026-11-12T20:00:00Z"},
                "teams": {"home": {"name": "Millonarios"}, "away": {"name": "Rival F"}},
                "goals": {"home": 0, "away": 2},
                "league": {"name": "Primera A", "id": 239},
            },
        ]
    }
    (tmp_path / "fixtures_test.json").write_text(json.dumps(payload), encoding="utf-8")

    loader = RecentFormLoader(raw_fixtures_dir=str(tmp_path), processed_datasets=[], window=5)
    # Consulta con nombre distinto ("Deportivo Pereira FC") para validar fallback fuzzy de TeamAligner
    pereira_form = loader.get_form("Deportivo Pereira FC", "2026-11-15")
    millos_form = loader.get_form("Millonarios", "2026-11-15")

    assert pereira_form["form_n"] == 3
    assert pereira_form["form_weighted_points_avg"] > pereira_form["form_points_avg"]
    assert millos_form["form_weighted_points_avg"] < millos_form["form_points_avg"]
    assert pereira_form["form_goal_diff_avg"] > 0
    assert millos_form["form_goal_diff_avg"] == 0.0

    adj = loader.get_form_adjustment("Deportivo Pereira FC", "Millonarios", "2026-11-15")
    assert 0.0 < adj["form_delta"] <= 0.10


