import re
from difflib import SequenceMatcher

def clean_name(name):
    """
    Cleans a team name by lowercasing, replacing accents (including Portuguese/Latin),
    and removing punctuation without stripping football club keywords.
    """
    if not name:
        return ""
    n = name.lower()
    accents = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n', 'ã': 'a', 'õ': 'o', 'â': 'a',
        'ê': 'e', 'ô': 'o', 'à': 'a', 'ç': 'c'
    }
    for char, replacement in accents.items():
        n = n.replace(char, replacement)
    n = re.sub(r'[^\w\s]', ' ', n)
    return re.sub(r'\s+', ' ', n).strip()

def normalize_name(name):
    """
    Normalizes a team name by lowercasing, removing special characters, accents,
    and stripping common football club prefixes/suffixes.
    """
    if not name:
        return ""
    
    n = clean_name(name)
    
    # Remove common football suffixes/prefixes
    terms_to_remove = [
        r'\bfc\b', r'\bcf\b', r'\bcd\b', r'\bfk\b', r'\bclub\b', r'\bclube\b',
        r'\bdeportivo\b', r'\bsc\b', r'\bsd\b', r'\batletico\b', 
        r'\breal\b', r'\bud\b', r'\brcd\b', r'\bca\b', r'\bac\b', r'\bec\b', r'\bse\b',
        r'\bunited\b', r'\bcity\b', r'\bsporting\b', r'\bdeportes\b',
        r'\bcorporacion\b', r'\bsad\b', r'\bjk\b', r'\btc\b', r'\bafc\b', r'\baa\b', r'\bcr\b'
    ]
    for term in terms_to_remove:
        n = re.sub(term, '', n)
        
    # Strip double spaces and surrounding spaces
    n = re.sub(r'\s+', ' ', n).strip()
    return n

def get_similarity(s1, s2):
    """
    Returns Jaro-Winkler-like similarity using SequenceMatcher.
    """
    return SequenceMatcher(None, s1, s2).ratio()

class TeamAligner:
    def __init__(self, static_mappings=None):
        # Dictionary of manual overrides for names that are hard to match automatically
        self.static_mappings = static_mappings or {
            # Format: 'wrong_or_api_name': 'canonical_impect_name'
            "manchester united": "Manchester United",
            "manchester city": "Manchester City",
            "munchen": "Bayern München",
            "bayern munich": "Bayern München",
            "milan": "AC Milan",
            "inter": "Inter Milan",
            "inter milan": "Inter Milan",
            "ind. medellin": "Independiente Medellín",
            "independiente medellin": "Independiente Medellín",
            "atletico nacional": "Atlético Nacional",
            "millonarios": "Millonarios FC",
            "dep. cali": "Deportivo Cali",
            "deportivo cali": "Deportivo Cali",
            "junior": "Junior FC",
            "tolima": "Deportes Tolima",
            "america de cali": "América de Cali",
            "santa fe": "Independiente Santa Fe",
            "bucaramanga": "Atlético Bucaramanga",
            "la equidad": "La Equidad",
            "pasto": "Deportivo Pasto",
            "boyaca chico": "Boyacá Chicó FC",
            "chico": "Boyacá Chicó FC",
            "patriotas": "Patriotas Boyacá",
            "jaguares de cordoba": "Jaguares de Córdoba",
            "jaguares": "Jaguares de Córdoba",
            "envigado": "Envigado FC",
            "fortaleza": "Fortaleza CEIF",
            "aguilas doradas": "Águilas Doradas",
            "rionegro aguilas": "Rionegro Águilas",
            "pereira": "Deportivo Pereira",
            "san lorenzo": "CA San Lorenzo de Almagro",
            "gimnasia m": "Gimnasia y Esgrima de Mendoza",
            "gimnasia l p": "Club de Gimnasia y Esgrima La Plata",
            "gimnasia lp": "Club de Gimnasia y Esgrima La Plata",
            "union santa fe": "Club Atlético Unión",
            "talleres cordoba": "Club Atlético Talleres",
            "pumas": "UNAM Pumas",
            "alianza valledupar": "Alianza Fútbol Club",
            # Brazilian Serie A & B canonical overrides
            "corinthians": "Corinthians São Paulo",
            "corinthians paulista": "Corinthians São Paulo",
            "sc corinthians paulista": "Corinthians São Paulo",
            "palmeiras": "SE Palmeiras São Paulo",
            "flamengo": "Flamengo Rio de Janeiro",
            "fluminense": "Fluminense Rio de Janeiro",
            "gremio": "Grêmio Porto Alegre",
            "internacional": "SC Internacional Porto Alegre",
            "inter porto alegre": "SC Internacional Porto Alegre",
            "sao paulo": "FC São Paulo",
            "santos": "FC Santos",
            "atletico mineiro": "Clube Atlético Mineiro",
            "atletico mg": "Clube Atlético Mineiro",
            "mineiro": "Clube Atlético Mineiro",
            "cruzeiro": "EC Cruzeiro Belo Horizonte",
            "botafogo": "Botafogo Rio de Janeiro",
            "botafogo rj": "Botafogo Rio de Janeiro",
            "vasco": "Vasco da Gama Rio de Janeiro",
            "vasco da gama": "Vasco da Gama Rio de Janeiro",
            "bahia": "EC Bahia",
            "vitoria": "Esporte Clube Vitória",
            "bragantino": "Red Bull Bragantino",
            "red bull bragantino": "Red Bull Bragantino",
            "rb bragantino": "Red Bull Bragantino",
            "athletico paranaense": "CA Paranaense",
            "atletico paranaense": "CA Paranaense",
            "athletico pr": "CA Paranaense",
            "atletico pr": "CA Paranaense",
            "paranaense": "CA Paranaense",
            "remo": "Clube do Remo (PA)",
            "coritiba": "Coritiba FC",
            "coritiba fbc": "Coritiba FC",
            "mirassol": "Mirassol FC (SP)",
            "mirassol sp": "Mirassol FC (SP)",
            "ponte preta": "AA Ponte Preta",
            "sport": "Sport Recife",
            "sport recife": "Sport Recife",
            "america mg": "América FC (MG)",
            "america mineiro": "América FC (MG)",
            "cuiaba": "Cuiabá EC (MT)",
            "avai": "Avaí FC (SC)",
            "atletico go": "AC Goianiense",
            "atletico goianiense": "AC Goianiense",
            "ceara": "Ceará SC",
            "fortaleza ec": "Fortaleza Esporte Clube",
            "fortaleza esporte clube": "Fortaleza Esporte Clube",
            "goias": "Goiás EC",
            "juventude": "EC Juventude",
            "operario": "Operário Ferroviário EC (PR)",
            "operario pr": "Operário Ferroviário EC (PR)",
            "operario ferroviario": "Operário Ferroviário EC (PR)",
            "londrina": "Londrina EC (PR)",
            "botafogo sp": "Botafogo FC (SP)",
            "crb": "Clube de Regatas Brasil (AL)",
            "athletic": "Athletic Club (MG)",
            "athletic club": "Athletic Club (MG)",
            "sao bernardo": "São Bernardo FC (SP)",
            "criciuma": "Criciúma Esporte Clube",
            "novorizontino": "Grêmio Novorizontino (SP)",
            "gremio novorizontino": "Grêmio Novorizontino (SP)",
            "vila nova": "Vila Nova FC (GO)",
            "nautico": "Clube Náutico Capibaribe",
            "nautico recife": "Clube Náutico Capibaribe",
            "chapecoense": "Chapecoense",
            # MLS & same-city derby disambiguation
            "new york city": "New York City FC",
            "new york city fc": "New York City FC",
            "nycfc": "New York City FC",
            "new york red bulls": "New York Red Bulls",
            "ny red bulls": "New York Red Bulls",
            "red bulls": "New York Red Bulls",
            "los angeles fc": "Los Angeles FC",
            "lafc": "Los Angeles FC",
            "los angeles galaxy": "Los Angeles Galaxy",
            "la galaxy": "Los Angeles Galaxy",
        }
        self._known_canonicals = set(self.static_mappings.values())

    @staticmethod
    def _has_conflicting_identity(clean_a: str, clean_b: str) -> bool:
        """Detect rival clubs from the same city with mutually exclusive identity tokens."""
        tokens_a = set(clean_a.split())
        tokens_b = set(clean_b.split())
        exclusive_groups = [
            ({"city"}, {"united", "red", "bulls", "galaxy", "rovers"}),
            ({"united"}, {"city", "red", "bulls", "galaxy", "rovers", "wednesday"}),
            ({"galaxy"}, {"city", "united", "lafc"}),
            ({"bulls"}, {"city", "united", "galaxy"}),
        ]
        for group_1, group_2 in exclusive_groups:
            if (tokens_a & group_1 and tokens_b & group_2) or (tokens_b & group_1 and tokens_a & group_2):
                return True
        # Same-city clubs where one is just "<city> fc" and the other has a distinct club name
        multi_club_cities = [
            {"new", "york"},
            {"los", "angeles"},
            {"sao", "paulo"},
            {"rio", "de", "janeiro"},
            {"porto", "alegre"},
            {"belo", "horizonte"},
            {"manchester"},
            {"madrid"},
            {"milan"},
        ]
        norm_a = set(normalize_name(clean_a).split())
        norm_b = set(normalize_name(clean_b).split())
        for city_tokens in multi_club_cities:
            if (norm_a == city_tokens and norm_b > city_tokens) or (norm_b == city_tokens and norm_a > city_tokens):
                return True
        return False

    def _resolve_canonical(self, raw_name: str) -> str | None:
        c_name = clean_name(raw_name)
        n_name = normalize_name(raw_name)
        if c_name in self.static_mappings:
            return self.static_mappings[c_name]
        if n_name in self.static_mappings:
            return self.static_mappings[n_name]
        if raw_name in self._known_canonicals:
            return raw_name
        return None
        
    def find_best_match(self, query_name, candidate_names, threshold=0.7):
        """
        Finds the best match for a query_name within a list of candidate_names.
        Uses static mappings first (matching clean and normalized forms),
        then normalized exact matching, token containment, and falls back to fuzzy matching.
        """
        if not query_name:
            return None, 0.0
            
        c_query = clean_name(query_name)
        norm_query = normalize_name(query_name)
        mapped_target = self._resolve_canonical(query_name)

        def _is_incompatible_candidate(candidate: str) -> bool:
            c_cand = clean_name(candidate)
            if self._has_conflicting_identity(c_query, c_cand):
                return True
            cand_canonical = self._resolve_canonical(candidate)
            if mapped_target and cand_canonical and cand_canonical != mapped_target:
                return True
            return False
        
        # 1. Check static mappings first (clean or normalized)
        if mapped_target:
            if mapped_target in candidate_names:
                return mapped_target, 1.0
            target_norm = normalize_name(mapped_target)
            for candidate in candidate_names:
                if _is_incompatible_candidate(candidate):
                    continue
                n_cand = normalize_name(candidate)
                cand_mapped = self._resolve_canonical(candidate)
                if n_cand == target_norm or cand_mapped == mapped_target:
                    return candidate, 1.0
            query_name = mapped_target
            c_query = clean_name(query_name)
            norm_query = target_norm
            
        # 2. Check for exact normalized match
        for candidate in candidate_names:
            if _is_incompatible_candidate(candidate):
                continue
            if normalize_name(candidate) == norm_query:
                return candidate, 1.0
                
        # 3. Token containment match (prevents false matches when city names skew fuzzy ratio)
        q_tokens = [t for t in norm_query.split() if len(t) > 2]
        if q_tokens:
            token_candidates = []
            for candidate in candidate_names:
                if _is_incompatible_candidate(candidate):
                    continue
                c_norm = normalize_name(candidate)
                c_tokens = [t for t in c_norm.split() if len(t) > 2]
                if set(q_tokens).issubset(set(c_tokens)):
                    sim = len(" ".join(q_tokens)) / max(len(c_norm), 1)
                    token_candidates.append((candidate, 0.85 + 0.14 * sim))
                elif c_tokens and set(c_tokens).issubset(set(q_tokens)):
                    sim = len(" ".join(c_tokens)) / max(len(norm_query), 1)
                    token_candidates.append((candidate, 0.85 + 0.14 * sim))
            if token_candidates:
                token_candidates.sort(key=lambda x: x[1], reverse=True)
                return token_candidates[0]

        # 4. Fuzzy matching fallback
        best_candidate = None
        best_score = 0.0
        
        for candidate in candidate_names:
            if _is_incompatible_candidate(candidate):
                continue
            score = get_similarity(norm_query, normalize_name(candidate))
            if score > best_score:
                best_score = score
                best_candidate = candidate
                
        if best_score >= threshold:
            return best_candidate, best_score
            
        return None, best_score
