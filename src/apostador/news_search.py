"""news_search.py — Cliente Brave News Search para contexto prepartido.

Basado en documentacion/bravesearch/Documentation.md:
  GET https://api.search.brave.com/res/v1/news/search
  Header: X-Subscription-Token: <BRAVEAPI>
  Params: q, count, search_lang, freshness, extra_snippets, safesearch.

Uso tipico:
  from src.apostador.news_search import BraveNewsClient
  client = BraveNewsClient()  # lee BRAVEAPI de .env
  result = client.search_match_news("Real Betis", "Real Madrid", "2026-09-04")
"""

import hashlib
import json
import os
import time
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
from dotenv import load_dotenv

from .config import RAW_DIR
from .duckdb_store import read_json, write_json

load_dotenv()

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
BRAVE_RAW_DIR = RAW_DIR / "brave"


NEWS_CACHE_VERSION = "v3"
NEWS_CACHE_TTL = timedelta(hours=6)
NEWS_CATEGORIES = {
    "lesion_baja": ("injur", "lesion", "lesionado", "baja", "ruled out", "doubtful", "unavailable", "misses next", "se perdera", "no estara"),
    "suspension_roja": ("suspend", "suspension", "red card", "sent off", "expuls", "tarjeta roja", "sancion", "banned"),
    "alineacion_nomina": ("lineup", "starting xi", "starting eleven", "starting 11", "alineacion", "convocatoria", "nomina", "roster", "squad list", "titular", "starter", "omitted"),
    "rotacion": ("rotation", "rotacion", "rested", "descanso", "descansara", "changes to the starting", "cambios en el once"),
    "formacion_tactica": ("formation", "formacion", "starting lineup", "probable lineup", "once inicial", "alineacion probable", "sistema tactico"),
    "cambio_tecnico": ("new manager", "new coach", "head coach", "sacked", "fired", "appointed", "interim coach", "nuevo entrenador", "tecnico interino", "destituido", "despedido", "nuevo tecnico"),
    "carga_competiciones": ("fixture congestion", "congested schedule", "three games in", "midweek match", "calendar congestion", "fixture pileup", "doble competencia", "calendario apretado", "acumulacion de partidos", "entre semana"),
    "problema_plantilla": ("dressing room", "internal conflict", "players strike", "wage dispute", "training ground row", "conflicto interno", "problemas internos", "huelga de jugadores", "problema salarial"),
}
GENERIC_TEAM_WORDS = {"club", "team", "deportivo", "athletic", "football", "futbol", "fc", "cf", "sc", "afc", "the", "united", "city", "real", "sporting"}
COUNTRY_SEARCH = {
    "colombia": ("CO", "es"), "mexico": ("MX", "es"), "brazil": ("BR", "pt"),
    "brasil": ("BR", "pt"), "usa": ("US", "en"), "united states": ("US", "en"),
    "argentina": ("AR", "es"), "spain": ("ES", "es"), "italy": ("IT", "it"),
    "belgium": ("BE", "nl"), "russia": ("RU", "ru"), "chile": ("CL", "es"),
    "peru": ("PE", "es"), "ecuador": ("EC", "es"), "uruguay": ("UY", "es"),
    "paraguay": ("PY", "es"), "bolivia": ("BO", "es"), "venezuela": ("VE", "es"),
    "england": ("GB", "en"), "germany": ("DE", "de"), "france": ("FR", "fr"),
    "netherlands": ("NL", "nl"), "portugal": ("PT", "pt"), "europe": ("GB", "en"),
    "world": ("GB", "en"), "saudi arabia": ("SA", "ar"),
}


def _normalize(value):
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _team_mentioned(text, team):
    normalized = _normalize(text)
    name = _normalize(team)
    name_parts = name.split()
    if len(name_parts) > 1 and name in normalized:
        return True
    words = set(normalized.split())
    team_words = {word for word in name_parts if len(word) >= 5 and word not in GENERIC_TEAM_WORDS}
    return bool(team_words & words)


def _claim_metadata(text):
    """Conservative lexical labels; these labels are evidence hints, not facts."""
    text = _normalize(text)
    if any(term in text for term in ("rumor", "rumour", "speculation", "se rumorea", "sin confirmar")):
        certainty = "rumor"
    elif any(term in text for term in ("officially confirmed", "confirmed by", "official statement",
                                       "ruled out", "will miss", "is suspended", "fue confirmado",
                                       "confirmo el club", "no jugara", "fue descartado")):
        certainty = "confirmado"
    elif any(term in text for term in ("expected to", "likely to", "doubtful", "probable", "could miss",
                                       "may miss", "reportedly", "se espera que", "podria perderse",
                                       "seria titular", "es duda")):
        certainty = "probable"
    else:
        certainty = "sin_verificar"

    negated_absence = any(term in text for term in (
        "no injury", "no injuries", "not injured", "without injury", "sin lesion", "sin bajas",
    ))
    unavailable = not negated_absence and any(term in text for term in (
        "injured", "injury", "ruled out", "unavailable", "suspended", "will miss", "doubtful",
        "lesionado", "lesion", "baja", "sancionado", "no jugara", "se perdera", "descartado",
    ))
    available = any(term in text for term in (
        "available again", "back in training", "fit again", "returns to the squad", "available for",
        "alta medica", "vuelve a estar disponible", "regresa a la convocatoria", "entreno con normalidad",
    ))
    claim_status = "ausencia" if unavailable and not available else (
        "disponible" if available and not unavailable else None
    )
    return certainty, claim_status


def _canonical_url(url):
    parts = urlsplit(str(url or ""))
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, "", ""))


def _domain_tier(domain, policy, country_code=None, league_id=None):
    if not domain:
        return "no_revisado"
    domain = str(domain).lower().removeprefix("www.")
    scopes = []
    if league_id is not None:
        scopes.append((policy.get("leagues") or {}).get(str(league_id), {}))
    if country_code:
        scopes.append((policy.get("countries") or {}).get(str(country_code).upper(), {}))
    for scope in scopes:
        for tier in ("official_domains", "trusted_domains"):
            if any(domain == item.lower().removeprefix("www.") or domain.endswith("." + item.lower().removeprefix("www."))
                   for item in scope.get(tier, [])):
                return "oficial" if tier == "official_domains" else "revisado"
    return "no_revisado"


def classify_news_evidence(home_team, away_team, items, policy=None, country_code=None, league_id=None):
    """Extract sourced candidates, label certainty, group duplicates, flag conflicts."""
    evidence = []
    for item in items or []:
        text = " ".join([item.get("title", ""), *[str(x) for x in item.get("snippets", [])]])
        normalized = _normalize(text)
        categories = [name for name, terms in NEWS_CATEGORIES.items()
                      if any(_normalize(term) in normalized for term in terms)]
        teams_mentioned = [team for team in (home_team, away_team) if _team_mentioned(text, team)]
        if categories and teams_mentioned and item.get("url"):
            certainty, claim_status = _claim_metadata(text)
            domain = item.get("domain") or urlsplit(item["url"]).hostname
            evidence.append({"url": item["url"], "title": item.get("title", ""),
                             "team": teams_mentioned[0] if len(teams_mentioned) == 1 else None,
                             "teams_mentioned": teams_mentioned, "categories": categories,
                             "snippet": " ".join(item.get("snippets", []))[:500],
                             "published_at": item.get("published_at"),
                             "age": item.get("age"), "domain": domain,
                             "certainty": certainty, "claim_status": claim_status,
                             "fetched_at": item.get("fetched_at"),
                             "domain_tier": _domain_tier(domain, policy or {}, country_code, league_id),
                             "duplicate_sources": [domain] if domain else [],
                             "duplicate_count": 1, "potential_conflict": False})

    # Collapse wire duplicates and syndicated/near-identical titles while
    # retaining the publishers for audit. The shortlist per fixture is small.
    grouped = []
    for item in evidence:
        title = _normalize(item["title"])
        duplicate = next((existing for existing in grouped
                          if _canonical_url(existing["url"]) == _canonical_url(item["url"])
                          or (title and SequenceMatcher(None, title, _normalize(existing["title"])).ratio() >= 0.93)), None)
        if duplicate is None:
            grouped.append(item)
        else:
            duplicate["duplicate_count"] += 1
            if item.get("domain") and item["domain"] not in duplicate["duplicate_sources"]:
                duplicate["duplicate_sources"].append(item["domain"])
            if item.get("published_at") and not duplicate.get("published_at"):
                duplicate["published_at"] = item["published_at"]

    # Opposing team-level availability claims need human/model review. Since
    # player identity is not reliably extractable here, this is only a warning.
    for team in (home_team, away_team):
        claims = [item for item in grouped if team in item["teams_mentioned"] and item["claim_status"]]
        statuses = {item["claim_status"] for item in claims}
        if len(statuses) > 1:
            for item in claims:
                item["potential_conflict"] = True

    categories = sorted({category for item in evidence for category in item["categories"]})
    return {"relevant": bool(grouped), "evidence": grouped,
            "categories": categories, "candidate_count": len(grouped)}


class BraveNewsClient:
    """Busca noticias frescas de un enfrentamiento y cachea el payload en DuckDB, esquema raw."""

    def __init__(self, api_key=None, cache_dir=None, timeout=30, sleep_between_calls=1.1,
                 source_policy_path=None):
        self.api_key = api_key or os.getenv("BRAVEAPI")
        if not self.api_key:
            raise ValueError("Falta BRAVEAPI en .env — ver documentacion/bravesearch/Documentation.md")
        self.cache_dir = Path(cache_dir) if cache_dir else BRAVE_RAW_DIR
        self.timeout = timeout
        self.sleep_between_calls = sleep_between_calls
        policy_path = Path(source_policy_path) if source_policy_path else Path(__file__).with_name("news_source_policy.json")
        try:
            self.source_policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.source_policy = {"countries": {}, "leagues": {}}

    # ------------------------------------------------------------------ API
    def _query_api(self, query, count=10, search_lang="es", freshness="pw", country=None):
        params = {
            "q": query,
            "count": count,
            "search_lang": search_lang,
            "freshness": freshness,
            "safesearch": "moderate",
            "extra_snippets": "true",
        }
        if country:
            params["country"] = country
        resp = requests.get(
            BRAVE_ENDPOINT,
            headers={"X-Subscription-Token": self.api_key},
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_items(payload):
        """Normalize Brave News results (and older LLM Context cached payloads)."""
        items = []
        for block in payload.get("results", []) or []:
            profile = block.get("profile") or {}
            meta_url = block.get("meta_url") or {}
            snippets = [block.get("description", ""), *(block.get("extra_snippets") or [])]
            items.append({
                "url": block.get("url", ""), "title": block.get("title", ""),
                "snippets": [snippet for snippet in snippets if snippet],
                "published_at": block.get("page_age"), "age": block.get("age"),
                "fetched_at": block.get("page_fetched"),
                "domain": meta_url.get("hostname") or profile.get("long_name") or profile.get("name"),
            })
        for block in (payload.get("grounding") or {}).get("generic", []):
            items.append({
                "url": block.get("url", ""),
                "title": block.get("title", ""),
                "snippets": list(block.get("snippets") or []),
                "published_at": None, "age": None, "domain": None,
            })
        return items

    @staticmethod
    def summarise(items, max_chars=2500):
        """Compacta snippets para caber en el state de JEV."""
        chunks = []
        total = 0
        for it in items:
            for snip in it["snippets"]:
                metadata = " | ".join(str(v) for v in (it.get("domain"), it.get("published_at") or it.get("age")) if v)
                text = f"[{it['title']}{' | ' + metadata if metadata else ''}] {snip}".strip()
                if total + len(text) > max_chars:
                    break
                chunks.append(text)
                total += len(text)
        digest = "\n---\n".join(chunks)
        urls = [it["url"] for it in items if it["url"]]
        return digest, urls

    # --------------------------------------------------------------- publico
    def _cache_path(self, home, away, match_date, country=None, league_id=None):
        slug = f"{NEWS_CACHE_VERSION}_{home}_vs_{away}_{match_date or 'nodate'}_{country or 'global'}_{league_id or 'league'}"
        slug = "".join(c if c.isalnum() else "_" for c in slug.lower())[:80]
        digest = hashlib.md5(f"{NEWS_CACHE_VERSION}|{home}|{away}|{match_date}|{country}|{league_id}".encode()).hexdigest()[:8]
        return self.cache_dir / f"brave_{slug}_{digest}.json"

    def search_match_news(self, home_team, away_team, match_date=None,
                          use_cache=True, max_chars=2500, country=None, league_id=None):
        """Devuelve dict con digest listo para JEV + items crudos + meta.

        Hace 2 busquedas (una por equipo con contexto del rival) para capturar
        lesiones/sanciones de ambos lados. Entre llamadas respeta 1.1s (ventana
        de rate-limit de 1s documentada por Brave).
        """
        cache_path = self._cache_path(home_team, away_team, match_date, country, league_id)
        cached = read_json(cache_path) if use_cache else None
        if cached is not None:
            try:
                fetched_at = datetime.fromisoformat(cached.get("fetched_at", ""))
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc) < NEWS_CACHE_TTL:
                    cached["from_cache"] = True
                    return cached
            except (TypeError, ValueError):
                pass  # Old or malformed cache: fetch fresh evidence.

        queries = [
            f'"{home_team}" "{away_team}" lesiones bajas sancion sancionados alineacion formacion rotacion tecnico calendario partidos descanso plantilla',
            f'"{away_team}" "{home_team}" injuries unavailable suspension lineup formation rotation coach schedule fixture congestion squad',
        ]
        all_items, seen = [], set()
        errors = []
        country_settings = COUNTRY_SEARCH.get(_normalize(country or ""))
        country_code, local_language = country_settings if country_settings else (None, "es")
        for index, q in enumerate(queries):
            try:
                payload = self._query_api(
                    q, search_lang=local_language if index == 0 else "en", country=country_code,
                )
                for it in self._extract_items(payload):
                    if it["url"] and it["url"] not in seen:
                        seen.add(it["url"])
                        all_items.append(it)
            except Exception as exc:  # noqa: BLE001 — Brave no debe romper el pipeline
                errors.append(f"{q}: {exc}")
            time.sleep(self.sleep_between_calls)

        digest, urls = self.summarise(all_items, max_chars=max_chars)
        gate = classify_news_evidence(home_team, away_team, all_items, self.source_policy,
                                      country_code=country_code, league_id=league_id)
        result = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "country": country_code,
            "league_id": str(league_id) if league_id is not None else None,
            "queries": queries,
            "digest": digest,
            "sources": urls,
            "n_sources": len(urls),
            "news_relevant": gate["relevant"],
            "news_evidence": gate["evidence"],
            "news_categories": gate["categories"],
            "news_candidate_count": gate["candidate_count"],
            "errors": errors,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from_cache": False,
        }
        try:
            write_json(cache_path, result)
            result["cache_path"] = str(cache_path)
        except OSError as exc:
            result["errors"].append(f"cache: {exc}")
        return result
