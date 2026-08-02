"""LLM-as-a-wiki: фоновая джоба, которая читает raw_locations и строит/обновляет
wiki_places. Отдельный процесс от бота — НЕ вызывается из request-path.

Иерархия: промпт просит LLM определить parent_region → находим/создаём
родительскую статью-регион, привязываем место к ней.
Кросс-ссылки: промпт просит связанные места → линкуем.
Цены: SearXNG (localhost:8888) → LLM экстракт.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from storage.raw import RawLocationRecord, list_raw_locations_without_wiki, list_all_raw_locations
from storage.wiki import (
    WikiPlaceRecord,
    find_candidates,
    get_wiki_place_by_name,
    make_wiki_place_id,
    normalize_key,
    rebuild_all,
    set_place_links,
    upsert_wiki_place,
    update_prices,
)

logger = logging.getLogger(__name__)


async def sync_all_coordinates(database_path: str) -> int:
    """Синхронизирует координаты wiki_places из locations.
    Для каждого wiki_place ищет matching locations по source_location_ids
    и обновляет координаты, если в locations они лучше/новее.
    Возвращает количество обновлённых записей.
    """

    def _sync():
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        try:
            wiki_places = conn.execute(
                "SELECT id, source_location_ids_json, lat, lng FROM wiki_places"
            ).fetchall()
            updated = 0
            for wp in wiki_places:
                source_ids = json.loads(wp["source_location_ids_json"]) if wp["source_location_ids_json"] else []
                if not source_ids:
                    # Fallback: match by canonical_name → locations.name
                    pass
                placeholders = ",".join("?" for _ in source_ids) if source_ids else None
                if placeholders:
                    rows = conn.execute(
                        f"SELECT lat, lng FROM locations WHERE id IN ({placeholders}) "
                        "AND lat IS NOT NULL AND lng IS NOT NULL",
                        source_ids,
                    ).fetchall()
                else:
                    rows = []
                if rows:
                    best = rows[0]
                    if wp["lat"] != best["lat"] or wp["lng"] != best["lng"]:
                        conn.execute(
                            "UPDATE wiki_places SET lat = ?, lng = ?, updated_at = ? WHERE id = ?",
                            (best["lat"], best["lng"], datetime.now(timezone.utc).isoformat(), wp["id"]),
                        )
                        updated += 1
            conn.commit()
            return updated
        finally:
            conn.close()

    import asyncio as _asyncio
    loop = _asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)

DISTILL_PROMPT_VERSION = "v2"
GLM_TEXT_MODEL = "glm-5"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0

SEARXNG_URL = "http://localhost:8888"

# Yandex Search API v2 — credentials from env
YANDEX_SEARCH_API_KEY = os.environ.get("YANDEX_SEARCH_API_KEY", "")
YANDEX_SEARCH_FOLDER_ID = os.environ.get("YANDEX_SEARCH_FOLDER_ID", "")
YANDEX_SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/search"

DISTILL_SYSTEM_PROMPT = (
    "Ты редактор путеводителя по Дагестану. На входе — одно или несколько "
    "упоминаний места из Instagram-постов. Напиши/обнови статью.\n"
    "Верни СТРОГО JSON без markdown:\n"
    '{\n'
    '  "canonical_name": "строка",\n'
    '  "region": "город/район или null",\n'
    '  "parent_region": "крупный регион для группировки (напр. Махачкала, Дербент, Горный Дагестан) или null",\n'
    '  "place_type": "природа|еда|жильё|достопримечательность|город|другое",\n'
    '  "description": "2-4 предложения, связный текст на русском",\n'
    '  "tags": ["тег1", "тег2"],\n'
    '  "related_places": ["название1", "название2"]\n'
    '}\n'
    "related_places — другие места из контекста, которые логически связаны "
    "(рядом, часть маршрута, похожий тип). Если таких нет — пустой массив.\n"
    "Не выдумывай факты, которых нет в источниках."
)

MATCH_SYSTEM_PROMPT = (
    "Определи, относится ли упоминание к существующей статье или это новое место.\n"
    'Верни JSON: {"is_same": true/false, "confidence": 0.0-1.0}\n'
    "Если сомневаешься — is_same=false."
)

DEDUP_SYSTEM_PROMPT = (
    "Ты редактор базы данных мест. На входе — список названий мест в одном регионе.\n"
    "Найди дубликаты — разные записи, которые описывают одно и то же место.\n"
    "Учитывай: опечатки, транслитерация, латиница/кириллица, сокращения, лишние пробелы.\n"
    'Верни СТРОГО JSON без markdown:\n'
    '{"groups": [["id1", "id2"], ["id3", "id4", "id5"]]}\n'
    "Каждая группа — массив ID записей, которые являются дубликатами.\n"
    "Записи, не имеющие дубликатов, не включай в ответ.\n"
    "Если дубликатов нет — верни {\"groups\": []}."
)


class DistillError(RuntimeError):
    pass


@dataclass
class DistillDraft:
    canonical_name: str
    region: str | None
    place_type: str
    description: str
    tags: list[str]
    parent_region: str | None = None
    related_places: list[str] = field(default_factory=list)


async def deduplicate_wiki_places(
    database_path: str, api_key: str, base_url: str,
) -> int:
    """Находит и сливает дубликаты wiki_places через LLM.
    Возвращает количество слитых записей.
    """

    def _get_groups():
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        try:
            # Группируем по parent_place_id
            rows = conn.execute(
                "SELECT id, canonical_name, parent_place_id "
                "FROM wiki_places WHERE parent_place_id IS NOT NULL "
                "ORDER BY parent_place_id, canonical_name"
            ).fetchall()
            groups: dict[str, list[dict]] = {}
            for r in rows:
                key = r["parent_place_id"]
                groups.setdefault(key, []).append(
                    {"id": r["id"], "name": r["canonical_name"]}
                )
            return groups
        finally:
            conn.close()

    groups = await asyncio.get_running_loop().run_in_executor(None, _get_groups)

    total_merged = 0

    for parent_id, places in groups.items():
        if len(places) < 2:
            continue

        # Pre-filter: простые дубли по нормализованному имени (без LLM)
        from difflib import SequenceMatcher
        simple_dupes = _find_simple_duplicates(places)
        for dup_ids in simple_dupes:
            merged = await _merge_wiki_places(database_path, dup_ids)
            if merged:
                total_merged += len(dup_ids) - 1
                logger.info("dedup (simple): слиты %s → %s", dup_ids, merged)
        # Удаляем уже слитые из списка для LLM
        merged_ids = set()
        for d in simple_dupes:
            merged_ids.update(d[1:])  # все кроме первого (keeper)
        remaining = [p for p in places if p["id"] not in merged_ids]

        if len(remaining) < 2:
            continue

        # LLM-фильтр для неточных дубликатов
        place_list = "\n".join(
            f'{p["id"]}  {p["name"]}' for p in remaining
        )
        try:
            response_text = _call_glm_dedup(
                [
                    {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Места в регионе:\n{place_list}"},
                ],
                api_key, base_url,
            )
            logger.info("dedup: LLM response for %s: %s", parent_id, response_text[:500])
            data = _parse_json_response(response_text)
            dup_groups = data.get("groups", [])
        except Exception as exc:
            logger.warning("dedup: LLM не отработал для региона %s: %s", parent_id, exc)
            continue

        if not dup_groups:
            continue

        for dup_ids in dup_groups:
            if len(dup_ids) < 2:
                continue
            # Проверяем что все ID существуют в этом регионе
            valid_ids = {p["id"] for p in places}
            dup_ids = [d for d in dup_ids if d in valid_ids]
            if len(dup_ids) < 2:
                continue

            merged = await _merge_wiki_places(database_path, dup_ids)
            if merged:
                total_merged += len(dup_ids) - 1
                logger.info("dedup: слиты %s → %s", dup_ids, merged)

    if total_merged:
        logger.info("dedup: всего слито %d записей", total_merged)
    return total_merged


def _find_simple_duplicates(places: list[dict]) -> list[list[str]]:
    """Находит очевидные дубли без LLM:
    - идентичные имена (case-insensitive)
    - имена с разницей в пробелах/пунктуации (ratio > 0.85)
    """
    from difflib import SequenceMatcher
    import re

    def normalize(name: str) -> str:
        return re.sub(r'[^\w]', '', name.lower())

    norm_map = [(normalize(p["name"]), p) for p in places]
    used = set()
    groups = []

    for i, (n1, p1) in enumerate(norm_map):
        if p1["id"] in used:
            continue
        cluster = [p1["id"]]
        for j in range(i + 1, len(norm_map)):
            n2, p2 = norm_map[j]
            if p2["id"] in used:
                continue
            # Точное совпадение нормализованного имени
            if n1 == n2:
                cluster.append(p2["id"])
                used.add(p2["id"])
            # Высокая похожесть
            elif n1 and n2 and SequenceMatcher(None, n1, n2).ratio() > 0.88:
                cluster.append(p2["id"])
                used.add(p2["id"])
        if len(cluster) > 1:
            groups.append(cluster)
            used.add(p1["id"])
    return groups


async def _merge_wiki_places(database_path: str, ids: list[str]) -> str | None:
    """Сливает несколько wiki_places в один. Оставляет запись с наибольшим post_count.
    Переносит sources, links, координаты. Удаляет лишние.
    """

    def _merge():
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        try:
            # Выбираем «главную» запись — max post_count, потом max description
            rows = conn.execute(
                f"SELECT * FROM wiki_places WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
            if len(rows) < 2:
                return None

            rows.sort(key=lambda r: (r["post_count"], len(r["description"] or "")), reverse=True)
            keeper = rows[0]
            dupes = rows[1:]
            keeper_id = keeper["id"]

            now = datetime.now(timezone.utc).isoformat()

            # Переносим wiki_place_sources
            for d in dupes:
                conn.execute(
                    "UPDATE OR IGNORE wiki_place_sources SET wiki_place_id = ? WHERE wiki_place_id = ?",
                    (keeper_id, d["id"]),
                )
                # Переносим links (source)
                conn.execute(
                    "UPDATE OR IGNORE wiki_place_links SET source_place_id = ? WHERE source_place_id = ?",
                    (keeper_id, d["id"]),
                )
                # Переносим links (target)
                conn.execute(
                    "UPDATE OR IGNORE wiki_place_links SET target_place_id = ? WHERE target_place_id = ?",
                    (keeper_id, d["id"]),
                )
                # Удаляем дубликат
                conn.execute("DELETE FROM wiki_places WHERE id = ?", (d["id"],))

            # Обновляем post_count и source_location_ids у keeper
            source_ids = [r["raw_location_id"] for r in conn.execute(
                "SELECT raw_location_id FROM wiki_place_sources WHERE wiki_place_id = ?", (keeper_id,)
            ).fetchall()]
            conn.execute(
                "UPDATE wiki_places SET source_location_ids_json = ?, post_count = ?, updated_at = ? WHERE id = ?",
                (json.dumps(source_ids), len(source_ids), now, keeper_id),
            )

            conn.commit()
            return keeper_id
        finally:
            conn.close()

    return await asyncio.get_running_loop().run_in_executor(None, _merge)


def _call_glm_dedup(messages: list[dict], api_key: str, base_url: str) -> str:
    """Отдельный вызов для дедупликации — с thinking enabled."""
    payload = {
        "model": GLM_TEXT_MODEL,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    delay = RETRY_BASE_DELAY_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("dedup LLM: попытка %d/%d не удалась: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return '{"groups": []}'


def _call_glm(messages: list[dict], api_key: str, base_url: str, max_tokens: int = 2048) -> str:
    payload = {
        "model": GLM_TEXT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    delay = RETRY_BASE_DELAY_SECONDS
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as exc:
            last_error = exc
            logger.warning("distill: попытка %d/%d: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise DistillError(f"GLM недоступен после {MAX_RETRIES} попыток: {last_error}")


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


def _build_sources_text(raw_locations: list[RawLocationRecord]) -> str:
    lines: list[str] = []
    for i, rl in enumerate(raw_locations, 1):
        parts: list[str] = []
        if rl.caption:
            parts.append(f"Подпись: {rl.caption}")
        if rl.transcript:
            parts.append(f"Транскрипт: {rl.transcript}")
        if rl.overlay_text:
            parts.append(f"Текст на кадрах: {rl.overlay_text}")
        if rl.recognized_place:
            parts.append(f"Распознанное место: {rl.recognized_place}")
        if rl.aggregated_name:
            parts.append(f"Название: {rl.aggregated_name}")
        if rl.aggregated_region:
            parts.append(f"Регион: {rl.aggregated_region}")
        if parts:
            lines.append(f"Упоминание {i}:\n" + "\n".join(parts))
    return "\n\n".join(lines) if lines else "Источников нет."


async def distill_place(
    existing: WikiPlaceRecord | None,
    raw_locations: list[RawLocationRecord],
    api_key: str,
    base_url: str,
) -> DistillDraft:
    sources_text = _build_sources_text(raw_locations)
    existing_text = ""
    if existing and existing.description:
        existing_text = f"Текущая статья:\n{existing.description}\n"

    raw_response = await asyncio.get_running_loop().run_in_executor(
        None, _call_glm,
        [
            {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
            {"role": "user", "content": f"{existing_text}Новые упоминания:\n{sources_text}"},
        ],
        api_key, base_url,
    )

    try:
        data = _parse_json_response(raw_response)
        return DistillDraft(
            canonical_name=data.get("canonical_name", raw_locations[0].aggregated_name or "Неизвестно"),
            region=data.get("region"),
            parent_region=data.get("parent_region"),
            place_type=data.get("place_type", "другое"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            related_places=data.get("related_places", []),
        )
    except json.JSONDecodeError as exc:
        raise DistillError(f"distill: неразборчивый ответ: {raw_response!r}") from exc


async def _ensure_parent_region(
    database_path: str, parent_region_name: str, api_key: str, base_url: str,
) -> str:
    """Находит или создаёт статью-регион (город/район) как родителя."""
    nkey = normalize_key(parent_region_name)
    place_id = make_wiki_place_id(nkey)

    existing = await get_wiki_place_by_name(database_path, parent_region_name)
    if existing:
        return existing.id

    now = datetime.now(timezone.utc)
    record = WikiPlaceRecord(
        id=place_id,
        normalized_key=nkey,
        canonical_name=parent_region_name,
        place_type="город",
        description=f"Места в регионе {parent_region_name}.",
        source_location_ids_json="[]",
        post_count=0,
        distill_prompt_version=DISTILL_PROMPT_VERSION,
        distilled_at=now,
        created_at=now,
        updated_at=now,
    )
    await upsert_wiki_place(database_path, record, [])
    logger.info("distill: создан регион-родитель %s (%s)", parent_region_name, place_id)
    return place_id


def _searxng_search(query: str, num_results: int = 5) -> list[dict]:
    """Ищет через локальный SearXNG."""
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "language": "ru"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])[:num_results]
    except Exception as exc:
        logger.warning("distill: SearXNG запрос не удался: %s", exc)
        return []


def _yandex_search(query: str, num_results: int = 5) -> list[dict]:
    """Yandex Search API v2 — качественнее SearXNG для российских заведений."""
    import base64
    import xml.etree.ElementTree as ET

    headers = {
        "Authorization": f"Api-Key {YANDEX_SEARCH_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "query": {
            "query_text": query,
            "search_type": "SEARCH_TYPE_RU",
        },
        "folderId": YANDEX_SEARCH_FOLDER_ID,
        "maxResultCount": num_results,
        "sortBy": "rlv",
    }
    try:
        resp = requests.post(YANDEX_SEARCH_URL, headers=headers, json=body, timeout=20)
        if resp.status_code != 200:
            logger.warning("distill: Yandex Search API %d: %s", resp.status_code, resp.text[:200])
            return []
        data = resp.json()
        raw_data = data.get("rawData", "")
        if not raw_data:
            return []
        xml_text = base64.b64decode(raw_data).decode("utf-8")
        results = []
        root = ET.fromstring(xml_text)
        for doc in root.findall(".//doc"):
            title_raw = doc.findtext("title", "") or ""
            title = re.sub(r"<[^>]+>", "", title_raw)
            url_el = doc.findtext("url", "") or ""
            # headline или passages
            headline = doc.findtext("headline", "") or ""
            headline = re.sub(r"<[^>]+>", "", headline)
            # passages — могут быть под doc/passages/passage
            if not headline:
                passage_els = doc.findall(".//passage")
                headline = " ".join((p.text or "") for p in passage_els[:2])
                headline = re.sub(r"<[^>]+>", "", headline)
            results.append({
                "title": title.strip(),
                "url": url_el.strip(),
                "content": headline.strip(),
            })
        logger.info("distill: Yandex Search нашёл %d результатов для %r", len(results), query[:50])
        return results
    except Exception as exc:
        logger.warning("distill: Yandex Search API не удался: %s", exc)
        return []


def _fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """Скачивает страницу и извлекает текст. Для Яндекс Карт / Zoon / 2GIS."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        import re
        # Убираем теги, скрипты, стили
        text = re.sub(r"<script[^>]*>.*?</script>", " ", resp.text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception:
        return ""


def _extract_prices_glm(
    place_name: str, region: str | None, search_results: list[dict],
    api_key: str, base_url: str,
) -> str | None:
    """LLM извлекает цены из выдачи SearXNG."""
    if not search_results:
        return None

    snippets = []
    for r in search_results:
        title = r.get('title', '')
        content = r.get('content', '')[:300]
        url = r.get('url', '')
        snippets.append(f"- [{title}] {content}")
        # Фетчим страницы с полезным контентом
        if any(d in url for d in ['yandex.ru/maps', 'zoon.ru', '2gis.ru', 'eda.yandex', 'vk.com', 't.me']):
            page_text = _fetch_page_text(url, max_chars=5000)
            if page_text:
                price_ctx = re.findall(r'.{0,80}(?:средн\w+\s*(?:чек|счет)|цена|стоимость|от\s*\d+|\d{3,5}\s*₽|\d{3,5}\s*руб).{0,80}', page_text, re.IGNORECASE)
                if price_ctx:
                    snippets.append(f"  Страница: {' | '.join(price_ctx[:3])}")

    snippets_text = "\n".join(snippets)

    prompt = (
        f"Вот поисковые результаты о месте «{place_name}»"
        + (f", {region}" if region else "")
        + ". Извлеки цены. Если есть цены блюд — вычисли средний чек (ориентируясь на горячие блюда).\n"
        + "Интересуют: средний чек, цены ключевых позиций, вход, парковка.\n"
        'Верни JSON: {"prices": [{"item": "название", "amount": "сумма", "currency": "RUB", "note": "примечание"}], '
        '"source_hint": "откуда информация"}\n'
        "Если цен совсем нет — верни {\"prices\": []}.\n"
        f"Источники:\n{snippets_text}"
    )

    try:
        raw = _call_glm(
            [{"role": "user", "content": prompt}],
            api_key, base_url,
        )
        data = _parse_json_response(raw)
        prices = data.get("prices", [])
        if prices:
            return json.dumps(prices, ensure_ascii=False)
    except Exception as exc:
        logger.warning("distill: извлечение цен не удалось для %s: %s", place_name, exc)
    return None


async def _enrich_prices(
    database_path: str, place: WikiPlaceRecord, api_key: str, base_url: str,
) -> None:
    """Ищет цены через Yandex Search API → SearXNG → LLM и сохраняет."""
    query = f"{place.canonical_name} {place.region or ''} средний чек цены стоимость"

    # Yandex Search API — приоритетный источник
    results = await asyncio.get_running_loop().run_in_executor(None, _yandex_search, query, 5)

    # Fallback на SearXNG если мало результатов
    if len(results) < 2:
        searxng_results = await asyncio.get_running_loop().run_in_executor(None, _searxng_search, query, 5)
        results = (results + searxng_results)[:5]

    if not results:
        logger.info("distill: нет результатов поиска для %s", place.canonical_name)
        return

    prices_json = await asyncio.get_running_loop().run_in_executor(
        None, _extract_prices_glm, place.canonical_name, place.region, results, api_key, base_url,
    )
    if prices_json:
        await update_prices(database_path, place.id, prices_json)
        logger.info("distill: цены обновлены для %s", place.canonical_name)
    else:
        logger.info("distill: цены не найдены для %s", place.canonical_name)


async def _copy_best_coordinates(
    database_path: str, place_id: str, source_location_ids: list[str],
) -> bool:
    """Копирует лучшие координаты из locations в wiki_place.
    Берём первую location с непустыми lat/lng (предпочтительно Nominatim/Yandex
    над LLM). Возвращает True если координаты обновлены.
    """
    if not source_location_ids:
        return False

    def _sync():
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        try:
            # Ищем location с координатами среди source_location_ids
            placeholders = ",".join("?" for _ in source_location_ids)
            rows = conn.execute(
                f"SELECT lat, lng FROM locations WHERE id IN ({placeholders}) "
                "AND lat IS NOT NULL AND lng IS NOT NULL",
                source_location_ids,
            ).fetchall()
            if not rows:
                return False
            # Берём первую (orchestrator сохраняет в порядке обработки)
            best = rows[0]
            conn.execute(
                "UPDATE wiki_places SET lat = ?, lng = ?, updated_at = ? WHERE id = ?",
                (best["lat"], best["lng"], datetime.now(timezone.utc).isoformat(), place_id),
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


async def run_distill_job(
    database_path: str, api_key: str, base_url: str, batch_size: int = 20,
) -> int:
    raw_locations = await list_raw_locations_without_wiki(database_path)
    if not raw_locations:
        return 0

    processed = 0
    for rl in raw_locations[:batch_size]:
        try:
            name = rl.aggregated_name or rl.recognized_place or ""
            if not name:
                continue

            nkey = normalize_key(name, rl.aggregated_region)
            candidates = await find_candidates(database_path, nkey, None, None)

            # Простой матч: точное совпадение ключа
            matched_id = None
            for c in candidates:
                if c.normalized_key == nkey:
                    matched_id = c.id
                    break

            if matched_id:
                # Обновляем существующее
                wiki_place = next(c for c in candidates if c.id == matched_id)
                existing_ids = json.loads(wiki_place.source_location_ids_json) if wiki_place.source_location_ids_json else []
                all_ids = list(set(existing_ids + [rl.id]))
                all_raws = [r for r in raw_locations if r.id in all_ids] or [rl]

                draft = await distill_place(wiki_place, all_raws, api_key, base_url)

                wiki_place.canonical_name = draft.canonical_name
                wiki_place.region = draft.region
                wiki_place.place_type = draft.place_type
                wiki_place.description = draft.description
                wiki_place.tags_json = json.dumps(draft.tags, ensure_ascii=False)
                wiki_place.source_location_ids_json = json.dumps(all_ids)
                wiki_place.post_count = len(all_ids)
                wiki_place.distill_prompt_version = DISTILL_PROMPT_VERSION
                wiki_place.distilled_at = datetime.now(timezone.utc)

                # Parent region
                if draft.parent_region:
                    parent_id = await _ensure_parent_region(database_path, draft.parent_region, api_key, base_url)
                    wiki_place.parent_place_id = parent_id

                nkey_new = normalize_key(draft.canonical_name, draft.region)
                wiki_place.normalized_key = nkey_new

                await upsert_wiki_place(database_path, wiki_place, all_ids)
                place_id = wiki_place.id
                await _copy_best_coordinates(database_path, place_id, all_ids)
            else:
                # Новое место
                draft = await distill_place(None, [rl], api_key, base_url)
                nkey_new = normalize_key(draft.canonical_name, draft.region)
                place_id = make_wiki_place_id(nkey_new)
                now = datetime.now(timezone.utc)

                parent_id = None
                if draft.parent_region:
                    parent_id = await _ensure_parent_region(database_path, draft.parent_region, api_key, base_url)

                record = WikiPlaceRecord(
                    id=place_id,
                    normalized_key=nkey_new,
                    canonical_name=draft.canonical_name,
                    region=draft.region,
                    place_type=draft.place_type,
                    description=draft.description,
                    tags_json=json.dumps(draft.tags, ensure_ascii=False),
                    parent_place_id=parent_id,
                    source_location_ids_json=json.dumps([rl.id]),
                    post_count=1,
                    distill_prompt_version=DISTILL_PROMPT_VERSION,
                    distilled_at=now,
                    created_at=now,
                    updated_at=now,
                )
                await upsert_wiki_place(database_path, record, [rl.id])
                logger.info("distill: создана статья %s (%s)", draft.canonical_name, place_id)
                await _copy_best_coordinates(database_path, place_id, [rl.id])

            # Кросс-ссылки
            if draft.related_places:
                links: list[tuple[str, str]] = []
                for rp_name in draft.related_places:
                    rp = await get_wiki_place_by_name(database_path, rp_name)
                    if rp:
                        links.append((rp.id, "related"))
                if links:
                    await set_place_links(database_path, place_id, links)

            # Цены через SearXNG (только для новых мест, не каждый раз)
            if not matched_id:
                place_record = await get_wiki_place_by_name(database_path, draft.canonical_name)
                if place_record:
                    await _enrich_prices(database_path, place_record, api_key, base_url)

            processed += 1
        except Exception as exc:
            logger.error("distill: ошибка %s: %s", rl.id, exc, exc_info=True)

    logger.info("distill: обработано %d", processed)
    return processed


async def rebuild_all_wiki(database_path: str, api_key: str, base_url: str) -> int:
    await rebuild_all(database_path)
    all_raws = await list_all_raw_locations(database_path)
    logger.info("distill: полный rebuild, %d raw_locations", len(all_raws))

    groups: dict[str, list[RawLocationRecord]] = {}
    for rl in all_raws:
        name = rl.aggregated_name or rl.recognized_place or ""
        if not name:
            continue
        nkey = normalize_key(name, rl.aggregated_region)
        groups.setdefault(nkey, []).append(rl)

    processed = 0
    for nkey, raws in groups.items():
        try:
            draft = await distill_place(None, raws, api_key, base_url)
            nkey_new = normalize_key(draft.canonical_name, draft.region)
            place_id = make_wiki_place_id(nkey_new)
            now = datetime.now(timezone.utc)
            all_ids = [r.id for r in raws]

            parent_id = None
            if draft.parent_region:
                parent_id = await _ensure_parent_region(database_path, draft.parent_region, api_key, base_url)

            record = WikiPlaceRecord(
                id=place_id,
                normalized_key=nkey_new,
                canonical_name=draft.canonical_name,
                region=draft.region,
                place_type=draft.place_type,
                description=draft.description,
                tags_json=json.dumps(draft.tags, ensure_ascii=False),
                parent_place_id=parent_id,
                source_location_ids_json=json.dumps(all_ids),
                post_count=len(all_ids),
                distill_prompt_version=DISTILL_PROMPT_VERSION,
                distilled_at=now,
                created_at=now,
                updated_at=now,
            )
            await upsert_wiki_place(database_path, record, all_ids)
            await _copy_best_coordinates(database_path, place_id, all_ids)

            # Кросс-ссылки
            if draft.related_places:
                links: list[tuple[str, str]] = []
                for rp_name in draft.related_places:
                    rp = await get_wiki_place_by_name(database_path, rp_name)
                    if rp:
                        links.append((rp.id, "related"))
                if links:
                    await set_place_links(database_path, place_id, links)

            # Цены
            await _enrich_prices(database_path, record, api_key, base_url)

            processed += 1
        except Exception as exc:
            logger.error("distill: rebuild %s: %s", nkey, exc, exc_info=True)

    logger.info("distill: rebuild завершён, %d статей", processed)
    return processed
