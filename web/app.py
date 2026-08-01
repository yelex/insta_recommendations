"""Веб-фронтенд для Wiki: FastAPI + минимальный HTML.

Иерархия: регионы → места внутри региона.
Хлебные крошки, кросс-ссылки, цены.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from bot.config import load_settings
from storage.wiki import (
    WikiPlaceRecord,
    get_wiki_place,
    get_wiki_place_by_name,
    get_place_links,
    list_wiki_places,
)

app = FastAPI(title="Insta Recommendations Wiki")

_settings = None


def _get_db_path() -> str:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings.database_url.replace("sqlite:///", "")


# Эмодзи-иконки по типу места
TYPE_EMOJI = {
    "город": "🏙️",
    "село": "🏘️",
    "аул": "🏘️",
    "регион": "🗺️",
    "кофейня": "☕",
    "ресторан": "🍽️",
    "кафе": "🍽️",
    "гостиница": "🏨",
    "отель": "🏨",
    "музей": "🏛️",
    "парк": "🌳",
    "гора": "⛰️",
    "озеро": "🏞️",
    "водопад": "💦",
    "пляж": "🏖️",
    "мечеть": "🕌",
    "храм": "⛪",
    "крепость": "🏰",
    "магазин": "🛍️",
    "рынок": "🛒",
    "достопримечательность": "📸",
    "природа": "🌿",
    "ущелье": "🏔️",
    "пещера": "🕳️",
    "смотровая площадка": "🔭",
    "село-музей": "🏘️",
}
DEFAULT_EMOJI = "📍"


def _emoji_for(place_type: str | None) -> str:
    if not place_type:
        return DEFAULT_EMOJI
    return TYPE_EMOJI.get(place_type.strip().lower(), DEFAULT_EMOJI)


def e(value) -> str:
    """html.escape для произвольных данных из БД/LLM."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


CSS = """
<style>
  :root {
    --terracotta: #c1550f;
    --terracotta-dark: #9c4109;
    --ochre: #d99a2b;
    --ochre-light: #f3e0b8;
    --green: #5c7a4f;
    --green-dark: #3f5636;
    --cream: #faf5ec;
    --card-bg: #fffdf8;
    --text: #3a2e22;
    --text-muted: #8a7a68;
    --border: #ecdfc9;
    --shadow: 0 2px 8px rgba(120, 78, 24, 0.10);
    --shadow-hover: 0 8px 24px rgba(120, 78, 24, 0.18);
    --radius: 14px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--cream);
    color: var(--text);
    line-height: 1.6;
  }
  .container { max-width: 1000px; margin: 0 auto; padding: 0 20px 40px; }

  .hero {
    background: linear-gradient(135deg, var(--terracotta) 0%, var(--ochre) 60%, var(--green) 130%);
    color: #fff8ec;
    padding: 48px 20px 40px;
    margin-bottom: 28px;
    box-shadow: 0 4px 18px rgba(120, 78, 24, 0.25);
  }
  .hero-inner { max-width: 1000px; margin: 0 auto; }
  .hero h1 {
    font-size: 2.1em;
    font-weight: 700;
    text-shadow: 0 1px 3px rgba(0,0,0,0.2);
  }
  .hero .subtitle {
    margin-top: 8px;
    font-size: 1em;
    opacity: 0.92;
  }
  .hero .stats-pill {
    display: inline-block;
    margin-top: 16px;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    backdrop-filter: blur(2px);
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.88em;
  }

  h2 {
    color: var(--green-dark);
    margin: 30px 0 14px;
    font-size: 1.25em;
    font-weight: 700;
  }
  a { color: var(--terracotta-dark); text-decoration: none; }
  a:hover { text-decoration: underline; }

  .breadcrumbs { font-size: 0.85em; color: var(--text-muted); margin-bottom: 18px; }
  .breadcrumbs a { color: var(--text-muted); }
  .breadcrumbs a:hover { color: var(--terracotta-dark); }

  .nav-top { margin-bottom: 20px; }
  .nav-top a {
    margin-right: 10px;
    font-size: 0.9em;
    background: var(--ochre-light);
    color: var(--terracotta-dark);
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
  }
  .nav-top a:hover { background: var(--ochre); color: #fff; text-decoration: none; }

  .place-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
  }
  .place-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
    border-color: var(--ochre);
  }
  .place-name { font-size: 1.2em; font-weight: 700; }
  .place-name a { color: var(--terracotta-dark); }
  .place-icon { margin-right: 6px; }
  .place-region { color: var(--text-muted); font-size: 0.88em; margin-top: 3px; }
  .place-type {
    display: inline-block;
    background: var(--green);
    color: #fff;
    padding: 2px 11px;
    border-radius: 20px;
    font-size: 0.76em;
    margin-left: 6px;
    font-weight: 600;
  }
  .place-desc { margin-top: 10px; color: var(--text); font-size: 0.95em; }
  .tags { margin-top: 10px; }
  .tag {
    display: inline-block;
    background: var(--ochre-light);
    color: var(--terracotta-dark);
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.78em;
    margin-right: 4px;
    margin-bottom: 4px;
    font-weight: 600;
  }
  .coords { color: var(--text-muted); font-family: monospace; font-size: 0.82em; margin-top: 8px; }
  .stats { color: var(--text-muted); font-size: 0.88em; margin-bottom: 14px; }
  .empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
    background: var(--card-bg);
    border-radius: var(--radius);
    border: 1px dashed var(--border);
  }
  .region-section { margin-bottom: 32px; }
  .region-title {
    font-size: 1.2em;
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 3px solid var(--ochre);
  }
  .region-title a { color: var(--terracotta-dark); }

  .price-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9em;
                 border-radius: 10px; overflow: hidden; box-shadow: var(--shadow); }
  .price-table th {
    text-align: left; padding: 8px 12px; background: var(--green);
    color: #fff; font-weight: 600;
  }
  .price-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); background: var(--card-bg); }
  .price-table tr:last-child td { border-bottom: none; }
  .price-amount { font-weight: 700; color: var(--terracotta-dark); }

  .related { margin-top: 20px; }
  .related-item {
    display: inline-block;
    background: var(--green);
    color: #fff;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.85em;
    margin: 3px 5px 3px 0;
    transition: background 0.15s ease;
  }
  .related-item a { color: #fff; font-weight: 600; }
  .related-item:hover { background: var(--green-dark); }

  .map-link { display: inline-block; margin-top: 10px; font-size: 0.85em; font-weight: 600; }

  /* Filters */
  .filter-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .filter-chip {
    cursor: pointer; user-select: none;
    padding: 6px 14px; border-radius: 999px;
    border: 1.5px solid var(--border); background: var(--card-bg);
    font-size: 0.88em; font-weight: 600; color: var(--text-muted);
    transition: all 0.15s ease;
  }
  .filter-chip:hover { border-color: var(--ochre); color: var(--terracotta-dark); }
  .filter-chip.active { background: var(--terracotta); color: #fff8ec; border-color: var(--terracotta); }

  /* Region collapsible */
  details.region-collapse {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 12px;
    box-shadow: var(--shadow); overflow: hidden;
  }
  details.region-collapse summary {
    cursor: pointer; padding: 14px 18px;
    font-size: 1.05em; font-weight: 700; color: var(--green-dark);
    user-select: none; transition: background 0.15s ease;
  }
  details.region-collapse summary:hover { background: var(--ochre-light); }
  details.region-collapse[open] summary { border-bottom: 1px solid var(--border); }
  .region-count { font-weight: 400; color: var(--text-muted); font-size: 0.85em; }

  /* Place expandable card */
  .place-expandable {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 12px; margin-bottom: 8px; overflow: hidden;
    transition: box-shadow 0.2s ease;
  }
  .place-expandable:hover { box-shadow: var(--shadow); }
  .place-expandable > summary {
    cursor: pointer; padding: 10px 16px; user-select: none;
    list-style: none; display: flex; align-items: center; gap: 10px;
  }
  .place-expandable > summary::-webkit-details-marker { display: none; }
  .place-expandable > summary::after {
    content: "\25B6"; margin-left: auto; font-size: 0.7em;
    color: var(--text-muted); transition: transform 0.2s ease;
  }
  .place-expandable[open] > summary::after { transform: rotate(90deg); }
  .pe-icon { font-size: 1.3em; }
  .pe-name { font-weight: 600; }
  .pe-name a { color: var(--text); text-decoration: none; }
  .pe-name a:hover { color: var(--terracotta-dark); }
  .pe-badge {
    background: var(--ochre-light); color: var(--terracotta-dark);
    padding: 2px 8px; border-radius: 10px; font-size: 0.76em; font-weight: 600;
  }
  .pe-body { padding: 12px 16px 16px; border-top: 1px solid var(--border); }
  .pe-desc { font-size: 0.93em; line-height: 1.55; color: var(--text); }
  .pe-tags { margin-top: 8px; }
  .pe-coords { margin-top: 8px; color: var(--text-muted); font-family: monospace; font-size: 0.78em; }
  .pe-link { display: inline-block; margin-top: 8px; font-size: 0.85em; font-weight: 600; }
  .pe-hidden { display: none !important; }

  #wiki-map {
    height: 480px;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    margin-bottom: 28px;
    z-index: 0;
  }
  #place-map {
    height: 320px;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    margin-top: 14px;
    margin-bottom: 28px;
    z-index: 0;
  }
  .leaflet-popup-content { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  .leaflet-popup-content b { color: var(--terracotta-dark); }
  .leaflet-popup-content a { color: var(--green-dark); font-weight: 600; }

  @media (max-width: 600px) {
    .hero { padding: 34px 16px 30px; }
    .hero h1 { font-size: 1.5em; }
    .container { padding: 0 12px 30px; }
    .filter-bar { gap: 6px; }
    .filter-chip { font-size: 0.82em; padding: 5px 10px; }
    #wiki-map { height: 340px; }
    #place-map { height: 240px; }
  }
</style>
"""


def _tags_html(tags: list[str]) -> str:
    return "".join(f'<span class="tag">{e(t)}</span>' for t in tags) if tags else ""


def _breadcrumbs(place: WikiPlaceRecord | None = None) -> str:
    parts = ['<div class="breadcrumbs"><a href="/wiki">📚 Wiki</a>']
    if place:
        if place.parent_place_id:
            parts.append(
                f' › <a href="/wiki/place/{e(place.parent_place_id)}">{e(place.region or "Регион")}</a>'
            )
        parts.append(f' › <span>{e(place.canonical_name)}</span>')
    parts.append('</div>')
    return "".join(parts)


def _place_expandable(p: WikiPlaceRecord) -> str:
    icon = _emoji_for(p.place_type)
    tags = json.loads(p.tags_json) if p.tags_json else []
    desc = (p.description or "")[:200]
    if len(p.description or "") > 200:
        desc += "…"
    coords = f"<div class='pe-coords'>📍 {p.lat:.4f}, {p.lng:.4f}</div>" if p.lat and p.lng else ""
    tags_html = "<div class='pe-tags'>" + _tags_html(tags) + "</div>" if tags else ""
    return f'''
    <details class="place-expandable" data-type="{e(p.place_type or "другое")}">
      <summary>
        <span class="pe-icon">{icon}</span>
        <span class="pe-name"><a href="/wiki/place/{e(p.id)}" onclick="event.stopPropagation()">{e(p.canonical_name)}</a></span>
        <span class="pe-badge">{e(p.place_type or "—")}</span>
      </summary>
      <div class="pe-body">
        {"<div class='pe-desc'>" + e(desc) + "</div>" if desc else ""}
        {tags_html}
        {coords}
        <a class="pe-link" href="/wiki/place/{e(p.id)}">Открыть полностью →</a>
      </div>
    </details>'''


LEAFLET_CSS = ""
LEAFLET_JS = ""
YANDEX_MAPS_JS = "<script src='https://api-maps.yandex.ru/2.1/?apikey=416d89a7-e586-4b4e-86a4-a0696e7a0f53&lang=ru_RU'></script>"


def _build_map_js(places: list[WikiPlaceRecord], div_id: str) -> str:
    """Yandex Maps JS API с маркерами."""
    markers_data = []
    for p in places:
        if not p.lat or not p.lng:
            continue
        icon = _emoji_for(p.place_type)
        markers_data.append({
            "lat": p.lat, "lng": p.lng,
            "name": p.canonical_name, "id": p.id,
            "icon": icon, "type": p.place_type or "",
        })
    import json as _json
    json_str = _json.dumps(markers_data, ensure_ascii=False)
    return f"""
<script>
ymaps.ready(function() {{
  var data = {json_str};
  if (!data.length) return;
  var map = new ymaps.Map('{div_id}', {{
    center: [42.52, 47.64],
    zoom: 8,
    controls: ['zoomControl', 'fullscreenControl']
  }});
  var bounds = [];
  data.forEach(function(d) {{
    var placemark = new ymaps.Placemark([d.lat, d.lng], {{
      balloonContent: '<b>' + d.icon + ' ' + d.name + '</b><br><a href="/wiki/place/' + d.id + '">Открыть →</a>',
      iconContent: d.icon
    }}, {{
      preset: 'islands#icon',
      iconColor: '#c1550f'
    }});
    map.geoObjects.add(placemark);
    bounds.push([d.lat, d.lng]);
  }});
  if (bounds.length > 0) {{
    var minLat = bounds[0][0], maxLat = bounds[0][0];
    var minLng = bounds[0][1], maxLng = bounds[0][1];
    for (var i = 1; i < bounds.length; i++) {{
      if (bounds[i][0] < minLat) minLat = bounds[i][0];
      if (bounds[i][0] > maxLat) maxLat = bounds[i][0];
      if (bounds[i][1] < minLng) minLng = bounds[i][1];
      if (bounds[i][1] > maxLng) maxLng = bounds[i][1];
    }}
    map.setBounds([[minLat, minLng], [maxLat, maxLng]], {{checkZoomRange: true, zoomMargin: [50, 50, 50, 50]}});
    if (map.getZoom() > 8) map.setZoom(8);
  }}
}});
</script>"""


def _page_head(title: str, with_map: bool = False) -> str:
    extra = YANDEX_MAPS_JS if with_map else ""
    return (
        f"<html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{e(title)}</title>{CSS}{extra}</head><body>"
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return '<html><head><meta http-equiv="refresh" content="0; url=/wiki"></head></html>'


@app.get("/wiki", response_class=HTMLResponse)
async def wiki_list(request: Request):
    db = _get_db_path()
    places = await list_wiki_places(db)

    # Группируем по parent_place_id
    regions: dict[str | None, list[WikiPlaceRecord]] = {}
    for p in places:
        key = p.parent_place_id or "__root__"
        regions.setdefault(key, []).append(p)

    # Region name lookup
    region_names: dict[str, str] = {}
    for p in places:
        if p.place_type == "город" or p.parent_place_id is None:
            region_names[p.id] = p.canonical_name

    # Маркеры для карты (только места с координатами)
    map_places = [p for p in places if p.lat and p.lng and p.parent_place_id is not None]
    map_js = _build_map_js(map_places, "wiki-map") if map_places else ""

    # Собираем типы для фильтров
    all_child_places = [p for p in places if p.parent_place_id is not None]
    type_counts: dict[str, int] = {}
    for p in all_child_places:
        t = p.place_type or "другое"
        type_counts[t] = type_counts.get(t, 0) + 1
    filter_types = sorted(type_counts.items(), key=lambda x: -x[1])
    filter_html = '<div class="filter-bar" id="filterBar">'
    filter_html += '<div class="filter-chip active" data-filter="all">📍 Все</div>'
    for t, cnt in filter_types:
        filter_html += f'<div class="filter-chip" data-filter="{e(t)}">{_emoji_for(t)} {e(t)} ({cnt})</div>'
    filter_html += '</div>'

    html_parts = [
        _page_head("Wiki — Дагестан", with_map=bool(map_places)),
        "<div class='hero'><div class='hero-inner'>",
        "<h1>🏔️ Wiki мест Дагестана</h1>",
        "<div class='subtitle'>Терракотовые горы, охристые плато, зелёные долины — гид по местам, собранным из постов</div>",
        f"<div class='stats-pill'>{len(all_child_places)} мест в базе</div>",
        "</div></div>",
        "<div class='container'>",
        "<div class='nav-top'><a href='/wiki'>Все места</a> <a href='/wiki/regions'>Регионы</a> <a href='#wiki-map'>🗺️ Карта</a></div>",
    ]

    if map_js:
        html_parts.append("<h2>🗺️ Карта мест</h2>")
        html_parts.append("<div id='wiki-map'></div>")

    html_parts.append(filter_html)

    root_places = regions.get("__root__", [])
    region_places = [p for p in root_places if p.parent_place_id is None and (p.place_type == "город" or p.post_count == 0)]
    regular_places = [p for p in root_places if p.parent_place_id is None and p not in region_places]

    for region in region_places:
        children = sorted(regions.get(region.id, []), key=lambda c: c.canonical_name)
        if not children:
            continue
        html_parts.append('<details class="region-collapse" open>')
        html_parts.append(
            f'<summary>{_emoji_for(region.place_type)} '
            f'<a href="/wiki/place/{e(region.id)}" onclick="event.stopPropagation()">{e(region.canonical_name)}</a> '
            f'<span class="region-count">({len(children)} мест)</span></summary>'
        )
        html_parts.append('<div>')
        for child in children:
            html_parts.append(_place_expandable(child))
        html_parts.append('</div></details>')

    if regular_places:
        html_parts.append('<details class="region-collapse"><summary>Без категории '
                          f'<span class="region-count">({len(regular_places)} мест)</span></summary>')
        html_parts.append('<div>')
        for p in sorted(regular_places, key=lambda c: c.canonical_name):
            html_parts.append(_place_expandable(p))
        html_parts.append('</div></details>')

    if not places:
        html_parts.append('<div class="empty">Wiki пока пуста.<br>Обработайте несколько постов и запустите distill.</div>')

    html_parts.append("</div>")
    if map_js:
        html_parts.append(map_js)
    html_parts.append("""
<script>
document.querySelectorAll('.filter-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    var filter = chip.dataset.filter;
    document.querySelectorAll('.place-expandable').forEach(function(p) {
      if (filter === 'all' || p.dataset.type === filter) {
        p.classList.remove('pe-hidden');
      } else {
        p.classList.add('pe-hidden');
      }
    });
  });
});
</script>
""")
    html_parts.append("</body></html>")
    return "".join(html_parts)
    return "".join(html_parts)


@app.get("/wiki/regions", response_class=HTMLResponse)
async def wiki_regions(request: Request):
    db = _get_db_path()
    places = await list_wiki_places(db)
    regions = [p for p in places if p.parent_place_id is None and any(c.parent_place_id == p.id for c in places)]

    html_parts = [
        _page_head("Регионы — Wiki Дагестан"),
        "<div class='hero'><div class='hero-inner'>",
        "<h1>🗺️ Регионы Дагестана</h1>",
        "<div class='subtitle'>Выберите регион, чтобы увидеть места внутри него</div>",
        "</div></div>",
        "<div class='container'>",
        _breadcrumbs(),
    ]

    for r in regions:
        children = [p for p in places if p.parent_place_id == r.id]
        if not children:
            continue
        types = {}
        for c in children:
            types.setdefault(c.place_type or "другое", 0)
            types[c.place_type or "другое"] += 1
        types_str = ", ".join(f"{e(k)}: {v}" for k, v in sorted(types.items(), key=lambda x: -x[1]))
        html_parts.append(f"""
        <div class="place-card">
          <div class="place-name"><span class="place-icon">{_emoji_for(r.place_type)}</span><a href="/wiki/place/{e(r.id)}">{e(r.canonical_name)}</a></div>
          <div class="place-region">{len(children)} мест · {types_str}</div>
          {'<div class="place-desc">' + e(r.description) + '</div>' if r.description else ''}
        </div>""")

    html_parts.append("</div></body></html>")
    return "".join(html_parts)


@app.get("/wiki/place/{place_id}", response_class=HTMLResponse)
async def wiki_place_detail(place_id: str, request: Request):
    db = _get_db_path()
    place = await get_wiki_place(db, place_id)
    if not place:
        return f"<html><body><h1>Не найдено</h1>{_breadcrumbs()}<a href='/wiki'>← Назад</a></body></html>", 404

    tags = json.loads(place.tags_json) if place.tags_json else []
    coords = f"{place.lat:.5f}, {place.lng:.5f}" if place.lat else "—"
    maps_link = ""
    if place.lat and place.lng:
        maps_link = (
            f'<div class="map-link"><a href="https://yandex.ru/maps/?pt={place.lng},{place.lat}&z=15&l=map" '
            f'target="_blank">📍 Открыть на карте</a></div>'
        )

    # Дочерние места (если это регион)
    all_places = await list_wiki_places(db)
    children = [p for p in all_places if p.parent_place_id == place.id]

    # Связанные места
    links = await get_place_links(db, place_id)

    # Цены
    prices_html = ""
    if place.prices_json:
        try:
            prices = json.loads(place.prices_json)
            if prices:
                rows = "".join(
                    f"<tr><td>{e(p.get('item',''))}</td><td class='price-amount'>{e(p.get('amount',''))}</td>"
                    f"<td>{e(p.get('currency','RUB'))}</td><td>{e(p.get('note',''))}</td></tr>"
                    for p in prices
                )
                prices_html = f"""
                <h2>💰 Цены</h2>
                <table class="price-table">
                  <tr><th>Что</th><th>Цена</th><th>Валюта</th><th>Примечание</th></tr>
                  {rows}
                </table>"""
        except json.JSONDecodeError:
            pass

    # Дети
    children_html = ""
    if children:
        children_html = f"<h2>📍 Места в регионе ({len(children)})</h2><ul class='place-list'>"
        for c in sorted(children, key=lambda x: x.canonical_name):
            children_html += _place_list_item(c)
        children_html += "</ul>"

    # Связи
    related_html = ""
    if links:
        items = "".join(
            f'<span class="related-item"><a href="/wiki/place/{e(lid)}">{e(lname)}</a></span>'
            for lid, lname, _ in links
        )
        related_html = f'<div class="related"><h2>🔗 Связанные места</h2>{items}</div>'

    icon = _emoji_for(place.place_type)

    # Карта места
    place_map_js = ""
    if place.lat and place.lng:
        place_map_js = _build_map_js([place], "place-map")

    return f"""
    {_page_head(place.canonical_name, with_map=bool(place_map_js))}
    <div class='hero'><div class='hero-inner'>
      <h1>{icon} {e(place.canonical_name)}</h1>
      <div class='subtitle'>{e(place.region or 'Дагестан')}</div>
    </div></div>
    <div class='container'>
      {_breadcrumbs(place)}
      <div class="place-card">
        <div class="place-region">{e(place.region or 'Дагестан')}
          <span class="place-type">{e(place.place_type or '—')}</span>
        </div>
        <div class="coords">Координаты: {e(coords)}</div>
        {maps_link}
        <div class="place-desc" style="font-size:1.05em">{e(place.description) or 'Описание пока не сгенерировано.'}</div>
        {'<div class="tags">' + _tags_html(tags) + '</div>' if tags else ''}
      </div>
      <div class="stats">Упоминаний: {place.post_count}</div>
      {prices_html}
      {place_map_js and '<h2>🗺️ На карте</h2><div id="place-map"></div>'}
      {children_html}
      {related_html}
    </div>
    {place_map_js}
    </body></html>
    """


@app.get("/api/places", response_class=JSONResponse)
async def api_places():
    db = _get_db_path()
    places = await list_wiki_places(db)
    return [
        {
            "id": p.id, "name": p.canonical_name, "region": p.region,
            "place_type": p.place_type, "lat": p.lat, "lng": p.lng,
            "description": p.description,
            "tags": json.loads(p.tags_json) if p.tags_json else [],
            "prices": json.loads(p.prices_json) if p.prices_json else None,
            "parent_place_id": p.parent_place_id,
            "post_count": p.post_count,
        }
        for p in places
    ]


def run():
    import uvicorn
    port = int(os.getenv("WIKI_WEB_PORT", "8088"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run()
