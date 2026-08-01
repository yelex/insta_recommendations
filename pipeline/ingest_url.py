"""url-ingest: скачивание видео/карусели по ссылке (Instagram и др.).

yt-dlp extract_flat=False с ignoreerrors даёт для каждого элемента карусели
свой CDN URL в requested_formats. Скачиваем напрямую.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import yt_dlp

from pipeline.models import MediaItem, RawItem
from pipeline.ingest import create_video_raw_item

logger = logging.getLogger(__name__)


def _pick_best_thumbnail(thumbnails: list[dict]) -> str | None:
    """Выбирает лучший thumbnail.

    Для Instagram фото-каруселей yt-dlp отдаёт ~13 thumbnails на каждое фото
    (разные размеры), обычно без width/height. id='0' — самый полноразмерный.
    """
    if not thumbnails:
        return None
    # Сначала пытаемся найти с максимальным разрешением
    sized = [t for t in thumbnails if t.get("width") and t.get("height")]
    if sized:
        best = max(sized, key=lambda t: t["width"] * t["height"])
        return best.get("url")
    # Fallback: id='0' — обычно full-res для Instagram
    for t in thumbnails:
        if str(t.get("id", "")) == "0":
            return t.get("url")
    # Last resort: просто первый
    return thumbnails[0].get("url")


def _download_image(url: str, destination: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("url-ingest: не удалось скачать изображение: %s", exc)
        return False


def _download_video_direct(url: str, destination: Path) -> bool:
    """Скачивает видео по прямому CDN URL."""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("url-ingest: не удалось скачать видео напрямую: %s", exc)
        return False


def download_video_from_url(
    url: str,
    item_id: str,
    media_storage_dir: Path,
    caption_text: str | None = None,
    source_message_id: str | None = None,
) -> RawItem:
    """Скачивает медиа по URL через yt-dlp и строит RawItem."""
    media_storage_dir.mkdir(parents=True, exist_ok=True)

    # Шаг 1: извлекаем инфо (не flat — нужны requested_formats с CDN URLs)
    # ignore_no_formats_error — критично для фото-каруселей: yt-dlp иначе
    # убивает entry целиком при отсутствии video formats, и мы теряем
    # thumbnail URLs (которые для фото и есть основной контент).
    ydl_opts_info = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "extract_flat": False,
        "noplaylist": False,
        "ignore_no_formats_error": True,
    }

    logger.info("url-ingest: извлечение инфо %s", url)

    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            video_info = ydl.extract_info(url, download=False)
    except Exception as exc:
        logger.error("url-ingest: не удалось получить инфо %s: %s", url, exc)
        raise RuntimeError(f"Не удалось получить информацию по ссылке: {exc}") from exc

    if video_info is None:
        raise RuntimeError("yt-dlp вернул None")

    # Описание поста
    ig_description = (video_info or {}).get("description") or ""
    ig_title = (video_info or {}).get("title") or ""
    parts: list[str] = []
    if ig_description:
        parts.append(ig_description.strip())
    if ig_title and ig_title != ig_description:
        parts.append(ig_title.strip())
    if caption_text and caption_text != url:
        parts.append(caption_text.strip())
    combined_caption = "\n".join(parts) if parts else None

    entries = (video_info or {}).get("entries")

    # Одиночное видео
    if not entries:
        output_path = media_storage_dir / f"{item_id}.mp4"
        ydl_opts_dl = {
            "outtmpl": str(output_path),
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
                ydl.download([url])
        except Exception as exc:
            logger.error("url-ingest: не удалось скачать %s: %s", url, exc)
            raise RuntimeError(f"Не удалось скачать видео по ссылке: {exc}") from exc

        if not output_path.exists():
            raise RuntimeError(f"Файл не создан: {output_path}")

        raw_item = create_video_raw_item(
            item_id=item_id,
            file_path=output_path,
            caption_text=combined_caption,
            source_message_id=source_message_id,
        )
        logger.info("url-ingest: скачано id=%s file_path=%s", raw_item.id, output_path)
        return raw_item

    # Карусель — у каждого элемента свой CDN URL в requested_formats
    entries = [e for e in entries if e is not None]
    if not entries:
        raise RuntimeError("Карусель пуста")

    media_items: list[MediaItem] = []

    for i, entry in enumerate(entries):
        # Пытаемся получить прямой CDN URL из requested_formats или formats
        cdn_url = None
        req_formats = entry.get("requested_formats") or []
        if req_formats:
            # Берём best video format
            for f in req_formats:
                if f.get("vcodec") and f["vcodec"] != "none":
                    cdn_url = f.get("url")
                    break
            if not cdn_url and req_formats:
                cdn_url = req_formats[0].get("url")

        if not cdn_url:
            # Fallback: ищем в formats
            formats = entry.get("formats") or []
            for f in formats:
                if f.get("vcodec") and f["vcodec"] != "none" and f.get("url"):
                    cdn_url = f["url"]
                    break

        is_video = entry.get("duration", 0) > 0 or cdn_url is not None

        if is_video and cdn_url:
            video_path = media_storage_dir / f"{item_id}_{i}.mp4"
            if _download_video_direct(cdn_url, video_path):
                media_items.append(MediaItem(path=video_path, kind="video"))
                logger.info("url-ingest: карусель [%d] видео: %s (%d bytes)",
                            i, video_path.name, video_path.stat().st_size)
            else:
                logger.warning("url-ingest: карусель [%d] не удалось скачать видео", i)
        else:
            # Фото — скачиваем лучший thumbnail (для фото это полноразмерное изображение)
            thumbnails = entry.get("thumbnails") or []
            thumb_url = _pick_best_thumbnail(thumbnails)
            if thumb_url:
                img_path = media_storage_dir / f"{item_id}_{i}.jpg"
                if _download_image(thumb_url, img_path):
                    media_items.append(MediaItem(path=img_path, kind="image"))
                    logger.info("url-ingest: карусель [%d] фото: %s (%d bytes)",
                                i, img_path.name, img_path.stat().st_size)
                else:
                    logger.warning("url-ingest: карусель [%d] не удалось скачать фото", i)
            else:
                logger.warning("url-ingest: карусель [%d] нет thumbnail URL", i)

    if not media_items:
        raise RuntimeError("Ни один элемент карусели не был скачан")

    logger.info(
        "url-ingest: карусель id=%s, %d элементов (%d видео, %d фото)",
        item_id, len(media_items),
        sum(1 for m in media_items if m.kind == "video"),
        sum(1 for m in media_items if m.kind == "image"),
    )

    return RawItem(
        id=item_id,
        type="carousel",
        received_at=datetime.now(timezone.utc),
        file_path=media_items[0].path if media_items else None,
        caption_text=combined_caption,
        source_message_id=source_message_id,
        media_items=media_items,
    )
