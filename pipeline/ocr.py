"""OCR-модуль: чтение текста с кадров через Tesseract.

Используется как дополнение к GLM vision — Tesseract точно читает
текстовые подписи/overlay, которые vision-модель может пропустить.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_frame(frame_path: Path, languages: str = "rus+eng") -> str:
    """Извлекает текст из одного кадра через Tesseract OCR."""
    try:
        img = Image.open(frame_path)
        text = pytesseract.image_to_string(img, lang=languages)
        return text.strip()
    except Exception as exc:
        logger.warning("ocr: не удалось обработать %s: %s", frame_path.name, exc)
        return ""


def extract_text_from_frames(frame_paths: list[Path], languages: str = "rus+eng") -> str:
    """Извлекает текст из списка кадров, склеивает уникальные строки.

    Возвращает общий текст со всех кадров (дедуплицированный по строкам).
    """
    seen_lines: dict[str, bool] = {}
    all_lines: list[str] = []

    for frame_path in frame_paths:
        text = extract_text_from_frame(frame_path, languages)
        if not text:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) > 1 and line not in seen_lines:
                seen_lines[line] = True
                all_lines.append(line)

    result = "\n".join(all_lines)
    if result:
        logger.info("ocr: извлечён текст из %d кадров, %d уникальных строк", len(frame_paths), len(all_lines))
    return result
