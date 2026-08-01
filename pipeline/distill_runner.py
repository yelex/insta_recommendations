"""Точка входа для distill-джобы: python -m pipeline.distill_runner

Запуск:
    cd /root/.openclaw/workspace/insta_recommendations
    .venv/bin/python -m pipeline.distill_runner           # инкрементально
    .venv/bin/python -m pipeline.distill_runner --rebuild  # полный пересбор
"""
import argparse
import asyncio
import logging
import sys

from bot.config import load_settings
from pipeline.distill import run_distill_job, rebuild_all_wiki


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Wiki distill job")
    parser.add_argument("--rebuild", action="store_true", help="Полный пересбор wiki")
    args = parser.parse_args()

    settings = load_settings()
    database_path = settings.database_url.replace("sqlite:///", "")

    if args.rebuild:
        processed = asyncio.run(
            rebuild_all_wiki(database_path, settings.glm_api_key, settings.glm_base_url)
        )
    else:
        processed = asyncio.run(
            run_distill_job(database_path, settings.glm_api_key, settings.glm_base_url)
        )
    print(f"Готово. Обработано: {processed}", file=sys.stderr)


if __name__ == "__main__":
    main()
