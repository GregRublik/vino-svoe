from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from config import settings


def build_wine_link(slug: str | None) -> str | None:
    """Формирует ссылку на карточку вина по стабильному slug."""
    normalized_slug = (slug or "").strip().strip("/")
    if not normalized_slug:
        return None

    base_url = settings.wine_base_url.rstrip("/")
    return f"{base_url}/wines/{quote(normalized_slug, safe='-._~')}"


def slug_from_filename(filename: str | None) -> str | None:
    """Извлекает slug из канонического имени файла `<slug>.<extension>`."""
    if not filename:
        return None

    slug = Path(filename).stem.strip()
    return slug or None


def slug_from_link(link: str | None) -> str | None:
    """Извлекает slug из URL карточки вина."""
    if not link:
        return None

    path = urlparse(link).path.rstrip("/")
    if not path:
        return None

    slug = unquote(path.rsplit("/", 1)[-1]).strip()
    return slug or None


def extract_wine_slug(
    content: dict | None = None,
    metadata: dict | None = None,
    link: str | None = None,
) -> str | None:
    """Находит slug в payload, ссылке или каноническом имени файла."""
    for payload in (content or {}, metadata or {}):
        slug = payload.get("slug")
        if isinstance(slug, str) and slug.strip():
            return slug.strip().strip("/")

    return slug_from_link(link) or slug_from_filename(
        (content or {}).get("filename") or (metadata or {}).get("filename")
    )
