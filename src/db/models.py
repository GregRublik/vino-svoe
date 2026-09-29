"""ORM-модели каталога вин."""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base


class WineCatalogRow(Base):
    """Строка каталога вина в PostgreSQL."""

    __tablename__ = "wine_catalog"

    slug: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    winery: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    grape_variety: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    photo_name: Mapped[str | None] = mapped_column(Text)
    link: Mapped[str | None] = mapped_column(Text)
