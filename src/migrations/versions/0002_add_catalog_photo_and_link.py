"""Store catalog photo names and wine links."""

import sqlalchemy as sa
from alembic import op

revision = "0002_add_catalog_photo_and_link"
down_revision = "0001_create_wine_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("wine_catalog")}

    if "photo_name" not in columns:
        op.add_column("wine_catalog", sa.Column("photo_name", sa.Text(), nullable=True))
    if "link" not in columns:
        op.add_column("wine_catalog", sa.Column("link", sa.Text(), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("wine_catalog")}

    if "link" in columns:
        op.drop_column("wine_catalog", "link")
    if "photo_name" in columns:
        op.drop_column("wine_catalog", "photo_name")
