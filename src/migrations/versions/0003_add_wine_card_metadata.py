"""Add optional wine rating and serving recommendation fields."""

import sqlalchemy as sa
from alembic import op

revision = "0003_add_wine_card_metadata"
down_revision = "0002_add_catalog_photo_and_link"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("wine_catalog")}

    if "roskachestvo_rating" not in columns:
        op.add_column(
            "wine_catalog",
            sa.Column("roskachestvo_rating", sa.Text(), nullable=True),
        )
    if "serving_recommendation" not in columns:
        op.add_column(
            "wine_catalog",
            sa.Column("serving_recommendation", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("wine_catalog")}

    if "serving_recommendation" in columns:
        op.drop_column("wine_catalog", "serving_recommendation")
    if "roskachestvo_rating" in columns:
        op.drop_column("wine_catalog", "roskachestvo_rating")
