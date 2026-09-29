"""Create the wine catalog table."""

import sqlalchemy as sa
from alembic import op

revision = "0001_create_wine_catalog"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("wine_catalog"):
        return

    op.create_table(
        "wine_catalog",
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("winery", sa.Text(), nullable=True),
        sa.Column("region", sa.Text(), nullable=True),
        sa.Column("grape_variety", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("color", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("slug"),
    )


def downgrade() -> None:
    op.drop_table("wine_catalog")
