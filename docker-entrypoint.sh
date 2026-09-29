#!/bin/sh
set -eu

/app/.venv/bin/alembic upgrade head
/app/.venv/bin/python -m services.catalog_import
exec "$@"
