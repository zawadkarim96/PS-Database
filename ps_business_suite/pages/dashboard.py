"""Dashboard page for PS Business Suite."""
from __future__ import annotations

from ps_business_suite.core import crm, db


def main() -> None:
    db.init_shared_schema()
    crm.render_page("Dashboard")


if __name__ == "__main__":
    main()
