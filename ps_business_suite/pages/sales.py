"""Sales module entry point."""
from __future__ import annotations

from ps_business_suite.core import db
from ps_business_suite.core import sales


def main() -> None:
    db.init_shared_schema()
    sales.render_page()


if __name__ == "__main__":
    main()
