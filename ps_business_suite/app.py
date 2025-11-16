"""PS Business Suite entry point (Dashboard page)."""
from __future__ import annotations

import streamlit as st

from ps_business_suite.core import db
from ps_business_suite.core import crm


def main() -> None:
    st.set_page_config(page_title="PS Business Suite", layout="wide")
    db.init_shared_schema()
    crm.render_page("Dashboard")


if __name__ == "__main__":
    main()
