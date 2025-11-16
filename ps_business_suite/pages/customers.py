"""Customers workspace page consolidating CRM tools."""
from __future__ import annotations

import streamlit as st

from ps_business_suite.core import crm, db


SECTION_LABELS = {
    "Customers": "Customers",
    "Customer summary": "Customer Summary",
    "Warranties": "Warranties",
    "Maintenance & Service": "Maintenance and Service",
    "Imports": "Import",
    "Scrap bin": "Scraps",
    "Duplicates": "Duplicates",
}


def main() -> None:
    db.init_shared_schema()
    choice = st.selectbox("Select CRM section", list(SECTION_LABELS.keys()))
    crm.render_page(SECTION_LABELS[choice])


if __name__ == "__main__":
    main()
