"""Database maintenance utilities."""
from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from ps_business_suite.core import db
from ps_business_suite.core.utils import render_signature


def _db_size(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return 0.0


def main() -> None:
    db.init_shared_schema()
    path = db.database_path()
    st.title("Database maintenance")
    st.info("This workspace stores both CRM and Sales data in a single SQLite file.")
    st.metric("Database location", str(path))
    st.metric("Size (MB)", f"{_db_size(path):.2f}")

    with db.connection_scope() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        st.write(f"Total tables: {len(tables)}")
        st.dataframe([{"table": row[0]} for row in tables], use_container_width=True)

    with open(path, "rb") as fh:
        st.download_button(
            "Download SQLite backup",
            data=fh.read(),
            file_name="ps_business_suite.db",
            mime="application/octet-stream",
        )

    if st.button("Run VACUUM/optimize"):
        with db.connection_scope() as conn:
            conn.execute("VACUUM")
        st.success("Database optimised successfully.")

    render_signature()


if __name__ == "__main__":
    main()
