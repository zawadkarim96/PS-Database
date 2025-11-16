"""PS Business Suite entry point (Dashboard page)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps_business_suite.core import crm, db


def _streamlit_runtime_active() -> bool:
    """Return ``True`` when running inside a Streamlit runtime."""

    runtime = None
    try:
        from streamlit import runtime as st_runtime

        runtime = st_runtime
    except Exception:  # pragma: no cover - best effort detection
        runtime = None

    if runtime is not None:
        try:
            if runtime.exists():
                return True
        except Exception:  # pragma: no cover - defensive
            pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover - defensive
        return False

    try:
        return get_script_run_ctx() is not None
    except Exception:  # pragma: no cover - defensive
        return False


def _streamlit_flag_options_from_env() -> dict[str, object]:
    """Map hosting environment variables to Streamlit CLI flags."""

    flag_options: dict[str, object] = {}

    port_env = os.getenv("PORT") or os.getenv("STREAMLIT_SERVER_PORT")
    if port_env:
        try:
            port = int(port_env)
        except (TypeError, ValueError):
            port = None
        if port and port > 0:
            flag_options["server.port"] = port

    address_env = (
        os.getenv("HOST")
        or os.getenv("BIND_ADDRESS")
        or os.getenv("STREAMLIT_SERVER_ADDRESS")
        or os.getenv("STREAMLIT_SERVER_HOST")
    )
    flag_options["server.address"] = address_env or "0.0.0.0"

    external_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
    if external_host:
        flag_options["browser.serverAddress"] = external_host

    headless_env = os.getenv("STREAMLIT_SERVER_HEADLESS")
    if headless_env is None:
        flag_options["server.headless"] = True
    else:
        flag_options["server.headless"] = headless_env.strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    return flag_options


def _bootstrap_streamlit_app() -> None:
    """Launch Streamlit via bootstrap when executed with plain ``python``."""

    try:
        from streamlit.web import bootstrap
    except Exception:  # pragma: no cover - Streamlit not installed
        return

    flag_options = _streamlit_flag_options_from_env()
    try:
        bootstrap.load_config_options(flag_options)
    except Exception:  # pragma: no cover - defensive best effort
        pass

    try:
        bootstrap.run(
            os.path.abspath(__file__),
            False,
            [],
            flag_options,
        )
    except Exception:  # pragma: no cover - Streamlit bootstrap failure
        pass


def main() -> None:
    st.set_page_config(page_title="PS Business Suite", layout="wide")
    db.init_shared_schema()
    crm.render_page("Dashboard")


if _streamlit_runtime_active():
    main()
elif __name__ == "__main__":
    _bootstrap_streamlit_app()
