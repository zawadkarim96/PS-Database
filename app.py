import io
import os
import sqlite3
import hashlib
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from textwrap import dedent
import pandas as pd


import streamlit as st

# ---------- Config ----------
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "ps_crm.db"))
DATE_FMT = "%d-%m-%Y"

UPLOADS_DIR = BASE_DIR / "uploads"
DELIVERY_ORDER_DIR = UPLOADS_DIR / "delivery_orders"
SERVICE_DOCS_DIR = UPLOADS_DIR / "service_documents"
MAINTENANCE_DOCS_DIR = UPLOADS_DIR / "maintenance_documents"

REQUIRED_CUSTOMER_FIELDS = {
    "name": "Name",
    "phone": "Phone",
    "address": "Address",
}


def customer_complete_clause(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return " AND ".join(
        [
            f"TRIM(COALESCE({prefix}name, '')) <> ''",
            f"TRIM(COALESCE({prefix}phone, '')) <> ''",
            f"TRIM(COALESCE({prefix}address, '')) <> ''",
        ]
    )


def customer_incomplete_clause(alias: str = "") -> str:
    return f"NOT ({customer_complete_clause(alias)})"

# ---------- Schema ----------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    pass_hash TEXT,
    role TEXT DEFAULT 'staff',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    email TEXT,
    address TEXT,
    city TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    dup_flag INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    model TEXT,
    serial TEXT,
    unit_price REAL,
    dup_flag INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    order_date TEXT,
    delivery_date TEXT,
    notes TEXT,
    dup_flag INTEGER DEFAULT 0,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER DEFAULT 1,
    unit_price REAL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS warranties (
    warranty_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product_id INTEGER,
    serial TEXT,
    issue_date TEXT,
    expiry_date TEXT,
    status TEXT DEFAULT 'active',
    dup_flag INTEGER DEFAULT 0,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS delivery_orders (
    do_number TEXT PRIMARY KEY,
    customer_id INTEGER,
    order_id INTEGER,
    description TEXT,
    sales_person TEXT,
    file_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    do_number TEXT,
    customer_id INTEGER,
    service_date TEXT,
    description TEXT,
    remarks TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(do_number) REFERENCES delivery_orders(do_number) ON DELETE SET NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS maintenance_records (
    maintenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    do_number TEXT,
    customer_id INTEGER,
    maintenance_date TEXT,
    description TEXT,
    remarks TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(do_number) REFERENCES delivery_orders(do_number) ON DELETE SET NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS service_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_id INTEGER,
    file_path TEXT,
    original_name TEXT,
    uploaded_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(service_id) REFERENCES services(service_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS maintenance_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    maintenance_id INTEGER,
    file_path TEXT,
    original_name TEXT,
    uploaded_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(maintenance_id) REFERENCES maintenance_records(maintenance_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS needs (
    need_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product TEXT,
    unit TEXT,
    unit_price REAL,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);
"""

# ---------- Helpers ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_schema(conn):
    ensure_upload_dirs()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    # bootstrap admin if empty
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_user = os.getenv("ADMIN_USER", "admin")
        admin_pass = os.getenv("ADMIN_PASS", "admin123")
        h = hashlib.sha256(admin_pass.encode("utf-8")).hexdigest()
        conn.execute("INSERT INTO users (username, pass_hash, role) VALUES (?, ?, 'admin')", (admin_user, h))
        conn.commit()

def df_query(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)

def fmt_dates(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(DATE_FMT)
    return df


def clean_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    return value or None


def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def _streamlit_runtime_active() -> bool:
    """Return True when running inside a Streamlit runtime."""

    runtime = None
    try:
        from streamlit import runtime as st_runtime

        runtime = st_runtime
    except Exception:
        runtime = None

    if runtime is not None:
        try:
            if runtime.exists():
                return True
        except Exception:
            pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def ensure_upload_dirs():
    for path in (UPLOADS_DIR, DELIVERY_ORDER_DIR, SERVICE_DOCS_DIR, MAINTENANCE_DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, target_dir: Path, filename: Optional[str] = None) -> Optional[Path]:
    if uploaded_file is None:
        return None
    ensure_upload_dirs()
    safe_name = filename or uploaded_file.name
    safe_name = "".join(ch for ch in safe_name if ch.isalnum() or ch in (".", "_", "-"))
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    dest = target_dir / safe_name
    counter = 1
    while dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.read())
    return dest


def resolve_upload_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _sanitize_path_component(value: Optional[str]) -> str:
    if not value:
        return "item"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.() ")
    cleaned = "".join(ch if ch in allowed else "_" for ch in str(value))
    cleaned = cleaned.strip()
    return cleaned or "item"


def build_customer_groups(conn, only_complete: bool = True):
    where_clause = ""
    params = ()
    if only_complete:
        where_clause = f"WHERE {customer_complete_clause()}"
    df = df_query(
        conn,
        f"SELECT customer_id, TRIM(name) AS name FROM customers {where_clause}",
        params,
    )
    if df.empty:
        return [], {}
    df["name"] = df["name"].fillna("")
    df["norm_name"] = df["name"].astype(str).str.strip()
    df.sort_values(by=["norm_name", "customer_id"], inplace=True)
    groups = []
    label_by_id = {}
    for norm_name, group in df.groupby("norm_name", sort=False):
        ids = group["customer_id"].astype(int).tolist()
        primary = ids[0]
        raw_name = clean_text(group.iloc[0].get("name"))
        count = len(ids)
        base_label = raw_name or f"Customer #{primary}"
        if raw_name and count > 1:
            display_label = f"{base_label} ({count} records)"
        else:
            display_label = base_label
        groups.append(
            {
                "norm_name": norm_name,
                "primary_id": primary,
                "ids": ids,
                "raw_name": raw_name,
                "label": display_label,
                "count": count,
            }
        )
        for cid in ids:
            label_by_id[int(cid)] = display_label
    groups.sort(key=lambda g: (g["norm_name"] or "").lower())
    return groups, label_by_id


def fetch_customer_choices(conn):
    groups, label_by_id = build_customer_groups(conn, only_complete=True)
    options = [None]
    labels = {None: "-- Select customer --"}
    group_map = {}
    for group in groups:
        primary = group["primary_id"]
        options.append(primary)
        labels[primary] = group["label"]
        group_map[primary] = group["ids"]
    return options, labels, group_map, label_by_id


def attach_documents(
    conn,
    table: str,
    fk_column: str,
    record_id: int,
    files,
    target_dir: Path,
    prefix: str,
):
    if not files:
        return 0
    saved = 0
    for idx, uploaded in enumerate(files, start=1):
        if uploaded is None:
            continue
        original_name = uploaded.name or f"{prefix}_{idx}.pdf"
        safe_original = Path(original_name).name
        filename = f"{prefix}_{idx}_{safe_original}"
        saved_path = save_uploaded_file(uploaded, target_dir, filename=filename)
        if not saved_path:
            continue
        try:
            stored_path = str(saved_path.relative_to(BASE_DIR))
        except ValueError:
            stored_path = str(saved_path)
        conn.execute(
            f"INSERT INTO {table} ({fk_column}, file_path, original_name) VALUES (?, ?, ?)",
            (int(record_id), stored_path, safe_original),
        )
        saved += 1
    return saved


def bundle_documents_zip(documents):
    if not documents:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in documents:
            path = doc.get("path")
            archive_name = doc.get("archive_name")
            if not path or not archive_name:
                continue
            if not path.exists():
                continue
            zf.write(path, archive_name)
    buffer.seek(0)
    return buffer


def dedupe_join(values: Iterable[Optional[str]]) -> str:
    seen = []
    for value in values:
        if value is None:
            continue
        val = str(value).strip()
        if not val:
            continue
        if val not in seen:
            seen.append(val)
    return ", ".join(seen)


def collapse_warranty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()
    work["description"] = work.apply(
        lambda row: dedupe_join(
            [
                clean_text(row.get("product")),
                clean_text(row.get("model")),
                clean_text(row.get("serial")),
            ]
        ),
        axis=1,
    )
    issue_dt = pd.to_datetime(work.get("issue_date"), errors="coerce")
    expiry_dt = pd.to_datetime(work.get("expiry_date"), errors="coerce")
    work["issue_fmt"] = issue_dt.dt.strftime(DATE_FMT)
    work.loc[issue_dt.isna(), "issue_fmt"] = None
    work["expiry_fmt"] = expiry_dt.dt.strftime(DATE_FMT)
    work.loc[expiry_dt.isna(), "expiry_fmt"] = None
    work["expiry_dt"] = expiry_dt

    grouped = (
        work.groupby("customer", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "description": dedupe_join(g["description"].tolist()),
                    "issue_date": dedupe_join(g["issue_fmt"].tolist()),
                    "expiry_date": dedupe_join(g["expiry_fmt"].tolist()),
                    "_sort": g["expiry_dt"].min(),
                }
            )
        )
        .reset_index()
    )
    grouped = grouped.sort_values("_sort", na_position="last").drop(columns=["_sort"])
    grouped.rename(columns={"customer": "Customer", "description": "Description", "issue_date": "Issue date", "expiry_date": "Expiry date"}, inplace=True)
    if "Customer" in grouped.columns:
        grouped["Customer"] = grouped["Customer"].fillna("(unknown)")
    return grouped


def export_database_to_excel(conn) -> bytes:
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'") if not str(row[0]).startswith("sqlite_")]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for table in tables:
            df = df_query(conn, f"SELECT * FROM {table}")
            if df.empty:
                cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                df = pd.DataFrame(columns=cols)
            sheet_name = table[:31] if table else "Sheet"
            if not sheet_name:
                sheet_name = "Sheet"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def fetch_warranty_window(conn, start_days: int, end_days: int) -> pd.DataFrame:
    query = dedent(
        """
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.status='active'
          AND date(w.expiry_date) BETWEEN date('now', ?) AND date('now', ?)
        ORDER BY date(w.expiry_date) ASC
        """
    )
    start = f"+{start_days} day"
    end = f"+{end_days} day"
    return df_query(conn, query, (start, end))


def format_warranty_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()
    expiry_raw = pd.to_datetime(work.get("expiry_date"), errors="coerce")
    today = pd.Timestamp.now().normalize()
    status_labels = []
    work["Description"] = work.apply(
        lambda row: dedupe_join(
            [
                clean_text(row.get("product")),
                clean_text(row.get("model")),
                clean_text(row.get("serial")),
            ]
        ),
        axis=1,
    )
    for idx in work.index:
        exp = expiry_raw.loc[idx] if expiry_raw is not None and idx in expiry_raw.index else pd.NaT
        if pd.notna(exp) and exp.normalize() < today:
            status_labels.append("Expired")
        else:
            base_status = clean_text(work.loc[idx, "status"]) if "status" in work.columns else None
            status_labels.append((base_status or "Active").title())
    work["Status"] = status_labels
    for col in ("product", "model", "serial"):
        if col in work.columns:
            work.drop(columns=[col], inplace=True)
    if "status" in work.columns:
        work.drop(columns=["status"], inplace=True)
    rename_map = {
        "customer": "Customer",
        "issue_date": "Issue date",
        "expiry_date": "Expiry date",
    }
    work.rename(columns={k: v for k, v in rename_map.items() if k in work.columns}, inplace=True)
    for col in ("dup_flag", "id", "duplicate"):
        if col in work.columns:
            work.drop(columns=[col], inplace=True)
    return work


def _pdf_escape_text(value: str) -> str:
    replacements = [("\\", "\\\\"), ("(", "\\("), (")", "\\)")]
    escaped = value
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def _build_simple_pdf_document(lines: list[str]) -> bytes:
    if not lines:
        lines = [""]
    commands = ["BT", "/F1 12 Tf", "72 770 Td"]
    for idx, line in enumerate(lines):
        escaped = _pdf_escape_text(line)
        if idx == 0:
            commands.append(f"({escaped}) Tj")
        else:
            commands.append("0 -14 Td")
            commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream_bytes = "\n".join(commands).encode("latin-1", "replace")

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = []

    def write_obj(obj_id: int, body: bytes) -> None:
        offsets.append(buffer.tell())
        buffer.write(f"{obj_id} 0 obj\n".encode("latin-1"))
        buffer.write(body)
        if not body.endswith(b"\n"):
            buffer.write(b"\n")
        buffer.write(b"endobj\n")

    write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")
    write_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n")
    write_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n",
    )
    stream_obj = b"<< /Length %d >>\nstream\n" % len(stream_bytes) + stream_bytes + b"\nendstream\n"
    write_obj(4, stream_obj)
    write_obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n")

    xref_offset = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets) + 1}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for off in offsets:
        buffer.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    buffer.write(b"trailer\n")
    buffer.write(f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n".encode("latin-1"))
    buffer.write(b"startxref\n")
    buffer.write(f"{xref_offset}\n".encode("latin-1"))
    buffer.write(b"%%EOF\n")
    return buffer.getvalue()


def generate_customer_summary_pdf(customer_name: str, info: dict, warranties: Optional[pd.DataFrame], services: pd.DataFrame, maintenance: pd.DataFrame) -> bytes:
    lines: list[str] = [f"Customer Summary – {customer_name}", ""]
    lines.extend(
        [
            f"Phone: {clean_text(info.get('phone')) or '-'}",
            f"Email: {clean_text(info.get('email')) or '-'}",
            f"Address: {clean_text(info.get('address')) or '-'}",
            f"City: {clean_text(info.get('city')) or '-'}",
            "",
        ]
    )

    def extend_section(title: str, rows: list[str]) -> None:
        lines.append(title)
        if not rows:
            lines.append("  (no records)")
        else:
            for row in rows:
                lines.append(f"  • {row}")
        lines.append("")

    warranty_rows: list[str] = []
    if warranties is not None and isinstance(warranties, pd.DataFrame) and not warranties.empty:
        for _, row in warranties.iterrows():
            warranty_rows.append(
                " | ".join(
                    [
                        f"Description: {clean_text(row.get('Description')) or '-'}",
                        f"Issue: {clean_text(row.get('Issue date')) or '-'}",
                        f"Expiry: {clean_text(row.get('Expiry date')) or '-'}",
                        f"Status: {clean_text(row.get('Status')) or '-'}",
                    ]
                )
            )

    service_rows: list[str] = []
    if isinstance(services, pd.DataFrame) and not services.empty:
        for _, row in services.iterrows():
            service_rows.append(
                " | ".join(
                    [
                        f"DO: {clean_text(row.get('do_number')) or '-'}",
                        f"Date: {clean_text(row.get('service_date')) or '-'}",
                        f"Desc: {clean_text(row.get('description')) or '-'}",
                        f"Remarks: {clean_text(row.get('remarks')) or '-'}",
                    ]
                )
            )

    maintenance_rows: list[str] = []
    if isinstance(maintenance, pd.DataFrame) and not maintenance.empty:
        for _, row in maintenance.iterrows():
            maintenance_rows.append(
                " | ".join(
                    [
                        f"DO: {clean_text(row.get('do_number')) or '-'}",
                        f"Date: {clean_text(row.get('maintenance_date')) or '-'}",
                        f"Desc: {clean_text(row.get('description')) or '-'}",
                        f"Remarks: {clean_text(row.get('remarks')) or '-'}",
                    ]
                )
            )

    extend_section("Warranties", warranty_rows)
    extend_section("Service history", service_rows)
    extend_section("Maintenance history", maintenance_rows)

    return _build_simple_pdf_document(lines)


def _streamlit_flag_options_from_env() -> dict[str, object]:
    """Derive Streamlit bootstrap flag options from environment variables."""

    flag_options: dict[str, object] = {}

    port_env = os.getenv("PORT")
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
        or os.getenv("RENDER_EXTERNAL_HOSTNAME")
    )
    flag_options["server.address"] = address_env or "0.0.0.0"

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
    """Launch the Streamlit app when executed via ``python app.py``."""

    try:
        from streamlit.web import bootstrap
    except Exception:
        return

    try:
        bootstrap.run(
            os.path.abspath(__file__),
            False,
            [],
            _streamlit_flag_options_from_env(),
        )
    except Exception:
        pass


def recalc_customer_duplicate_flag(conn, phone):
    if not phone or str(phone).strip() == "":
        return
    cur = conn.execute("SELECT customer_id FROM customers WHERE phone = ?", (str(phone).strip(),))
    ids = [int(row[0]) for row in cur.fetchall()]
    dup = 1 if len(ids) > 1 else 0
    conn.executemany("UPDATE customers SET dup_flag=? WHERE customer_id=?", [(dup, cid) for cid in ids])


def init_ui():
    st.set_page_config(page_title="PS Mini CRM", page_icon="🧰", layout="wide")
    st.title("PS Engineering – Mini CRM")
    st.caption("Customers • Warranties • Needs • Summaries")
    if "user" not in st.session_state:
        st.session_state.user = None

# ---------- Auth ----------
def login_box(conn):
    st.sidebar.markdown("### Login")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user['username']} ({st.session_state.user['role']})")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "Dashboard"
            _safe_rerun()
        return True
    with st.sidebar.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        row = df_query(conn, "SELECT user_id, username, pass_hash, role FROM users WHERE username = ?", (u,))
        if not row.empty and hashlib.sha256(p.encode("utf-8")).hexdigest() == row.iloc[0]["pass_hash"]:
            st.session_state.user = {"user_id": int(row.iloc[0]["user_id"]), "username": row.iloc[0]["username"], "role": row.iloc[0]["role"]}
            st.session_state.page = "Dashboard"
            st.session_state.just_logged_in = True
            _safe_rerun()
        else:
            st.sidebar.error("Invalid credentials")
    st.stop()

def ensure_auth(role=None):
    if role and st.session_state.user and st.session_state.user["role"] != role:
        st.warning("You do not have permission to access this page.")
        st.stop()

# ---------- Pages ----------
def dashboard(conn):
    st.subheader("📊 Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    complete_count = int(
        df_query(conn, f"SELECT COUNT(*) c FROM customers WHERE {customer_complete_clause()}").iloc[0]["c"]
    )
    scrap_count = int(
        df_query(conn, f"SELECT COUNT(*) c FROM customers WHERE {customer_incomplete_clause()}").iloc[0]["c"]
    )
    with col1:
        st.metric("Customers", complete_count)
    with col2:
        st.metric("Scraps", scrap_count)
    with col3:
        st.metric(
            "Active Warranties",
            int(
                df_query(
                    conn,
                    "SELECT COUNT(*) c FROM warranties WHERE date(expiry_date) >= date('now')",
                ).iloc[0]["c"]
            ),
        )
    with col4:
        expired_count = int(
            df_query(
                conn,
                "SELECT COUNT(*) c FROM warranties WHERE date(expiry_date) < date('now')",
            ).iloc[0]["c"]
        )
        st.metric("Expired", expired_count)

    month_expired = int(
        df_query(
            conn,
            """
            SELECT COUNT(*) c
            FROM warranties
            WHERE date(expiry_date) < date('now')
              AND strftime('%Y-%m', expiry_date) = strftime('%Y-%m', 'now')
            """,
        ).iloc[0]["c"]
    )
    service_month = int(
        df_query(
            conn,
            """
            SELECT COUNT(*) c
            FROM services
            WHERE service_date IS NOT NULL
              AND strftime('%Y-%m', service_date) = strftime('%Y-%m', 'now')
            """,
        ).iloc[0]["c"]
    )
    maintenance_month = int(
        df_query(
            conn,
            """
            SELECT COUNT(*) c
            FROM maintenance_records
            WHERE maintenance_date IS NOT NULL
              AND strftime('%Y-%m', maintenance_date) = strftime('%Y-%m', 'now')
            """,
        ).iloc[0]["c"]
    )
    today_expired_df = df_query(
        conn,
        """
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE date(w.expiry_date) = date('now')
        ORDER BY date(w.expiry_date) ASC
        """,
    )
    today_expired_count = len(today_expired_df.index)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Expired this month", month_expired)
    with col6:
        st.metric("Services this month", service_month)
    with col7:
        st.metric("Maintenance this month", maintenance_month)
    with col8:
        st.metric("Expired today", today_expired_count)

    if not today_expired_df.empty:
        notice = collapse_warranty_rows(today_expired_df)
        lines = []
        for _, row in notice.iterrows():
            customer = row.get("Customer") or "(unknown)"
            description = row.get("Description") or ""
            if description:
                lines.append(f"- {customer}: {description}")
            else:
                lines.append(f"- {customer}")
        st.warning("⚠️ Warranties expiring today:\n" + "\n".join(lines))

    if st.session_state.user and st.session_state.user["role"] == "admin":
        excel_bytes = export_database_to_excel(conn)
        st.download_button(
            "⬇️ Download full database (Excel)",
            excel_bytes,
            file_name="ps_crm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")
    st.subheader("🔎 Quick snapshots")
    tab1, tab2, tab3 = st.tabs(["Upcoming expiries", "Recent services", "Recent maintenance"])

    with tab1:
        upcoming = fetch_warranty_window(conn, 0, 30)
        upcoming = format_warranty_table(upcoming)
        st.dataframe(upcoming.head(10), use_container_width=True)

    with tab2:
        recent_services = df_query(
            conn,
            """
            SELECT s.do_number,
                   s.service_date,
                   COALESCE(c.name, cdo.name, '(unknown)') AS customer,
                   s.description
            FROM services s
            LEFT JOIN customers c ON c.customer_id = s.customer_id
            LEFT JOIN delivery_orders d ON d.do_number = s.do_number
            LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
            ORDER BY datetime(s.service_date) DESC, s.service_id DESC
            LIMIT 10
            """,
        )
        recent_services = fmt_dates(recent_services, ["service_date"])
        st.dataframe(
            recent_services.rename(
                columns={
                    "do_number": "DO Serial",
                    "service_date": "Service date",
                    "customer": "Customer",
                    "description": "Description",
                }
            ),
            use_container_width=True,
        )

    with tab3:
        recent_maintenance = df_query(
            conn,
            """
            SELECT m.do_number,
                   m.maintenance_date,
                   COALESCE(c.name, cdo.name, '(unknown)') AS customer,
                   m.description
            FROM maintenance_records m
            LEFT JOIN customers c ON c.customer_id = m.customer_id
            LEFT JOIN delivery_orders d ON d.do_number = m.do_number
            LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
            ORDER BY datetime(m.maintenance_date) DESC, m.maintenance_id DESC
            LIMIT 10
            """,
        )
        recent_maintenance = fmt_dates(recent_maintenance, ["maintenance_date"])
        st.dataframe(
            recent_maintenance.rename(
                columns={
                    "do_number": "DO Serial",
                    "maintenance_date": "Maintenance date",
                    "customer": "Customer",
                    "description": "Description",
                }
            ),
            use_container_width=True,
        )


def show_expiry_notifications(conn):
    if not st.session_state.get("just_logged_in"):
        return

    total_expired = int(
        df_query(
            conn,
            "SELECT COUNT(*) c FROM warranties WHERE date(expiry_date) < date('now')",
        ).iloc[0]["c"]
    )
    if total_expired == 0:
        st.session_state.just_logged_in = False
        return
    month_expired = int(
        df_query(
            conn,
            """
            SELECT COUNT(*) c
            FROM warranties
            WHERE date(expiry_date) < date('now')
              AND strftime('%Y-%m', expiry_date) = strftime('%Y-%m', 'now')
            """,
        ).iloc[0]["c"]
    )
    expired_recent = df_query(
        conn,
        """
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE date(w.expiry_date) < date('now')
        ORDER BY date(w.expiry_date) DESC
        LIMIT 12
        """,
    )
    formatted = format_warranty_table(expired_recent)
    try:
        with st.modal("Expired warranties alert"):
            st.markdown("### Warranties needing attention")
            st.write(
                f"Total expired: **{total_expired}**, expired this month: **{month_expired}**."
            )
            if formatted is None or formatted.empty:
                st.info("No expired warranties found.")
            else:
                st.dataframe(formatted, use_container_width=True)
    except Exception:
        if total_expired > 0:
            st.warning(
                f"{total_expired} warranties are past expiry. {month_expired} expired this month."
            )
    try:
        if total_expired > 0:
            st.toast(f"{total_expired} warranties require attention.")
    except Exception:
        pass

    st.session_state.just_logged_in = False

def customers_page(conn):
    st.subheader("👥 Customers")
    with st.expander("Add new customer"):
        with st.form("new_customer"):
            name = st.text_input("Name *")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            address = st.text_area("Address")
            city = st.text_input("City")
            st.markdown("**Purchase / Warranty (optional)**")
            pur_date = st.date_input("Purchase/Issue date", value=datetime.now().date())
            prod_name = st.text_input("Product")
            prod_model = st.text_input("Model")
            prod_serial = st.text_input("Serial")
            unit = st.text_input("Unit (e.g., pcs, set)")
            price = st.number_input("Unit price", min_value=0.0, value=0.0, step=0.01)
            submitted = st.form_submit_button("Save")
            if submitted and name.strip():
                cur = conn.cursor()
                dupc = 0
                if phone and phone.strip():
                    cur.execute("SELECT 1 FROM customers WHERE phone=? LIMIT 1", (phone.strip(),))
                    dupc = 1 if cur.fetchone() else 0
                cur.execute("INSERT INTO customers (name, phone, email, address, city, dup_flag) VALUES (?, ?, ?, ?, ?, ?)",
                            (name.strip(), phone, email, address, city, dupc))
                cid = cur.lastrowid
                conn.commit()
                # If a product name is provided, create product and warranty
                if prod_name.strip():
                    # check for existing product by name+model
                    cur.execute("SELECT product_id FROM products WHERE name=? AND IFNULL(model,'')=IFNULL(?, '') LIMIT 1", (prod_name.strip(), prod_model.strip() or None))
                    row = cur.fetchone()
                    if row:
                        pid = row[0]
                    else:
                        cur.execute("INSERT INTO products (name, model, serial, unit_price) VALUES (?, ?, ?, ?)", (prod_name.strip(), prod_model.strip() or None, prod_serial.strip() or None, float(price) if price else None))
                        pid = cur.lastrowid
                    # create warranty 1-year from purchase date
                    issue = pur_date.strftime("%Y-%m-%d")
                    expiry = (pur_date + timedelta(days=365)).strftime("%Y-%m-%d")
                    cur.execute("INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, 'active')",
                                (cid, pid, prod_serial.strip() or None, issue, expiry))
                    conn.commit()
                st.success("Customer saved")

    sort_dir = st.radio("Sort by created date", ["Newest first", "Oldest first"], horizontal=True)
    order = "DESC" if sort_dir == "Newest first" else "ASC"
    q = st.text_input("Search (name/phone/email/city)")
    df = df_query(conn, f"""
        SELECT customer_id as id, name, phone, email, address, city, created_at, dup_flag
        FROM customers
        WHERE (? = '' OR name LIKE '%'||?||'%' OR phone LIKE '%'||?||'%' OR email LIKE '%'||?||'%' OR city LIKE '%'||?||'%')
        ORDER BY datetime(created_at) {order}
    """, (q,q,q,q,q))
    df = fmt_dates(df, ["created_at"])
    if "dup_flag" in df.columns:
        df = df.assign(duplicate=df["dup_flag"].apply(lambda x: "🔁 duplicate phone" if int(x)==1 else ""))
    if not df.empty:
        df = df.assign(
            scrap=df.apply(
                lambda row: "🗂 scrap"
                if any(
                    clean_text(row.get(col)) is None
                    for col in REQUIRED_CUSTOMER_FIELDS
                    if col in row
                )
                else "",
                axis=1,
            )
        )
    st.dataframe(df)
    if not df.empty and 'dup_flag' in df.columns:
        st.info("🔁 = duplicate phone detected")
    if not df.empty:
        st.caption("🗂 scrap = missing mandatory details. Fix these from the Scraps page.")

    st.markdown("**Recently Added Customers**")
    recent_df = df_query(conn, """
        SELECT customer_id as id, name, phone, email, city, created_at
        FROM customers
        ORDER BY datetime(created_at) DESC LIMIT 200
    """)
    recent_df = fmt_dates(recent_df, ["created_at"])
    st.dataframe(recent_df)

    if st.session_state.user and st.session_state.user.get("role") == "admin":
        st.markdown("---")
        st.markdown("**Delete Customers**")
        all_cust = df_query(
            conn,
            f"SELECT customer_id, name FROM customers WHERE {customer_complete_clause()} ORDER BY name ASC",
        )
        if all_cust.empty:
            st.info("No customers to delete.")
        else:
            ids = all_cust["customer_id"].astype(int).tolist()
            name_map = {int(i): str(n) for i, n in zip(all_cust["customer_id"].astype(int), all_cust["name"].fillna("").astype(str))}
            select_all = st.checkbox("Select all customers", key="del_all_customers")
            selected = st.multiselect(
                "Customers to delete",
                ids,
                default=ids if select_all else [],
                format_func=lambda i: name_map.get(int(i), str(i)),
                key="del_customers",
            )
            if selected and st.button("Delete selected", type="primary"):
                cur = conn.cursor()
                cur.executemany("DELETE FROM customers WHERE customer_id=?", [(int(i),) for i in selected])
                conn.commit()
                st.success(f"Deleted {len(selected)} customers.")
                _safe_rerun()

def warranties_page(conn):
    st.subheader("🛡️ Warranties")
    sort_dir = st.radio("Sort by expiry date", ["Soonest first", "Latest first"], horizontal=True)
    order = "ASC" if sort_dir == "Soonest first" else "DESC"
    q = st.text_input("Search (customer/product/model/serial)")

    base = dedent(
        """
        SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial,
               w.issue_date, w.expiry_date, w.status, w.dup_flag
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE (? = '' OR c.name LIKE '%'||?||'%' OR p.name LIKE '%'||?||'%' OR p.model LIKE '%'||?||'%' OR w.serial LIKE '%'||?||'%')
          AND {date_cond}
        ORDER BY date(w.expiry_date) {order}
        """
    )

    active = df_query(conn, base.format(date_cond="date(w.expiry_date) >= date('now')", order=order), (q,q,q,q,q))
    active = fmt_dates(active, ["issue_date","expiry_date"])
    if "dup_flag" in active.columns:
        active = active.assign(Duplicate=active["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        active.drop(columns=["dup_flag"], inplace=True)
    active = format_warranty_table(active)
    st.markdown("**Active Warranties**")
    st.dataframe(active, use_container_width=True)

    expired = df_query(conn, base.format(date_cond="date(w.expiry_date) < date('now')", order="DESC"), (q,q,q,q,q))
    expired = fmt_dates(expired, ["issue_date","expiry_date"])
    if "dup_flag" in expired.columns:
        expired = expired.assign(Duplicate=expired["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        expired.drop(columns=["dup_flag"], inplace=True)
    expired = format_warranty_table(expired)
    st.markdown("**Expired Warranties**")
    st.dataframe(expired, use_container_width=True)

    st.markdown("---")
    st.subheader("🔔 Upcoming Expiries")
    col1, col2 = st.columns(2)
    soon3 = collapse_warranty_rows(fetch_warranty_window(conn, 0, 3))
    soon60 = collapse_warranty_rows(fetch_warranty_window(conn, 0, 60))
    with col1:
        st.caption("Next **3** days")
        st.dataframe(soon3, use_container_width=True)
    with col2:
        st.caption("Next **60** days")
        st.dataframe(soon60, use_container_width=True)


def delivery_orders_page(conn):
    st.subheader("🚚 Delivery Orders")
    customer_options, customer_labels, _, _ = fetch_customer_choices(conn)

    with st.form("delivery_order_form"):
        do_number = st.text_input("Delivery Order Serial *")
        selected_customer_index = 0
        selected_customer = st.selectbox(
            "Customer",
            options=customer_options,
            index=selected_customer_index,
            format_func=lambda cid: customer_labels.get(cid, str(cid)),
        )
        description = st.text_area("Description")
        sales_person = st.text_input("Sales person")
        uploaded_pdf = st.file_uploader("Upload Delivery Order (PDF)", type=["pdf"])
        submit = st.form_submit_button("Save Delivery Order", type="primary")

    if submit:
        serial = clean_text(do_number)
        if not serial:
            st.error("Delivery Order serial is required.")
        else:
            cur = conn.execute("SELECT 1 FROM delivery_orders WHERE do_number = ?", (serial,))
            if cur.fetchone():
                st.error("A Delivery Order with this serial already exists.")
            else:
                stored_path = None
                if uploaded_pdf is not None:
                    saved = save_uploaded_file(uploaded_pdf, DELIVERY_ORDER_DIR, filename=f"{serial}.pdf")
                    if saved:
                        try:
                            stored_path = str(saved.relative_to(BASE_DIR))
                        except ValueError:
                            stored_path = str(saved)
                conn.execute(
                    "INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        serial,
                        int(selected_customer) if selected_customer else None,
                        None,
                        clean_text(description),
                        clean_text(sales_person),
                        stored_path,
                    ),
                )
                conn.commit()
                st.success("Delivery Order saved.")
                _safe_rerun()

    do_df = df_query(
        conn,
        """
        SELECT d.do_number, c.name AS customer, d.description, d.sales_person, d.created_at, d.file_path, d.order_id
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    if not do_df.empty:
        display = do_df.copy()
        display = fmt_dates(display, ["created_at"])
        display.rename(
            columns={
                "do_number": "DO Serial",
                "customer": "Customer",
                "description": "Description",
                "sales_person": "Sales person",
                "created_at": "Created",
            },
            inplace=True,
        )
        st.markdown("### Recorded Delivery Orders")
        st.dataframe(
            display.drop(columns=["file_path", "order_id"], errors="ignore"),
            use_container_width=True,
        )

        downloadable = do_df.to_dict("records")
        with st.expander("Delivery Order files", expanded=False):
            has_files = False
            for record in downloadable:
                path = resolve_upload_path(record.get("file_path"))
                if path and path.exists():
                    has_files = True
                    st.download_button(
                        f"Download {record['do_number']}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"do_dl_{record['do_number']}",
                    )
            if not has_files:
                st.info("No Delivery Order files uploaded yet.")
    else:
        st.info("No Delivery Orders recorded yet.")


def services_page(conn):
    st.subheader("🛠️ Service Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    do_df = df_query(
        conn,
        """
        SELECT d.do_number, d.customer_id, COALESCE(c.name, '(unknown)') AS customer_name, d.description
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    do_options = [None]
    do_labels = {None: "-- Select delivery order --"}
    do_customer_map = {}
    do_customer_name_map = {}
    for _, row in do_df.iterrows():
        do_num = clean_text(row.get("do_number"))
        if not do_num:
            continue
        cust_id = int(row["customer_id"]) if not pd.isna(row.get("customer_id")) else None
        summary = clean_text(row.get("description"))
        cust_name = customer_label_map.get(cust_id) if cust_id else clean_text(row.get("customer_name"))
        label = do_num
        if summary:
            label = f"{do_num} – {summary[:40]}" + ("…" if len(summary) > 40 else "")
        do_options.append(do_num)
        do_labels[do_num] = label
        do_customer_map[do_num] = cust_id
        do_customer_name_map[do_num] = cust_name or "(not linked)"

    with st.form("service_form"):
        selected_do = st.selectbox(
            "Delivery Order *",
            options=do_options,
            format_func=lambda do: do_labels.get(do, str(do)),
        )
        customer_name_display = do_customer_name_map.get(selected_do)
        st.text_input(
            "Customer",
            value=customer_name_display or "(not linked)",
            disabled=True,
        )
        service_date = st.date_input("Service date", value=datetime.now().date())
        description = st.text_area("Service description")
        remarks = st.text_area("Remarks / updates")
        service_files = st.file_uploader(
            "Attach service documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="service_new_docs",
        )
        submit = st.form_submit_button("Log service", type="primary")

    if submit:
        if not selected_do:
            st.error("Delivery Order is required for service records.")
        else:
            selected_customer = do_customer_map.get(selected_do)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO services (do_number, customer_id, service_date, description, remarks, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    selected_do,
                    int(selected_customer) if selected_customer else None,
                    service_date.strftime("%Y-%m-%d") if service_date else None,
                    clean_text(description),
                    clean_text(remarks),
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            service_id = cur.lastrowid
            saved_docs = attach_documents(
                conn,
                "service_documents",
                "service_id",
                service_id,
                service_files,
                SERVICE_DOCS_DIR,
                f"service_{service_id}",
            )
            conn.commit()
            message = "Service record saved."
            if saved_docs:
                message = f"{message} Attached {saved_docs} document(s)."
            st.success(message)
            _safe_rerun()

    service_df = df_query(
        conn,
        """
        SELECT s.service_id,
               s.do_number,
               s.service_date,
               s.description,
               s.remarks,
               s.updated_at,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(sd.document_id) AS doc_count
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN service_documents sd ON sd.service_id = s.service_id
        GROUP BY s.service_id
        ORDER BY datetime(s.service_date) DESC, s.service_id DESC
        """,
    )
    if not service_df.empty:
        service_df = fmt_dates(service_df, ["service_date"])
        service_df["Last update"] = pd.to_datetime(service_df.get("updated_at"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M")
        service_df.loc[service_df["Last update"].isna(), "Last update"] = None
        display = service_df.rename(
            columns={
                "do_number": "DO Serial",
                "service_date": "Service date",
                "description": "Description",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.markdown("### Service history")
        st.dataframe(
            display.drop(columns=["updated_at", "service_id"], errors="ignore"),
            use_container_width=True,
        )

        records = service_df.to_dict("records")
        st.markdown("#### Update remarks")
        options = [int(r["service_id"]) for r in records]
        labels = {
            int(r["service_id"]): f"#{int(r['service_id'])} – {r.get('do_number')}"
            for r in records
        }
        selected_service_id = st.selectbox(
            "Select service entry",
            options,
            format_func=lambda rid: labels.get(rid, f"#{rid}"),
        )
        selected_record = next(r for r in records if int(r["service_id"]) == int(selected_service_id))
        new_remarks = st.text_area(
            "Remarks",
            value=clean_text(selected_record.get("remarks")) or "",
            key=f"service_edit_{selected_service_id}",
        )
        if st.button("Save remarks", key="save_service_remarks"):
            conn.execute(
                "UPDATE services SET remarks = ?, updated_at = datetime('now') WHERE service_id = ?",
                (clean_text(new_remarks), int(selected_service_id)),
            )
            conn.commit()
            st.success("Service remarks updated.")
            _safe_rerun()

        attachments_df = df_query(
            conn,
            """
            SELECT document_id, file_path, original_name, uploaded_at
            FROM service_documents
            WHERE service_id = ?
            ORDER BY datetime(uploaded_at) DESC, document_id DESC
            """,
            (int(selected_service_id),),
        )
        st.markdown("**Attached documents**")
        if attachments_df.empty:
            st.caption("No documents attached yet.")
        else:
            for _, doc_row in attachments_df.iterrows():
                path = resolve_upload_path(doc_row.get("file_path"))
                display_name = clean_text(doc_row.get("original_name"))
                if path and path.exists():
                    label = display_name or path.name
                    st.download_button(
                        f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"service_doc_dl_{int(doc_row['document_id'])}",
                    )
                else:
                    label = display_name or "Document"
                    st.caption(f"⚠️ Missing file: {label}")

        with st.form(f"service_doc_upload_{selected_service_id}"):
            more_docs = st.file_uploader(
                "Add more service documents (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"service_doc_files_{selected_service_id}",
            )
            upload_docs = st.form_submit_button("Upload documents")
        if upload_docs:
            if more_docs:
                saved = attach_documents(
                    conn,
                    "service_documents",
                    "service_id",
                    int(selected_service_id),
                    more_docs,
                    SERVICE_DOCS_DIR,
                    f"service_{selected_service_id}",
                )
                conn.commit()
                st.success(f"Uploaded {saved} document(s).")
                _safe_rerun()
            else:
                st.info("Select at least one PDF to upload.")
    else:
        st.info("No service records yet. Log one using the form above.")


def maintenance_page(conn):
    st.subheader("🔧 Maintenance Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    do_df = df_query(
        conn,
        """
        SELECT d.do_number, d.customer_id, COALESCE(c.name, '(unknown)') AS customer_name, d.description
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    do_options = [None]
    do_labels = {None: "-- Select delivery order --"}
    do_customer_map = {}
    do_customer_name_map = {}
    for _, row in do_df.iterrows():
        do_num = clean_text(row.get("do_number"))
        if not do_num:
            continue
        cust_id = int(row["customer_id"]) if not pd.isna(row.get("customer_id")) else None
        summary = clean_text(row.get("description"))
        cust_name = customer_label_map.get(cust_id) if cust_id else clean_text(row.get("customer_name"))
        label = do_num
        if summary:
            label = f"{do_num} – {summary[:40]}" + ("…" if len(summary) > 40 else "")
        do_options.append(do_num)
        do_labels[do_num] = label
        do_customer_map[do_num] = cust_id
        do_customer_name_map[do_num] = cust_name or "(not linked)"

    with st.form("maintenance_form"):
        selected_do = st.selectbox(
            "Delivery Order *",
            options=do_options,
            format_func=lambda do: do_labels.get(do, str(do)),
        )
        st.text_input(
            "Customer",
            value=do_customer_name_map.get(selected_do) or "(not linked)",
            disabled=True,
        )
        maintenance_date = st.date_input("Maintenance date", value=datetime.now().date())
        description = st.text_area("Maintenance description")
        remarks = st.text_area("Remarks / updates")
        maintenance_files = st.file_uploader(
            "Attach maintenance documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="maintenance_new_docs",
        )
        submit = st.form_submit_button("Log maintenance", type="primary")

    if submit:
        if not selected_do:
            st.error("Delivery Order is required for maintenance records.")
        else:
            selected_customer = do_customer_map.get(selected_do)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO maintenance_records (do_number, customer_id, maintenance_date, description, remarks, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    selected_do,
                    int(selected_customer) if selected_customer else None,
                    maintenance_date.strftime("%Y-%m-%d") if maintenance_date else None,
                    clean_text(description),
                    clean_text(remarks),
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            maintenance_id = cur.lastrowid
            saved_docs = attach_documents(
                conn,
                "maintenance_documents",
                "maintenance_id",
                maintenance_id,
                maintenance_files,
                MAINTENANCE_DOCS_DIR,
                f"maintenance_{maintenance_id}",
            )
            conn.commit()
            message = "Maintenance record saved."
            if saved_docs:
                message = f"{message} Attached {saved_docs} document(s)."
            st.success(message)
            _safe_rerun()

    maintenance_df = df_query(
        conn,
        """
        SELECT m.maintenance_id,
               m.do_number,
               m.maintenance_date,
               m.description,
               m.remarks,
               m.updated_at,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(md.document_id) AS doc_count
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN maintenance_documents md ON md.maintenance_id = m.maintenance_id
        GROUP BY m.maintenance_id
        ORDER BY datetime(m.maintenance_date) DESC, m.maintenance_id DESC
        """,
    )
    if not maintenance_df.empty:
        maintenance_df = fmt_dates(maintenance_df, ["maintenance_date"])
        maintenance_df["Last update"] = pd.to_datetime(maintenance_df.get("updated_at"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M")
        maintenance_df.loc[maintenance_df["Last update"].isna(), "Last update"] = None
        display = maintenance_df.rename(
            columns={
                "do_number": "DO Serial",
                "maintenance_date": "Maintenance date",
                "description": "Description",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.markdown("### Maintenance history")
        st.dataframe(
            display.drop(columns=["updated_at", "maintenance_id"], errors="ignore"),
            use_container_width=True,
        )

        records = maintenance_df.to_dict("records")
        st.markdown("#### Update remarks")
        options = [int(r["maintenance_id"]) for r in records]
        labels = {
            int(r["maintenance_id"]): f"#{int(r['maintenance_id'])} – {r.get('do_number')}"
            for r in records
        }
        selected_maintenance_id = st.selectbox(
            "Select maintenance entry",
            options,
            format_func=lambda rid: labels.get(rid, f"#{rid}"),
        )
        selected_record = next(r for r in records if int(r["maintenance_id"]) == int(selected_maintenance_id))
        new_remarks = st.text_area(
            "Remarks",
            value=clean_text(selected_record.get("remarks")) or "",
            key=f"maintenance_edit_{selected_maintenance_id}",
        )
        if st.button("Save maintenance remarks", key="save_maintenance_remarks"):
            conn.execute(
                "UPDATE maintenance_records SET remarks = ?, updated_at = datetime('now') WHERE maintenance_id = ?",
                (clean_text(new_remarks), int(selected_maintenance_id)),
            )
            conn.commit()
            st.success("Maintenance remarks updated.")
            _safe_rerun()

        attachments_df = df_query(
            conn,
            """
            SELECT document_id, file_path, original_name, uploaded_at
            FROM maintenance_documents
            WHERE maintenance_id = ?
            ORDER BY datetime(uploaded_at) DESC, document_id DESC
            """,
            (int(selected_maintenance_id),),
        )
        st.markdown("**Attached documents**")
        if attachments_df.empty:
            st.caption("No documents attached yet.")
        else:
            for _, doc_row in attachments_df.iterrows():
                path = resolve_upload_path(doc_row.get("file_path"))
                display_name = clean_text(doc_row.get("original_name"))
                if path and path.exists():
                    label = display_name or path.name
                    st.download_button(
                        f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"maintenance_doc_dl_{int(doc_row['document_id'])}",
                    )
                else:
                    label = display_name or "Document"
                    st.caption(f"⚠️ Missing file: {label}")

        with st.form(f"maintenance_doc_upload_{selected_maintenance_id}"):
            more_docs = st.file_uploader(
                "Add more maintenance documents (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"maintenance_doc_files_{selected_maintenance_id}",
            )
            upload_docs = st.form_submit_button("Upload documents")
        if upload_docs:
            if more_docs:
                saved = attach_documents(
                    conn,
                    "maintenance_documents",
                    "maintenance_id",
                    int(selected_maintenance_id),
                    more_docs,
                    MAINTENANCE_DOCS_DIR,
                    f"maintenance_{selected_maintenance_id}",
                )
                conn.commit()
                st.success(f"Uploaded {saved} document(s).")
                _safe_rerun()
            else:
                st.info("Select at least one PDF to upload.")
    else:
        st.info("No maintenance records yet. Log one using the form above.")


def customer_summary_page(conn):
    st.subheader("📒 Customer Summary")
    blank_label = "(blank)"
    complete_clause = customer_complete_clause()
    customers = df_query(
        conn,
        f"""
        SELECT TRIM(name) AS name, GROUP_CONCAT(customer_id) AS ids, COUNT(*) AS cnt
        FROM customers
        WHERE {complete_clause}
        GROUP BY TRIM(name)
        ORDER BY TRIM(name) ASC
        """,
    )
    if customers.empty:
        st.info("No complete customers yet. Check the Scraps page for records that need details.")
        return

    names = customers["name"].tolist()
    name_map = {
        row["name"]: f"{row['name']} ({int(row['cnt'])} records)" if int(row["cnt"]) > 1 else row["name"]
        for _, row in customers.iterrows()
    }
    sel_name = st.selectbox("Select customer", names, format_func=lambda n: name_map.get(n, n))
    row = customers[customers["name"] == sel_name].iloc[0]
    ids = [int(i) for i in str(row["ids"]).split(",") if i]
    cnt = int(row["cnt"])

    info = df_query(
        conn,
        f"""
        SELECT
            MAX(name) AS name,
            GROUP_CONCAT(DISTINCT phone) AS phone,
            GROUP_CONCAT(DISTINCT email) AS email,
            GROUP_CONCAT(DISTINCT address) AS address,
            GROUP_CONCAT(DISTINCT city) AS city
        FROM customers
        WHERE customer_id IN ({','.join('?'*len(ids))})
        """,
        ids,
    ).iloc[0].to_dict()

    st.write("**Name:**", info.get("name") or blank_label)
    st.write("**Phone:**", info.get("phone"))
    st.write("**Email:**", info.get("email"))
    st.write("**Address:**", info.get("address"))
    st.write("**City:**", info.get("city"))
    if cnt > 1:
        st.caption(f"Merged from {cnt} duplicates")

    st.markdown("---")
    placeholders = ",".join("?" * len(ids))

    warr = df_query(
        conn,
        f"""
        SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date, w.status, w.dup_flag
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.customer_id IN ({placeholders})
        ORDER BY date(w.expiry_date) DESC
        """,
        ids,
    )
    warr = fmt_dates(warr, ["issue_date", "expiry_date"])
    if "dup_flag" in warr.columns:
        warr = warr.assign(duplicate=warr["dup_flag"].apply(lambda x: "🔁 duplicate" if int(x) == 1 else ""))
    warr_display = format_warranty_table(warr)

    service_df = df_query(
        conn,
        f"""
        SELECT s.service_id,
               s.do_number,
               s.service_date,
               s.description,
               s.remarks,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(sd.document_id) AS doc_count
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN service_documents sd ON sd.service_id = s.service_id
        WHERE COALESCE(s.customer_id, d.customer_id) IN ({placeholders})
        GROUP BY s.service_id
        ORDER BY datetime(s.service_date) DESC, s.service_id DESC
        """,
        ids,
    )
    service_df = fmt_dates(service_df, ["service_date"])

    maintenance_df = df_query(
        conn,
        f"""
        SELECT m.maintenance_id,
               m.do_number,
               m.maintenance_date,
               m.description,
               m.remarks,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(md.document_id) AS doc_count
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN maintenance_documents md ON md.maintenance_id = m.maintenance_id
        WHERE COALESCE(m.customer_id, d.customer_id) IN ({placeholders})
        GROUP BY m.maintenance_id
        ORDER BY datetime(m.maintenance_date) DESC, m.maintenance_id DESC
        """,
        ids,
    )
    maintenance_df = fmt_dates(maintenance_df, ["maintenance_date"])

    do_df = df_query(
        conn,
        f"""
        SELECT d.do_number,
               COALESCE(c.name, '(unknown)') AS customer,
               d.description,
               d.sales_person,
               d.created_at,
               d.file_path
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        WHERE d.customer_id IN ({placeholders})
        ORDER BY datetime(d.created_at) DESC
        """,
        ids,
    )
    if not do_df.empty:
        do_df = fmt_dates(do_df, ["created_at"])
        do_df["do_number"] = do_df["do_number"].apply(clean_text)
        do_df["Document"] = do_df["file_path"].apply(lambda fp: "📎" if clean_text(fp) else "")

    do_numbers = set()
    if not do_df.empty and "do_number" in do_df.columns:
        do_numbers.update(val for val in do_df["do_number"].tolist() if val)
    if not service_df.empty and "do_number" in service_df.columns:
        do_numbers.update(clean_text(val) for val in service_df["do_number"].tolist() if clean_text(val))
    if not maintenance_df.empty and "do_number" in maintenance_df.columns:
        do_numbers.update(clean_text(val) for val in maintenance_df["do_number"].tolist() if clean_text(val))
    do_numbers = {val for val in do_numbers if val}

    present_dos = set()
    if not do_df.empty and "do_number" in do_df.columns:
        present_dos.update(val for val in do_df["do_number"].tolist() if val)
    missing_dos = sorted(do for do in do_numbers if do not in present_dos)
    if missing_dos:
        extra_df = df_query(
            conn,
            f"""
            SELECT d.do_number,
                   COALESCE(c.name, '(unknown)') AS customer,
                   d.description,
                   d.sales_person,
                   d.created_at,
                   d.file_path
            FROM delivery_orders d
            LEFT JOIN customers c ON c.customer_id = d.customer_id
            WHERE d.do_number IN ({','.join('?' * len(missing_dos))})
            """,
            missing_dos,
        )
        if not extra_df.empty:
            extra_df = fmt_dates(extra_df, ["created_at"])
            extra_df["do_number"] = extra_df["do_number"].apply(clean_text)
            extra_df["Document"] = extra_df["file_path"].apply(lambda fp: "📎" if clean_text(fp) else "")
            do_df = pd.concat([do_df, extra_df], ignore_index=True) if not do_df.empty else extra_df
            present_dos.update(val for val in extra_df["do_number"].tolist() if val)
    orphan_dos = sorted(do for do in do_numbers if do not in present_dos)

    st.markdown("**Delivery orders**")
    if (do_df is None or do_df.empty) and not orphan_dos:
        st.info("No delivery orders found for this customer.")
    else:
        if do_df is not None and not do_df.empty:
            st.dataframe(
                do_df.rename(
                    columns={
                        "do_number": "DO Serial",
                        "customer": "Customer",
                        "description": "Description",
                        "sales_person": "Sales person",
                        "created_at": "Created",
                        "Document": "Document",
                    }
                ).drop(columns=["file_path"], errors="ignore"),
                use_container_width=True,
            )
        if orphan_dos:
            st.caption("Referenced DO codes without a recorded delivery order: " + ", ".join(orphan_dos))

    st.markdown("**Warranties**")
    if warr_display is None or warr_display.empty:
        st.info("No warranties recorded for this customer.")
    else:
        st.dataframe(warr_display)

    st.markdown("**Service records**")
    if service_df.empty:
        st.info("No service records found for this customer.")
    else:
        service_display = service_df.rename(
            columns={
                "do_number": "DO Serial",
                "service_date": "Service date",
                "description": "Description",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.dataframe(
            service_display.drop(columns=["service_id"], errors="ignore"),
            use_container_width=True,
        )

    st.markdown("**Maintenance records**")
    if maintenance_df.empty:
        st.info("No maintenance records found for this customer.")
    else:
        maintenance_display = maintenance_df.rename(
            columns={
                "do_number": "DO Serial",
                "maintenance_date": "Maintenance date",
                "description": "Description",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.dataframe(
            maintenance_display.drop(columns=["maintenance_id"], errors="ignore"),
            use_container_width=True,
        )

    documents = []
    if do_df is not None and not do_df.empty:
        for _, row in do_df.iterrows():
            path = resolve_upload_path(row.get("file_path"))
            if not path or not path.exists():
                continue
            do_ref = clean_text(row.get("do_number")) or "delivery_order"
            display_name = path.name
            archive_name = "/".join(
                [
                    _sanitize_path_component("delivery_orders"),
                    f"{_sanitize_path_component(do_ref)}_{_sanitize_path_component(display_name)}",
                ]
            )
            documents.append(
                {
                    "source": "Delivery order",
                    "reference": do_ref,
                    "display": display_name,
                    "path": path,
                    "archive_name": archive_name,
                    "key": f"do_{do_ref}",
                }
            )

    service_docs = pd.DataFrame()
    if "service_id" in service_df.columns and not service_df.empty:
        service_ids = [int(val) for val in service_df["service_id"].dropna().astype(int).tolist()]
        if service_ids:
            service_docs = df_query(
                conn,
                f"""
                SELECT document_id, service_id, file_path, original_name, uploaded_at
                FROM service_documents
                WHERE service_id IN ({','.join('?' * len(service_ids))})
                ORDER BY datetime(uploaded_at) DESC, document_id DESC
                """,
                service_ids,
            )
    service_lookup = {}
    if "service_id" in service_df.columns and not service_df.empty:
        for _, row in service_df.iterrows():
            if pd.isna(row.get("service_id")):
                continue
            service_lookup[int(row["service_id"])] = row
    if not service_docs.empty:
        for _, doc_row in service_docs.iterrows():
            path = resolve_upload_path(doc_row.get("file_path"))
            if not path or not path.exists():
                continue
            service_id = int(doc_row.get("service_id"))
            record = service_lookup.get(service_id, {})
            reference = clean_text(record.get("do_number")) or f"Service #{service_id}"
            display_name = clean_text(doc_row.get("original_name")) or path.name
            uploaded = pd.to_datetime(doc_row.get("uploaded_at"), errors="coerce")
            uploaded_fmt = uploaded.strftime("%d-%m-%Y %H:%M") if pd.notna(uploaded) else None
            archive_name = "/".join(
                [
                    _sanitize_path_component("service"),
                    f"{_sanitize_path_component(reference)}_{_sanitize_path_component(display_name)}",
                ]
            )
            documents.append(
                {
                    "source": "Service",
                    "reference": reference,
                    "display": display_name,
                    "uploaded": uploaded_fmt,
                    "path": path,
                    "archive_name": archive_name,
                    "key": f"service_{service_id}_{int(doc_row['document_id'])}",
                }
            )

    maintenance_docs = pd.DataFrame()
    if "maintenance_id" in maintenance_df.columns and not maintenance_df.empty:
        maintenance_ids = [int(val) for val in maintenance_df["maintenance_id"].dropna().astype(int).tolist()]
        if maintenance_ids:
            maintenance_docs = df_query(
                conn,
                f"""
                SELECT document_id, maintenance_id, file_path, original_name, uploaded_at
                FROM maintenance_documents
                WHERE maintenance_id IN ({','.join('?' * len(maintenance_ids))})
                ORDER BY datetime(uploaded_at) DESC, document_id DESC
                """,
                maintenance_ids,
            )
    maintenance_lookup = {}
    if "maintenance_id" in maintenance_df.columns and not maintenance_df.empty:
        for _, row in maintenance_df.iterrows():
            if pd.isna(row.get("maintenance_id")):
                continue
            maintenance_lookup[int(row["maintenance_id"])] = row
    if not maintenance_docs.empty:
        for _, doc_row in maintenance_docs.iterrows():
            path = resolve_upload_path(doc_row.get("file_path"))
            if not path or not path.exists():
                continue
            maintenance_id = int(doc_row.get("maintenance_id"))
            record = maintenance_lookup.get(maintenance_id, {})
            reference = clean_text(record.get("do_number")) or f"Maintenance #{maintenance_id}"
            display_name = clean_text(doc_row.get("original_name")) or path.name
            uploaded = pd.to_datetime(doc_row.get("uploaded_at"), errors="coerce")
            uploaded_fmt = uploaded.strftime("%d-%m-%Y %H:%M") if pd.notna(uploaded) else None
            archive_name = "/".join(
                [
                    _sanitize_path_component("maintenance"),
                    f"{_sanitize_path_component(reference)}_{_sanitize_path_component(display_name)}",
                ]
            )
            documents.append(
                {
                    "source": "Maintenance",
                    "reference": reference,
                    "display": display_name,
                    "uploaded": uploaded_fmt,
                    "path": path,
                    "archive_name": archive_name,
                    "key": f"maintenance_{maintenance_id}_{int(doc_row['document_id'])}",
                }
            )

    documents.sort(key=lambda d: (d["source"], d.get("reference") or "", d.get("display") or ""))

    st.markdown("**Documents**")
    if not documents:
        st.info("No documents attached for this customer.")
    else:
        for idx, doc in enumerate(documents, start=1):
            path = doc.get("path")
            if not path or not path.exists():
                continue
            label = f"{doc['source']}: {doc['reference']} – {doc['display']}"
            if doc.get("uploaded"):
                label = f"{label} (uploaded {doc['uploaded']})"
            st.download_button(
                f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                key=f"cust_doc_{doc['key']}_{idx}",
            )
        zip_buffer = bundle_documents_zip(documents)
        if zip_buffer is not None:
            archive_title = _sanitize_path_component(info.get("name") or blank_label)
            st.download_button(
                "⬇️ Download all documents (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"{archive_title}_documents.zip",
                mime="application/zip",
                key="cust_docs_zip",
            )

    pdf_bytes = generate_customer_summary_pdf(
        info.get("name") or blank_label,
        info,
        warr_display,
        service_df,
        maintenance_df,
    )
    st.download_button(
        "⬇️ Download summary (PDF)",
        data=pdf_bytes,
        file_name=f"customer_summary_{clean_text(info.get('name')) or 'customer'}.pdf",
        mime="application/pdf",
    )


def scraps_page(conn):
    st.subheader("🗂️ Scraps (Incomplete Records)")
    st.caption(
        "Rows listed here are missing key details (name, phone, or address). They stay hidden from summaries until completed."
    )
    scraps = df_query(
        conn,
        f"""
        SELECT customer_id as id, name, phone, email, address, city, created_at
        FROM customers
        WHERE {customer_incomplete_clause()}
        ORDER BY datetime(created_at) DESC
        """,
    )
    scraps = fmt_dates(scraps, ["created_at"])
    if scraps.empty:
        st.success("No scraps! All customer rows have the required details.")
        return

    def missing_fields(row):
        missing = []
        for col, label in REQUIRED_CUSTOMER_FIELDS.items():
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                missing.append(label)
        return ", ".join(missing)

    scraps = scraps.assign(missing=scraps.apply(missing_fields, axis=1))
    display_cols = ["id", "name", "phone", "email", "address", "city", "missing", "created_at"]
    st.dataframe(scraps[display_cols])

    st.markdown("### Update scrap record")
    records = scraps.to_dict("records")
    option_keys = [int(r["id"]) for r in records]
    option_labels = {}
    for r in records:
        rid = int(r["id"])
        name_label = clean_text(r.get("name")) or "(no name)"
        missing_label = clean_text(r.get("missing")) or "—"
        option_labels[rid] = f"#{rid} – {name_label} (missing: {missing_label})"
    selected_id = st.selectbox(
        "Choose a record to fix",
        option_keys,
        format_func=lambda k: option_labels[k],
    )
    selected = next(r for r in records if int(r["id"]) == selected_id)

    def existing_value(key):
        return clean_text(selected.get(key)) or ""

    with st.form("scrap_update_form"):
        name = st.text_input("Name", existing_value("name"))
        phone = st.text_input("Phone", existing_value("phone"))
        email = st.text_input("Email", existing_value("email"))
        address = st.text_area("Address", existing_value("address"))
        city = st.text_input("City", existing_value("city"))
        col1, col2 = st.columns(2)
        save = col1.form_submit_button("Save changes", type="primary")
        delete = col2.form_submit_button("Delete scrap")

    if save:
        new_name = clean_text(name)
        new_phone = clean_text(phone)
        new_email = clean_text(email)
        new_address = clean_text(address)
        new_city = clean_text(city)
        old_phone = clean_text(selected.get("phone"))
        conn.execute(
            "UPDATE customers SET name=?, phone=?, email=?, address=?, city=?, dup_flag=0 WHERE customer_id=?",
            (new_name, new_phone, new_email, new_address, new_city, int(selected_id)),
        )
        if old_phone and old_phone != new_phone:
            recalc_customer_duplicate_flag(conn, old_phone)
        if new_phone:
            recalc_customer_duplicate_flag(conn, new_phone)
        conn.commit()
        if new_name and new_phone and new_address:
            st.success("Details saved. This record is now complete and will appear in other pages.")
        else:
            st.info("Details saved, but the record is still incomplete and will remain in Scraps until all required fields are filled.")
        _safe_rerun()

    if delete:
        conn.execute("DELETE FROM customers WHERE customer_id=?", (int(selected_id),))
        conn.commit()
        st.warning("Scrap record deleted.")
        _safe_rerun()

# ---------- Import helpers ----------
def refine_multiline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: [s.strip() for s in str(x).splitlines() if s.strip()] if isinstance(x, str) else [x]
        )
    df = df.explode(list(df.columns)).reset_index(drop=True)
    return df

def normalize_headers(cols):
    norm = []
    for c in cols:
        s = str(c).strip().lower().replace(" ", "_")
        norm.append(s)
    return norm

HEADER_MAP = {
    "date": {"date", "delivery_date", "issue_date", "order_date", "dt", "d_o", "d", "sale_date"},
    "customer_name": {"customer_name", "customer", "company", "company_name", "client", "party", "name"},
    "address": {"address", "addr", "street", "location"},
    "phone": {"phone", "mobile", "contact", "contact_no", "phone_no", "phone_number", "cell", "whatsapp"},
    "product": {"product", "item", "generator", "model", "description"},
    "price": {"price", "amount", "unit_price", "rate", "value", "tk", "bdt"},
    "do_code": {"do_code", "delivery_order", "delivery_order_code", "delivery_order_no", "do", "d_o_code", "do_number"},
}

def map_headers_guess(cols):
    cols_norm = normalize_headers(cols)
    mapping = {k: None for k in HEADER_MAP.keys()}
    for i, cn in enumerate(cols_norm):
        for target, aliases in HEADER_MAP.items():
            if cn in aliases and mapping[target] is None:
                mapping[target] = i
                break
    default_order = ["date", "customer_name", "address", "phone", "product", "price", "do_code"]
    if cols_norm[: len(default_order)] == default_order:
        mapping = {field: idx for idx, field in enumerate(default_order)}
    return mapping

def coerce_excel_date(series):
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if s.isna().mean() > 0.5:
        try:
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().sum() > 0 and (num.dropna().median() > 20000):
                s = pd.to_datetime(num, unit="d", origin="1899-12-30", errors="coerce")
        except Exception:
            pass
    return s

def import_page(conn):
    st.subheader("⬆️ Import from Excel/CSV (append)")
    st.caption("We’ll auto-detect columns; you can override mapping. Dates accept DD-MM-YYYY or Excel serials.")
    f = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"])
    if f is None:
        return
    if f.name.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    st.write("Preview:", df.head())

    guess = map_headers_guess(list(df.columns))
    cols = list(df.columns)
    opts = ["(blank)"] + cols
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, _, _ = st.columns(3)
    sel_date = col1.selectbox(
        "Date", options=opts, index=(guess["date"] + 1) if guess.get("date") is not None else 0
    )
    sel_name = col2.selectbox(
        "Customer name", options=opts, index=(guess["customer_name"] + 1) if guess.get("customer_name") is not None else 0
    )
    sel_addr = col3.selectbox(
        "Address", options=opts, index=(guess["address"] + 1) if guess.get("address") is not None else 0
    )
    sel_phone = col4.selectbox(
        "Phone", options=opts, index=(guess["phone"] + 1) if guess.get("phone") is not None else 0
    )
    sel_prod = col5.selectbox(
        "Product", options=opts, index=(guess["product"] + 1) if guess.get("product") is not None else 0
    )
    sel_price = col6.selectbox(
        "Price", options=opts, index=(guess["price"] + 1) if guess.get("price") is not None else 0
    )
    sel_do = col7.selectbox(
        "Delivery order code", options=opts, index=(guess["do_code"] + 1) if guess.get("do_code") is not None else 0
    )

    def pick(col_name):
        return df[col_name] if col_name != "(blank)" else pd.Series([None] * len(df))

    df_norm = pd.DataFrame(
        {
            "date": pick(sel_date),
            "customer_name": pick(sel_name),
            "address": pick(sel_addr),
            "phone": pick(sel_phone),
            "product": pick(sel_prod),
            "price": pick(sel_price),
            "do_code": pick(sel_do),
        }
    )
    skip_blanks = st.checkbox("Skip blank rows", value=True)
    df_norm = refine_multiline(df_norm)
    df_norm["date"] = coerce_excel_date(df_norm["date"])
    df_norm = df_norm.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    if skip_blanks:
        df_norm = df_norm.dropna(how="all")
    df_norm = df_norm.drop_duplicates().sort_values(by=["date", "customer_name", "phone", "do_code"]).reset_index(drop=True)
    st.markdown("#### Dry-run preview (first 10 rows)")
    st.dataframe(df_norm.head(10))

    if st.button("Append into database"):
        seeded, d_c, d_p = _import_clean6(conn, df_norm, tag="Manual import (mapped)")
        if seeded == 0:
            st.warning("No rows added (rows empty/invalid). Check mapping or file.")
        else:
            st.success(f"Imported {seeded} rows. Duplicates flagged — customers: {d_c}, products: {d_p}.")

def duplicates_page(conn):
    st.subheader("⚠️ Possible Duplicates")
    cust = df_query(conn, "SELECT customer_id as id, name, phone, email, city, dup_flag, created_at FROM customers ORDER BY datetime(created_at) DESC")
    cust = fmt_dates(cust, ["created_at"])
    prod = df_query(conn, "SELECT product_id as id, name, model, serial, unit_price, dup_flag FROM products ORDER BY name ASC")
    warr = df_query(conn, "SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date, w.dup_flag FROM warranties w LEFT JOIN customers c ON c.customer_id = w.customer_id LEFT JOIN products p ON p.product_id = w.product_id ORDER BY date(w.issue_date) DESC")
    warr = fmt_dates(warr, ["issue_date","expiry_date"])

    if not cust.empty:
        cust = cust.assign(duplicate=cust["dup_flag"].apply(lambda x: "🔁 duplicate phone" if int(x)==1 else ""))
        st.markdown("**Customers (duplicate phone)**")
        st.dataframe(cust[cust["dup_flag"]==1])
    if not prod.empty:
        prod = prod.assign(duplicate=prod["dup_flag"].apply(lambda x: "🔁 duplicate name+model" if int(x)==1 else ""))
        st.markdown("**Products (duplicate name+model)**")
        st.dataframe(prod[prod["dup_flag"]==1])
    if not warr.empty:
        warr = warr.assign(duplicate=warr["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        st.markdown("**Warranties (duplicate serial)**")
        st.dataframe(warr[warr["dup_flag"]==1])

def users_admin_page(conn):
    ensure_auth(role="admin")
    st.subheader("👤 Users (Admin)")
    users = df_query(conn, "SELECT user_id as id, username, role, created_at FROM users ORDER BY datetime(created_at) DESC")
    users = users.assign(created_at=pd.to_datetime(users["created_at"], errors="coerce").dt.strftime(DATE_FMT))
    st.dataframe(users)

    with st.expander("Add user"):
        with st.form("add_user"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["staff", "admin"])
            ok = st.form_submit_button("Create")
            if ok and u.strip() and p.strip():
                h = hashlib.sha256(p.encode("utf-8")).hexdigest()
                try:
                    conn.execute("INSERT INTO users (username, pass_hash, role) VALUES (?, ?, ?)", (u.strip(), h, role))
                    conn.commit()
                    st.success("User added")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")

    with st.expander("Reset password / delete"):
        uid = st.number_input("User ID", min_value=1, step=1)
        newp = st.text_input("New password", type="password")
        col1, col2 = st.columns(2)
        if col1.button("Set new password"):
            h = hashlib.sha256(newp.encode("utf-8")).hexdigest()
            conn.execute("UPDATE users SET pass_hash=? WHERE user_id=?", (h, int(uid)))
            conn.commit()
            st.success("Password updated")
        if col2.button("Delete user"):
            conn.execute("DELETE FROM users WHERE user_id=?", (int(uid),))
            conn.commit()
            st.warning("User deleted")

# ---------- Import engine ----------
def _import_clean6(conn, df, tag="Import"):
    """Import cleaned dataframe into database.

    The function is resilient to messy input: it normalizes and sorts the
    dataframe internally so callers can pass raw data without pre-processing.
    """
    # ensure dataframe is normalized even if caller didn't pre-clean
    df = df.copy()
    df = refine_multiline(df)
    if "date" in df.columns:
        df["date"] = coerce_excel_date(df["date"])
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(how="all").drop_duplicates().sort_values(by=["date", "customer_name", "phone"])\
           .reset_index(drop=True)

    cur = conn.cursor()
    seeded = 0
    d_c = d_p = 0
    for _, r in df.iterrows():
        d = r.get("date", pd.NaT)
        cust = r.get("customer_name"); addr = r.get("address")
        phone = r.get("phone"); prod = r.get("product")
        price = r.get("price")
        do_code = r.get("do_code")
        if pd.isna(cust) and pd.isna(phone) and pd.isna(prod):
            continue
        cust = str(cust) if pd.notna(cust) else None
        addr = str(addr) if pd.notna(addr) else None
        phone = str(phone) if pd.notna(phone) else None
        prod = str(prod) if pd.notna(prod) else None
        try:
            price = float(price) if pd.notna(price) else None
        except Exception:
            price = None
        # dup checks
        def exists_phone(phone):
            if not phone or str(phone).lower() == "nan":
                return False
            cur.execute("SELECT 1 FROM customers WHERE phone = ? LIMIT 1", (str(phone),))
            return cur.fetchone() is not None

        dupc = 1 if exists_phone(phone) else 0
        cur.execute(
            "INSERT INTO customers (name, phone, address, dup_flag) VALUES (?, ?, ?, ?)",
            (cust, phone, addr, dupc),
        )
        cid = cur.lastrowid
        if dupc:
            d_c += 1

        name, model = prod, None
        if isinstance(prod, str) and "-" in prod:
            name, model = prod.split("-", 1)
            name, model = name.strip(), model.strip()

        def exists_prod(name, model):
            if not name:
                return False
            cur.execute(
                "SELECT 1 FROM products WHERE name = ? AND IFNULL(model,'') = IFNULL(?, '') LIMIT 1",
                (name, model),
            )
            return cur.fetchone() is not None

        dupp = 1 if exists_prod(name, model) else 0
        cur.execute(
            "INSERT INTO products (name, model, unit_price, dup_flag) VALUES (?, ?, ?, ?)",
            (name, model, price, dupp),
        )
        pid = cur.lastrowid
        if dupp:
            d_p += 1

        # we still record orders (hidden) to keep a timeline if needed
        order_date = d if pd.notna(d) else None
        delivery_date = d if pd.notna(d) else None
        cur.execute(
            "INSERT INTO orders (customer_id, order_date, delivery_date, notes) VALUES (?, ?, ?, ?)",
            (
                cid,
                order_date.strftime("%Y-%m-%d") if order_date is not None else None,
                delivery_date.strftime("%Y-%m-%d") if delivery_date is not None else None,
                f"Imported ({tag})",
            ),
        )
        oid = cur.lastrowid
        cur.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
            (oid, pid, 1, price),
        )

        base = d if pd.notna(d) else pd.Timestamp.now()
        expiry = base + pd.Timedelta(days=365)
        cur.execute(
            "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status, dup_flag) VALUES (?, ?, ?, ?, ?, 'active', 0)",
            (cid, pid, None, base.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d")),
        )

        do_serial = clean_text(do_code)
        if do_serial:
            description = clean_text(prod)
            cur.execute(
                "INSERT OR IGNORE INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    do_serial,
                    cid,
                    oid,
                    description,
                    None,
                    None,
                ),
            )
        seeded += 1
    conn.commit()
    return seeded, d_c, d_p

# ---------- Main ----------
def main():
    init_ui()
    conn = get_conn()
    init_schema(conn)
    login_box(conn)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    with st.sidebar:
        pages = ["Dashboard", "Customers", "Customer Summary", "Scraps", "Warranties", "Import", "Duplicates"]
        if st.session_state.user and st.session_state.user["role"] == "admin":
            pages.append("Users (Admin)")
        pages.extend(["Delivery Orders", "Service", "Maintenance"])
        current_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0
        page = st.radio("Navigate", pages, index=current_index, key="nav_page")
        st.session_state.page = page

    show_expiry_notifications(conn)

    if page == "Dashboard":
        dashboard(conn)
    elif page == "Customers":
        customers_page(conn)
    elif page == "Customer Summary":
        customer_summary_page(conn)
    elif page == "Scraps":
        scraps_page(conn)
    elif page == "Warranties":
        warranties_page(conn)
    elif page == "Import":
        import_page(conn)
    elif page == "Duplicates":
        duplicates_page(conn)
    elif page == "Users (Admin)":
        users_admin_page(conn)
    elif page == "Delivery Orders":
        delivery_orders_page(conn)
    elif page == "Service":
        services_page(conn)
    elif page == "Maintenance":
        maintenance_page(conn)

if _streamlit_runtime_active():
    main()
elif __name__ == "__main__":
    _bootstrap_streamlit_app()
