import io
import os
import sqlite3
import hashlib
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
    for path in (UPLOADS_DIR, DELIVERY_ORDER_DIR):
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


def fetch_customer_choices(conn):
    df = df_query(
        conn,
        f"SELECT customer_id, name FROM customers WHERE {customer_complete_clause()} ORDER BY name COLLATE NOCASE",
    )
    options = [None]
    labels = {None: "-- Select customer --"}
    for _, row in df.iterrows():
        cid = int(row["customer_id"])
        name = clean_text(row.get("name")) or f"Customer #{cid}"
        options.append(cid)
        labels[cid] = name
    return options, labels


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
    for col in ("product", "model", "serial"):
        if col in work.columns:
            work.drop(columns=[col], inplace=True)
    rename_map = {
        "customer": "Customer",
        "issue_date": "Issue date",
        "expiry_date": "Expiry date",
        "status": "Status",
    }
    work.rename(columns={k: v for k, v in rename_map.items() if k in work.columns}, inplace=True)
    return work


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

    today_expired = df_query(
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
    if not today_expired.empty:
        notice = collapse_warranty_rows(today_expired)
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
    st.subheader("🛠️ Warranties Expiring Soon")
    colA, colB = st.columns(2)

    soon3 = fetch_warranty_window(conn, 0, 3)
    soon3 = collapse_warranty_rows(soon3)
    with colA:
        st.caption("Next **3** days")
        st.dataframe(soon3, use_container_width=True)

    soon60 = fetch_warranty_window(conn, 0, 60)
    soon60 = collapse_warranty_rows(soon60)
    with colB:
        st.caption("Next **60** days")
        st.dataframe(soon60, use_container_width=True)

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
    customer_options, customer_labels = fetch_customer_choices(conn)
    orders_df = df_query(conn, "SELECT order_id, order_date FROM orders ORDER BY datetime(order_date) DESC")
    order_options = [None]
    order_labels = {None: "-- Not linked --"}
    for _, row in orders_df.iterrows():
        oid = int(row["order_id"])
        date_label = clean_text(row.get("order_date"))
        if date_label:
            dt = pd.to_datetime(date_label, errors="coerce")
            if pd.notna(dt):
                date_label = dt.strftime(DATE_FMT)
        label = f"Order #{oid}" + (f" – {date_label}" if date_label else "")
        order_options.append(oid)
        order_labels[oid] = label

    with st.form("delivery_order_form"):
        do_number = st.text_input("Delivery Order Serial *")
        selected_customer_index = 0
        selected_customer = st.selectbox(
            "Customer",
            options=customer_options,
            index=selected_customer_index,
            format_func=lambda cid: customer_labels.get(cid, str(cid)),
        )
        selected_order = st.selectbox(
            "Related Order",
            options=order_options,
            format_func=lambda oid: order_labels.get(oid, str(oid)),
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
                        int(selected_order) if selected_order else None,
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
                "order_id": "Order",
            },
            inplace=True,
        )
        st.markdown("### Recorded Delivery Orders")
        st.dataframe(display.drop(columns=["file_path"], errors="ignore"), use_container_width=True)

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
    customer_options, customer_labels = fetch_customer_choices(conn)
    do_df = df_query(
        conn,
        "SELECT do_number, customer_id, description FROM delivery_orders ORDER BY datetime(created_at) DESC",
    )
    do_options = [None]
    do_labels = {None: "-- Select delivery order --"}
    do_customer_map = {}
    for _, row in do_df.iterrows():
        do_num = clean_text(row.get("do_number"))
        if not do_num:
            continue
        cust_id = int(row["customer_id"]) if not pd.isna(row.get("customer_id")) else None
        summary = clean_text(row.get("description"))
        label = do_num
        if summary:
            label = f"{do_num} – {summary[:40]}" + ("…" if len(summary) > 40 else "")
        do_options.append(do_num)
        do_labels[do_num] = label
        do_customer_map[do_num] = cust_id

    with st.form("service_form"):
        selected_do = st.selectbox(
            "Delivery Order *",
            options=do_options,
            format_func=lambda do: do_labels.get(do, str(do)),
        )
        default_customer = do_customer_map.get(selected_do)
        try:
            customer_index = customer_options.index(default_customer) if default_customer in customer_options else 0
        except ValueError:
            customer_index = 0
        selected_customer = st.selectbox(
            "Customer",
            options=customer_options,
            index=customer_index,
            format_func=lambda cid: customer_labels.get(cid, str(cid)),
        )
        service_date = st.date_input("Service date", value=datetime.now().date())
        description = st.text_area("Service description")
        remarks = st.text_area("Remarks / updates")
        submit = st.form_submit_button("Log service", type="primary")

    if submit:
        if not selected_do:
            st.error("Delivery Order is required for service records.")
        else:
            conn.execute(
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
            conn.commit()
            st.success("Service record saved.")
            _safe_rerun()

    service_df = df_query(
        conn,
        """
        SELECT s.service_id, s.do_number, s.service_date, s.description, s.remarks, s.updated_at, c.name AS customer
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
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
            }
        )
        st.markdown("### Service history")
        st.dataframe(display.drop(columns=["updated_at"], errors="ignore"), use_container_width=True)

        records = service_df.to_dict("records")
        st.markdown("#### Update remarks")
        options = [int(r["service_id"]) for r in records]
        labels = {
            int(r["service_id"]): f"#{int(r['service_id'])} – {r.get('DO Serial', r.get('do_number'))}"
            for r in display.to_dict("records")
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
    else:
        st.info("No service records yet. Log one using the form above.")


def maintenance_page(conn):
    st.subheader("🔧 Maintenance Records")
    customer_options, customer_labels = fetch_customer_choices(conn)
    do_df = df_query(
        conn,
        "SELECT do_number, customer_id, description FROM delivery_orders ORDER BY datetime(created_at) DESC",
    )
    do_options = [None]
    do_labels = {None: "-- Select delivery order --"}
    do_customer_map = {}
    for _, row in do_df.iterrows():
        do_num = clean_text(row.get("do_number"))
        if not do_num:
            continue
        cust_id = int(row["customer_id"]) if not pd.isna(row.get("customer_id")) else None
        summary = clean_text(row.get("description"))
        label = do_num
        if summary:
            label = f"{do_num} – {summary[:40]}" + ("…" if len(summary) > 40 else "")
        do_options.append(do_num)
        do_labels[do_num] = label
        do_customer_map[do_num] = cust_id

    with st.form("maintenance_form"):
        selected_do = st.selectbox(
            "Delivery Order *",
            options=do_options,
            format_func=lambda do: do_labels.get(do, str(do)),
        )
        default_customer = do_customer_map.get(selected_do)
        try:
            customer_index = customer_options.index(default_customer) if default_customer in customer_options else 0
        except ValueError:
            customer_index = 0
        selected_customer = st.selectbox(
            "Customer",
            options=customer_options,
            index=customer_index,
            format_func=lambda cid: customer_labels.get(cid, str(cid)),
        )
        maintenance_date = st.date_input("Maintenance date", value=datetime.now().date())
        description = st.text_area("Maintenance description")
        remarks = st.text_area("Remarks / updates")
        submit = st.form_submit_button("Log maintenance", type="primary")

    if submit:
        if not selected_do:
            st.error("Delivery Order is required for maintenance records.")
        else:
            conn.execute(
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
            conn.commit()
            st.success("Maintenance record saved.")
            _safe_rerun()

    maintenance_df = df_query(
        conn,
        """
        SELECT m.maintenance_id, m.do_number, m.maintenance_date, m.description, m.remarks, m.updated_at, c.name AS customer
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
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
            }
        )
        st.markdown("### Maintenance history")
        st.dataframe(display.drop(columns=["updated_at"], errors="ignore"), use_container_width=True)

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
    st.write("**Warranties**")
    placeholders = ",".join("?" * len(ids))
    warr = df_query(
        conn,
        f"""
        SELECT w.warranty_id as id, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date, w.status, w.dup_flag
        FROM warranties w
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.customer_id IN ({placeholders})
        ORDER BY date(w.expiry_date) DESC
        """,
        ids,
    )
    warr = fmt_dates(warr, ["issue_date","expiry_date"])
    if "dup_flag" in warr.columns:
        warr = warr.assign(duplicate=warr["dup_flag"].apply(lambda x: "🔁 duplicate" if int(x) == 1 else ""))
    st.dataframe(warr)


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
    "date": {"date","delivery_date","issue_date","order_date","dt","d_o","d","sale_date"},
    "customer_name": {"customer_name","customer","company","company_name","client","party","name"},
    "address": {"address","addr","street","location"},
    "phone": {"phone","mobile","contact","contact_no","phone_no","phone_number","cell","whatsapp"},
    "product": {"product","item","generator","model","description"},
    "price": {"price","amount","unit_price","rate","value","tk","bdt"}
}

def map_headers_guess(cols):
    cols_norm = normalize_headers(cols)
    mapping = {"date":None,"customer_name":None,"address":None,"phone":None,"product":None,"price":None}
    for i,cn in enumerate(cols_norm):
        for target, aliases in HEADER_MAP.items():
            if cn in aliases and mapping[target] is None:
                mapping[target] = i
                break
    if cols_norm[:6] == ["date","customer_name","address","phone","product","price"]:
        mapping = {k:i for i,k in enumerate(["date","customer_name","address","phone","product","price"])}
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
    col1, col2, col3 = st.columns(3); col4, col5, col6 = st.columns(3)
    sel_date = col1.selectbox(
        "Date", options=opts, index=(guess["date"] + 1) if guess["date"] is not None else 0
    )
    sel_name = col2.selectbox(
        "Customer name", options=opts, index=(guess["customer_name"] + 1) if guess["customer_name"] is not None else 0
    )
    sel_addr = col3.selectbox(
        "Address", options=opts, index=(guess["address"] + 1) if guess["address"] is not None else 0
    )
    sel_phone = col4.selectbox(
        "Phone", options=opts, index=(guess["phone"] + 1) if guess["phone"] is not None else 0
    )
    sel_prod = col5.selectbox(
        "Product", options=opts, index=(guess["product"] + 1) if guess["product"] is not None else 0
    )
    sel_price = col6.selectbox(
        "Price", options=opts, index=(guess["price"] + 1) if guess["price"] is not None else 0
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
        }
    )
    skip_blanks = st.checkbox("Skip blank rows", value=True)
    df_norm = refine_multiline(df_norm)
    df_norm["date"] = coerce_excel_date(df_norm["date"])
    df_norm = df_norm.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    if skip_blanks:
        df_norm = df_norm.dropna(how="all")
    df_norm = df_norm.drop_duplicates().sort_values(by=["date", "customer_name", "phone"]).reset_index(drop=True)
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
        seeded += 1
    conn.commit()
    return seeded, d_c, d_p

# ---------- Main ----------
def main():
    init_ui()
    conn = get_conn()
    init_schema(conn)
    login_box(conn)

    with st.sidebar:
        pages = ["Dashboard", "Customers", "Customer Summary", "Scraps", "Warranties", "Import", "Duplicates"]
        if st.session_state.user and st.session_state.user["role"] == "admin":
            pages.append("Users (Admin)")
        pages.extend(["Delivery Orders", "Service", "Maintenance"])
        page = st.radio("Navigate", pages)

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
