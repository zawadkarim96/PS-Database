import hashlib
import hmac
import os
import sqlite3
import string
from datetime import datetime, timedelta
from dotenv import load_dotenv
from textwrap import dedent
import pandas as pd


import bcrypt
import streamlit as st

# ---------- Config ----------
load_dotenv()
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "ps_crm.db"))
DATE_FMT = "%d-%m-%Y"

REQUIRED_CUSTOMER_FIELDS = {
    "name": "Name",
    "phone": "Phone",
    "address": "Address",
}


def hash_password(password: str) -> str:
    """Create a salted password hash."""

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def is_bcrypt_hash(value: str) -> bool:
    return isinstance(value, str) and value.startswith("$2")


def is_legacy_hash(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in string.hexdigits for c in value)


def verify_password(conn, user_id: int, password: str, stored_hash: str) -> bool:
    """Verify password for user, migrating legacy SHA-256 hashes when encountered."""

    if not stored_hash:
        return False
    password_bytes = password.encode("utf-8")
    if is_bcrypt_hash(stored_hash):
        try:
            return bcrypt.checkpw(password_bytes, stored_hash.encode("utf-8"))
        except ValueError:
            return False
    if is_legacy_hash(stored_hash):
        legacy_hash = hashlib.sha256(password_bytes).hexdigest()
        if hmac.compare_digest(legacy_hash, stored_hash):
            new_hash = hash_password(password)
            conn.execute("UPDATE users SET pass_hash=? WHERE user_id=?", (new_hash, int(user_id)))
            conn.commit()
            return True
    return False


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
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    # bootstrap admin if empty
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_user = os.getenv("ADMIN_USER", "admin")
        admin_pass = os.getenv("ADMIN_PASS", "admin123")
        h = hash_password(admin_pass)
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
        if not row.empty:
            uid = int(row.iloc[0]["user_id"])
            stored_hash = row.iloc[0]["pass_hash"]
            if verify_password(conn, uid, p, stored_hash):
                st.session_state.user = {"user_id": uid, "username": row.iloc[0]["username"], "role": row.iloc[0]["role"]}
                _safe_rerun()
                return
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

    if st.session_state.user and st.session_state.user["role"] == "admin":
        with open(DB_PATH, "rb") as dbfile:
            st.download_button("⬇️ Download full database", dbfile, file_name=os.path.basename(DB_PATH))

    st.markdown("---")
    st.subheader("🛠️ Warranties Expiring Soon")
    colA, colB = st.columns(2)

    soon3 = df_query(conn, """
        SELECT c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.status='active'
          AND date(w.expiry_date) BETWEEN date('now') AND date('now', '+3 day')
        ORDER BY date(w.expiry_date) ASC
    """)
    soon3 = fmt_dates(soon3, ["issue_date","expiry_date"])
    with colA:
        st.caption("Next **3** days")
        st.dataframe(soon3)

    soon60 = df_query(conn, """
        SELECT c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.status='active'
          AND date(w.expiry_date) BETWEEN date('now') AND date('now', '+60 day')
        ORDER BY date(w.expiry_date) ASC
    """)
    soon60 = fmt_dates(soon60, ["issue_date","expiry_date"])
    with colB:
        st.caption("Next **60** days")
        st.dataframe(soon60)

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
        active = active.assign(duplicate=active["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
    st.markdown("**Active Warranties**")
    st.dataframe(active)

    expired = df_query(conn, base.format(date_cond="date(w.expiry_date) < date('now')", order="DESC"), (q,q,q,q,q))
    expired = fmt_dates(expired, ["issue_date","expiry_date"])
    if "dup_flag" in expired.columns:
        expired = expired.assign(duplicate=expired["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
    st.markdown("**Expired Warranties**")
    st.dataframe(expired)

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
                h = hash_password(p)
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
            if newp.strip():
                h = hash_password(newp)
                conn.execute("UPDATE users SET pass_hash=? WHERE user_id=?", (h, int(uid)))
                conn.commit()
                st.success("Password updated")
            else:
                st.error("Password cannot be blank")
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

if _streamlit_runtime_active():
    main()
elif __name__ == "__main__":
    _bootstrap_streamlit_app()
