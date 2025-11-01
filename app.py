import io
import os
import re
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

try:
    from storage_paths import get_storage_dir
except ModuleNotFoundError:  # pragma: no cover - defensive for bundled test imports
    import importlib.util

    _storage_module_path = Path(__file__).resolve().parent / "storage_paths.py"
    spec = importlib.util.spec_from_file_location("storage_paths", _storage_module_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:  # pragma: no cover - should not happen
        raise
    loader.exec_module(module)
    get_storage_dir = module.get_storage_dir

# ---------- Config ----------
load_dotenv()
DEFAULT_BASE_DIR = get_storage_dir()
BASE_DIR = Path(os.getenv("APP_STORAGE_DIR", DEFAULT_BASE_DIR))
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "ps_crm.db"))
DATE_FMT = "%d-%m-%Y"

UPLOADS_DIR = BASE_DIR / "uploads"
DELIVERY_ORDER_DIR = UPLOADS_DIR / "delivery_orders"
SERVICE_DOCS_DIR = UPLOADS_DIR / "service_documents"
MAINTENANCE_DOCS_DIR = UPLOADS_DIR / "maintenance_documents"
CUSTOMER_DOCS_DIR = UPLOADS_DIR / "customer_documents"

REQUIRED_CUSTOMER_FIELDS = {
    "name": "Name",
    "phone": "Phone",
    "address": "Address",
}

SERVICE_STATUS_OPTIONS = ["In progress", "Completed", "Haven't started"]
DEFAULT_SERVICE_STATUS = SERVICE_STATUS_OPTIONS[0]


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
    purchase_date TEXT,
    product_info TEXT,
    delivery_order_code TEXT,
    sales_person TEXT,
    attachment_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    dup_flag INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    model TEXT,
    serial TEXT,
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
    service_start_date TEXT,
    service_end_date TEXT,
    description TEXT,
    status TEXT DEFAULT 'In progress',
    remarks TEXT,
    service_product_info TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(do_number) REFERENCES delivery_orders(do_number) ON DELETE SET NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS maintenance_records (
    maintenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    do_number TEXT,
    customer_id INTEGER,
    maintenance_date TEXT,
    maintenance_start_date TEXT,
    maintenance_end_date TEXT,
    description TEXT,
    status TEXT DEFAULT 'In progress',
    remarks TEXT,
    maintenance_product_info TEXT,
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
CREATE TABLE IF NOT EXISTS import_history (
    import_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product_id INTEGER,
    order_id INTEGER,
    order_item_id INTEGER,
    warranty_id INTEGER,
    do_number TEXT,
    import_tag TEXT,
    imported_at TEXT DEFAULT (datetime('now')),
    original_date TEXT,
    customer_name TEXT,
    address TEXT,
    phone TEXT,
    product_label TEXT,
    notes TEXT,
    deleted_at TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE SET NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE SET NULL,
    FOREIGN KEY(order_item_id) REFERENCES order_items(order_item_id) ON DELETE SET NULL,
    FOREIGN KEY(warranty_id) REFERENCES warranties(warranty_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS needs (
    need_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product TEXT,
    unit TEXT,
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
    ensure_schema_upgrades(conn)
    conn.commit()
    # bootstrap admin if empty
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_user = os.getenv("ADMIN_USER", "admin")
        admin_pass = os.getenv("ADMIN_PASS", "admin123")
        h = hashlib.sha256(admin_pass.encode("utf-8")).hexdigest()
        conn.execute("INSERT INTO users (username, pass_hash, role) VALUES (?, ?, 'admin')", (admin_user, h))
        conn.commit()


def ensure_schema_upgrades(conn):
    def has_column(table: str, column: str) -> bool:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(str(row[1]) == column for row in cur.fetchall())

    def add_column(table: str, column: str, definition: str) -> None:
        if not has_column(table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    add_column("customers", "purchase_date", "TEXT")
    add_column("customers", "product_info", "TEXT")
    add_column("customers", "delivery_order_code", "TEXT")
    add_column("customers", "attachment_path", "TEXT")
    add_column("customers", "sales_person", "TEXT")
    add_column("services", "status", "TEXT DEFAULT 'In progress'")
    add_column("services", "service_start_date", "TEXT")
    add_column("services", "service_end_date", "TEXT")
    add_column("services", "service_product_info", "TEXT")
    add_column("maintenance_records", "status", "TEXT DEFAULT 'In progress'")
    add_column("maintenance_records", "maintenance_start_date", "TEXT")
    add_column("maintenance_records", "maintenance_end_date", "TEXT")
    add_column("maintenance_records", "maintenance_product_info", "TEXT")

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


def normalize_product_entries(
    entries: Iterable[dict[str, object]]
) -> tuple[list[dict[str, object]], list[str]]:
    cleaned: list[dict[str, object]] = []
    labels: list[str] = []
    for entry in entries:
        name_clean = clean_text(entry.get("name")) if isinstance(entry, dict) else None
        model_clean = clean_text(entry.get("model")) if isinstance(entry, dict) else None
        serial_clean = clean_text(entry.get("serial")) if isinstance(entry, dict) else None
        quantity_raw = entry.get("quantity") if isinstance(entry, dict) else None
        try:
            qty_val = int(quantity_raw or 1)
        except Exception:
            qty_val = 1
        qty_val = max(qty_val, 1)
        if not any([name_clean, model_clean, serial_clean]):
            continue
        cleaned.append(
            {
                "name": name_clean,
                "model": model_clean,
                "serial": serial_clean,
                "quantity": qty_val,
            }
        )
        label_parts = [val for val in [name_clean, model_clean] if val]
        label = " - ".join(label_parts)
        if qty_val > 1:
            label = f"{label} ×{qty_val}" if label else f"×{qty_val}"
        if serial_clean:
            label = f"{label} (Serial: {serial_clean})" if label else f"Serial: {serial_clean}"
        if label:
            labels.append(label)
    return cleaned, labels


def format_period_label(
    start: Optional[str], end: Optional[str], *, joiner: str = " → "
) -> Optional[str]:
    start_clean = clean_text(start)
    end_clean = clean_text(end)
    if not start_clean and not end_clean:
        return None
    if start_clean and end_clean:
        if start_clean == end_clean:
            return start_clean
        return f"{start_clean}{joiner}{end_clean}"
    return start_clean or end_clean


def status_input_widget(prefix: str, default_status: Optional[str] = None) -> str:
    lookup = {opt.lower(): opt for opt in SERVICE_STATUS_OPTIONS}
    default_choice = DEFAULT_SERVICE_STATUS
    custom_default = "Haven't started"
    default_clean = clean_text(default_status)
    if default_clean:
        normalized = default_clean.lower()
        if normalized in lookup and lookup[normalized] != "Haven't started":
            default_choice = lookup[normalized]
        elif normalized == "haven't started":
            default_choice = lookup[normalized]
            custom_default = lookup[normalized]
        else:
            default_choice = "Haven't started"
            custom_default = default_clean

    choice = st.selectbox(
        "Status",
        SERVICE_STATUS_OPTIONS,
        index=SERVICE_STATUS_OPTIONS.index(default_choice),
        key=f"{prefix}_status_choice",
    )
    if choice == "Haven't started":
        custom_value = st.text_input(
            "Custom status label",
            value=custom_default or "Haven't started",
            key=f"{prefix}_status_custom",
            help="Customize the saved status when a record hasn't started yet.",
        )
        return clean_text(custom_value) or "Haven't started"
    return choice


def link_delivery_order_to_customer(
    conn: sqlite3.Connection, do_number: Optional[str], customer_id: Optional[int]
) -> None:
    do_serial = clean_text(do_number)
    if not do_serial:
        return
    cur = conn.cursor()
    row = cur.execute(
        "SELECT customer_id FROM delivery_orders WHERE do_number = ?",
        (do_serial,),
    ).fetchone()
    if row is None:
        if customer_id is not None:
            cur.execute(
                "UPDATE customers SET delivery_order_code = ? WHERE customer_id = ?",
                (do_serial, int(customer_id)),
            )
        return
    previous_customer = int(row[0]) if row and row[0] is not None else None
    if customer_id is not None:
        cur.execute(
            "UPDATE delivery_orders SET customer_id = ? WHERE do_number = ?",
            (int(customer_id), do_serial),
        )
        cur.execute(
            "UPDATE customers SET delivery_order_code = ? WHERE customer_id = ?",
            (do_serial, int(customer_id)),
        )
    else:
        cur.execute(
            "UPDATE delivery_orders SET customer_id = NULL WHERE do_number = ?",
            (do_serial,),
        )
    if previous_customer and previous_customer != (int(customer_id) if customer_id is not None else None):
        cur.execute(
            "UPDATE customers SET delivery_order_code = NULL WHERE customer_id = ? AND delivery_order_code = ?",
            (previous_customer, do_serial),
        )


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
    for path in (
        UPLOADS_DIR,
        DELIVERY_ORDER_DIR,
        SERVICE_DOCS_DIR,
        MAINTENANCE_DOCS_DIR,
        CUSTOMER_DOCS_DIR,
    ):
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


def merge_customer_records(conn, customer_ids) -> bool:
    ids = []
    for cid in customer_ids:
        cid_int = int_or_none(cid)
        if cid_int is not None and cid_int not in ids:
            ids.append(cid_int)
    if len(ids) < 2:
        return False

    placeholders = ",".join(["?"] * len(ids))
    query = dedent(
        f"""
        SELECT customer_id, name, phone, address, purchase_date, product_info, delivery_order_code, sales_person, created_at
        FROM customers
        WHERE customer_id IN ({placeholders})
        """
    )
    df = df_query(conn, query, params=tuple(ids))
    if df.empty:
        return False

    df["created_at_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df.sort_values(by=["created_at_dt", "customer_id"], inplace=True, na_position="last")
    base_row = df.iloc[0]
    base_id = int(base_row.get("customer_id"))
    other_ids = []
    for row in df.get("customer_id", pd.Series(dtype=object)).tolist():
        rid = int_or_none(row)
        if rid is not None and rid != base_id and rid not in other_ids:
            other_ids.append(rid)
    if not other_ids:
        return False

    name_values = [clean_text(v) for v in df.get("name", pd.Series(dtype=object)).tolist()]
    name_values = [v for v in name_values if v]
    address_values = [clean_text(v) for v in df.get("address", pd.Series(dtype=object)).tolist()]
    address_values = [v for v in address_values if v]
    phone_values = [clean_text(v) for v in df.get("phone", pd.Series(dtype=object)).tolist()]
    phone_values = [v for v in phone_values if v]
    phones_to_recalc: set[str] = set(phone_values)

    base_name = clean_text(base_row.get("name")) or (name_values[0] if name_values else None)
    base_address = clean_text(base_row.get("address")) or (address_values[0] if address_values else None)
    base_phone = clean_text(base_row.get("phone")) or (phone_values[0] if phone_values else None)

    do_codes = []
    product_lines = []
    fallback_products = []
    purchase_dates = []
    sales_people = []

    for record in df.to_dict("records"):
        date_raw = clean_text(record.get("purchase_date"))
        product_raw = clean_text(record.get("product_info"))
        do_raw = clean_text(record.get("delivery_order_code"))
        sales_raw = clean_text(record.get("sales_person"))
        if do_raw:
            do_codes.append(do_raw)
        if product_raw:
            fallback_products.append(product_raw)
        dt = parse_date_value(record.get("purchase_date"))
        if dt is not None:
            purchase_dates.append(dt)
            date_label = dt.strftime(DATE_FMT)
        else:
            date_label = date_raw
        if date_label and product_raw:
            product_lines.append(f"{date_label} – {product_raw}")
        elif product_raw:
            product_lines.append(product_raw)
        elif date_label:
            product_lines.append(date_label)
        if sales_raw:
            sales_people.append(sales_raw)

    earliest_purchase = min(purchase_dates).strftime("%Y-%m-%d") if purchase_dates else None
    combined_products = dedupe_join(product_lines or fallback_products)
    combined_do_codes = dedupe_join(do_codes)
    combined_sales = dedupe_join(sales_people)

    conn.execute(
        """
        UPDATE customers
        SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, dup_flag=0
        WHERE customer_id=?
        """,
        (
            base_name,
            base_phone,
            base_address,
            earliest_purchase,
            clean_text(combined_products),
            clean_text(combined_do_codes),
            clean_text(combined_sales),
            base_id,
        ),
    )

    related_tables = (
        "orders",
        "warranties",
        "delivery_orders",
        "services",
        "maintenance_records",
        "needs",
    )
    for cid in other_ids:
        for table in related_tables:
            conn.execute(f"UPDATE {table} SET customer_id=? WHERE customer_id=?", (base_id, cid))
        conn.execute("UPDATE import_history SET customer_id=? WHERE customer_id=?", (base_id, cid))
        conn.execute("DELETE FROM customers WHERE customer_id=?", (cid,))

    if base_phone:
        phones_to_recalc.add(base_phone)
    if phones_to_recalc:
        for phone in phones_to_recalc:
            recalc_customer_duplicate_flag(conn, phone)
    conn.commit()
    return True


def delete_customer_record(conn, customer_id: int) -> None:
    """Delete a customer and related records, recalculating duplicate flags."""

    try:
        cid = int(customer_id)
    except (TypeError, ValueError):
        return

    cur = conn.execute(
        "SELECT phone, delivery_order_code, attachment_path FROM customers WHERE customer_id=?",
        (cid,),
    )
    row = cur.fetchone()
    if not row:
        return

    phone_val = clean_text(row[0])
    do_code = clean_text(row[1])
    attachment_path = row[2]

    conn.execute("DELETE FROM customers WHERE customer_id=?", (cid,))
    if do_code:
        conn.execute(
            "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
            (do_code, cid),
        )
    conn.execute(
        "UPDATE import_history SET deleted_at = datetime('now') WHERE customer_id=? AND deleted_at IS NULL",
        (cid,),
    )
    conn.commit()

    if phone_val:
        recalc_customer_duplicate_flag(conn, phone_val)
        conn.commit()

    if attachment_path:
        path = resolve_upload_path(attachment_path)
        if path and path.exists():
            try:
                path.unlink()
            except Exception:
                pass


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


def _build_customers_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT customer_id,
               name,
               phone,
               email,
               address,
               purchase_date,
               product_info,
               delivery_order_code,
               sales_person,
               created_at
        FROM customers
        ORDER BY datetime(created_at) DESC, customer_id DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["purchase_date", "created_at"])
    return df.rename(
        columns={
            "customer_id": "Customer ID",
            "name": "Customer",
            "phone": "Phone",
            "email": "Email",
            "address": "Address",
            "purchase_date": "Purchase date",
            "product_info": "Product info",
            "delivery_order_code": "Delivery order",
            "sales_person": "Sales person",
            "created_at": "Created at",
        }
    )


def _build_delivery_orders_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT d.do_number,
               COALESCE(c.name, '(unknown)') AS customer,
               d.description,
               d.sales_person,
               d.created_at
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC, d.do_number DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["created_at"])
    return df.rename(
        columns={
            "do_number": "DO number",
            "customer": "Customer",
            "description": "Description",
            "sales_person": "Sales person",
            "created_at": "Created at",
        }
    )


def _build_warranties_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT w.warranty_id,
               COALESCE(c.name, '(unknown)') AS customer,
               COALESCE(p.name, '') AS product,
               p.model,
               w.serial,
               w.issue_date,
               w.expiry_date,
               w.status
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        ORDER BY date(w.expiry_date) ASC, w.warranty_id ASC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["issue_date", "expiry_date"])
    if "status" in df.columns:
        df["status"] = df["status"].fillna("Active").apply(lambda x: str(x).title())
    return df.rename(
        columns={
            "warranty_id": "Warranty ID",
            "customer": "Customer",
            "product": "Product",
            "model": "Model",
            "serial": "Serial",
            "issue_date": "Issue date",
            "expiry_date": "Expiry date",
            "status": "Status",
        }
    )


def _build_services_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT s.service_id,
               s.do_number,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               s.service_date,
               s.service_start_date,
               s.service_end_date,
               s.service_product_info,
               s.description,
               s.status,
               s.remarks,
               s.updated_at
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        ORDER BY datetime(s.service_date) DESC, s.service_id DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["service_date", "service_start_date", "service_end_date", "updated_at"])
    if "status" in df.columns:
        df["status"] = df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
    return df.rename(
        columns={
            "service_id": "Service ID",
            "do_number": "DO number",
            "customer": "Customer",
            "service_date": "Service date",
            "service_start_date": "Service start date",
            "service_end_date": "Service end date",
            "service_product_info": "Products sold",
            "description": "Description",
            "status": "Status",
            "remarks": "Remarks",
            "updated_at": "Updated at",
        }
    )


def _build_maintenance_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT m.maintenance_id,
               m.do_number,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               m.maintenance_date,
               m.maintenance_start_date,
               m.maintenance_end_date,
               m.maintenance_product_info,
               m.description,
               m.status,
               m.remarks,
               m.updated_at
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        ORDER BY datetime(m.maintenance_date) DESC, m.maintenance_id DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["maintenance_date", "maintenance_start_date", "maintenance_end_date", "updated_at"])
    if "status" in df.columns:
        df["status"] = df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
    return df.rename(
        columns={
            "maintenance_id": "Maintenance ID",
            "do_number": "DO number",
            "customer": "Customer",
            "maintenance_date": "Maintenance date",
            "maintenance_start_date": "Maintenance start date",
            "maintenance_end_date": "Maintenance end date",
            "maintenance_product_info": "Products sold",
            "description": "Description",
            "status": "Status",
            "remarks": "Remarks",
            "updated_at": "Updated at",
        }
    )


def _build_master_sheet(sheets: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = [
        {
            "Sheet": "Export generated at",
            "Details": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
    for sheet_name, df in sheets:
        count = len(df.index) if df is not None else 0
        label = "record" if count == 1 else "records"
        rows.append({"Sheet": sheet_name, "Details": f"{count} {label}"})
    return pd.DataFrame(rows, columns=["Sheet", "Details"])


def export_database_to_excel(conn) -> bytes:
    sheet_builders = [
        ("Customers", _build_customers_export),
        ("Delivery orders", _build_delivery_orders_export),
        ("Warranties", _build_warranties_export),
        ("Services", _build_services_export),
        ("Maintenance", _build_maintenance_export),
    ]

    sheet_data: list[tuple[str, pd.DataFrame]] = []
    for name, builder in sheet_builders:
        df = builder(conn)
        sheet_data.append((name, df))

    master_df = _build_master_sheet(sheet_data)
    ordered_sheets = [("Master", master_df)] + sheet_data

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in ordered_sheets[:6]:
            safe_name = sheet_name[:31] if sheet_name else "Sheet"
            if not safe_name:
                safe_name = "Sheet"
            if df is None or df.empty:
                df_to_write = pd.DataFrame()
            else:
                df_to_write = df
            df_to_write.to_excel(writer, sheet_name=safe_name, index=False)
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
            f"Address: {clean_text(info.get('address')) or '-'}",
            f"Purchase: {clean_text(info.get('purchase_dates')) or '-'}",
            f"Product: {clean_text(info.get('products')) or '-'}",
            f"Delivery order: {clean_text(info.get('do_codes')) or '-'}",
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

    address_env = os.getenv("HOST") or os.getenv("BIND_ADDRESS")
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
    cur = conn.execute(
        "SELECT customer_id, purchase_date FROM customers WHERE phone = ?",
        (str(phone).strip(),),
    )
    rows = cur.fetchall()
    if not rows:
        return

    grouped: dict[Optional[str], list[int]] = {}
    for cid, purchase_date in rows:
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        key = clean_text(purchase_date) or None
        grouped.setdefault(key, []).append(cid_int)

    updates: list[tuple[int, int]] = []
    for cid_list in grouped.values():
        dup_flag = 1 if len(cid_list) > 1 else 0
        updates.extend((dup_flag, cid) for cid in cid_list)

    if updates:
        conn.executemany(
            "UPDATE customers SET dup_flag=? WHERE customer_id=?",
            updates,
        )


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
    st.markdown(
        "<div style='text-align: right; font-size: 0.6rem; color: #888;'>by ZAD</div>",
        unsafe_allow_html=True,
    )
    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"

    if "show_today_expired" not in st.session_state:
        st.session_state.show_today_expired = False

    if is_admin:
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
                        "SELECT COUNT(*) c FROM warranties WHERE status='active' AND date(expiry_date) >= date('now')",
                    ).iloc[0]["c"]
                ),
            )
        with col4:
            expired_count = int(
                df_query(
                    conn,
                    "SELECT COUNT(*) c FROM warranties WHERE status='active' AND date(expiry_date) < date('now')",
                ).iloc[0]["c"]
            )
            st.metric("Expired", expired_count)
    else:
        st.info("Staff view: focus on upcoming activities below. Metrics are available to admins only.")

    month_expired = int(
        df_query(
            conn,
            """
            SELECT COUNT(*) c
            FROM warranties
            WHERE status='active'
              AND date(expiry_date) < date('now')
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
        WHERE w.status='active' AND date(w.expiry_date) = date('now')
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
        toggle_label = "Show list" if not st.session_state.get("show_today_expired") else "Hide list"
        if st.button(toggle_label, key="toggle_expired_today"):
            st.session_state.show_today_expired = not st.session_state.get("show_today_expired")
            show_today_expired = st.session_state.show_today_expired
        else:
            show_today_expired = st.session_state.get("show_today_expired")

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

    show_today_expired = st.session_state.get("show_today_expired")
    if show_today_expired:
        if today_expired_df.empty:
            st.info("No warranties expire today.")
        else:
            today_detail = fmt_dates(today_expired_df, ["issue_date", "expiry_date"])
            today_table = format_warranty_table(today_detail)
            st.markdown("#### Warranties expiring today")
            st.dataframe(today_table, use_container_width=True)

    if is_admin:
        if "show_deleted_panel" not in st.session_state:
            st.session_state.show_deleted_panel = False

        excel_bytes = export_database_to_excel(conn)
        admin_action_cols = st.columns([0.78, 0.22])
        with admin_action_cols[0]:
            st.download_button(
                "⬇️ Download full database (Excel)",
                excel_bytes,
                file_name="ps_crm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        toggle_label = (
            "🗑️ Deleted data"
            if not st.session_state.get("show_deleted_panel")
            else "Hide deleted data"
        )
        with admin_action_cols[1]:
            if st.button(
                toggle_label,
                key="toggle_deleted_panel",
                help="Admins can review deleted import records here.",
            ):
                st.session_state.show_deleted_panel = not st.session_state.get(
                    "show_deleted_panel", False
                )

        if st.session_state.get("show_deleted_panel"):
            deleted_df = df_query(
                conn,
                """
                SELECT import_id, imported_at, customer_name, phone, product_label, original_date, do_number, deleted_at
                FROM import_history
                WHERE deleted_at IS NOT NULL
                ORDER BY datetime(deleted_at) DESC
                """,
            )

            if deleted_df.empty:
                st.info("No deleted import entries found.")
            else:
                formatted_deleted = fmt_dates(
                    deleted_df,
                    ["imported_at", "original_date", "deleted_at"],
                )
                deleted_bytes = io.BytesIO()
                with pd.ExcelWriter(deleted_bytes, engine="openpyxl") as writer:
                    formatted_deleted.to_excel(
                        writer, index=False, sheet_name="deleted_imports"
                    )
                deleted_bytes.seek(0)

                st.markdown("#### Deleted import history")
                st.caption(
                    "Only administrators can access this view. Download the Excel file for a full audit trail."
                )
                st.download_button(
                    "Download deleted imports",
                    deleted_bytes.getvalue(),
                    file_name="deleted_imports.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="deleted_imports_dl",
                )
                preview_cols = [
                    "import_id",
                    "imported_at",
                    "customer_name",
                    "phone",
                    "product_label",
                    "do_number",
                    "original_date",
                    "deleted_at",
                ]
                st.dataframe(
                    formatted_deleted[preview_cols],
                    use_container_width=True,
                )

    st.markdown("---")
    st.subheader("🔎 Quick snapshots")
    tab1, tab2, tab3 = st.tabs([
        "Upcoming expiries (next 60 days)",
        "Recent services",
        "Recent maintenance",
    ])

    with tab1:
        days_window = st.slider(
            "Upcoming window (days)",
            min_value=7,
            max_value=180,
            value=60,
            step=1,
            help="Adjust how far ahead to look for upcoming warranty expiries.",
        )
        upcoming = fetch_warranty_window(conn, 0, int(days_window))
        upcoming = format_warranty_table(upcoming)
        st.caption(f"Active warranties scheduled to expire in the next {int(days_window)} days.")
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
            address = st.text_area("Address")
            purchase_date = st.date_input("Purchase/Issue date", value=datetime.now().date())
            product_count = st.number_input(
                "Number of products",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="new_customer_product_count",
                help="Add additional rows when a customer has purchased multiple products.",
            )
            product_entries: list[dict[str, object]] = []
            for idx in range(int(product_count)):
                cols = st.columns((2, 2, 2, 1))
                with cols[0]:
                    product_name = st.text_input(
                        f"Product {idx + 1} details",
                        key=f"new_customer_product_name_{idx}",
                    )
                with cols[1]:
                    product_model = st.text_input(
                        f"Model {idx + 1}",
                        key=f"new_customer_product_model_{idx}",
                    )
                with cols[2]:
                    product_serial = st.text_input(
                        f"Serial {idx + 1}",
                        key=f"new_customer_product_serial_{idx}",
                    )
                with cols[3]:
                    product_quantity = st.number_input(
                        f"Qty {idx + 1}",
                        min_value=1,
                        max_value=999,
                        value=1,
                        step=1,
                        key=f"new_customer_product_quantity_{idx}",
                    )
                product_entries.append(
                    {
                        "name": product_name,
                        "model": product_model,
                        "serial": product_serial,
                        "quantity": int(product_quantity),
                    }
                )
            do_code = st.text_input("Delivery order code")
            sales_person_input = st.text_input("Sales person")
            customer_pdf = st.file_uploader("Attach customer PDF", type=["pdf"], key="new_customer_pdf")
            do_pdf = st.file_uploader("Attach Delivery Order (PDF)", type=["pdf"], key="new_customer_do_pdf")
            submitted = st.form_submit_button("Save")
            if submitted and name.strip():
                cur = conn.cursor()
                name_val = clean_text(name)
                phone_val = clean_text(phone)
                address_val = clean_text(address)
                cleaned_products, product_labels = normalize_product_entries(product_entries)
                product_label = "\n".join(product_labels) if product_labels else None
                purchase_str = purchase_date.strftime("%Y-%m-%d") if purchase_date else None
                do_serial = clean_text(do_code)
                cur.execute(
                    "INSERT INTO customers (name, phone, address, purchase_date, product_info, delivery_order_code, sales_person, dup_flag) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        name_val,
                        phone_val,
                        address_val,
                        purchase_str,
                        product_label,
                        do_serial,
                        clean_text(sales_person_input),
                    ),
                )
                cid = cur.lastrowid
                conn.commit()
                if cleaned_products:
                    for prod in cleaned_products:
                        if not prod.get("name"):
                            continue
                        cur.execute(
                            "SELECT product_id FROM products WHERE name=? AND IFNULL(model,'')=IFNULL(?, '') LIMIT 1",
                            (prod.get("name"), prod.get("model")),
                        )
                        row = cur.fetchone()
                        if row:
                            pid = row[0]
                        else:
                            cur.execute(
                                "INSERT INTO products (name, model, serial) VALUES (?, ?, ?)",
                                (
                                    prod.get("name"),
                                    prod.get("model"),
                                    prod.get("serial"),
                                ),
                            )
                            pid = cur.lastrowid
                        issue = purchase_date.strftime("%Y-%m-%d") if purchase_date else None
                        expiry = (
                            (purchase_date + timedelta(days=365)).strftime("%Y-%m-%d")
                            if purchase_date
                            else None
                        )
                        cur.execute(
                            "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, 'active')",
                            (cid, pid, prod.get("serial"), issue, expiry),
                        )
                    conn.commit()
                if do_serial:
                    cur = conn.cursor()
                    cur.execute("SELECT 1 FROM delivery_orders WHERE do_number = ?", (do_serial,))
                    if cur.fetchone():
                        st.warning("Delivery order code already exists. Skipped linking.")
                    else:
                        stored_path = None
                        if do_pdf is not None:
                            safe_name = _sanitize_path_component(do_serial)
                            saved = save_uploaded_file(do_pdf, DELIVERY_ORDER_DIR, filename=f"{safe_name}.pdf")
                            if saved:
                                try:
                                    stored_path = str(saved.relative_to(BASE_DIR))
                                except ValueError:
                                    stored_path = str(saved)
                        conn.execute(
                            "INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path) VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                do_serial,
                                cid,
                                None,
                                cleaned_products[0].get("name") if cleaned_products else None,
                                clean_text(sales_person_input),
                                stored_path,
                            ),
                        )
                        conn.commit()
                if customer_pdf is not None:
                    saved = save_uploaded_file(customer_pdf, CUSTOMER_DOCS_DIR, filename=f"customer_{cid}.pdf")
                    if saved:
                        try:
                            stored_path = str(saved.relative_to(BASE_DIR))
                        except ValueError:
                            stored_path = str(saved)
                        conn.execute(
                            "UPDATE customers SET attachment_path=? WHERE customer_id=?",
                            (stored_path, cid),
                        )
                        conn.commit()
                if phone_val:
                    recalc_customer_duplicate_flag(conn, phone_val)
                    conn.commit()
                st.success("Customer saved")
                _safe_rerun()
    sort_dir = st.radio("Sort by created date", ["Newest first", "Oldest first"], horizontal=True)
    order = "DESC" if sort_dir == "Newest first" else "ASC"
    q = st.text_input("Search (name/phone/address/product/DO)")
    df_raw = df_query(conn, f"""
        SELECT customer_id as id, name, phone, address, purchase_date, product_info, delivery_order_code, sales_person, attachment_path, created_at, dup_flag
        FROM customers
        WHERE (? = '' OR name LIKE '%'||?||'%' OR phone LIKE '%'||?||'%' OR address LIKE '%'||?||'%' OR product_info LIKE '%'||?||'%' OR delivery_order_code LIKE '%'||?||'%' OR sales_person LIKE '%'||?||'%')
        ORDER BY datetime(created_at) {order}
    """, (q, q, q, q, q, q, q))
    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"
    st.markdown("### Quick edit or delete")
    if df_raw.empty:
        st.info("No customers found for the current filters.")
    else:
        original_map: dict[int, dict] = {}
        for record in df_raw.to_dict("records"):
            cid = int_or_none(record.get("id"))
            if cid is not None:
                original_map[cid] = record
        editor_df = df_raw.copy()
        editor_df["purchase_date"] = pd.to_datetime(editor_df["purchase_date"], errors="coerce")
        editor_df["created_at"] = pd.to_datetime(editor_df["created_at"], errors="coerce")
        if "dup_flag" in editor_df.columns:
            editor_df["duplicate"] = editor_df["dup_flag"].apply(lambda x: "🔁 duplicate phone" if int_or_none(x) == 1 else "")
        else:
            editor_df["duplicate"] = ""
        editor_df["Action"] = "Keep"
        column_order = [
            col
            for col in [
                "id",
                "name",
                "phone",
                "address",
                "purchase_date",
                "product_info",
                "delivery_order_code",
                "sales_person",
                "duplicate",
                "created_at",
                "Action",
            ]
            if col in editor_df.columns
        ]
        editor_df = editor_df[column_order]
        editor_state = st.data_editor(
            editor_df,
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "id": st.column_config.Column("ID", disabled=True),
                "name": st.column_config.TextColumn("Name"),
                "phone": st.column_config.TextColumn("Phone"),
                "address": st.column_config.TextColumn("Address"),
                "purchase_date": st.column_config.DateColumn("Purchase date", format="DD-MM-YYYY", required=False),
                "product_info": st.column_config.TextColumn("Product"),
                "delivery_order_code": st.column_config.TextColumn("DO code"),
                "sales_person": st.column_config.TextColumn("Sales person"),
                "duplicate": st.column_config.Column("Duplicate", disabled=True),
                "created_at": st.column_config.DatetimeColumn("Created", format="DD-MM-YYYY HH:mm", disabled=True),
                "Action": st.column_config.SelectboxColumn("Action", options=["Keep", "Delete"], required=True),
            },
        )
        if not is_admin:
            st.caption("Set Action to “Delete” requires admin access; non-admin changes will be ignored.")
        if st.button("Apply table updates", type="primary"):
            editor_result = editor_state if isinstance(editor_state, pd.DataFrame) else pd.DataFrame(editor_state)
            if editor_result.empty:
                st.info("No rows to update.")
            else:
                phones_to_recalc: set[str] = set()
                updates = deletes = 0
                errors: list[str] = []
                made_updates = False
                for row in editor_result.to_dict("records"):
                    cid = int_or_none(row.get("id"))
                    if cid is None or cid not in original_map:
                        continue
                    action = str(row.get("Action") or "Keep").strip().lower()
                    if action == "delete":
                        if is_admin:
                            delete_customer_record(conn, cid)
                            deletes += 1
                        else:
                            errors.append(f"Only admins can delete customers (ID #{cid}).")
                        continue
                    new_name = clean_text(row.get("name"))
                    new_phone = clean_text(row.get("phone"))
                    new_address = clean_text(row.get("address"))
                    purchase_str, _ = date_strings_from_input(row.get("purchase_date"))
                    product_label = clean_text(row.get("product_info"))
                    new_do = clean_text(row.get("delivery_order_code"))
                    new_sales_person = clean_text(row.get("sales_person"))
                    original_row = original_map[cid]
                    old_name = clean_text(original_row.get("name"))
                    old_phone = clean_text(original_row.get("phone"))
                    old_address = clean_text(original_row.get("address"))
                    old_purchase = clean_text(original_row.get("purchase_date"))
                    old_product = clean_text(original_row.get("product_info"))
                    old_do = clean_text(original_row.get("delivery_order_code"))
                    old_sales_person = clean_text(original_row.get("sales_person"))
                    if (
                        new_name == old_name
                        and new_phone == old_phone
                        and new_address == old_address
                        and purchase_str == old_purchase
                        and product_label == old_product
                        and new_do == old_do
                        and new_sales_person == old_sales_person
                    ):
                        continue
                    conn.execute(
                        "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, dup_flag=0 WHERE customer_id=?",
                        (
                            new_name,
                            new_phone,
                            new_address,
                            purchase_str,
                            product_label,
                            new_do,
                            new_sales_person,
                            cid,
                        ),
                    )
                    if new_do:
                        conn.execute(
                            """
                            INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(do_number) DO UPDATE SET
                                customer_id=excluded.customer_id,
                                description=excluded.description,
                                sales_person=excluded.sales_person
                            """,
                            (
                                new_do,
                                cid,
                                None,
                                product_label,
                                new_sales_person,
                                None,
                            ),
                        )
                    if old_do and old_do != new_do:
                        conn.execute(
                            "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                            (old_do, cid),
                        )
                    conn.execute(
                        "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                        (
                            new_name,
                            new_phone,
                            new_address,
                            product_label,
                            new_do,
                            purchase_str,
                            cid,
                        ),
                    )
                    if old_phone and old_phone != new_phone:
                        phones_to_recalc.add(old_phone)
                    if new_phone:
                        phones_to_recalc.add(new_phone)
                    updates += 1
                    made_updates = True
                if made_updates:
                    conn.commit()
                if phones_to_recalc:
                    for phone_value in phones_to_recalc:
                        recalc_customer_duplicate_flag(conn, phone_value)
                    conn.commit()
                if errors:
                    for err in errors:
                        st.error(err)
                if updates or deletes:
                    st.success(f"Updated {updates} row(s) and deleted {deletes} row(s).")
                    if not errors:
                        _safe_rerun()
                elif not errors:
                    st.info("No changes detected.")
    st.markdown("### Detailed editor & attachments")
    with st.expander("Open detailed editor", expanded=False):
        df_form = fmt_dates(df_raw.copy(), ["created_at", "purchase_date"])
        if df_form.empty:
            st.info("No customers to edit yet.")
        else:
            records_fmt = df_form.to_dict("records")
            raw_map = {int(row["id"]): row for row in df_raw.to_dict("records") if int_or_none(row.get("id")) is not None}
            option_ids = [int(row["id"]) for row in records_fmt]
            labels = {}
            for row in records_fmt:
                cid = int(row["id"])
                label_name = clean_text(row.get("name")) or "(no name)"
                label_phone = clean_text(row.get("phone")) or "-"
                labels[cid] = f"{label_name} – {label_phone}"
            selected_customer_id = st.selectbox(
                "Select customer",
                option_ids,
                format_func=lambda cid: labels.get(int(cid), str(cid)),
            )
            selected_raw = raw_map[int(selected_customer_id)]
            selected_fmt = next(r for r in records_fmt if int(r["id"]) == int(selected_customer_id))
            attachment_path = selected_raw.get("attachment_path")
            resolved_attachment = resolve_upload_path(attachment_path)
            if resolved_attachment and resolved_attachment.exists():
                st.download_button(
                    "Download current PDF",
                    data=resolved_attachment.read_bytes(),
                    file_name=resolved_attachment.name,
                    key=f"cust_pdf_dl_{selected_customer_id}",
                )
            else:
                st.caption("No customer PDF attached yet.")
            is_admin = user.get("role") == "admin"
            with st.form(f"edit_customer_{selected_customer_id}"):
                name_edit = st.text_input("Name", value=clean_text(selected_raw.get("name")) or "")
                phone_edit = st.text_input("Phone", value=clean_text(selected_raw.get("phone")) or "")
                address_edit = st.text_area("Address", value=clean_text(selected_raw.get("address")) or "")
                purchase_edit = st.text_input(
                    "Purchase date (DD-MM-YYYY)", value=clean_text(selected_fmt.get("purchase_date")) or ""
                )
                product_edit = st.text_input("Product", value=clean_text(selected_raw.get("product_info")) or "")
                do_edit = st.text_input(
                    "Delivery order code", value=clean_text(selected_raw.get("delivery_order_code")) or ""
                )
                sales_person_edit = st.text_input(
                    "Sales person", value=clean_text(selected_raw.get("sales_person")) or ""
                )
                new_pdf = st.file_uploader(
                    "Attach/replace customer PDF", type=["pdf"], key=f"edit_customer_pdf_{selected_customer_id}"
                )
                col1, col2 = st.columns(2)
                save_customer = col1.form_submit_button("Save changes", type="primary")
                delete_customer = col2.form_submit_button("Delete customer", disabled=not is_admin)
            if save_customer:
                old_phone = clean_text(selected_raw.get("phone"))
                new_name = clean_text(name_edit)
                new_phone = clean_text(phone_edit)
                new_address = clean_text(address_edit)
                purchase_str, _ = date_strings_from_input(purchase_edit)
                product_label = clean_text(product_edit)
                new_do = clean_text(do_edit)
                old_do = clean_text(selected_raw.get("delivery_order_code"))
                new_sales_person = clean_text(sales_person_edit)
                new_attachment_path = attachment_path
                if new_pdf is not None:
                    saved = save_uploaded_file(new_pdf, CUSTOMER_DOCS_DIR, filename=f"customer_{selected_customer_id}.pdf")
                    if saved:
                        try:
                            stored_path = str(saved.relative_to(BASE_DIR))
                        except ValueError:
                            stored_path = str(saved)
                        new_attachment_path = stored_path
                        if attachment_path:
                            old_path = resolve_upload_path(attachment_path)
                            if old_path and old_path.exists() and old_path != saved:
                                try:
                                    old_path.unlink()
                                except Exception:
                                    pass
                conn.execute(
                    "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, attachment_path=?, dup_flag=0 WHERE customer_id=?",
                    (
                        new_name,
                        new_phone,
                        new_address,
                        purchase_str,
                        product_label,
                        new_do,
                        new_sales_person,
                        new_attachment_path,
                        int(selected_customer_id),
                    ),
                )
                if new_do:
                    conn.execute(
                        """
                        INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(do_number) DO UPDATE SET
                            customer_id=excluded.customer_id,
                            description=excluded.description,
                            sales_person=excluded.sales_person
                        """,
                        (
                            new_do,
                            int(selected_customer_id),
                            None,
                            product_label,
                            new_sales_person,
                            None,
                        ),
                    )
                if old_do and old_do != new_do:
                    conn.execute(
                        "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                        (old_do, int(selected_customer_id)),
                    )
                conn.execute(
                    "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                    (
                        new_name,
                        new_phone,
                        new_address,
                        product_label,
                        new_do,
                        purchase_str,
                        int(selected_customer_id),
                    ),
                )
                conn.commit()
                if old_phone and old_phone != new_phone:
                    recalc_customer_duplicate_flag(conn, old_phone)
                if new_phone:
                    recalc_customer_duplicate_flag(conn, new_phone)
                conn.commit()
                st.success("Customer updated.")
                _safe_rerun()
            if delete_customer:
                if is_admin:
                    delete_customer_record(conn, int(selected_customer_id))
                    st.warning("Customer deleted.")
                    _safe_rerun()
                else:
                    st.error("Only admins can delete customers.")
    st.markdown("**Recently Added Customers**")
    recent_df = df_query(conn, """
        SELECT customer_id as id, name, phone, purchase_date, product_info, delivery_order_code, sales_person, created_at
        FROM customers
        ORDER BY datetime(created_at) DESC LIMIT 200
    """)
    recent_df = fmt_dates(recent_df, ["created_at", "purchase_date"])
    st.dataframe(recent_df.drop(columns=["id"], errors="ignore"))
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
          AND (w.status IS NULL OR w.status <> 'deleted')
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


def _render_service_section(conn, *, show_heading: bool = True):
    if show_heading:
        st.subheader("🛠️ Service Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    customer_options, customer_labels, _, label_by_id = fetch_customer_choices(conn)
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
        label_parts = [do_num]
        if cust_name:
            label_parts.append(f"({cust_name})")
        if summary:
            snippet = summary[:40]
            if len(summary) > 40:
                snippet += "…"
            label_parts.append(f"– {snippet}")
        label = " ".join(part for part in label_parts if part)
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
        default_customer = do_customer_map.get(selected_do)
        state_key = "service_customer_link"
        last_do_key = "service_customer_last_do"
        linked_customer = default_customer
        if default_customer is not None:
            st.session_state[last_do_key] = selected_do
            st.session_state[state_key] = default_customer
            customer_label = (
                customer_labels.get(default_customer)
                or customer_label_map.get(default_customer)
                or label_by_id.get(default_customer)
                or do_customer_name_map.get(selected_do)
                or f"Customer #{default_customer}"
            )
            st.text_input("Customer", value=customer_label, disabled=True)
        else:
            choices = list(customer_options)
            if st.session_state.get(last_do_key) != selected_do:
                st.session_state[last_do_key] = selected_do
                st.session_state[state_key] = None
            linked_customer = st.selectbox(
                "Customer",
                options=choices,
                format_func=lambda cid: customer_labels.get(cid, "-- Select customer --"),
                key=state_key,
            )
        today = datetime.now().date()
        service_period_value = st.date_input(
            "Service period",
            value=(today, today),
            help="Select the start and end dates for the service visit.",
        )
        description = st.text_area("Service description")
        status_value = status_input_widget("service_new", DEFAULT_SERVICE_STATUS)
        remarks = st.text_area("Remarks / updates")
        service_product_count = st.number_input(
            "Products sold during service",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="service_additional_product_count",
            help="Capture any new items sold while this service was in progress.",
        )
        service_product_entries: list[dict[str, object]] = []
        for idx in range(int(service_product_count)):
            cols = st.columns((2, 2, 2, 1))
            with cols[0]:
                product_name = st.text_input(
                    f"Product {idx + 1} details",
                    key=f"service_product_name_{idx}",
                )
            with cols[1]:
                product_model = st.text_input(
                    f"Model {idx + 1}",
                    key=f"service_product_model_{idx}",
                )
            with cols[2]:
                product_serial = st.text_input(
                    f"Serial {idx + 1}",
                    key=f"service_product_serial_{idx}",
                )
            with cols[3]:
                product_quantity = st.number_input(
                    f"Qty {idx + 1}",
                    min_value=1,
                    max_value=999,
                    value=1,
                    step=1,
                    key=f"service_product_quantity_{idx}",
                )
            service_product_entries.append(
                {
                    "name": product_name,
                    "model": product_model,
                    "serial": product_serial,
                    "quantity": int(product_quantity),
                }
            )
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
            selected_customer = linked_customer if linked_customer is not None else do_customer_map.get(selected_do)
            selected_customer = int(selected_customer) if selected_customer is not None else None
            cur = conn.cursor()
            if isinstance(service_period_value, (list, tuple)):
                if len(service_period_value) >= 1:
                    service_start_date = service_period_value[0]
                    service_end_date = (
                        service_period_value[-1]
                        if len(service_period_value) > 1
                        else service_period_value[0]
                    )
                else:
                    service_start_date = service_end_date = None
            else:
                service_start_date = service_end_date = service_period_value
            if service_start_date and service_end_date and service_end_date < service_start_date:
                service_start_date, service_end_date = service_end_date, service_start_date
            service_start_str = (
                service_start_date.strftime("%Y-%m-%d") if service_start_date else None
            )
            service_end_str = (
                service_end_date.strftime("%Y-%m-%d") if service_end_date else None
            )
            service_date_str = service_start_str or service_end_str
            _cleaned_service_products, service_product_labels = normalize_product_entries(
                service_product_entries
            )
            service_product_label = (
                "\n".join(service_product_labels) if service_product_labels else None
            )
            cur.execute(
                """
                INSERT INTO services (
                    do_number,
                    customer_id,
                    service_date,
                    service_start_date,
                    service_end_date,
                    description,
                    status,
                    remarks,
                    service_product_info,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_do,
                    selected_customer,
                    service_date_str,
                    service_start_str,
                    service_end_str,
                    clean_text(description),
                    status_value,
                    clean_text(remarks),
                    service_product_label,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            service_id = cur.lastrowid
            link_delivery_order_to_customer(conn, selected_do, selected_customer)
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
               s.service_start_date,
               s.service_end_date,
               s.service_product_info,
               s.description,
               s.status,
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
        ORDER BY datetime(COALESCE(s.service_start_date, s.service_date)) DESC, s.service_id DESC
        """,
    )
    if not service_df.empty:
        service_df = fmt_dates(service_df, ["service_date", "service_start_date", "service_end_date"])
        service_df["service_period"] = service_df.apply(
            lambda row: format_period_label(
                row.get("service_start_date"), row.get("service_end_date")
            ),
            axis=1,
        )
        service_df["Last update"] = pd.to_datetime(service_df.get("updated_at"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M")
        service_df.loc[service_df["Last update"].isna(), "Last update"] = None
        if "status" in service_df.columns:
            service_df["status"] = service_df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
        display = service_df.rename(
            columns={
                "do_number": "DO Serial",
                "service_date": "Service date",
                "service_start_date": "Service start date",
                "service_end_date": "Service end date",
                "service_period": "Service period",
                "service_product_info": "Products sold",
                "description": "Description",
                "status": "Status",
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
        st.markdown("#### Update status & remarks")
        options = [int(r["service_id"]) for r in records]
        def service_label(record):
            do_ref = clean_text(record.get("do_number")) or "(no DO)"
            date_ref = clean_text(record.get("service_period")) or clean_text(
                record.get("service_date")
            )
            customer_ref = clean_text(record.get("customer"))
            parts = [do_ref]
            if date_ref:
                parts.append(f"· {date_ref}")
            if customer_ref:
                parts.append(f"· {customer_ref}")
            return " ".join(parts)

        labels = {int(r["service_id"]): service_label(r) for r in records}
        selected_service_id = st.selectbox(
            "Select service entry",
            options,
            format_func=lambda rid: labels.get(rid, str(rid)),
        )
        selected_record = next(r for r in records if int(r["service_id"]) == int(selected_service_id))
        new_status = status_input_widget(
            f"service_edit_{selected_service_id}", selected_record.get("status")
        )
        new_remarks = st.text_area(
            "Remarks",
            value=clean_text(selected_record.get("remarks")) or "",
            key=f"service_edit_{selected_service_id}",
        )
        if st.button("Save updates", key="save_service_updates"):
            conn.execute(
                "UPDATE services SET status = ?, remarks = ?, updated_at = datetime('now') WHERE service_id = ?",
                (
                    new_status,
                    clean_text(new_remarks),
                    int(selected_service_id),
                ),
            )
            conn.commit()
            st.success("Service record updated.")
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


def _render_maintenance_section(conn, *, show_heading: bool = True):
    if show_heading:
        st.subheader("🔧 Maintenance Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    customer_options, customer_labels, _, label_by_id = fetch_customer_choices(conn)
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
        label_parts = [do_num]
        if cust_name:
            label_parts.append(f"({cust_name})")
        if summary:
            snippet = summary[:40]
            if len(summary) > 40:
                snippet += "…"
            label_parts.append(f"– {snippet}")
        label = " ".join(part for part in label_parts if part)
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
        default_customer = do_customer_map.get(selected_do)
        state_key = "maintenance_customer_link"
        last_do_key = "maintenance_customer_last_do"
        linked_customer = default_customer
        if default_customer is not None:
            st.session_state[last_do_key] = selected_do
            st.session_state[state_key] = default_customer
            customer_label = (
                customer_labels.get(default_customer)
                or customer_label_map.get(default_customer)
                or label_by_id.get(default_customer)
                or do_customer_name_map.get(selected_do)
                or f"Customer #{default_customer}"
            )
            st.text_input("Customer", value=customer_label, disabled=True)
        else:
            choices = list(customer_options)
            if st.session_state.get(last_do_key) != selected_do:
                st.session_state[last_do_key] = selected_do
                st.session_state[state_key] = None
            linked_customer = st.selectbox(
                "Customer",
                options=choices,
                format_func=lambda cid: customer_labels.get(cid, "-- Select customer --"),
                key=state_key,
            )
        today = datetime.now().date()
        maintenance_period_value = st.date_input(
            "Maintenance period",
            value=(today, today),
            help="Select when this maintenance work started and finished.",
        )
        description = st.text_area("Maintenance description")
        status_value = status_input_widget("maintenance_new", DEFAULT_SERVICE_STATUS)
        remarks = st.text_area("Remarks / updates")
        maintenance_product_count = st.number_input(
            "Products sold during maintenance",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="maintenance_additional_product_count",
            help="Track any new items purchased while maintenance was carried out.",
        )
        maintenance_product_entries: list[dict[str, object]] = []
        for idx in range(int(maintenance_product_count)):
            cols = st.columns((2, 2, 2, 1))
            with cols[0]:
                product_name = st.text_input(
                    f"Product {idx + 1} details",
                    key=f"maintenance_product_name_{idx}",
                )
            with cols[1]:
                product_model = st.text_input(
                    f"Model {idx + 1}",
                    key=f"maintenance_product_model_{idx}",
                )
            with cols[2]:
                product_serial = st.text_input(
                    f"Serial {idx + 1}",
                    key=f"maintenance_product_serial_{idx}",
                )
            with cols[3]:
                product_quantity = st.number_input(
                    f"Qty {idx + 1}",
                    min_value=1,
                    max_value=999,
                    value=1,
                    step=1,
                    key=f"maintenance_product_quantity_{idx}",
                )
            maintenance_product_entries.append(
                {
                    "name": product_name,
                    "model": product_model,
                    "serial": product_serial,
                    "quantity": int(product_quantity),
                }
            )
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
            selected_customer = linked_customer if linked_customer is not None else do_customer_map.get(selected_do)
            selected_customer = int(selected_customer) if selected_customer is not None else None
            cur = conn.cursor()
            if isinstance(maintenance_period_value, (list, tuple)):
                if len(maintenance_period_value) >= 1:
                    maintenance_start_date = maintenance_period_value[0]
                    maintenance_end_date = (
                        maintenance_period_value[-1]
                        if len(maintenance_period_value) > 1
                        else maintenance_period_value[0]
                    )
                else:
                    maintenance_start_date = maintenance_end_date = None
            else:
                maintenance_start_date = maintenance_end_date = maintenance_period_value
            if (
                maintenance_start_date
                and maintenance_end_date
                and maintenance_end_date < maintenance_start_date
            ):
                maintenance_start_date, maintenance_end_date = (
                    maintenance_end_date,
                    maintenance_start_date,
                )
            maintenance_start_str = (
                maintenance_start_date.strftime("%Y-%m-%d")
                if maintenance_start_date
                else None
            )
            maintenance_end_str = (
                maintenance_end_date.strftime("%Y-%m-%d")
                if maintenance_end_date
                else None
            )
            maintenance_date_str = maintenance_start_str or maintenance_end_str
            _cleaned_maintenance_products, maintenance_product_labels = normalize_product_entries(
                maintenance_product_entries
            )
            maintenance_product_label = (
                "\n".join(maintenance_product_labels)
                if maintenance_product_labels
                else None
            )
            cur.execute(
                """
                INSERT INTO maintenance_records (
                    do_number,
                    customer_id,
                    maintenance_date,
                    maintenance_start_date,
                    maintenance_end_date,
                    description,
                    status,
                    remarks,
                    maintenance_product_info,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_do,
                    selected_customer,
                    maintenance_date_str,
                    maintenance_start_str,
                    maintenance_end_str,
                    clean_text(description),
                    status_value,
                    clean_text(remarks),
                    maintenance_product_label,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            maintenance_id = cur.lastrowid
            link_delivery_order_to_customer(conn, selected_do, selected_customer)
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
               m.maintenance_start_date,
               m.maintenance_end_date,
               m.maintenance_product_info,
               m.description,
               m.status,
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
        ORDER BY datetime(COALESCE(m.maintenance_start_date, m.maintenance_date)) DESC, m.maintenance_id DESC
        """,
    )
    if not maintenance_df.empty:
        maintenance_df = fmt_dates(
            maintenance_df,
            ["maintenance_date", "maintenance_start_date", "maintenance_end_date"],
        )
        maintenance_df["maintenance_period"] = maintenance_df.apply(
            lambda row: format_period_label(
                row.get("maintenance_start_date"), row.get("maintenance_end_date")
            ),
            axis=1,
        )
        maintenance_df["Last update"] = pd.to_datetime(maintenance_df.get("updated_at"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M")
        maintenance_df.loc[maintenance_df["Last update"].isna(), "Last update"] = None
        if "status" in maintenance_df.columns:
            maintenance_df["status"] = maintenance_df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
        display = maintenance_df.rename(
            columns={
                "do_number": "DO Serial",
                "maintenance_date": "Maintenance date",
                "maintenance_start_date": "Maintenance start date",
                "maintenance_end_date": "Maintenance end date",
                "maintenance_period": "Maintenance period",
                "maintenance_product_info": "Products sold",
                "description": "Description",
                "status": "Status",
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
        st.markdown("#### Update status & remarks")
        options = [int(r["maintenance_id"]) for r in records]
        def maintenance_label(record):
            do_ref = clean_text(record.get("do_number")) or "(no DO)"
            date_ref = clean_text(record.get("maintenance_period")) or clean_text(
                record.get("maintenance_date")
            )
            customer_ref = clean_text(record.get("customer"))
            parts = [do_ref]
            if date_ref:
                parts.append(f"· {date_ref}")
            if customer_ref:
                parts.append(f"· {customer_ref}")
            return " ".join(parts)

        labels = {int(r["maintenance_id"]): maintenance_label(r) for r in records}
        selected_maintenance_id = st.selectbox(
            "Select maintenance entry",
            options,
            format_func=lambda rid: labels.get(rid, str(rid)),
        )
        selected_record = next(r for r in records if int(r["maintenance_id"]) == int(selected_maintenance_id))
        new_status = status_input_widget(
            f"maintenance_edit_{selected_maintenance_id}",
            selected_record.get("status"),
        )
        new_remarks = st.text_area(
            "Remarks",
            value=clean_text(selected_record.get("remarks")) or "",
            key=f"maintenance_edit_{selected_maintenance_id}",
        )
        if st.button("Save maintenance updates", key="save_maintenance_updates"):
            conn.execute(
                "UPDATE maintenance_records SET status = ?, remarks = ?, updated_at = datetime('now') WHERE maintenance_id = ?",
                (
                    new_status,
                    clean_text(new_remarks),
                    int(selected_maintenance_id),
                ),
            )
            conn.commit()
            st.success("Maintenance record updated.")
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


def service_maintenance_page(conn):
    st.subheader("🛠️ Service & Maintenance")
    service_tab, maintenance_tab = st.tabs(["Service", "Maintenance"])
    with service_tab:
        _render_service_section(conn, show_heading=False)
    with maintenance_tab:
        _render_maintenance_section(conn, show_heading=False)


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
            GROUP_CONCAT(DISTINCT address) AS address,
            GROUP_CONCAT(DISTINCT purchase_date) AS purchase_dates,
            GROUP_CONCAT(DISTINCT product_info) AS products,
            GROUP_CONCAT(DISTINCT delivery_order_code) AS do_codes
        FROM customers
        WHERE customer_id IN ({','.join('?'*len(ids))})
        """,
        ids,
    ).iloc[0].to_dict()

    st.write("**Name:**", info.get("name") or blank_label)
    st.write("**Phone:**", info.get("phone"))
    st.write("**Address:**", info.get("address"))
    st.write("**Purchase:**", info.get("purchase_dates"))
    st.write("**Product:**", info.get("products"))
    st.write("**Delivery order:**", info.get("do_codes"))
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
               s.service_start_date,
               s.service_end_date,
               s.service_product_info,
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
        ORDER BY datetime(COALESCE(s.service_start_date, s.service_date)) DESC, s.service_id DESC
        """,
        ids,
    )
    service_df = fmt_dates(service_df, ["service_date", "service_start_date", "service_end_date"])
    if not service_df.empty:
        service_df["service_period"] = service_df.apply(
            lambda row: format_period_label(
                row.get("service_start_date"), row.get("service_end_date")
            ),
            axis=1,
        )

    maintenance_df = df_query(
        conn,
        f"""
        SELECT m.maintenance_id,
               m.do_number,
               m.maintenance_date,
               m.maintenance_start_date,
               m.maintenance_end_date,
               m.maintenance_product_info,
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
        ORDER BY datetime(COALESCE(m.maintenance_start_date, m.maintenance_date)) DESC, m.maintenance_id DESC
        """,
        ids,
    )
    maintenance_df = fmt_dates(
        maintenance_df,
        ["maintenance_date", "maintenance_start_date", "maintenance_end_date"],
    )
    if not maintenance_df.empty:
        maintenance_df["maintenance_period"] = maintenance_df.apply(
            lambda row: format_period_label(
                row.get("maintenance_start_date"), row.get("maintenance_end_date")
            ),
            axis=1,
        )

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
                "service_start_date": "Service start date",
                "service_end_date": "Service end date",
                "service_period": "Service period",
                "service_product_info": "Products sold",
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
                "maintenance_start_date": "Maintenance start date",
                "maintenance_end_date": "Maintenance end date",
                "maintenance_period": "Maintenance period",
                "maintenance_product_info": "Products sold",
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
        SELECT customer_id as id, name, phone, address, purchase_date, product_info, delivery_order_code, created_at
        FROM customers
        WHERE {customer_incomplete_clause()}
        ORDER BY datetime(created_at) DESC
        """,
    )
    scraps = fmt_dates(scraps, ["created_at", "purchase_date"])
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
    display_cols = ["name", "phone", "address", "purchase_date", "product_info", "delivery_order_code", "missing", "created_at"]
    st.dataframe(scraps[display_cols])

    st.markdown("### Update scrap record")
    records = scraps.to_dict("records")
    option_keys = [int(r["id"]) for r in records]
    option_labels = {}
    for r in records:
        rid = int(r["id"])
        name_label = clean_text(r.get("name")) or "(no name)"
        missing_label = clean_text(r.get("missing")) or "—"
        details = missing_label or "complete"
        created = clean_text(r.get("created_at"))
        created_fmt = f" – added {created}" if created else ""
        option_labels[rid] = f"{name_label or '(no name)'} (missing: {details}){created_fmt}"
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
        address = st.text_area("Address", existing_value("address"))
        purchase = st.text_input("Purchase date (DD-MM-YYYY)", existing_value("purchase_date"))
        product = st.text_input("Product", existing_value("product_info"))
        do_code = st.text_input("Delivery order code", existing_value("delivery_order_code"))
        col1, col2 = st.columns(2)
        save = col1.form_submit_button("Save changes", type="primary")
        delete = col2.form_submit_button("Delete scrap")

    if save:
        new_name = clean_text(name)
        new_phone = clean_text(phone)
        new_address = clean_text(address)
        purchase_str, _ = date_strings_from_input(purchase)
        new_product = clean_text(product)
        new_do = clean_text(do_code)
        old_phone = clean_text(selected.get("phone"))
        conn.execute(
            "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
            (
                new_name,
                new_phone,
                new_address,
                purchase_str,
                new_product,
                new_do,
                int(selected_id),
            ),
        )
        old_do = clean_text(selected.get("delivery_order_code"))
        if new_do:
            conn.execute(
                """
                INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(do_number) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    description=excluded.description
                """,
                (
                    new_do,
                    int(selected_id),
                    None,
                    new_product,
                    None,
                    None,
                ),
            )
        if old_do and old_do != new_do:
            conn.execute(
                "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                (old_do, int(selected_id)),
            )
        if old_phone and old_phone != new_phone:
            recalc_customer_duplicate_flag(conn, old_phone)
        if new_phone:
            recalc_customer_duplicate_flag(conn, new_phone)
        conn.commit()
        conn.execute(
            "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
            (
                new_name,
                new_phone,
                new_address,
                new_product,
                new_do,
                purchase_str,
                int(selected_id),
            ),
        )
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
    default_order = ["date", "customer_name", "address", "phone", "product", "do_code"]
    if cols_norm[: len(default_order)] == default_order:
        mapping = {field: idx for idx, field in enumerate(default_order)}
    return mapping


def split_product_label(label: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if label is None:
        return None, None
    text = clean_text(label)
    if not text:
        return None, None
    if "-" in text:
        left, right = text.split("-", 1)
        return clean_text(left), clean_text(right)
    return text, None


def parse_date_value(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    dt = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    if isinstance(dt, pd.DatetimeIndex):
        dt = dt[0]
    return dt.normalize()


def date_strings_from_input(value) -> tuple[Optional[str], Optional[str]]:
    dt = parse_date_value(value)
    if dt is None:
        return None, None
    expiry = dt + pd.Timedelta(days=365)
    return dt.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d")


def int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

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
        st.markdown("---")
        manage_import_history(conn)
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
    col4, col5 = st.columns(2)
    col6, _ = st.columns(2)
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
    sel_do = col6.selectbox(
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
    st.markdown("#### Review & edit rows before importing")
    preview = df_norm.copy()
    preview["Action"] = "Import"
    editor = st.data_editor(
        preview,
        key="import_editor",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD-MM-YYYY", required=False),
            "Action": st.column_config.SelectboxColumn("Action", options=["Import", "Skip"], required=True),
        },
    )

    if st.button("Append into database"):
        editor = editor if isinstance(editor, pd.DataFrame) else pd.DataFrame(editor)
        ready = editor[editor["Action"].fillna("Import").str.lower() == "import"].copy()
        ready.drop(columns=["Action"], inplace=True, errors="ignore")
        seeded, d_c, d_p = _import_clean6(conn, ready, tag="Manual import (mapped)")
        if seeded == 0:
            st.warning("No rows added (rows empty/invalid). Check mapping or file.")
        else:
            st.success(f"Imported {seeded} rows. Duplicates flagged — customers: {d_c}, products: {d_p}.")

    st.markdown("---")
    manage_import_history(conn)

def manual_merge_section(conn, customers_df: pd.DataFrame) -> None:
    if customers_df is None or customers_df.empty:
        return

    if "id" not in customers_df.columns:
        return

    work_df = customers_df.copy()
    work_df["id"] = work_df["id"].apply(int_or_none)
    work_df = work_df[work_df["id"].notna()]
    if work_df.empty:
        return

    work_df["id"] = work_df["id"].astype(int)

    def build_label(row):
        name_val = clean_text(row.get("name")) or "(no name)"
        phone_val = clean_text(row.get("phone")) or "(no phone)"
        address_val = clean_text(row.get("address")) or "-"
        product_val = clean_text(row.get("product_info")) or "-"
        do_val = clean_text(row.get("delivery_order_code")) or "-"
        date_dt = parse_date_value(row.get("purchase_date"))
        if date_dt is not None:
            date_label = date_dt.strftime(DATE_FMT)
        else:
            date_label = clean_text(row.get("purchase_date")) or "-"
        return f"#{row['id']} – {name_val} | Phone: {phone_val} | Date: {date_label} | Product: {product_val} | DO: {do_val}"

    work_df["_label"] = work_df.apply(build_label, axis=1)
    work_df["_search_blob"] = work_df.apply(
        lambda row: " ".join(
            filter(
                None,
                [
                    clean_text(row.get("name")),
                    clean_text(row.get("phone")),
                    clean_text(row.get("address")),
                    clean_text(row.get("product_info")),
                    clean_text(row.get("delivery_order_code")),
                ],
            )
        ),
        axis=1,
    )
    work_df["_search_blob"] = work_df["_search_blob"].fillna("").str.lower()

    label_map = {row["id"]: row["_label"] for row in work_df.to_dict("records")}

    st.divider()
    st.markdown("#### Manual customer merge")
    st.caption(
        "Select multiple customer records that refer to the same person even if the phone number or purchase date differs. "
        "The earliest record will be kept and enriched with the combined details."
    )

    filter_value = st.text_input(
        "Filter customers by name, phone, address, product, or DO (optional)",
        key="manual_merge_filter",
    ).strip()

    filtered_df = work_df
    if filter_value:
        escaped = re.escape(filter_value.lower())
        mask = filtered_df["_search_blob"].str.contains(escaped, regex=True, na=False)
        filtered_df = filtered_df[mask]

    options = filtered_df["id"].tolist()
    if not options:
        st.info("No customer records match the current filter.")
        return

    with st.form("manual_merge_form"):
        selected_ids = st.multiselect(
            "Select customer records to merge",
            options=options,
            format_func=lambda cid: label_map.get(cid, f"#{cid}"),
        )

        preview_df = work_df[work_df["id"].isin(selected_ids)]
        if not preview_df.empty:
            preview_df = preview_df.copy()
            preview_df["purchase_date"] = pd.to_datetime(preview_df["purchase_date"], errors="coerce")
            preview_df["purchase_date"] = preview_df["purchase_date"].dt.strftime(DATE_FMT)
            preview_df["purchase_date"] = preview_df["purchase_date"].fillna("-")
            preview_cols = [
                col
                for col in [
                    "id",
                    "name",
                    "phone",
                    "address",
                    "purchase_date",
                    "product_info",
                    "delivery_order_code",
                    "created_at",
                ]
                if col in preview_df.columns
            ]
            st.dataframe(
                preview_df[preview_cols]
                .rename(
                    columns={
                        "id": "ID",
                        "name": "Name",
                        "phone": "Phone",
                        "address": "Address",
                        "purchase_date": "Purchase date",
                        "product_info": "Product",
                        "delivery_order_code": "DO code",
                        "created_at": "Created",
                    }
                )
                .sort_values("ID"),
                use_container_width=True,
                hide_index=True,
            )

        submitted = st.form_submit_button("Merge selected customers", type="primary")

    if submitted:
        if len(selected_ids) < 2:
            st.warning("Select at least two customers to merge.")
            return
        if merge_customer_records(conn, selected_ids):
            st.success(f"Merged {len(selected_ids)} customer records.")
            _safe_rerun()
        else:
            st.error("Could not merge the selected customers. Please try again.")


def duplicates_page(conn):
    st.subheader("⚠️ Possible Duplicates")
    cust_raw = df_query(
        conn,
        "SELECT customer_id as id, name, phone, address, purchase_date, product_info, delivery_order_code, dup_flag, created_at FROM customers ORDER BY datetime(created_at) DESC",
    )
    warr = df_query(
        conn,
        "SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date, w.dup_flag FROM warranties w LEFT JOIN customers c ON c.customer_id = w.customer_id LEFT JOIN products p ON p.product_id = w.product_id ORDER BY date(w.issue_date) DESC",
    )
    duplicate_customers = pd.DataFrame()
    if not cust_raw.empty:
        duplicate_customers = cust_raw[cust_raw["dup_flag"] == 1].copy()
    if duplicate_customers.empty:
        st.success("No customer duplicates detected at the moment.")
    else:
        editor_df = duplicate_customers.copy()
        editor_df["__group_key"] = [
            " | ".join(
                [
                    clean_text(row.get("phone")) or "(no phone)",
                    (
                        parse_date_value(row.get("purchase_date")).strftime(DATE_FMT)
                        if parse_date_value(row.get("purchase_date")) is not None
                        else "-"
                    ),
                    clean_text(row.get("product_info")) or "-",
                ]
            )
            for _, row in editor_df.iterrows()
        ]
        preview_df = editor_df.assign(
            duplicate="🔁 duplicate phone",
            purchase_date_fmt=pd.to_datetime(editor_df["purchase_date"], errors="coerce").dt.strftime(DATE_FMT),
            created_at_fmt=pd.to_datetime(editor_df["created_at"], errors="coerce").dt.strftime("%d-%m-%Y %H:%M"),
        )
        preview_cols = [
            col
            for col in [
                "__group_key",
                "name",
                "phone",
                "address",
                "purchase_date_fmt",
                "product_info",
                "delivery_order_code",
                "duplicate",
                "created_at_fmt",
            ]
            if col in preview_df.columns
        ]
        if preview_cols:
            display_df = (
                preview_df[preview_cols]
                .rename(
                    columns={
                        "__group_key": "Duplicate set",
                        "purchase_date_fmt": "Purchase date",
                        "product_info": "Product",
                        "delivery_order_code": "DO code",
                        "created_at_fmt": "Created",
                    }
                )
                .sort_values(by=["Duplicate set", "Created"], na_position="last")
            )
            display_df["Purchase date"] = display_df["Purchase date"].fillna("-")
            display_df["Created"] = display_df["Created"].fillna("-")
            st.markdown("#### Duplicate rows")
            st.caption(
                "Each duplicate set groups rows sharing the same phone, purchase date, and product so you can double-check real multi-unit sales."
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        group_counts = editor_df.groupby("__group_key").size().to_dict()
        selection_options = [(None, "All duplicate rows")] + [
            (label, f"{label} ({group_counts.get(label, 0)} row(s))") for label in sorted(editor_df["__group_key"].unique())
        ]
        selected_group, _ = st.selectbox(
            "Focus on a duplicate set (optional)",
            options=selection_options,
            index=0,
            format_func=lambda opt: opt[1],
        )
        if selected_group:
            editor_df = editor_df[editor_df["__group_key"] == selected_group]
        if editor_df.empty:
            st.info("No rows match the selected duplicate set.")
        else:
            editor_df["duplicate"] = "🔁 duplicate phone"
            editor_df["purchase_date"] = pd.to_datetime(editor_df["purchase_date"], errors="coerce")
            editor_df["created_at"] = pd.to_datetime(editor_df["created_at"], errors="coerce")
            editor_df["Action"] = "Keep"
            column_order = [
                col
                for col in [
                    "id",
                    "name",
                    "phone",
                    "address",
                    "purchase_date",
                    "product_info",
                    "delivery_order_code",
                    "duplicate",
                    "created_at",
                    "Action",
                ]
                if col in editor_df.columns
            ]
            editor_df = editor_df[column_order]
            st.markdown("#### Edit duplicate entries")
            editor_state = st.data_editor(
                editor_df,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "id": st.column_config.Column("ID", disabled=True),
                    "name": st.column_config.TextColumn("Name"),
                    "phone": st.column_config.TextColumn("Phone"),
                    "address": st.column_config.TextColumn("Address"),
                    "purchase_date": st.column_config.DateColumn("Purchase date", format="DD-MM-YYYY", required=False),
                    "product_info": st.column_config.TextColumn("Product"),
                    "delivery_order_code": st.column_config.TextColumn("DO code"),
                    "duplicate": st.column_config.Column("Duplicate", disabled=True),
                    "created_at": st.column_config.DatetimeColumn("Created", format="DD-MM-YYYY HH:mm", disabled=True),
                    "Action": st.column_config.SelectboxColumn("Action", options=["Keep", "Delete"], required=True),
                },
            )
            user = st.session_state.user or {}
            is_admin = user.get("role") == "admin"
            if not is_admin:
                st.caption("Deleting rows requires admin privileges; non-admin delete actions will be ignored.")
            raw_map = {int(row["id"]): row for row in duplicate_customers.to_dict("records") if int_or_none(row.get("id")) is not None}
            if st.button("Apply duplicate table updates", type="primary"):
                editor_result = editor_state if isinstance(editor_state, pd.DataFrame) else pd.DataFrame(editor_state)
                if editor_result.empty:
                    st.info("No rows to update.")
                else:
                    phones_to_recalc: set[str] = set()
                    updates = deletes = 0
                    errors: list[str] = []
                    made_updates = False
                    for row in editor_result.to_dict("records"):
                        cid = int_or_none(row.get("id"))
                        if cid is None or cid not in raw_map:
                            continue
                        action = str(row.get("Action") or "Keep").strip().lower()
                        if action == "delete":
                            if is_admin:
                                delete_customer_record(conn, cid)
                                deletes += 1
                            else:
                                errors.append(f"Only admins can delete customers (ID #{cid}).")
                            continue
                        new_name = clean_text(row.get("name"))
                        new_phone = clean_text(row.get("phone"))
                        new_address = clean_text(row.get("address"))
                        purchase_str, _ = date_strings_from_input(row.get("purchase_date"))
                        product_label = clean_text(row.get("product_info"))
                        new_do = clean_text(row.get("delivery_order_code"))
                        original_row = raw_map[cid]
                        old_name = clean_text(original_row.get("name"))
                        old_phone = clean_text(original_row.get("phone"))
                        old_address = clean_text(original_row.get("address"))
                        old_purchase = clean_text(original_row.get("purchase_date"))
                        old_product = clean_text(original_row.get("product_info"))
                        old_do = clean_text(original_row.get("delivery_order_code"))
                        if (
                            new_name == old_name
                            and new_phone == old_phone
                            and new_address == old_address
                            and purchase_str == old_purchase
                            and product_label == old_product
                            and new_do == old_do
                        ):
                            continue
                        conn.execute(
                            "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
                            (
                                new_name,
                                new_phone,
                                new_address,
                                purchase_str,
                                product_label,
                                new_do,
                                cid,
                            ),
                        )
                        if new_do:
                            conn.execute(
                                """
                                INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path)
                                VALUES (?, ?, ?, ?, ?, ?)
                                ON CONFLICT(do_number) DO UPDATE SET
                                    customer_id=excluded.customer_id,
                                    description=excluded.description
                                """,
                                (
                                    new_do,
                                    cid,
                                    None,
                                    product_label,
                                    None,
                                    None,
                                ),
                            )
                        if old_do and old_do != new_do:
                            conn.execute(
                                "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                                (old_do, cid),
                            )
                        conn.execute(
                            "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                            (
                                new_name,
                                new_phone,
                                new_address,
                                product_label,
                                new_do,
                                purchase_str,
                                cid,
                            ),
                        )
                        if old_phone and old_phone != new_phone:
                            phones_to_recalc.add(old_phone)
                        if new_phone:
                            phones_to_recalc.add(new_phone)
                        updates += 1
                        made_updates = True
                    if made_updates:
                        conn.commit()
                    if phones_to_recalc:
                        for phone_value in phones_to_recalc:
                            recalc_customer_duplicate_flag(conn, phone_value)
                        conn.commit()
                    if errors:
                        for err in errors:
                            st.error(err)
                    if updates or deletes:
                        st.success(f"Updated {updates} row(s) and deleted {deletes} row(s).")
                        if not errors:
                            _safe_rerun()
                    elif not errors:
                        st.info("No changes detected.")
    manual_merge_section(conn, cust_raw)

    if not warr.empty:
        warr = fmt_dates(warr, ["issue_date", "expiry_date"])
        warr = warr.assign(duplicate=warr["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        st.markdown("**Warranties (duplicate serial)**")
        st.dataframe(
            warr[warr["dup_flag"] == 1].drop(columns=["id", "dup_flag"], errors="ignore"),
            use_container_width=True,
        )
def users_admin_page(conn):
    ensure_auth(role="admin")
    st.subheader("👤 Users (Admin)")
    users = df_query(conn, "SELECT user_id as id, username, role, created_at FROM users ORDER BY datetime(created_at) DESC")
    users = users.assign(created_at=pd.to_datetime(users["created_at"], errors="coerce").dt.strftime(DATE_FMT))
    st.dataframe(users.drop(columns=["id"], errors="ignore"))

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
    sort_cols = [col for col in ["date", "customer_name", "phone", "do_code"] if col in df.columns]
    if not sort_cols:
        sort_cols = df.columns.tolist()
    df = df.dropna(how="all").drop_duplicates().sort_values(by=sort_cols).reset_index(drop=True)

    cur = conn.cursor()
    seeded = 0
    d_c = d_p = 0
    phones_to_recalc: set[str] = set()
    for _, r in df.iterrows():
        d = r.get("date", pd.NaT)
        cust = r.get("customer_name"); addr = r.get("address")
        phone = r.get("phone"); prod = r.get("product")
        do_code = r.get("do_code")
        if pd.isna(cust) and pd.isna(phone) and pd.isna(prod):
            continue
        cust = str(cust) if pd.notna(cust) else None
        addr = str(addr) if pd.notna(addr) else None
        phone = str(phone) if pd.notna(phone) else None
        prod = str(prod) if pd.notna(prod) else None
        purchase_dt = parse_date_value(d)
        purchase_str = purchase_dt.strftime("%Y-%m-%d") if isinstance(purchase_dt, pd.Timestamp) else None
        # dup checks
        def exists_phone(phone_value, purchase_value):
            normalized_phone = clean_text(phone_value)
            if not normalized_phone:
                return False
            if purchase_value:
                cur.execute(
                    "SELECT 1 FROM customers WHERE phone = ? AND IFNULL(purchase_date, '') = ? LIMIT 1",
                    (normalized_phone, purchase_value),
                )
            else:
                cur.execute(
                    "SELECT 1 FROM customers WHERE phone = ? AND (purchase_date IS NULL OR purchase_date = '') LIMIT 1",
                    (normalized_phone,),
                )
            return cur.fetchone() is not None

        dupc = 1 if exists_phone(phone, purchase_str) else 0
        cur.execute(
            "INSERT INTO customers (name, phone, address, dup_flag) VALUES (?, ?, ?, ?)",
            (cust, phone, addr, dupc),
        )
        cid = cur.lastrowid
        if dupc:
            d_c += 1
        if phone:
            normalized_phone = clean_text(phone)
            if normalized_phone:
                phones_to_recalc.add(normalized_phone)

        name, model = split_product_label(prod)

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
            "INSERT INTO products (name, model, dup_flag) VALUES (?, ?, ?)",
            (name, model, dupp),
        )
        pid = cur.lastrowid
        if dupp:
            d_p += 1

        # we still record orders (hidden) to keep a timeline if needed
        base_dt = purchase_dt or pd.Timestamp.now().normalize()
        order_date = base_dt
        delivery_date = base_dt
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
            "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)",
            (oid, pid, 1),
        )
        order_item_id = cur.lastrowid

        base = base_dt
        expiry = base + pd.Timedelta(days=365)
        cur.execute(
            "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status, dup_flag) VALUES (?, ?, ?, ?, ?, 'active', 0)",
            (cid, pid, None, base.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d")),
        )
        warranty_id = cur.lastrowid

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
        purchase_date = purchase_str or (base.strftime("%Y-%m-%d") if isinstance(base, pd.Timestamp) else None)
        cur.execute(
            "UPDATE customers SET purchase_date=?, product_info=?, delivery_order_code=? WHERE customer_id=?",
            (
                purchase_date,
                prod,
                do_serial,
                cid,
            ),
        )
        cur.execute(
            "INSERT INTO import_history (customer_id, product_id, order_id, order_item_id, warranty_id, do_number, import_tag, original_date, customer_name, address, phone, product_label, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid,
                pid,
                oid,
                order_item_id,
                warranty_id,
                do_serial,
                tag,
                purchase_date,
                cust,
                addr,
                phone,
                prod,
                None,
            ),
        )
        seeded += 1
    conn.commit()
    for p in phones_to_recalc:
        recalc_customer_duplicate_flag(conn, p)
    conn.commit()
    return seeded, d_c, d_p


def update_import_entry(conn, record: dict, updates: dict) -> None:
    cur = conn.cursor()
    import_id = int_or_none(record.get("import_id"))
    if import_id is None:
        return

    customer_id = int_or_none(record.get("customer_id"))
    product_id = int_or_none(record.get("product_id"))
    order_id = int_or_none(record.get("order_id"))
    order_item_id = int_or_none(record.get("order_item_id"))
    warranty_id = int_or_none(record.get("warranty_id"))

    old_phone = clean_text(record.get("live_phone")) or clean_text(record.get("phone"))
    old_do = clean_text(record.get("do_number"))

    new_name = clean_text(updates.get("customer_name"))
    new_phone = clean_text(updates.get("phone"))
    new_address = clean_text(updates.get("address"))
    purchase_date_str, expiry_str = date_strings_from_input(updates.get("purchase_date"))
    product_label = clean_text(updates.get("product_label"))
    new_do = clean_text(updates.get("do_number"))
    product_name, product_model = split_product_label(product_label)

    if customer_id is not None:
        cur.execute(
            "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
            (
                new_name,
                new_phone,
                new_address,
                purchase_date_str,
                product_label,
                new_do,
                customer_id,
            ),
        )

    if order_id is not None:
        cur.execute(
            "UPDATE orders SET order_date=?, delivery_date=? WHERE order_id=?",
            (purchase_date_str, purchase_date_str, order_id),
        )

    if order_item_id is not None:
            cur.execute(
                "UPDATE order_items SET quantity=? WHERE order_item_id=?",
                (1, order_item_id),
            )

    if product_id is not None:
            cur.execute(
                "UPDATE products SET name=?, model=? WHERE product_id=?",
                (product_name, product_model, product_id),
            )

    if warranty_id is not None:
        cur.execute(
            "UPDATE warranties SET issue_date=?, expiry_date=?, status='active' WHERE warranty_id=?",
            (purchase_date_str, expiry_str, warranty_id),
        )

    if new_do:
        cur.execute(
            """
            INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(do_number) DO UPDATE SET
                customer_id=excluded.customer_id,
                order_id=excluded.order_id,
                description=excluded.description
            """,
            (
                new_do,
                customer_id,
                order_id,
                product_label,
                None,
                None,
            ),
        )
    if old_do and old_do != new_do:
        params = [old_do]
        query = "DELETE FROM delivery_orders WHERE do_number=?"
        if order_id is not None:
            query += " AND (order_id IS NULL OR order_id=?)"
            params.append(order_id)
        cur.execute(query, tuple(params))

        cur.execute(
            "UPDATE import_history SET original_date=?, customer_name=?, address=?, phone=?, product_label=?, do_number=? WHERE import_id=?",
            (
                purchase_date_str,
                new_name,
                new_address,
                new_phone,
                product_label,
                new_do,
                import_id,
            ),
        )
    conn.commit()

    if old_phone and old_phone != new_phone:
        recalc_customer_duplicate_flag(conn, old_phone)
    if new_phone:
        recalc_customer_duplicate_flag(conn, new_phone)
    conn.commit()


def delete_import_entry(conn, record: dict) -> None:
    cur = conn.cursor()
    import_id = int_or_none(record.get("import_id"))
    if import_id is None:
        return

    customer_id = int_or_none(record.get("customer_id"))
    product_id = int_or_none(record.get("product_id"))
    order_id = int_or_none(record.get("order_id"))
    order_item_id = int_or_none(record.get("order_item_id"))
    warranty_id = int_or_none(record.get("warranty_id"))
    do_number = clean_text(record.get("do_number"))
    attachment_path = record.get("live_attachment_path")

    old_phone = clean_text(record.get("live_phone")) or clean_text(record.get("phone"))

    if do_number:
        params = [do_number]
        query = "DELETE FROM delivery_orders WHERE do_number=?"
        if order_id is not None:
            query += " AND (order_id IS NULL OR order_id=?)"
            params.append(order_id)
        cur.execute(query, tuple(params))

    if warranty_id is not None:
        cur.execute("DELETE FROM warranties WHERE warranty_id=?", (warranty_id,))
    if order_item_id is not None:
        cur.execute("DELETE FROM order_items WHERE order_item_id=?", (order_item_id,))
    if order_id is not None:
        cur.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
    if product_id is not None:
        cur.execute("DELETE FROM products WHERE product_id=?", (product_id,))
    if customer_id is not None:
        cur.execute("DELETE FROM customers WHERE customer_id=?", (customer_id,))

    cur.execute("UPDATE import_history SET deleted_at = datetime('now') WHERE import_id=?", (import_id,))
    conn.commit()

    if attachment_path:
        path = resolve_upload_path(attachment_path)
        if path and path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    if old_phone:
        recalc_customer_duplicate_flag(conn, old_phone)
        conn.commit()


def manage_import_history(conn):
    st.subheader("🗃️ Manage imported rows")
    hist = df_query(
        conn,
        """
        SELECT ih.*, c.name AS live_customer_name, c.address AS live_address, c.phone AS live_phone,
               c.purchase_date AS live_purchase_date, c.product_info AS live_product_info,
               c.delivery_order_code AS live_do_code, c.attachment_path AS live_attachment_path
        FROM import_history ih
        LEFT JOIN customers c ON c.customer_id = ih.customer_id
        WHERE ih.deleted_at IS NULL
        ORDER BY ih.import_id DESC
        LIMIT 200
        """,
    )
    if hist.empty:
        st.info("No imported rows yet. Upload a file to get started.")
        return

    display_cols = [
        "import_id",
        "import_tag",
        "imported_at",
        "customer_name",
        "phone",
        "product_label",
        "do_number",
    ]
    display = hist[display_cols].copy()
    display = fmt_dates(display, ["imported_at"])
    display.rename(
        columns={
            "import_id": "ID",
            "import_tag": "Tag",
            "imported_at": "Imported",
            "customer_name": "Customer",
            "phone": "Phone",
            "product_label": "Product",
            "do_number": "DO code",
        },
        inplace=True,
    )
    st.dataframe(display, use_container_width=True)

    ids = hist["import_id"].astype(int).tolist()
    label_map = {}
    for _, row in hist.iterrows():
        name = clean_text(row.get("customer_name")) or clean_text(row.get("live_customer_name")) or "(no name)"
        tag = clean_text(row.get("import_tag")) or "import"
        label_map[int(row["import_id"])] = f"#{int(row['import_id'])} – {name} ({tag})"

    selected_id = st.selectbox(
        "Select an import entry",
        ids,
        format_func=lambda x: label_map.get(int(x), str(x)),
    )
    selected = hist[hist["import_id"] == selected_id].iloc[0].to_dict()
    current_name = clean_text(selected.get("live_customer_name")) or clean_text(selected.get("customer_name")) or ""
    current_phone = clean_text(selected.get("live_phone")) or clean_text(selected.get("phone")) or ""
    current_address = clean_text(selected.get("live_address")) or clean_text(selected.get("address")) or ""
    current_product = clean_text(selected.get("live_product_info")) or clean_text(selected.get("product_label")) or ""
    current_do = clean_text(selected.get("live_do_code")) or clean_text(selected.get("do_number")) or ""
    purchase_seed = selected.get("live_purchase_date") or selected.get("original_date")
    purchase_str = clean_text(purchase_seed) or ""

    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"

    with st.form(f"manage_import_{selected_id}"):
        name_input = st.text_input("Customer name", value=current_name)
        phone_input = st.text_input("Phone", value=current_phone)
        address_input = st.text_area("Address", value=current_address)
        purchase_input = st.text_input("Purchase date (DD-MM-YYYY)", value=purchase_str)
        product_input = st.text_input("Product", value=current_product)
        do_input = st.text_input("Delivery order code", value=current_do)
        notes_input = st.text_area("Notes", value=clean_text(selected.get("notes")) or "", help="Optional remarks stored with this import entry.")
        col1, col2 = st.columns(2)
        save_btn = col1.form_submit_button("Save changes", type="primary")
        delete_btn = col2.form_submit_button("Delete import", disabled=not is_admin)

    if save_btn:
        update_import_entry(
            conn,
            selected,
            {
                "customer_name": name_input,
                "phone": phone_input,
                "address": address_input,
                "purchase_date": purchase_input,
                "product_label": product_input,
                "do_number": do_input,
            },
        )
        conn.execute(
            "UPDATE import_history SET notes=? WHERE import_id=?",
            (clean_text(notes_input), int(selected_id)),
        )
        conn.commit()
        st.success("Import entry updated.")
        _safe_rerun()

    if delete_btn and is_admin:
        delete_import_entry(conn, selected)
        st.warning("Import entry deleted.")
        _safe_rerun()
    elif delete_btn and not is_admin:
        st.error("Only admins can delete import rows.")
# ---------- Main ----------
def main():
    init_ui()
    conn = get_conn()
    init_schema(conn)
    login_box(conn)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    user = st.session_state.user or {}
    role = user.get("role")
    with st.sidebar:
        if role == "admin":
            pages = [
                "Dashboard",
                "Customers",
                "Customer Summary",
                "Scraps",
                "Warranties",
                "Import",
                "Duplicates",
                "Users (Admin)",
                "Service & Maintenance",
            ]
        else:
            pages = ["Dashboard", "Warranties", "Import", "Service & Maintenance"]
        if st.session_state.page not in pages:
            st.session_state.page = pages[0]
        current_index = pages.index(st.session_state.page)
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
    elif page == "Service & Maintenance":
        service_maintenance_page(conn)

if _streamlit_runtime_active():
    main()
elif __name__ == "__main__":
    _bootstrap_streamlit_app()
