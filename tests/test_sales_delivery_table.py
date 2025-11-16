import importlib
import sqlite3
import sys

import pytest


@pytest.fixture()
def sales_module(monkeypatch, tmp_path):
    monkeypatch.setenv("PS_SALES_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PS_SALES_DB_URL", f"sqlite:///{(tmp_path / 'suite.db').as_posix()}")
    module_name = "ps_business_suite.core.sales"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_delivery_table_created_when_crm_table_exists(sales_module):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE delivery_orders (
            do_number TEXT PRIMARY KEY,
            customer_id INTEGER,
            order_id INTEGER,
            description TEXT
        )
        """
    )

    sales_module._ensure_sales_delivery_orders_table(cur)

    cur.execute("SELECT name FROM sqlite_master WHERE name='delivery_orders'")
    assert cur.fetchone() is not None

    cur.execute(
        "SELECT name FROM sqlite_master WHERE name=?",
        (sales_module.DELIVERY_TABLE,),
    )
    assert cur.fetchone() is not None

    cur.execute(f"PRAGMA table_info({sales_module.DELIVERY_TABLE})")
    columns = {row[1] for row in cur.fetchall()}
    assert {"source_type", "salesperson_id", "receipt_path"}.issubset(columns)

    conn.close()


def test_legacy_sales_delivery_table_is_migrated(sales_module):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE quotations (
            quotation_id INTEGER PRIMARY KEY,
            salesperson_id INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE work_orders (
            work_order_id INTEGER PRIMARY KEY,
            quotation_id INTEGER
        )
        """
    )
    cur.execute("INSERT INTO quotations(quotation_id, salesperson_id) VALUES (1, 7)")
    cur.execute("INSERT INTO work_orders(work_order_id, quotation_id) VALUES (5, 1)")

    cur.execute(
        """
        CREATE TABLE delivery_orders (
            do_id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_order_id INTEGER NOT NULL,
            do_number TEXT NOT NULL,
            upload_date TEXT,
            pdf_path TEXT,
            price REAL DEFAULT 0,
            payment_received INTEGER DEFAULT 0,
            payment_date TEXT,
            notes TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        INSERT INTO delivery_orders (work_order_id, do_number, upload_date, price)
        VALUES (5, 'DO-1', '2024-01-01', 123.0)
        """
    )

    sales_module._ensure_sales_delivery_orders_table(cur)

    cur.execute(
        "SELECT name FROM sqlite_master WHERE name='delivery_orders'"
    )
    assert cur.fetchone() is None

    cur.execute(f"PRAGMA table_info({sales_module.DELIVERY_TABLE})")
    columns = {row[1] for row in cur.fetchall()}
    assert {"source_type", "quotation_id", "salesperson_id"}.issubset(columns)

    row = cur.execute(
        f"SELECT source_type, quotation_id, price FROM {sales_module.DELIVERY_TABLE}"
    ).fetchone()
    assert row[0] == "work_order"
    assert row[1] == 1
    assert pytest.approx(row[2], rel=1e-9) == 123.0

    conn.close()
