import hashlib
import io

import pandas as pd


def test_admin_user_seeded(db_conn, app_module):
    cur = db_conn.execute("SELECT username, pass_hash, role FROM users")
    row = cur.fetchone()
    assert row == ("test_admin", hashlib.sha256("secret123".encode("utf-8")).hexdigest(), "admin")


def test_customer_creation_and_duplicate_flag(db_conn, app_module):
    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO customers (name, phone, email, address, dup_flag) VALUES (?, ?, ?, ?, 0)",
        ("Alice", "555-0000", "alice@example.com", "123 Road"),
    )
    cur.execute(
        "INSERT INTO customers (name, phone, email, address, dup_flag) VALUES (?, ?, ?, ?, 0)",
        ("Bob", "555-0000", "bob@example.com", "456 Lane"),
    )
    db_conn.commit()

    app_module.recalc_customer_duplicate_flag(db_conn, "555-0000")

    complete_count = db_conn.execute(
        f"SELECT COUNT(*) FROM customers WHERE {app_module.customer_complete_clause()}"
    ).fetchone()[0]
    assert complete_count == 2

    flags = [row[0] for row in db_conn.execute("SELECT dup_flag FROM customers WHERE phone=?", ("555-0000",))]
    assert flags == [1, 1]


def test_scrap_record_completion_moves_out_of_scraps(db_conn, app_module):
    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO customers (name, phone, email, address, dup_flag) VALUES (?, ?, ?, ?, 0)",
        ("Scrappy", None, None, ""),
    )
    scrap_id = cur.lastrowid
    db_conn.commit()

    incomplete_before = db_conn.execute(
        f"SELECT COUNT(*) FROM customers WHERE {app_module.customer_incomplete_clause()}"
    ).fetchone()[0]
    assert incomplete_before == 1

    new_name = app_module.clean_text(" Scrappy Doo ")
    new_phone = app_module.clean_text("777-8888")
    new_address = app_module.clean_text(" 42 Hero Lane ")
    db_conn.execute(
        "UPDATE customers SET name=?, phone=?, address=?, dup_flag=0 WHERE customer_id=?",
        (new_name, new_phone, new_address, scrap_id),
    )
    app_module.recalc_customer_duplicate_flag(db_conn, new_phone)
    db_conn.commit()

    incomplete_after = db_conn.execute(
        f"SELECT COUNT(*) FROM customers WHERE {app_module.customer_incomplete_clause()}"
    ).fetchone()[0]
    assert incomplete_after == 0

    complete_after = db_conn.execute(
        f"SELECT COUNT(*) FROM customers WHERE {app_module.customer_complete_clause()}"
    ).fetchone()[0]
    assert complete_after == 1


def test_streamlit_flag_options_from_env_uses_port_and_host(monkeypatch, app_module):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("HOST", "1.2.3.4")
    flags = app_module._streamlit_flag_options_from_env()
    assert flags["server.port"] == 9999
    assert flags["server.address"] == "1.2.3.4"
    assert flags["server.headless"] is True


def test_streamlit_flag_options_from_env_respects_headless(monkeypatch, app_module):
    monkeypatch.setenv("STREAMLIT_SERVER_HEADLESS", "false")
    flags = app_module._streamlit_flag_options_from_env()
    assert flags["server.headless"] is False


def test_streamlit_flag_options_from_env_handles_invalid_port(monkeypatch, app_module):
    monkeypatch.setenv("PORT", "not-a-number")
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("BIND_ADDRESS", raising=False)
    monkeypatch.delenv("RENDER_EXTERNAL_HOSTNAME", raising=False)
    monkeypatch.delenv("STREAMLIT_SERVER_HEADLESS", raising=False)
    flags = app_module._streamlit_flag_options_from_env()
    assert "server.port" not in flags
    assert flags["server.address"] == "0.0.0.0"
    assert flags["server.headless"] is True


def test_export_database_to_excel_has_curated_sheets(db_conn, app_module):
    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO customers (name, phone, email, address, dup_flag) VALUES (?, ?, ?, ?, 0)",
        ("Charlie", "123", "charlie@example.com", "42 Test Way"),
    )
    customer_id = cur.lastrowid
    cur.execute(
        "INSERT INTO products (name, model, serial, dup_flag) VALUES (?, ?, ?, 0)",
        ("Air Conditioner", "AC-01", "SER123"),
    )
    product_id = cur.lastrowid
    cur.execute(
        "INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person) VALUES (?, ?, NULL, ?, ?)",
        ("DO-1", customer_id, "Main unit", "Sam"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (customer_id, product_id, "SER123", "2024-01-01", "2025-01-01", "active"),
    )
    cur.execute(
        "INSERT INTO services (do_number, customer_id, service_date, description, remarks) VALUES (?, ?, ?, ?, ?)",
        ("DO-1", customer_id, "2024-06-01", "Installation", "All good"),
    )
    cur.execute(
        "INSERT INTO maintenance_records (do_number, customer_id, maintenance_date, description, remarks) VALUES (?, ?, ?, ?, ?)",
        ("DO-1", customer_id, "2024-07-01", "Checkup", "No issues"),
    )
    db_conn.commit()

    excel_bytes = app_module.export_database_to_excel(db_conn)
    workbook = pd.ExcelFile(io.BytesIO(excel_bytes))

    assert workbook.sheet_names == [
        "Master",
        "Customers",
        "Delivery orders",
        "Warranties",
        "Services",
        "Maintenance",
    ]

    master_df = workbook.parse("Master")
    assert list(master_df.columns) == ["Sheet", "Details"]
    summary = dict(zip(master_df["Sheet"], master_df["Details"]))
    assert "Customers" in summary and summary["Customers"].startswith("1 ")
    assert "Warranties" in summary and summary["Warranties"].startswith("1 ")
