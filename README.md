# PS Business Suite

The PS Business Suite merges the legacy **PS Mini CRM** and **PS Sales Manager** systems into a single multi-page Streamlit experience powered by one SQLite database. Admins and staff log in once to work on customers, service operations, sales quotations, letters, and analytics without switching apps.

Every Streamlit page shows the credit text **“by Zad”** in a compact footer as requested.

## Project layout

```
ps_business_suite/
├── app.py                 # Dashboard landing page
├── core/
│   ├── __init__.py
│   ├── auth.py (future expansion placeholder)
│   ├── config.py          # Runtime settings & storage paths
│   ├── crm.py             # Legacy CRM logic (customers, warranties, service, reports)
│   ├── db.py              # Shared SQLite helpers + schema bootstrapper
│   ├── sales.py           # Legacy PS-SALES UI and services
│   └── utils.py           # UI helpers (includes the “by Zad” footer)
├── pages/
│   ├── dashboard.py       # CRM dashboard
│   ├── customers.py       # Customers/Warranties/Imports tabs
│   ├── database.py        # SQLite inspector & backup tools
│   ├── sales.py           # PS Sales Manager workspace
│   ├── reports.py         # Work reports
│   └── admin.py           # User admin utilities
└── static/
    ├── css/
    └── uploads/
```

The shared SQLite file defaults to `<APP_STORAGE_DIR>/ps_business_suite.db` (per `storage_paths.py`). Both CRM and Sales schemas live inside this database.

## Running locally

1. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Launch Streamlit:

   ```bash
   streamlit run ps_business_suite/app.py
   ```

3. Sign in with the default admin (`admin` / `admin123`). Create staff accounts from **Admin ➜ Users**.

The app automatically creates the SQLite schema, uploads folders, and default lookups on first launch.

## Railway deployment

1. Push this repository to GitHub.
2. Create a new Railway project and connect it to the repo.
3. Set the service environment variables if you need custom paths:
   - `APP_STORAGE_DIR` – persistent volume mount (`/data/ps-suite` recommended)
   - `ADMIN_USER`, `ADMIN_PASS` – bootstrap credentials
4. Railway detects the `Procfile` and runs `streamlit run ps_business_suite/app.py` listening on `$PORT`.

## Database migrations

The CRM logic now ensures the shared `users` table contains the extra `display_name`, `designation`, and `phone` columns required by the Sales module. On startup the app:

1. Runs `ps_business_suite/core/crm.init_schema(conn)` to create/upgrade CRM tables.
2. Runs `ps_business_suite/core/sales.init_schema(conn)` to create Sales tables in the same file.
3. Keeps uploads under `<APP_STORAGE_DIR>/uploads` and Sales attachments under `<APP_STORAGE_DIR>/sales_data/uploads`.

To migrate an existing PS Mini CRM database, copy `ps_crm.db` to the storage directory and rename it to `ps_business_suite.db`. Run the new app once; the upgrade routine will add the new Sales columns automatically.

## Combined features

- CRM: Dashboards, customers, warranty tracking, maintenance/service tickets, Excel importer, duplicate inspector, work reports, and admin tools.
- Sales: Quotation letters, quotations, work/delivery orders, document uploads, payment tracking, notifications, and account lockout rules.
- Shared login with admin/staff roles and one credentials table.
- Database maintenance page to download backups or vacuum the file.

## Tests

Use `python -m compileall ps_business_suite` for a quick syntax validation pass. Full unit tests from each legacy project still run under `pytest` if needed.
