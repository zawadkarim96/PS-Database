# PS Mini CRM v7.4
- Fixed syntax and navigation chain
- Dashboard shows only upcoming warranties (3 & 60 days)
- Removed Products/Orders pages, kept tables for import integrity
- Customers page supports Needs (product/unit)
- DD-MM-YYYY display, flexible importer + Excel serial dates
- Login safe rerun for all Streamlit versions

v7.5: Fixed selectbox errors; 'Add Customer' now supports Purchase/Warranty fields (date, product, model, serial, unit) and auto-creates warranty.
v7.6: Admins can bulk-delete customers from the Customers page with a select-all option.
v7.7: Importer allows blank column mappings and optional skipping of blank rows. Dashboard counts active warranties by expiry date. Customer summaries merge duplicate records and report how many were combined.
v7.8: Customer summary groups unnamed customers under "(blank)" so their contact info is still accessible.