
import pandas as pd
import numpy as np

def load_data(file_url_meta, file_url_sales):
    raw_meta_data = pd.read_excel(file_url_meta, sheet_name=None)
    raw_sales_data = pd.read_excel(file_url_sales, sheet_name=None)
    return raw_meta_data, raw_sales_data

def clean_data(raw_meta_data, raw_sales_data):
    category_map = {
        "F - Adult Fiction": "F",
        "S Adult Non-Fiction Specialist": "S",
        "T Adult Non-Fiction Trade": "T",
        "Y Childrens, YA & Educational": "Y_YA"
    }
    for sheet, cat in category_map.items():
        raw_meta_data[sheet]["product_category"] = cat
        raw_sales_data[sheet]["product_category"] = cat

    meta_data = pd.concat(raw_meta_data.values(), ignore_index=True)
    sales_data = pd.concat(raw_sales_data.values(), ignore_index=True)

    meta_data = meta_data.rename(columns={
        "ISBN": "isbn", "Title": "title", "Author": "author",
        "Imprint": "imprint", "Publisher Group": "publisher",
        "RRP": "rrp_gbp", "Binding": "binding",
        "Publication Date": "pub_date", "Product Class": "product_class",
        "Country of Publication": "origin_country"
    })

    sales_data = sales_data.rename(columns={
        "ISBN": "isbn", "Title": "title", "Author": "author",
        "Interval": "interval_weeks", "End Date": "period_end",
        "Volume": "units_sold", "Value": "revenue_gbp",
        "ASP": "asp_gbp", "RRP": "rrp_gbp", "Binding": "binding",
        "Imprint": "imprint", "Publisher Group": "publisher",
        "Product Class": "product_class"
    })

    meta_data = meta_data.drop("product_category", axis=1)
    sales_data = sales_data.drop("product_category", axis=1)

    meta_data["isbn"] = meta_data["isbn"].astype(str)
    sales_data["isbn"] = sales_data["isbn"].astype(str)

    sales_data["period_end"] = pd.to_datetime(sales_data["period_end"], errors="coerce")
    sales_data = sales_data.set_index("period_end").sort_index()

    return meta_data, sales_data

def get_book_sales(sales_data, meta_data, title, cutoff_date, binding=None):
    isbns = meta_data[meta_data["title"].str.contains(title, case=False, na=False)]["isbn"]
    mask = sales_data["isbn"].isin(isbns)
    book_sales = sales_data[mask].loc[cutoff_date:]

    if binding:
        book_sales = book_sales[book_sales["binding"] == binding]

    book_sales = (
        book_sales["units_sold"]
        .resample("W-SAT")
        .sum()
        .fillna(0)
    )
    return pd.DataFrame(book_sales)
