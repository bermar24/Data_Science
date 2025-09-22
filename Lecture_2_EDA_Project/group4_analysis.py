import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def iqr_filter(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    return (series >= lower) & (series <= upper)

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    num = ["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors",
           "waterfront","view","condition","sqft_above","sqft_basement",
           "yr_built","yr_renovated"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=[c for c in ["price","sqft_living","yr_built","city"] if c in df.columns])
    df = df[iqr_filter(df["price"]) & iqr_filter(df["sqft_living"])].copy()
    return df

def add_group4_flags(df):
    df["city_lower"] = df["city"].str.lower().str.strip()
    urban_cities = {"seattle","bellevue","redmond","kirkland"}
    df["is_urban"] = df["city_lower"].isin(urban_cities)
    distance_proxy = {"seattle":5,"bellevue":11,"kirkland":14,"redmond":18}
    df["distance_proxy_km"] = df["city_lower"].map(distance_proxy).fillna(28)
    df["is_modern"] = df["yr_built"] >= 2000
    if "floors" in df.columns:
        df["is_condo_like"] = (df["sqft_living"] <= df["sqft_living"].median()) & (df["floors"] <= df["floors"].median())
    else:
        df["is_condo_like"] = (df["sqft_living"] <= df["sqft_living"].median())
    return df

def hypothesis_tests(df):
    res = {}
    urban = df.loc[df["is_urban"], "price"]
    suburb = df.loc[~df["is_urban"], "price"]
    t, p2 = stats.ttest_ind(urban, suburb, equal_var=False, nan_policy="omit")
    md = urban.mean() - suburb.mean()
    p1 = p2/2.0 if md > 0 else 1 - (p2/2.0)
    res["Urban vs Suburb"] = dict(t_stat=float(t), p_value_one_sided=float(p1), mean_diff=float(md))
    modern = df.loc[df["yr_built"] >= 2000, "price"]
    old = df.loc[df["yr_built"] < 2000, "price"]
    t2, p22 = stats.ttest_ind(modern, old, equal_var=False, nan_policy="omit")
    md2 = modern.mean() - old.mean()
    p12 = p22/2.0 if md2 > 0 else 1 - (p22/2.0)
    res["Modern vs Old"] = dict(t_stat=float(t2), p_value_one_sided=float(p12), mean_diff=float(md2))
    return res

def make_plots(df, outdir="."):
    import os
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    df["price"].plot(kind="hist", bins=40, edgecolor="white")
    plt.title("Distribution of Home Prices")
    plt.xlabel("Price (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_price_hist.png"))
    plt.figure()
    plt.scatter(df["distance_proxy_km"], df["price"], s=10, alpha=0.6)
    plt.title("Price vs Distance-to-Tech-Hub (proxy)")
    plt.xlabel("Distance to Seattle/Tech Hub (proxy km)")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_price_vs_distance_proxy.png"))
    city_stats = df.groupby("city", as_index=False).agg(count=("price","size"),avg_price=("price","mean")).sort_values("count", ascending=False).head(10)
    plt.figure()
    plt.bar(city_stats["city"], city_stats["avg_price"])
    plt.title("Average Price by City (Top 10 by Listing Count)")
    plt.xlabel("City")
    plt.ylabel("Average Price (USD)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_avg_price_by_city.png"))

if __name__ == "__main__":
    df = load_and_clean("KC_housing_data.csv")
    df = add_group4_flags(df)
    res = hypothesis_tests(df)
    make_plots(df, outdir="figs")
    print(res)
