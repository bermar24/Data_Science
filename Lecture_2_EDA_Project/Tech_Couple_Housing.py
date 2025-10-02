import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from IPython.core.display_functions import display
from scipy import stats

# === Paths ===
# If you're on Windows, you can set e.g. Path(r"C:\\Users\\you\\Desktop\\da\\KC_housing_data.csv") or use forward slashes.
DATA_PATH = Path('./KC_housing_data.csv')
CLEAN_PATH = Path('./KC_housing_clean.csv')
SHORTLIST_PATH = Path('.shortlist_group4.csv')

plt.rcParams.update({'figure.figsize': (8,5)})

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
df.head()


desc_numeric_before = df.describe().T
missing_before = df.isna().sum().sort_values(ascending=False)

print("Descriptive statistics (numeric, before cleaning):")
display(desc_numeric_before)

print("\nMissing values per column (before cleaning):")
display(missing_before)


# Date
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ZIP extraction from 'statezip' (e.g., 'WA 98119' -> '98119')
def extract_zip(s):
    if pd.isna(s):
        return np.nan
    parts = str(s).split()
    for p in parts[::-1]:
        if p.isdigit():
            return p
    return np.nan

df['zip'] = df['statezip'].apply(extract_zip)

# Price per sqft
df['price_psf'] = df['price'] / df['sqft_living'].replace(0, np.nan)

# City normalization
df['city'] = df['city'].astype(str).str.title()

df[['date','zip','price','sqft_living','price_psf','bedrooms','bathrooms']].head()

box_cols = ['price','price_psf','sqft_living','sqft_lot','bedrooms','bathrooms']
for c in box_cols:
    plt.figure()
    plt.boxplot(df[c].dropna(), vert=True, showfliers=True)
    plt.title(f'Boxplot BEFORE cleaning — {c}')
    plt.ylabel(c)
    plt.show()


def iqr_bounds(s, k=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr

df_iqr = df.copy()
removed_stats = []
for c in box_cols:
    s = df_iqr[c].dropna()
    if s.empty:
        continue
    lo, up = iqr_bounds(s, k=1.5)
    before_n = len(df_iqr)
    df_iqr = df_iqr[(df_iqr[c].isna()) | ((df_iqr[c] >= lo) & (df_iqr[c] <= up))]
    after_n = len(df_iqr)
    removed_stats.append({'column': c, 'lower': lo, 'upper': up, 'removed_rows': before_n - after_n})
removed_log = pd.DataFrame(removed_stats)
print("IQR removal summary:")
display(removed_log)
print("Shape after IQR removal:", df_iqr.shape)


for c in box_cols:
    plt.figure()
    plt.boxplot(df_iqr[c].dropna(), vert=True, showfliers=True)
    plt.title(f'Boxplot AFTER IQR removal — {c}')
    plt.ylabel(c)
    plt.show()


def apply_anomaly_rules(frame):
    rules = []
    cur = frame.copy()
    def apply_and_log(mask, reason):
        nonlocal cur, rules
        removed = cur[~mask]
        rules.append({'rule': reason, 'removed_count': removed.shape[0]})
        cur = cur[mask]
    # Domain sanity checks (tweakable)
    apply_and_log(cur['price'] > 50000, "price > 50,000")
    apply_and_log(cur['sqft_living'] > 200, "sqft_living > 200")
    apply_and_log(cur['bedrooms'] > 0, "bedrooms > 0")
    apply_and_log(cur['bathrooms'] > 0, "bathrooms > 0")
    apply_and_log(cur['floors'] > 0, "floors > 0")
    return cur, pd.DataFrame(rules)

df_clean, anomaly_log = apply_anomaly_rules(df_iqr)
print("Anomaly filtering summary:")
display(anomaly_log)
print("Final shape:", df_clean.shape)


final_desc = df_clean.describe().T
display(final_desc)
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"Saved cleaned dataset to: {CLEAN_PATH}")


seattle = df_clean[df_clean['city']=="Seattle"].copy()
zip_med_psf = (seattle.groupby('zip')['price_psf']
               .median().dropna().sort_values(ascending=False).head(15))

plt.figure()
zip_med_psf.sort_values().plot(kind='barh')
plt.title('Seattle — Top 15 ZIPs by Median Price per Sqft')
plt.xlabel('Median Price per Sqft'); plt.ylabel('ZIP')
plt.tight_layout(); plt.show()

zip_med_psf.head()


sea = seattle.copy()
sea['ym'] = sea['date'].dt.to_period('M').astype(str)
monthly = sea.groupby('ym').agg(median_price=('price','median'),
                                sales=('price','count')).reset_index()
monthly['ym'] = pd.to_datetime(monthly['ym'])

plt.figure()
plt.plot(monthly['ym'], monthly['median_price'])
plt.title('Seattle — Monthly Median Price')
plt.xlabel('Year-Month'); plt.ylabel('Median Price')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(monthly['ym'], monthly['sales'])
plt.title('Seattle — Monthly Sales Count')
plt.xlabel('Year-Month'); plt.ylabel('Sales Count')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

monthly.tail()


proxy = seattle[
    (seattle['sqft_lot'] <= 1200) &
    ((seattle['yr_built'] >= 2000) | (seattle['yr_renovated'] >= 2000)) &
    (seattle['sqft_living'].between(600,1100)) &
    (seattle['bedrooms'] <= 2)
    ].copy()

def summarize(frame, name):
    return pd.Series({
        'count': len(frame),
        'median_price': frame['price'].median(),
        'median_psf': frame['price_psf'].median(),
        'median_sqft': frame['sqft_living'].median(),
        'median_yr_built': frame['yr_built'].median()
    }, name=name)

summary_tbl = pd.concat([summarize(seattle,'Seattle All'),
                         summarize(proxy,'Proxy Condo')], axis=1)
display(summary_tbl)

pz = proxy.groupby('zip')['price_psf'].median().dropna().sort_values()
plt.figure()
pz.plot(kind='bar', rot=90)
plt.title('Proxy Condo — Median PSF by ZIP')
plt.xlabel('ZIP'); plt.ylabel('Median PSF')
plt.tight_layout(); plt.show()

# Core vs Non-core
core_zips = ['98101','98109','98121','98122','98119']
seattle['is_core'] = seattle['zip'].isin(core_zips)

core_psf = seattle[seattle['is_core']]['price_psf'].dropna()
noncore_psf = seattle[~seattle['is_core']]['price_psf'].dropna()

t1, p1 = stats.ttest_ind(core_psf, noncore_psf, equal_var=False)
print("Welch's t-test (core vs non-core PSF): t =", t1, "p =", p1)

# New vs Old
seattle['is_new'] = (seattle['yr_built'] >= 2000) | (seattle['yr_renovated'] >= 2000)
new_psf = seattle[seattle['is_new']]['price_psf'].dropna()
old_psf = seattle[~seattle['is_new']]['price_psf'].dropna()

t2, p2 = stats.ttest_ind(new_psf, old_psf, equal_var=False)
print("Welch's t-test (new vs old PSF): t =", t2, "p =", p2)

# ANOVA across months
sea['month'] = sea['date'].dt.month
groups = [g['price'].dropna().values for _, g in sea.groupby('month') if len(g)>=20]
if len(groups) >= 2:
    f3, p3 = stats.f_oneway(*groups)
    print("ANOVA (price across months): F =", f3, "p =", p3)
else:
    print("Not enough monthly groups for ANOVA.")



# Core vs Non-Core boxplot
plt.figure()
seattle.boxplot(column='price_psf', by='is_core')
plt.title('Price per Sqft — Core vs Non-Core (Seattle)')
plt.suptitle("")
plt.xlabel('Core Area?'); plt.ylabel('Price per Sqft')
plt.show()

# New vs Old boxplot
plt.figure()
seattle.boxplot(column='price_psf', by='is_new')
plt.title('Price per Sqft — New vs Old (Seattle)')
plt.suptitle("")
plt.xlabel('New Build (>=2000)?'); plt.ylabel('Price per Sqft')
plt.show()

# Seasonality (two charts for clarity)
plt.figure()
plt.plot(monthly['ym'], monthly['median_price'])
plt.title('Seattle — Monthly Median Price')
plt.xlabel('Year-Month'); plt.ylabel('Median Price')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(monthly['ym'], monthly['sales'])
plt.title('Seattle — Monthly Sales Count')
plt.xlabel('Year-Month'); plt.ylabel('Sales Count')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()


proxy['rank'] = proxy['price_psf'].rank(method='min')*0.6 + proxy['price'].rank(method='min')*0.4
cols = ['date','price','price_psf','bedrooms','bathrooms','sqft_living','sqft_lot',
        'yr_built','yr_renovated','street','zip']
shortlist = proxy.sort_values('rank').head(20)[cols].reset_index(drop=True)
shortlist.to_csv(SHORTLIST_PATH, index=False)
print(f"Saved shortlist to: {SHORTLIST_PATH}")
shortlist.head(10)




















