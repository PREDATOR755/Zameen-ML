"""
train.py — Automated Retraining Pipeline
========================================
Reads the latest zameen_islamabad.csv, engineers geospatial features,
trains the Random Forest model, and saves the artifacts.
"""

import re
import time
import warnings
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

DATA_FILE = "zameen_islamabad.csv"   # Output of the scraper

# ──────────────────────────────────────────────────────────────────────────
# 1. Price parsing (handles "Crore", "Lakh", "Million")
# ──────────────────────────────────────────────────────────────────────────
def parse_price(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).lower().replace("pkr", "").replace(",", "").strip()
    multipliers = {"crore": 1e7, "lakh": 1e5, "million": 1e6, "billion": 1e9}
    for word, factor in multipliers.items():
        if word in s:
            try:
                num_str = re.sub(word, "", s).strip()
                return float(num_str) * factor
            except:
                return np.nan
    try:
        return float(s)
    except:
        return np.nan

# ──────────────────────────────────────────────────────────────────────────
# 2. Area conversion (Kanal → Marla, Sq.Ft. → Marla)
# ──────────────────────────────────────────────────────────────────────────
def convert_area_to_marla(row):
    area = pd.to_numeric(row.get("area"), errors="coerce")
    unit = row.get("area_unit", "Marla")
    if pd.isna(area):
        return np.nan
    unit_clean = str(unit).lower().strip()
    if unit_clean in ["kanal"]:
        return area * 20.0
    if unit_clean in ["sq. ft.", "sq ft", "square feet", "sqft"]:
        return area / 272.25
    return area   # assume already in Marla

# ──────────────────────────────────────────────────────────────────────────
# 3. Location cleaning (removes trailing area noise)
# ──────────────────────────────────────────────────────────────────────────
def clean_location(loc):
    if pd.isna(loc):
        return "Unknown"
    s = str(loc)
    # Remove noise from scraper (newlines, area leftovers)
    s = re.sub(r"\n.*$", "", s)
    s = re.sub(r"\d+[.\d]*\s*(Marla|Kanal)", "", s, flags=re.I)
    s = re.sub(r",?\s*Islamabad\s*\d*\s*(Marla|Kanal)?", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip().strip(",")
    return s if s else "Unknown"

# ──────────────────────────────────────────────────────────────────────────
# 4. Load, clean, and prepare the dataset
# ──────────────────────────────────────────────────────────────────────────
def load_and_clean():
    print(f"📥 Loading freshly scraped data from {DATA_FILE}...")
    try:
        raw_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"🚨 Cannot find {DATA_FILE}. Did the scraper run successfully?")

    # Parse price and area using the correct units
    raw_df["price_pkr"] = raw_df["price"].apply(parse_price)
    raw_df["area_marla"] = raw_df.apply(convert_area_to_marla, axis=1)

    # Drop rows without essential info
    raw_df = raw_df.dropna(subset=["price_pkr", "area_marla"])

    # Clean location string
    raw_df["location_clean"] = raw_df["location"].fillna("Unknown").apply(clean_location)

    # Keep only houses (property_type may be missing; assume "House" if empty)
    if "property_type" in raw_df.columns:
        raw_df = raw_df[raw_df["property_type"].str.lower() == "house"].copy()

    # Basic outlier filtering (price and area)
    df = raw_df[(raw_df["price_pkr"] >= 5_000_000) & (raw_df["area_marla"] <= 500)].copy().reset_index(drop=True)

    # =====================================================================
    # CRITICAL FIX: Minimum Frequency Filtering
    # Keep only locations with 5 or more listings to stabilize the model
    # =====================================================================
    loc_counts = df["location_clean"].value_counts()
    valid_locations = loc_counts[loc_counts >= 5].index
    df = df[df["location_clean"].isin(valid_locations)].copy().reset_index(drop=True)

    # Fill missing amenity columns with 0 (matching UI expectations)
    amenity_cols = ["parking_spaces", "servant_quarters", "bedrooms", "bathrooms", "kitchens"]
    for col in amenity_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0  # create column if missing

    # Built year fallback
    if "built_in_year" in df.columns:
        df["built_in_year"] = pd.to_numeric(df["built_in_year"], errors="coerce").fillna(2015)
    else:
        df["built_in_year"] = 2015

    # Remove extreme price outliers using IQR on log scale
    y_log = np.log1p(df["price_pkr"])
    q1, q3 = y_log.quantile(0.25), y_log.quantile(0.75)
    iqr = q3 - q1
    df = df[(y_log >= q1 - 2 * iqr) & (y_log <= q3 + 2 * iqr)].reset_index(drop=True)

    print(f"✅ After cleaning: {len(df)} rows, {df['location_clean'].nunique()} robust sectors.")
    return df

# ──────────────────────────────────────────────────────────────────────────
# 5. Geocode unique locations (cached to avoid repeated API calls)
# ──────────────────────────────────────────────────────────────────────────
def get_spatial_data(unique_locs):
    print("🌍 Fetching Geopy coordinates (respecting API limits)...")
    geolocator = Nominatim(user_agent="islamabad_ml_portfolio_v2")
    BLUE_AREA = (33.7077, 73.0498)
    METRO = (33.7075, 73.0501)

    db = {}
    for loc in unique_locs:
        try:
            loc_obj = geolocator.geocode(f"{loc}, Islamabad, Pakistan", timeout=5)
            if loc_obj:
                coords = (loc_obj.latitude, loc_obj.longitude)
                db[loc] = {
                    "dist_blue_area": geodesic(coords, BLUE_AREA).kilometers,
                    "dist_metro": geodesic(coords, METRO).kilometers
                }
            else:
                db[loc] = {"dist_blue_area": 12.0, "dist_metro": 4.5}
        except Exception:
            db[loc] = {"dist_blue_area": 12.0, "dist_metro": 4.5}
        time.sleep(1.2)   # Be nice to Nominatim
    return db

# ──────────────────────────────────────────────────────────────────────────
# 6. Feature engineering (mirrors the Streamlit app exactly)
# ──────────────────────────────────────────────────────────────────────────
def build_features(df, loc_map, glob_med, spatial_db):
    X = pd.DataFrame(index=range(len(df)))
    area = df["area_marla"].values

    X["log_area"] = np.log1p(area)
    X["area_marla"] = area
    X["area_sq"] = area ** 2
    X["area_band"] = pd.cut(pd.Series(area), bins=[0,4,7,12,22,50,500],
                            labels=[1,2,3,4,5,6]).astype(float).values

    # Amenities
    for col in ["parking_spaces", "servant_quarters", "built_in_year", "bedrooms", "bathrooms", "kitchens"]:
        X[col] = df[col].values

    # Geospatial distances
    X["dist_blue_area"] = df["location_clean"].map(lambda x: spatial_db.get(x, {"dist_blue_area": 12.0})["dist_blue_area"]).values
    X["dist_metro"] = df["location_clean"].map(lambda x: spatial_db.get(x, {"dist_metro": 4.5})["dist_metro"]).values

    # Target encoding (leakage‑free: fit on train only, but here we pass pre‑computed maps)
    raw_te = df["location_clean"].map(loc_map).fillna(glob_med).values
    X["loc_te"] = raw_te
    X["log_loc_te"] = np.log1p(raw_te)
    X["area_x_loc"] = area * np.log1p(raw_te) / 10.0

    return X

# ──────────────────────────────────────────────────────────────────────────
# 7. Main training pipeline
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("🚀 Starting Automated Retraining Pipeline...")
    df = load_and_clean()

    # Geocode unique locations
    unique_locs = df["location_clean"].unique()
    spatial_db = get_spatial_data(unique_locs)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Target encoding maps (based only on training data)
    loc_map = train_df.groupby("location_clean")["price_pkr"].median()
    glob_med = train_df["price_pkr"].median()

    X_train = build_features(train_df, loc_map, glob_med, spatial_db)
    X_test = build_features(test_df, loc_map, glob_med, spatial_db)
    y_train = np.log1p(train_df["price_pkr"])
    y_test = np.log1p(test_df["price_pkr"])

    print("🌲 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        max_features=0.6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
    print(f"✅ Pipeline Complete! Test R²: {r2:.3f} | MAE: PKR {mae:,.0f}")

    # Save model and encoders
    joblib.dump(model, "rf_model.pkl")
    joblib.dump({
        "loc_map": loc_map,
        "glob_med": glob_med,
        "spatial_db": spatial_db
    }, "encoders.pkl")
    print("💾 Models saved locally! The GitHub Action will now push them.")

if __name__ == "__main__":
    main()
