"""Feature engineering shared by training, batch scoring and the web service.

Rules we applied (see README "Decisions"):
  * Only fields known at dispatch time are used.
  * `last_service_event_type` and `pickup_scheduled_at` are NOT used: they are written
    AFTER a return is raised (policy s7) and the training and test distributions differ.
  * `order_value_inr` as stored is NOT used: October 2025 rows are 100x too large
    (gateway change). We recompute value from list price, discount and qty instead.
"""
import re
import numpy as np
import pandas as pd

CAT_COLS = ["family", "sku", "sales_channel", "payment_mode", "state", "city",
            "pin2", "note_type"]
NUM_COLS = ["discount_pct", "qty", "list_price_inr", "calc_value_inr", "promised_delivery_days",
            "pin_default", "is_gift_flag", "customer_prior_orders", "customer_prior_returns",
            "prior_return_rate", "shield_flag", "tenure_days", "product_age_days",
            "warranty_months", "hour", "dow", "high_discount", "long_promise"]
FEATURES = CAT_COLS + NUM_COLS


def note_type(s):
    if not isinstance(s, str) or not s.strip():
        return "none"
    s = s.lower()
    if s.startswith("landmark"):
        return "landmark"
    if "neighbour" in s:
        return "neighbour"
    for key in ["fragile", "security", "office", "lift", "whatsapp", "ring twice",
                "call before", "morning", "after 6"]:
        if key in s:
            return key
    return "other"


def build_features(orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    df["order_placed_at"] = pd.to_datetime(df["order_placed_at"])
    df = df.merge(products, on="sku", how="left").merge(customers, on="customer_id", how="left")
    pin = df["delivery_pincode"].astype(str).str.zfill(6)
    df["pin_default"] = (pin == "000000").astype(int)
    df["pin2"] = np.where(df["pin_default"] == 1, "default", pin.str[:2])
    df["calc_value_inr"] = df["list_price_inr"] * df["qty"] * (1 - df["discount_pct"] / 100.0)
    df["is_gift_flag"] = (df["is_gift"] == "Y").astype(int)
    df["shield_flag"] = (df["shield_member"] == "Y").astype(int)
    df["prior_return_rate"] = df["customer_prior_returns"] / df["customer_prior_orders"].clip(lower=1)
    df["tenure_days"] = (df["order_placed_at"] - pd.to_datetime(df["signup_date"])).dt.days
    df["product_age_days"] = (df["order_placed_at"] - pd.to_datetime(df["launch_date"])).dt.days
    df["hour"] = df["order_placed_at"].dt.hour
    df["dow"] = df["order_placed_at"].dt.dayofweek
    df["high_discount"] = (df["discount_pct"] >= 25).astype(int)
    df["long_promise"] = (df["promised_delivery_days"] >= 8).astype(int)
    df["note_type"] = df["delivery_note"].map(note_type)
    for c in CAT_COLS:
        df[c] = df[c].fillna("unknown").astype(str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_train(path, customers, products):
    tr = pd.read_csv(path)
    tr = tr.drop_duplicates("order_id", keep="first")  # partner_feed re-imports are exact copies
    return tr
