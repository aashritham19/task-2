"""Kestrel returns-risk service.  python app.py  ->  http://localhost:8000
No external API or key is used anywhere; the model is a local logistic regression."""
import os, sys, joblib, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from returns_risk.features import build_features
from returns_risk.model import contributions, load_ref, RETURN_COST, CALL_COST, CALL_PREVENTS

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
CUSTOMERS, PRODUCTS = load_ref(os.path.join(HERE, "data"))
try:
    BUNDLE = joblib.load(os.path.join(HERE, "model", "model.joblib"))
except Exception as e:  # polite failure
    BUNDLE = None; print("MODEL NOT FOUND - run `python train.py` first:", e, file=sys.stderr)

CALL_AT = 0.15        # our recommendation (break-even is 0.112; see memo)
HIGH_AT = 0.30
REQUIRED = ["order_placed_at", "sku", "sales_channel", "payment_mode", "discount_pct", "qty",
            "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"]
DEFAULTS = {"is_gift": "N", "delivery_pincode": "000000", "delivery_note": "", "customer_id": ""}


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return "an unknown number of"


def phrase(col, val, up, row):
    d = "raises" if up else "lowers"
    try:
        return _phrase(col, d, up, row)
    except Exception:
        return f"{col.replace('_', ' ')} {d} the risk."


def _phrase(col, d, up, row):
    return {
        "payment_mode": f"Payment by {str(row['payment_mode']).replace('_', ' ').upper()} {d} the risk" + (" (cash-on-delivery orders come back most)." if row['payment_mode'] == 'cod' else "."),
        "customer_prior_returns": f"This customer has {int(row['customer_prior_returns'])} earlier return(s), which {d} the risk.",
        "prior_return_rate": f"Their past return rate ({row['prior_return_rate']:.0%}) {d} the risk.",
        "customer_prior_orders": f"{int(row['customer_prior_orders'])} earlier order(s) {d} the risk.",
        "shield_flag": "Kestrel Shield member (free returns) - these customers return more." if up else "Not a Shield member (returns are not free) - lowers the risk.",
        "promised_delivery_days": f"{int(row['promised_delivery_days'])}-day promised delivery {d} the risk (longer waits get returned more).",
        "discount_pct": f"A {row['discount_pct']:.0f}% discount {d} the risk (deep discounts bring buyer's remorse).",
        "family": f"{row['family']} {d} the risk (product-line pattern).",
        "sku": f"Model {row['sku']} {d} the risk.",
        "list_price_inr": f"The list price (Rs {row['list_price_inr']:,.0f}) {d} the risk.",
        "calc_value_inr": f"Order value (Rs {row['calc_value_inr']:,.0f}) {d} the risk.",
        "sales_channel": f"Sold via {row['sales_channel'].replace('_', ' ')} {d} the risk.",
        "is_gift_flag": "Gift order raises the risk." if up else "Not a gift lowers the risk.",
        "warranty_months": f"{int(row['warranty_months'])}-month warranty {d} the risk.",
        "pin_default": "No delivery address captured (walk-in order)." ,
        "tenure_days": f"Customer for {_i(row['tenure_days'])} days {d} the risk.",
        "city": f"Delivery city ({row['city']}) {d} the risk.", "state": f"Delivery state ({row['state']}) {d} the risk.",
        "pin2": f"Delivery area {d} the risk.", "note_type": f"The delivery note {d} the risk.",
        "product_age_days": f"Product has been on sale {_i(row['product_age_days'])} days: {d} the risk.",
        "high_discount": f"Discount level {d} the risk.", "long_promise": f"Long promised delivery {d} the risk.",
        "hour": "Time of order " + d + " the risk.", "dow": "Day of week " + d + " the risk.", "qty": f"Quantity {d} the risk.",
    }.get(col, f"{col} {d} the risk.")


def score_record(rec):
    missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
    if missing:
        return {"error": "missing_fields", "message": "These fields are needed: " + ", ".join(missing)}, 400
    if BUNDLE is None:
        return {"error": "model_unavailable", "message": "The model file is not built yet. Run `python train.py` and restart."}, 503
    rec = {**DEFAULTS, "order_id": "NEW", **rec}
    if rec["sku"] not in set(PRODUCTS.sku):
        return {"error": "unknown_sku", "message": f"SKU {rec['sku']} is not in products.csv."}, 422
    try:
        df = pd.DataFrame([rec])
        for c in ["discount_pct", "qty", "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"]:
            df[c] = pd.to_numeric(df[c])
        F = build_features(df, CUSTOMERS, PRODUCTS)
        p = float(BUNDLE["model"].predict_proba(F)[0, 1])
        contrib = contributions(BUNDLE["model"], BUNDLE["mean_x"], F)
    except Exception as e:
        return {"error": "bad_input", "message": f"Could not read this order: {e}"}, 422
    row = F.iloc[0]
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    known_customer = rec.get("customer_id") in set(CUSTOMERS.customer_id)
    if not known_customer:
        top = [(c, v) for c, v in top if c not in ("shield_flag", "tenure_days", "city", "state")]
    top = [(c, v) for c, v in top if abs(v) >= 0.05][:4]
    reasons = [{"effect": "raises risk" if v > 0 else "lowers risk", "text": phrase(c, row[c], v > 0, row)} for c, v in top]
    notes = []
    if not known_customer:
        notes.append("Customer not found in customers.csv: Shield status, tenure and city were treated as unknown, so the score is less reliable.")
    if row["pin_default"] == 1:
        notes.append("No delivery address (default pincode): area was ignored.")
    gain = p * CALL_PREVENTS * RETURN_COST - CALL_COST
    if p >= CALL_AT:
        action = "CALL_BEFORE_DISPATCH"
        msg = "Make a confirmation call before dispatch. Do not hold the order if the customer does not pick up - ship it."
    else:
        action = "SHIP_NORMALLY"; msg = "Ship normally. Not worth a Rs 45 call."
    return {"order_id": rec["order_id"], "return_probability": round(p, 4),
            "risk_band": "high" if p >= HIGH_AT else "medium" if p >= CALL_AT else "low",
            "recommended_action": action, "action_text": msg,
            "expected_net_rupees_of_calling": round(gain),
            "average_order_return_probability": round(BUNDLE["return_rate"], 3),
            "reasons": reasons, "notes": notes,
            "disclaimer": "Probability, not a verdict: about 4 in 10 of the highest-risk orders are NOT returned."}, 200


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok" if BUNDLE else "model_missing", model_loaded=BUNDLE is not None, uses_external_api=False)


@app.post("/api/score")
def score():
    rec = request.get_json(silent=True)
    if not isinstance(rec, dict):
        return jsonify(error="bad_json", message="Send one order as a JSON object."), 400
    body, code = score_record(rec)
    return jsonify(body), code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
