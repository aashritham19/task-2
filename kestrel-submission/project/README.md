# Kestrel Home - returns risk (Variant A)

**CLIENT DATA - PRIVATE.** `data/` holds Kestrel's files. Do not push this folder to a public repository (a `.gitignore` excludes it).

## Run it (clean machine, Python 3.10+, no API key, no internet needed after install)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # http://localhost:8000
```
The trained model (`model/model.joblib`) is included, so this starts immediately. If it is missing the app still starts and answers `503 model_unavailable` politely.

* **Screen:** open http://localhost:8000, press "Example: risky", then "Score this order".
* **Endpoint:** `POST /api/score` with one order as JSON, e.g.
```bash
curl -s localhost:8000/api/score -H 'Content-Type: application/json' -d '{
 "order_placed_at":"2026-10-04 11:30","customer_id":"KC100001","sku":"KH-RV-02","sales_channel":"marketplace",
 "payment_mode":"cod","discount_pct":25,"qty":1,"promised_delivery_days":8,
 "customer_prior_orders":3,"customer_prior_returns":1,"delivery_pincode":"440365","is_gift":"N","delivery_note":"Call before delivery"}'
```
Returns `return_probability`, `risk_band`, `recommended_action` (`CALL_BEFORE_DISPATCH` or `SHIP_NORMALLY` - never "hold"), `expected_net_rupees_of_calling`, plain-English `reasons`, and `notes` about data problems (unknown customer, no address).
Bad input gives a 4xx with a readable `message`. `GET /api/health` reports status.

## Rebuild everything (about 20 seconds)
```bash
python train.py       # model/model.joblib, predictions.csv, evidence/*.csv|json
python evidence.py    # evidence/EVIDENCE.md and evidence.png
```

## Files
| File | What |
|---|---|
| `predictions.csv` | Score (return probability) for each of the 2,096 test orders |
| `evidence/EVIDENCE.md` | Does it work and how often it does not (read this) |
| `memo-to-ritu.md` | One-page memo for the client |
| `submission-form.md` | Completed form |
| `returns_risk/` | Feature code (shared by training and service) and model |
| `experiments/` | The scratch experiments (leakage test, regularisation, boosted trees) |

## Decisions I made (the brief is not a spec)
1. **Ignored the 95% target as stated; reported what is achievable.** Do-nothing scores 88.5% accuracy; the honest model reaches ~89.7%. Accuracy is the wrong yardstick for a 1-in-9 event, so I score by ranking (AUC) and by rupees.
2. **Dropped `last_service_event_type` and `pickup_scheduled_at`.** Policy s7: the pickup and REVERSE_PICKUP are written after approval of a return. In train, REVERSE_PICKUP = 100% returned; with them accuracy is 98.9%. In test no row has either, and train has no INSTALL_BOOKED at all, so the fields are not comparable. (Also INSTALL_DONE/DEMO_DONE have zero returns - after-the-fact status.) TECH_VISIT is also post-dispatch.
3. **De-duplicated** the 651 partner-feed copies of orders (exact copies, label identical) so validation is not contaminated.
4. **Did not use `order_value_inr`.** Every October 2025 row is exactly 100x too high. Value is recomputed as list price x qty x (1 - discount). Verified: all other months match exactly.
5. **Default pincode 000000** is treated as "no address", not as a place.
6. **Validated by time** (train on past, score next 3 months), never random splits. Final holdout Apr-Jun 2026 mimics the Jul-Sep 2026 test.
7. **Model = regularised logistic regression**, not boosted trees. Trees scored worse on the holdout (AUC 0.75-0.77 vs 0.78), and logistic regression gives honest reasons for each order. Probabilities are calibrated.
8. **Recommend calls, not holds.** Policy figures: a hold only gives a 12% chance of cancelling; it does not stop returns. A Rs 45 call prevents ~35% of returns (pilot figure, unverified), so calling pays when score >= 11.2% (45 / (0.35 x 1,150)). I set the threshold at 15% for a safety margin.
9. **Return cost Rs 1,150** (policy, Finance) not Rs 600 (Ritu's estimate).
10. **Shield members:** kept in the model as a feature (they return ~2x more often); the recommended action for them is also a call, never a hold. I did not add special rules.
11. **No model API anywhere** - the service runs locally; per-order cost is zero (Farhan's condition).
12. **UTC timestamps (Zoho events):** only the service-event columns are affected and those are unused. Order timestamps are IST in both files.
