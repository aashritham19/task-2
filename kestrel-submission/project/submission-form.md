# Submission form - Kestrel Home Returns Risk (Variant A)

> NOTE: `submission-form.md` was not in the zip I received, so I could not use your template. This follows the headings the brief asks for. Please paste into the official form if it differs.

**Name:** Aashritha  **Variant:** A  **Repo / zip:** private zip (client data included, not published)

## 1. What I expect `predictions.csv` to score, and why
- **Ranking (AUC): about 0.78** (plausible range 0.75-0.81). On the last 3 months of train (trained only on earlier orders) the AUC was 0.784; two earlier rolling windows gave 0.769 and 0.780. 90% bootstrap range 0.758-0.811.
- **PR-AUC: about 0.40.** Return rate in test I expect about 11-12% (train 11.4%; the model's own scores sum to 242 of 2,096 = 11.5%).
- **Accuracy: about 89-90% at the best threshold (about 0.44), against 88.5% for "no order returns".** I do NOT expect 95%, and I believe no honest model can reach it from dispatch-time information. The only way I found to reach 95%+ was using `last_service_event_type` / `pickup_scheduled_at` (98.9% on holdout), which are written after the return and absent in the test file, so I rejected it.
- Lower-than-expected risk: October-2026 festive orders (new payment gateway; I recomputed value instead of using the stored value, but behaviour may differ). Test starts 1 Jul 2026, i.e. right after the train period, as in my validation.

## 2. Decisions I made where the brief was unclear
See `README.md` "Decisions". The main ones: dropped post-return fields; de-duplicated partner-feed rows; ignored stored order value (October 2025 is 100x); recommend calls not holds; Rs 1,150 not Rs 600; threshold 15% (break-even 11.2%).

## 3. AI tools - used, cost, discarded
- **Used:** Claude (Anthropic) in the claude.ai chat with its code sandbox, for data exploration, modelling, writing the Flask service/UI, evidence and memo drafts. No other assistants. **No AI/model API is called by the product** (local logistic regression), so it starts and runs without any key and per-order cost is zero.
- **Cost:** no API credits used. Chat usage is covered by my existing Claude plan: [Aashritha - fill in plan / any cost]. Running the product costs nothing per order.
- **Discarded:** (1) the leaky model using service/pickup fields (98.9% holdout accuracy, useless in production); (2) gradient-boosted trees (AUC 0.75-0.77, worse than logistic regression, less explainable); (3) using stored `order_value_inr`; (4) hold-based policy (loses money); (5) free-text delivery note parsing beyond coarse types (no signal found).
- **Verified by me, not just generated:** service output matches batch predictions (max difference 0.00005); metrics rerun from `train.py` / `evidence.py`.

## 4. Where it fails / honest limits
Signal is modest; 4 in 10 of the riskiest orders are not returned; 35% call effect unverified; no margin data; unknown customers or SKUs handled with warnings/422, not guesses. Details: `evidence/EVIDENCE.md`.

## 5. Checklist
- [x] predictions.csv (2,096 rows, one per order_id, higher = more likely returned)
- [x] Service (endpoint + one screen), starts from README, no key
- [x] Evidence (`evidence/`)
- [x] One-page memo (`memo-to-ritu.md`)
- [ ] Screen recording (max 3 min) - NOT yet recorded; script in `recording-script.md`
- [x] This form
