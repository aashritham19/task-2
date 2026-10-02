"""python evidence.py -> evidence/EVIDENCE.md + charts (needs `python train.py` first)"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from returns_risk.model import RETURN_COST, CALL_COST, BREAK_EVEN
h = pd.read_csv("evidence/holdout_scores.csv"); S = json.load(open("evidence/summary.json"))
rng = np.random.default_rng(0)

def net(df, th, prev):  # call every order with p>=th
    s = df[df.p >= th]; return s.y.sum() * prev * RETURN_COST - len(s) * CALL_COST
rows = []
for prev in [0.35, 0.20, 0.10]:
    for th in [0.0, BREAK_EVEN, 0.15, 0.20, 0.30, 0.40]:
        bs = [net(h.sample(len(h), replace=True, random_state=int(rng.integers(1e9))), th, prev) for _ in range(300)]
        s = h[h.p >= th]
        rows.append(dict(call_effect=f"{int(prev*100)}%", threshold=round(th, 3), orders_called=len(s), share_of_orders=len(s)/len(h),
                         returners_reached=int(s.y.sum()), net_rupees_3mo=net(h, th, prev), ci_low=np.percentile(bs, 5), ci_high=np.percentile(bs, 95)))
T = pd.DataFrame(rows); T.to_csv("evidence/call_threshold_table.csv", index=False)
aucs = [roc_auc_score(*(lambda d: (d.y, d.p))(h.sample(len(h), replace=True, random_state=i))) for i in range(300)]
sh = h.groupby("shield").apply(lambda d: pd.Series(dict(orders=len(d), return_rate=d.y.mean(), auc=roc_auc_score(d.y, d.p),
        called_at_015=(d.p >= .15).mean())), include_groups=False)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cal = pd.read_csv("evidence/calibration_deciles.csv")
ax[0].plot([0, .5], [0, .5], "k--", lw=1); ax[0].plot(cal.avg_predicted, cal.actual_return_rate, "o-"); ax[0].set(title="Calibration (holdout deciles)", xlabel="predicted", ylabel="actual return rate")
ft = pd.read_csv("evidence/flag_tradeoff.csv"); ax[1].bar(ft.flag_top, ft.caught, label="returners caught"); ax[1].bar(ft.flag_top, ft.wrong_flags, bottom=ft.caught, label="wrongly flagged", alpha=.6); ax[1].set(title="Flag the top X% of orders"); ax[1].legend()
t35 = T[T.call_effect == "35%"]; ax[2].plot(t35.threshold.astype(str), t35.net_rupees_3mo, "o-", label="35% prevented"); t20 = T[T.call_effect == "20%"]; ax[2].plot(t20.threshold.astype(str), t20.net_rupees_3mo, "s-", label="20% prevented"); ax[2].axhline(0, c="k", lw=1); ax[2].set(title="Net Rs per quarter by call threshold", xlabel="call if score >="); ax[2].legend()
plt.tight_layout(); plt.savefig("evidence/evidence.png", dpi=110)
md = f"""# Evidence: does it work, and how often does it not?

All numbers are from **time-based validation**: model trained only on orders before 1 Apr 2026, scored on the {S['holdout_n']:,} orders of Apr-Jun 2026 (the last 3 months, same shape as the test file). Random splits are not used: they flatter the model.

## Headline
| | Value |
|---|---|
| Return rate in holdout | {S['holdout_return_rate']:.1%} |
| Accuracy of "nothing is ever returned" | **{S['baseline_accuracy_never_returned']:.1%}** |
| Best accuracy our model can reach (flag if score >= {S['best_accuracy_threshold']:.2f}) | **{S['best_accuracy']:.1%}** |
| Accuracy needed for the board promise | 95.0% |
| AUC (ranking quality, 0.5 = coin flip) | {S['validation'][-1]['auc']:.3f} (90% bootstrap range {np.percentile(aucs,5):.3f}-{np.percentile(aucs,95):.3f}) |
| Two earlier rolling windows (AUC) | {S['validation'][0]['auc']:.3f}, {S['validation'][1]['auc']:.3f} |

**95% accuracy is not reachable with the information available at dispatch.** An earlier version that used `last_service_event_type` and `pickup_scheduled_at` scored 98.9% on the same holdout - because those fields are written *after* the return is raised (policy s7). In the test file they are never REVERSE_PICKUP and pickups are blank, so that model would be useless on the orders that matter. We threw it away.

## How often it is wrong (holdout, {S['holdout_n']:,} orders, {int(round(S['holdout_return_rate']*S['holdout_n']))} real returns)
{ft.to_markdown(index=False, floatfmt='.2f')}

Read: flag the riskiest 10% and 42% of flags are real returns (caught 89 of 245), but 123 flags are wrong and 156 returns are still missed. Accuracy is the wrong yardstick here; what matters is rupees (below).

## Calibration
When the model says "20% chance", about 20% come back (deciles table: `calibration_deciles.csv`). Slightly under-confident in the 6th-7th deciles. The score is usable as a probability.

## Rupees: call vs hold (policy figures: return Rs {RETURN_COST:,.0f}, call Rs {CALL_COST:.0f}, call prevents 35% of returns, hold -> 12% cancel)
Break-even score for a call = {CALL_COST:.0f} / (0.35 x {RETURN_COST:,.0f}) = **{BREAK_EVEN:.1%}** - almost exactly the average return rate, so calling everyone roughly breaks even (net Rs {S['policies']['call_all']['net']:,.0f}); calling the riskier third nets about Rs {S['policies']['call_p_ge_breakeven']['net']:,.0f} per quarter.

{T.to_markdown(index=False, floatfmt='.2f')}

**Holding does not pay.** A hold does not stop a return - it only gives the customer a chance to cancel (12%). Holding the top 10% saves at most Rs {S['policies']['hold_top10pct']['saved']:,.0f} per quarter in avoided return costs while putting about Rs {S['policies']['hold_top10pct']['lost_sales_revenue']/1000:,.0f}k of expected sales revenue at risk of cancellation (we were not given margins; at any plausible margin this is a loss).

## Shield members
{sh.to_markdown(floatfmt='.2f')}

Shield members really do return more (about 20% vs 9%) and are flagged more often. The model ranks well inside both groups. We recommend a call, never a hold, for them.

## Known limits
- Signal is modest: age of the customer relationship, prior returns, COD, long promises, discounts, Shield. No one can predict remorse from this file at >90% precision.
- The 35% call effect comes from a spring pilot (policy s7). We do not know how those orders were chosen. At a 20% effect the break-even rises to ~20% and savings shrink (table above). Run a proper test first.
- Cannot see: reasons, product defects, delivery quality, margins.
- Partner-outlet orders appear twice in train (exact copies): de-duplicated, otherwise the same order would sit in both train and validation.
- Oct 2025 `order_value_inr` is 100x too high (every row): not used; value recomputed from price, qty and discount.
- Unconverted UTC timestamps (Zoho resolution events) affect only the service-event columns we do not use.
"""
open("evidence/EVIDENCE.md", "w").write(md); print(md[:1500]); print(T.to_string()); print(sh)
