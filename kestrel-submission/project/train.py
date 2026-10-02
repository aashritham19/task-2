"""python train.py  -> trains on all labelled orders, writes model/model.joblib, predictions.csv, evidence/*"""
import json, sys, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from returns_risk.features import FEATURES, build_features
from returns_risk.model import *
import os
os.makedirs("model", exist_ok=True); os.makedirs("evidence", exist_ok=True)
cu, pr = load_ref()
tr = pd.read_csv("data/train.csv").drop_duplicates("order_id")
te = pd.read_csv("data/test_unlabelled.csv")
F = build_features(tr, cu, pr); y = F.returned.values; t = F.order_placed_at
T = build_features(te, cu, pr)

# ---------- honest validation: train on the past, score the following 3 months ----------
res = []; hold = None
for cut in ["2025-12-31", "2026-03-31"]:
    a = (t <= cut).values; b = ((t > cut) & (t <= pd.Timestamp(cut) + pd.Timedelta(days=92))).values
    m = make_model().fit(F[a][FEATURES], y[a]); p = m.predict_proba(F[b][FEATURES])[:, 1]
    res.append(dict(cutoff=cut, n=int(b.sum()), base_rate=float(y[b].mean()), auc=roc_auc_score(y[b], p),
                    pr_auc=average_precision_score(y[b], p), brier=brier_score_loss(y[b], p)))
a = (t < "2026-04-01").values; b = ~a      # final mimic of the test period
m = make_model().fit(F[a][FEATURES], y[a]); p = m.predict_proba(F[b][FEATURES])[:, 1]; yb = y[b]
hold = pd.DataFrame({"order_id": F.order_id[b].values, "p": p, "y": yb, "shield": F.shield_flag[b].values,
                     "value": F.calc_value_inr[b].values})
hold.to_csv("evidence/holdout_scores.csv", index=False)
res.append(dict(cutoff="2026-03-31 (final)", n=int(b.sum()), base_rate=float(yb.mean()), auc=roc_auc_score(yb, p),
                pr_auc=average_precision_score(yb, p), brier=brier_score_loss(yb, p)))

# accuracy at the best threshold vs baseline
ths = np.linspace(0.05, 0.95, 91)
acc = [(float(np.mean((p >= th) == yb)), float(th)) for th in ths]
best_acc, best_th = max(acc)
summary = dict(validation=res, holdout_n=int(b.sum()), holdout_return_rate=float(yb.mean()),
               baseline_accuracy_never_returned=float(1 - yb.mean()), best_accuracy=best_acc, best_accuracy_threshold=best_th,
               accuracy_at_break_even=float(np.mean((p >= BREAK_EVEN) == yb)), break_even_prob=BREAK_EVEN)

# ---------- calibration & lift (deciles, top-k) ----------
hold["decile"] = pd.qcut(hold.p.rank(method="first"), 10, labels=False) + 1
cal = hold.groupby("decile").agg(orders=("y", "size"), avg_predicted=("p", "mean"), actual_return_rate=("y", "mean")).reset_index()
cal.to_csv("evidence/calibration_deciles.csv", index=False)
rows = []
for k in [0.05, 0.10, 0.20, 0.30, 0.50]:
    n = int(len(hold) * k); top = hold.sort_values("p", ascending=False).head(n)
    rows.append(dict(flag_top=f"{int(k*100)}%", flagged=n, caught=int(top.y.sum()), precision=top.y.mean(),
                     recall=top.y.sum() / hold.y.sum(), wrong_flags=int(n - top.y.sum()), missed=int(hold.y.sum() - top.y.sum())))
pd.DataFrame(rows).to_csv("evidence/flag_tradeoff.csv", index=False)

# ---------- rupees on the holdout: three policies ----------
def policy(sel, mode):
    s = hold[sel]; n_s = len(s); ret = s.y.sum()
    if mode == "call":
        saved = ret * CALL_PREVENTS * RETURN_COST; cost = n_s * CALL_COST; net = saved - cost
        return dict(contacted=n_s, returners_in_group=int(ret), saved=saved, cost=cost, net=net)
    if mode == "hold":   # best case for hold: each cancelled held returner avoids RETURN_COST; every cancelled non-returner loses margin
        saved = ret * HOLD_CANCEL * RETURN_COST
        lost_sale_value = (n_s - ret) * HOLD_CANCEL * s[s.y == 0].value.mean() if (s.y == 0).any() else 0
        return dict(contacted=n_s, returners_in_group=int(ret), saved=saved, lost_sales_revenue=lost_sale_value)
pol = {}
for name, sel in [("call_all", hold.p >= -1), ("call_p_ge_breakeven", hold.p >= BREAK_EVEN), ("call_top20pct", hold.p >= hold.p.quantile(0.8)),
                  ("call_top10pct", hold.p >= hold.p.quantile(0.9))]:
    pol[name] = policy(sel, "call")
for name, sel in [("hold_top10pct", hold.p >= hold.p.quantile(0.9)), ("hold_p_ge_breakeven", hold.p >= BREAK_EVEN)]:
    pol[name] = policy(sel, "hold")
# shield
for sh in [0, 1]:
    s = hold[hold.shield == sh]
    pol[f"shield={sh}_call_breakeven"] = policy(s.p >= BREAK_EVEN, "call") if False else None
summary["policies"] = {k: v for k, v in pol.items() if v}
summary["holdout_months"] = 3
summary["shield_split"] = hold.groupby("shield").agg(orders=("y", "size"), return_rate=("y", "mean"), avg_score=("p", "mean")).reset_index().to_dict("records")
json.dump(summary, open("evidence/summary.json", "w"), indent=2, default=float)
pd.DataFrame(res).to_csv("evidence/validation_by_cutoff.csv", index=False)

# ---------- final fit on ALL labelled orders, score test ----------
final = make_model().fit(F[FEATURES], y)
joblib.dump(dict(model=final, mean_x=mean_design(final, F), break_even=BREAK_EVEN, trained_on=int(len(F)), return_rate=float(y.mean())), "model/model.joblib")
ts = final.predict_proba(T[FEATURES])[:, 1]
sub = pd.read_csv("data/sample_submission.csv")[["order_id"]].merge(pd.DataFrame({"order_id": T.order_id, "score": ts}), on="order_id", how="left")
assert sub.score.notna().all() and len(sub) == len(te) and sub.order_id.is_unique
sub.to_csv("predictions.csv", index=False)
print(json.dumps(summary, indent=1, default=float)[:3500])
print(pd.DataFrame(rows)); print(cal)
print("test score mean", ts.mean(), "share>=breakeven", (ts >= BREAK_EVEN).mean())
