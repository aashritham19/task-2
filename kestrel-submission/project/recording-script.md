# 3-minute screen recording - run sheet
**0:00-0:30 What I tried.** Show `experiments/exp.py` output: logistic regression vs boosted trees (AUC 0.78 vs 0.75), time-based split not random.
**0:30-1:15 What I changed.** Show the leaky result (98.9% with `last_service_event_type`/`pickup_scheduled_at`), explain they are written after the return (policy s7), removed them. Show Oct 2025 values 100x, and the 651 duplicated partner rows.
**1:15-1:45 What I threw away.** Leaky model, boosted trees, stored order value, hold policy (EVIDENCE.md rupees table).
**1:45-2:40 The product.** `python app.py`, open the screen, score the risky and safe examples, show reasons; run the curl endpoint; stop nothing - show polite error for unknown SKU.
**2:40-3:00 The answer.** 95% is not reachable; call do not hold; about Rs 35k per quarter if calls work; run a randomised test next week.
