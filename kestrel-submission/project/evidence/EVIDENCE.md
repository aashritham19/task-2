# Evidence: does it work, and how often does it not?

All numbers are from **time-based validation**: model trained only on orders before 1 Apr 2026, scored on the 2,126 orders of Apr-Jun 2026 (the last 3 months, same shape as the test file). Random splits are not used: they flatter the model.

## Headline
| | Value |
|---|---|
| Return rate in holdout | 11.5% |
| Accuracy of "nothing is ever returned" | **88.5%** |
| Best accuracy our model can reach (flag if score >= 0.44) | **89.7%** |
| Accuracy needed for the board promise | 95.0% |
| AUC (ranking quality, 0.5 = coin flip) | 0.784 (90% bootstrap range 0.758-0.811) |
| Two earlier rolling windows (AUC) | 0.769, 0.780 |

**95% accuracy is not reachable with the information available at dispatch.** An earlier version that used `last_service_event_type` and `pickup_scheduled_at` scored 98.9% on the same holdout - because those fields are written *after* the return is raised (policy s7). In the test file they are never REVERSE_PICKUP and pickups are blank, so that model would be useless on the orders that matter. We threw it away.

## How often it is wrong (holdout, 2,126 orders, 245 real returns)
| flag_top   |   flagged |   caught |   precision |   recall |   wrong_flags |   missed |
|:-----------|----------:|---------:|------------:|---------:|--------------:|---------:|
| 5%         |       106 |       61 |        0.58 |     0.25 |            45 |      184 |
| 10%        |       212 |       89 |        0.42 |     0.36 |           123 |      156 |
| 20%        |       425 |      132 |        0.31 |     0.54 |           293 |      113 |
| 30%        |       637 |      159 |        0.25 |     0.65 |           478 |       86 |
| 50%        |      1063 |      207 |        0.19 |     0.84 |           856 |       38 |

Read: flag the riskiest 10% and 42% of flags are real returns (caught 89 of 245), but 123 flags are wrong and 156 returns are still missed. Accuracy is the wrong yardstick here; what matters is rupees (below).

## Calibration
When the model says "20% chance", about 20% come back (deciles table: `calibration_deciles.csv`). Slightly under-confident in the 6th-7th deciles. The score is usable as a probability.

## Rupees: call vs hold (policy figures: return Rs 1,150, call Rs 45, call prevents 35% of returns, hold -> 12% cancel)
Break-even score for a call = 45 / (0.35 x 1,150) = **11.2%** - almost exactly the average return rate, so calling everyone roughly breaks even (net Rs 2,942); calling the riskier third nets about Rs 35,555 per quarter.

| call_effect   |   threshold |   orders_called |   share_of_orders |   returners_reached |   net_rupees_3mo |    ci_low |   ci_high |
|:--------------|------------:|----------------:|------------------:|--------------------:|-----------------:|----------:|----------:|
| 35%           |        0.00 |            2126 |              1.00 |                 245 |          2942.50 |  -6757.75 |  13045.25 |
| 35%           |        0.11 |             641 |              0.30 |                 160 |         35555.00 |  28486.12 |  42394.75 |
| 35%           |        0.15 |             455 |              0.21 |                 138 |         35070.00 |  27198.00 |  42217.25 |
| 35%           |        0.20 |             312 |              0.15 |                 112 |         31040.00 |  24681.38 |  37569.50 |
| 35%           |        0.30 |             154 |              0.07 |                  74 |         22855.00 |  17888.38 |  27592.50 |
| 35%           |        0.40 |              83 |              0.04 |                  51 |         16792.50 |  13307.38 |  20686.87 |
| 20%           |        0.00 |            2126 |              1.00 |                 245 |        -39320.00 | -44840.00 | -33570.00 |
| 20%           |        0.11 |             641 |              0.30 |                 160 |          7955.00 |   3629.00 |  12420.00 |
| 20%           |        0.15 |             455 |              0.21 |                 138 |         11265.00 |   7449.75 |  15100.50 |
| 20%           |        0.20 |             312 |              0.15 |                 112 |         11720.00 |   8374.00 |  15143.50 |
| 20%           |        0.30 |             154 |              0.07 |                  74 |         10090.00 |   7537.50 |  12870.00 |
| 20%           |        0.40 |              83 |              0.04 |                  51 |          7995.00 |   5929.50 |  10160.50 |
| 10%           |        0.00 |            2126 |              1.00 |                 245 |        -67495.00 | -70370.00 | -64850.00 |
| 10%           |        0.11 |             641 |              0.30 |                 160 |        -10445.00 | -12595.00 |  -8380.75 |
| 10%           |        0.15 |             455 |              0.21 |                 138 |         -4605.00 |  -6423.00 |  -2648.25 |
| 10%           |        0.20 |             312 |              0.15 |                 112 |         -1160.00 |  -2835.50 |    556.00 |
| 10%           |        0.30 |             154 |              0.07 |                  74 |          1580.00 |    393.50 |   2668.00 |
| 10%           |        0.40 |              83 |              0.04 |                  51 |          2130.00 |   1319.50 |   3010.75 |

**Holding does not pay.** A hold does not stop a return - it only gives the customer a chance to cancel (12%). Holding the top 10% saves at most Rs 12,282 per quarter in avoided return costs while putting about Rs 204k of expected sales revenue at risk of cancellation (we were not given margins; at any plausible margin this is a loss).

## Shield members
|   shield |   orders |   return_rate |   auc |   called_at_015 |
|---------:|---------:|--------------:|------:|----------------:|
|        0 |  1648.00 |          0.09 |  0.76 |            0.15 |
|        1 |   478.00 |          0.21 |  0.76 |            0.42 |

Shield members really do return more (about 20% vs 9%) and are flagged more often. The model ranks well inside both groups. We recommend a call, never a hold, for them.

## Known limits
- Signal is modest: age of the customer relationship, prior returns, COD, long promises, discounts, Shield. No one can predict remorse from this file at >90% precision.
- The 35% call effect comes from a spring pilot (policy s7). We do not know how those orders were chosen. At a 20% effect the break-even rises to ~20% and savings shrink (table above). Run a proper test first.
- Cannot see: reasons, product defects, delivery quality, margins.
- Partner-outlet orders appear twice in train (exact copies): de-duplicated, otherwise the same order would sit in both train and validation.
- Oct 2025 `order_value_inr` is 100x too high (every row): not used; value recomputed from price, qty and discount.
- Unconverted UTC timestamps (Zoho resolution events) affect only the service-event columns we do not use.
