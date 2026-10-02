**To:** Ritu Deshpande, Head of D2C Operations  **Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni
**Subject:** Returns model - what it can and cannot do, and what to do next week

**The decision.** Do not hold orders. Call the customer before dispatch on the riskiest fifth of orders, and ship everything else as normal. I would also not take 95% accuracy to the board.

**The number.** 95% is not achievable from what we know when an order is placed. About 11 in 100 orders come back, so a "nothing ever returns" guess is already right 88.5% of the time; the best this model does is 89.7%. It ranks orders well (the riskiest 10% are returned 42% of the time, against 11.5% overall) but it cannot say with certainty which one will return. A version that scored 98.9% was only reading fields the system fills in *after* a customer has already asked for the return, so it would have been useless on live orders. I removed it.

**The rupees** (last three months, 2,126 orders, 245 returns at Rs 1,150 each = about Rs 2.8 lakh):
- **Calling** the 455 orders scoring 15% or higher (about 5 calls a day, 21% of orders) reaches 138 of the 245 returns. If calls prevent 35% of returns, as the spring pilot suggested, that is **about Rs 35,000 net per quarter** (about Rs 1.4 lakh a year) after paying Rs 45 a call. It is a gain of roughly an eighth of the return bill, not a fix.
- That 35% is the soft spot. At 20% the gain falls to about Rs 11,000 a quarter; at 10% calling loses money.
- **Holding** does not pay. A hold only gives the customer a chance to cancel (about 12%); it does not stop a return. Holding the top 10% would save at most about Rs 12,000 a quarter in avoided returns, while putting about Rs 2 lakh of sales at risk of cancelling.
- Shield members return about twice as often (20% vs 9%), so they will be called more. A call is far less intrusive than a hold and keeps Meenal's best customers on side.
- The model costs nothing per order to run (it uses no paid service), so Farhan's condition is met.

**Next week.**
1. Tell the board the target is rupees saved, not 95% accuracy. Share the first number above.
2. Start the call programme on orders scoring 15%+ for four to six weeks, and **randomly leave half of them uncalled**. That is the only way to learn the true effect of a call; the whole case rests on it.
3. Hold nothing. If a customer does not pick up, ship.
4. Ask Tanmay to (a) correct the October 2025 order values (all 100 times too high), (b) capture the delivery address on walk-in orders, and (c) log service and pickup events with dates, so a later version can be checked properly.
5. Review in six weeks: if calls prevent less than about 20% of returns, stop; if more, widen the net.

*Open assumptions:* Kestrel margins were not provided, so I have not netted cancelled sales against anything; the 35% call effect is unverified; the Rs 1,150 policy figure is used rather than Rs 600.
