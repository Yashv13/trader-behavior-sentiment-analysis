# Write-up: Trader Performance vs Market Sentiment

---

## Methodology

Two datasets were combined: Bitcoin Fear/Greed index (daily) and Hyperliquid historical trade data (~211k trades across 32 unique trader accounts). After standardizing timestamps and joining on date, trades were labeled with the day's sentiment — Fear or Greed.

The core design decision was to build a **fixed trader identity profile** first — grouping by account across the entire dataset to compute each trader's lifetime average leverage, trade frequency, win rate, and PnL volatility. These become permanent segment labels. This avoids the problem of a trader flipping between "high leverage" and "low leverage" day to day depending on a single trade, which would make behavioral segmentation meaningless.

Three segmentations were created:
- **Leverage tier** — top/mid/bottom 33% by lifetime average leverage
- **Frequency tier** — top/mid/bottom 33% by average trades per day
- **Winner tier** — Consistent Winner (profitable + low PnL volatility), Volatile Winner (profitable + high volatility), Loser

These labels were merged back into the trader×day table, so every daily row carries the trader's fixed segment. Sentiment comparison was then run *within* each segment — not across the whole dataset — to isolate segment-specific effects.

Metrics used: median daily PnL (robust to outliers), win rate, and PnL standard deviation as a drawdown proxy. True max drawdown was also computed per trader from cumulative PnL curves.

---

## Insights

**1. Fear increases volatility for high-leverage traders specifically**
High-leverage traders show meaningfully higher PnL standard deviation on Fear days compared to Greed days. The same spike does not appear in the low-leverage segment. This suggests Fear doesn't just reduce returns — it amplifies downside swings for traders already carrying large positions. Median PnL also drops for this group on Fear days, confirmed by the leverage tier chart.

**2. Low-frequency traders degrade on Fear days when they overtrade**
Low-frequency traders show a declining win rate on Fear days, with higher PnL volatility. The pattern does not appear for high-frequency traders. The likely explanation is that infrequent traders react to sentiment-driven market noise rather than their usual signal, increasing activity at exactly the wrong time.

**3. Consistent winners are largely sentiment-neutral; losers are not**
Consistent winners — traders with positive total PnL and below-median volatility — maintain stable median PnL across both sentiment regimes. Losers show wider swings between Fear and Greed days. This confirms that the sentiment effect is concentrated in weaker traders, not the whole market.

---

## Strategy Recommendations

**Strategy 1 — Leverage cap during Fear**
If sentiment = Fear and trader segment = High Leverage, cap leverage to the median leverage of the Low Leverage segment. High-leverage traders show increased volatility and reduced median PnL specifically on Fear days. The cap level is data-derived, not arbitrary. Low-leverage traders are unaffected so the rule only applies to this cohort.

**Strategy 2 — Frequency guardrail during Fear**
If sentiment = Fear and trader segment = Low Frequency, limit daily trades to the trader's own historical median. Low-frequency traders show win rate degradation when they trade above their normal pace on Fear days. Restricting activity to baseline prevents performance loss from sentiment-driven overtrading. High-frequency traders do not need this constraint.

---

## Limitations

- Leverage is estimated from position size — actual leverage data would sharpen results
- Daily granularity hides intra-day behavior
- Predictive models do not include market-wide variables like realized volatility or funding rates
- Fear/Greed is binary here — Extreme Fear may behave differently from mild Fear
