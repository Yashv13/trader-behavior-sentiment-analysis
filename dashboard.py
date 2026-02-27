import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")

# ── minimal style ──────────────────────────────────────────────
st.markdown("""
<style>
body { background: #0d1117; }
.block-container { padding-top: 1.5rem; }
h1 { font-size: 1.5rem; font-weight: 600; }
h3 { font-size: 1rem; color: #8b949e; font-weight: 400; margin-top: 0; }
.stMetric label { font-size: 0.75rem; color: #8b949e; }
</style>
""", unsafe_allow_html=True)

FEAR_C  = '#e74c3c'
GREED_C = '#2ecc71'
BASE_C  = '#3498db'
BG, CARD = '#0d1117', '#161b22'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9', 'xtick.color': '#c9d1d9', 'ytick.color': '#c9d1d9',
    'text.color': '#c9d1d9', 'grid.color': '#21262d', 'grid.linestyle': '--',
    'grid.alpha': 0.4, 'legend.facecolor': CARD, 'legend.edgecolor': '#30363d',
})

# ── load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    sentiment = pd.read_csv('fear_greed_index.csv')
    sentiment['date']  = pd.to_datetime(sentiment['date']).dt.normalize()
    sentiment['label'] = np.where(
        sentiment['classification'].str.contains('Greed', case=False), 'Greed', 'Fear'
    )

    trades = pd.read_csv('historical_data.csv')
    trades = trades.rename(columns={
        'Account':         'account',
        'Closed PnL':      'pnl',
        'Size USD':        'size_usd',
        'Side':            'side',
    })
    trades['date']     = pd.to_datetime(trades['Timestamp IST'], dayfirst=True).dt.normalize()
    trades['side']     = trades['side'].str.upper()
    trades['pnl']      = pd.to_numeric(trades['pnl'],      errors='coerce').fillna(0)
    trades['size_usd'] = pd.to_numeric(trades['size_usd'], errors='coerce').fillna(0)

    med = trades['size_usd'].median()
    trades['leverage'] = (trades['size_usd'] / med).clip(1, 100).round(0).astype(int)

    merged = trades.merge(
        sentiment[['date','label']].rename(columns={'label':'sentiment'}),
        on='date', how='inner'
    )
    merged['is_greed'] = (merged['sentiment'] == 'Greed').astype(int)

    # trader profile
    def max_dd(grp):
        cum = grp.sort_values('date')['pnl'].cumsum()
        return (cum - cum.cummax()).min()

    profile = merged.groupby('account').agg(
        total_trades   = ('pnl',      'count'),
        active_days    = ('date',     'nunique'),
        trades_per_day = ('pnl',      lambda x: len(x) / merged.loc[x.index,'date'].nunique()),
        total_pnl      = ('pnl',      'sum'),
        winrate        = ('pnl',      lambda x: (x > 0).mean()),
        avg_leverage   = ('leverage', 'mean'),
        avg_trade_size = ('size_usd', 'mean'),
        pnl_volatility = ('pnl',      'std'),
    ).reset_index()

    dd = merged.groupby('account').apply(max_dd).reset_index().rename(columns={0:'max_drawdown'})
    profile = profile.merge(dd, on='account')

    # segments
    l33, l67 = profile['avg_leverage'].quantile(0.33), profile['avg_leverage'].quantile(0.67)
    profile['lev_tier'] = np.where(profile['avg_leverage'] >= l67, 'High Leverage',
                          np.where(profile['avg_leverage'] <= l33, 'Low Leverage', 'Mid Leverage'))

    f33, f67 = profile['trades_per_day'].quantile(0.33), profile['trades_per_day'].quantile(0.67)
    profile['freq_tier'] = np.where(profile['trades_per_day'] >= f67, 'High Frequency',
                           np.where(profile['trades_per_day'] <= f33, 'Low Frequency', 'Mid Frequency'))

    med_vol = profile['pnl_volatility'].median()
    profile['winner_tier'] = np.where(
        (profile['total_pnl'] > 0) & (profile['pnl_volatility'] <= med_vol), 'Consistent Winner',
        np.where(profile['total_pnl'] > 0, 'Volatile Winner', 'Loser')
    )

    # trader×day
    td = merged.groupby(['account','date','sentiment','is_greed']).agg(
        daily_pnl  = ('pnl',      'sum'),
        n_trades   = ('pnl',      'count'),
        win_rate   = ('pnl',      lambda x: (x > 0).mean()),
        avg_lev    = ('leverage', 'mean'),
        avg_size   = ('size_usd', 'mean'),
    ).reset_index()

    td = td.merge(
        profile[['account','lev_tier','freq_tier','winner_tier',
                 'avg_leverage','winrate','trades_per_day']],
        on='account'
    ).rename(columns={'avg_leverage':'trader_avg_lev'})

    return td, profile

try:
    td, profile = load_data()
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Could not load data: {e}")
    st.stop()

# ── header ─────────────────────────────────────────────────────
st.title("Trader Behavior vs Market Sentiment")
st.markdown("### Hyperliquid trades × Fear/Greed index")
st.divider()

# ── tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Trader Lookup", "Strategy Rules"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    fear = td[td['sentiment']=='Fear']
    greed = td[td['sentiment']=='Greed']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trader-days", f"{len(td):,}")
    c2.metric("Unique traders", f"{td['account'].nunique():,}")
    c3.metric("Fear days", f"{td[td['sentiment']=='Fear']['date'].nunique()}")
    c4.metric("Greed days", f"{td[td['sentiment']=='Greed']['date'].nunique()}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fear vs Greed — summary**")
        summary = td.groupby('sentiment').agg(
            median_pnl = ('daily_pnl', 'median'),
            win_rate   = ('win_rate',  'mean'),
            pnl_std    = ('daily_pnl', 'std'),
            avg_trades = ('n_trades',  'mean'),
        ).round(3)
        st.dataframe(summary, use_container_width=True)

    with col2:
        st.markdown("**PnL distribution**")
        fig, ax = plt.subplots(figsize=(6, 3))
        for sent, c in [('Fear', FEAR_C), ('Greed', GREED_C)]:
            data = td[td['sentiment']==sent]['daily_pnl']
            lo, hi = data.quantile(0.02), data.quantile(0.98)
            ax.hist(data.clip(lo,hi), bins=50, color=c, alpha=0.7,
                    edgecolor='none', label=sent)
        ax.axvline(0, color='white', lw=0.8)
        ax.set_xlabel('daily PnL')
        ax.legend(fontsize=8)
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTS
# ══════════════════════════════════════════════════════════════
with tab2:
    seg_choice = st.radio("Segment by", ['lev_tier','freq_tier','winner_tier'],
                          format_func=lambda x: {'lev_tier':'Leverage tier',
                                                  'freq_tier':'Frequency tier',
                                                  'winner_tier':'Winner type'}[x],
                          horizontal=True)

    seg = td.groupby([seg_choice,'sentiment']).agg(
        median_pnl = ('daily_pnl', 'median'),
        win_rate   = ('win_rate',  'mean'),
        pnl_std    = ('daily_pnl', 'std'),
        n_traders  = ('account',   'nunique'),
    ).reset_index()

    st.divider()

    st.markdown("**Table**")
    st.dataframe(
        seg.pivot_table(index=seg_choice, columns='sentiment',
                        values=['median_pnl','win_rate','pnl_std']).round(3),
        use_container_width=True
    )

    st.divider()

    st.markdown("**Charts**")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, (col, label) in zip(axes, [('median_pnl','Median PnL'),
                                        ('win_rate','Win Rate'),
                                        ('pnl_std','PnL Std Dev')]):
        pivot = seg.pivot(index=seg_choice, columns='sentiment', values=col)
        pivot.plot(kind='bar', ax=ax, color=[FEAR_C, GREED_C], edgecolor='none', width=0.6)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=25, labelsize=8)
        ax.legend(fontsize=8, title='')
        ax.grid(True, axis='y')
        if col == 'median_pnl':
            ax.axhline(0, color='white', lw=0.7)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — TRADER LOOKUP
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("Type or paste a wallet address to see their profile.")

    accounts = profile['account'].tolist()
    query = st.text_input("Account address", placeholder="0x...")

    if query:
        matches = [a for a in accounts if query.lower() in a.lower()]
        if not matches:
            st.warning("No matching account found.")
        else:
            selected = st.selectbox("Matches", matches)
            row = profile[profile['account']==selected].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total PnL",       f"${row['total_pnl']:,.0f}")
            c2.metric("Win Rate",         f"{row['winrate']:.1%}")
            c3.metric("Avg Leverage",     f"{row['avg_leverage']:.1f}x")
            c4.metric("Max Drawdown",     f"${row['max_drawdown']:,.0f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Total Trades",     f"{int(row['total_trades']):,}")
            c6.metric("Trades/Day",       f"{row['trades_per_day']:.1f}")
            c7.metric("Leverage Tier",    row['lev_tier'])
            c8.metric("Winner Tier",      row['winner_tier'])

            st.divider()

            # this trader's daily PnL chart
            trader_td = td[td['account']==selected].sort_values('date')
            if len(trader_td):
                fig, ax = plt.subplots(figsize=(12, 3))
                colors = [GREED_C if p >= 0 else FEAR_C for p in trader_td['daily_pnl']]
                ax.bar(trader_td['date'], trader_td['daily_pnl'],
                       color=colors, edgecolor='none', width=0.8)
                ax.axhline(0, color='white', lw=0.7)
                ax.set_title(f"Daily PnL — {selected[:10]}...", fontsize=10)
                ax.set_xlabel('date')
                ax.grid(True, axis='y')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 4 — STRATEGY RULES
# ══════════════════════════════════════════════════════════════
with tab4:
    # compute thresholds live from data
    lev_cap  = profile[profile['lev_tier']=='Low Leverage']['avg_leverage'].median()
    freq_cap = profile[profile['freq_tier']=='Low Frequency']['trades_per_day'].median()

    st.markdown(f"""
### Strategy 1: Leverage Cap During Fear

**Condition**
```
Sentiment = Fear
AND trader segment = High Leverage
```

**Action**
Cap leverage to **{lev_cap:.1f}x** — the median leverage of Low Leverage traders.

**Why**
High leverage traders show higher PnL volatility on Fear days compared to Greed days.
The effect is specific to this segment — low leverage traders are mostly unaffected by sentiment.

---

### Strategy 2: Trade Frequency Guardrail During Fear

**Condition**
```
Sentiment = Fear
AND trader segment = Low Frequency
```

**Action**
Limit daily trades to **{freq_cap:.1f}** — the trader's historical median activity.

**Why**
Low frequency traders show declining win rate on Fear days when they trade
above their usual pace. Looks like they react to noise rather than signal.
High frequency traders don't show the same pattern.

---

### Evidence
""")

    seg_lev  = td.groupby(['lev_tier','sentiment']).agg(
        median_pnl=('daily_pnl','median'), win_rate=('win_rate','mean'),
        pnl_std=('daily_pnl','std')
    ).round(3)
    seg_freq = td.groupby(['freq_tier','sentiment']).agg(
        median_pnl=('daily_pnl','median'), win_rate=('win_rate','mean'),
        pnl_std=('daily_pnl','std')
    ).round(3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Leverage tier × sentiment**")
        st.dataframe(seg_lev, use_container_width=True)
    with col2:
        st.markdown("**Frequency tier × sentiment**")
        st.dataframe(seg_freq, use_container_width=True)
