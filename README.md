# 🎯 HEDGE FUND RADAR PRO v2.0

Bloomberg-terminal style market scanner with real-time TradingView charts, quantitative scoring, AI probability model, fundamental analysis, and price projections.

**LONG ONLY** · Anti-repaint guaranteed · 120+ tickers across global markets

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build scanner data (runs ~5-10 minutes)
python scripts/build_data.py --out-dir data

# 3. Preview locally
python -m http.server 8000
# Open http://localhost:8000
```

## Deploy to GitHub Pages (Recommended)

1. Push repo to GitHub
2. **Settings > Pages** → deploy from `main` branch
3. The GitHub Action refreshes data daily at 16:30 ET (after market close)
4. Run manually: **Actions > Refresh Scanner Data > Run workflow**

## Deploy to Streamlit Cloud (Alternative)

```bash
# Requires: streamlit in requirements
pip install streamlit
streamlit run app.py
```

## Features

| Feature | Description |
|---------|-------------|
| **Technical Score (0-100)** | EMA 50/200, RSI, ADX, Bollinger compression, OBV, volume, breakout setup |
| **AI Probability (5-95%)** | Sigmoid model: momentum Z-score, trend strength, volume regime |
| **Fundamental Grade** | P/E, ROE, ROA, EPS growth with traffic-light grading |
| **State Machine** | WAIT → ACCUM → ENTRY → ENTRY+ (fundamental-enhanced) |
| **Price Projections** | Quantitative targets based on ATR, momentum, and valuation |
| **TradingView Charts** | Real-time professional charts with MA% Ribbon + SPY overlay |
| **Multi-Chart Grid** | View entire sector at once |
| **Keyboard Navigation** | ↑↓ browse, M multichart, Esc single |
| **Anti-Repaint** | Only uses closed candles (verified with look-ahead test) |

## Fundamental Reference

| Metric | 🟢 Excellent | 🟡 Good | 🟠 Fair | 🔴 Weak |
|--------|-------------|---------|---------|---------|
| **P/E** | < 15 (Cheap) | 15-25 (Fair) | 25-40 (Pricey) | > 40 (Overvalued) |
| **ROE** | > 20% | 15-20% | 10-15% | < 10% |
| **ROA** | > 10% | 5-10% | 3-5% | < 3% |
| **EPS Growth** | > 25% | 10-25% | 0-10% | < 0% |

## Architecture

```
hedge-fund-radar-pro/
├── index.html                 # Bloomberg terminal dashboard
├── scripts/build_data.py      # Data pipeline (technical + AI + fundamental)
├── data/snapshot.json          # Pre-computed scanner data
├── app.py                     # Streamlit wrapper (optional)
├── .github/workflows/         # Auto-refresh Mon-Fri 16:30 ET
└── requirements.txt
```

## Disclaimer

This tool is for informational and educational purposes only. Not financial advice. All trading involves risk of capital loss.
