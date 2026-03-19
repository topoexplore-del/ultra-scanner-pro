"""
ULTRA SCANNER PRO — Build Data Pipeline v5.0
Generates pre-computed scanner data for the static dashboard.
Run: python scripts/build_data.py --out-dir data
"""
import argparse, json, os, time, warnings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# ── TICKER UNIVERSE ──────────────────────────────────────────────
# Mantener: Watchlist Core + AI & Semiconductors (US principales)
# Nuevos: LATAM, Colombia, España, Reino Unido, China, Hong Kong,
#          Índices Mundiales, ETFs Globales
GROUPS = {
    # ── US CORE ──
    "🎯 Watchlist Core": [
        "VRT","POWL","ETN","ANET","MPWR","PWR","CAT","FCX",
        "NVDA","PLTR","AVGO","AMD","LMT","NOC","CEG","SMCI",
        "GE","ROK","URI","DE"
    ],
    "🤖 AI & Semiconductors": [
        "TSM","ASML","LRCX","KLAC","MU","ARM","ON",
        "QCOM","AMAT","MRVL","ADI","NXPI"
    ],
    # ── LATAM (acciones rentables con buenos fundamentales) ──
    "🌎 LATAM": [
        "NU","MELI","VALE","PBR","SQM","BSBR","ABEV","ITUB",
        "GLOB","STNE","AMX","FMX","BIDU","PAC","VIST","SUPV"
    ],
    # ── COLOMBIA (ADRs y acciones listadas en US/OTC) ──
    "🇨🇴 Colombia": [
        "EC","CIB","AVAL","CNNE","CRGIY"
    ],
    # ── ESPAÑA (ADRs y acciones en Madrid/US) ──
    "🇪🇸 España": [
        "SAN","TEF","BBVA","IBE","ITX","REP",
    ],
    # ── REINO UNIDO (ADRs listados en US) ──
    "🇬🇧 Reino Unido": [
        "SHEL","AZN","HSBC","BP","RIO","LSEG",
        "GSK","UL","DEO","BCS","NWG","VOD"
    ],
    # ── CHINA CONTINENTAL (ADRs listados en US) ──
    "🇨🇳 China Continental": [
        "BABA","PDD","JD","BIDU","NIO","LI",
        "XPEV","ZK","FUTU","TME","BILI","YMM"
    ],
    # ── HONG KONG (tickers .HK para TradingView) ──
    "🇭🇰 Hong Kong": [
        "0700.HK","9988.HK","1299.HK","0005.HK","2318.HK",
        "0388.HK","0941.HK","1810.HK","9618.HK","3690.HK"
    ],
    # ── ÍNDICES MUNDIALES (ETFs que replican índices) ──
    "📊 Índices Mundiales": [
        "SPY","QQQ","DIA","IWM",          # US
        "EWZ","EWW",                        # LATAM: Brasil, México
        "FXI","MCHI",                       # China
        "EWU","EWG","EWQ",                  # UK, Alemania, Francia
        "EWJ","EWY",                        # Japón, Corea
        "EWA","EZA","EWT",                  # Australia, Sudáfrica, Taiwán
    ],
    # ── ETFs GLOBALES (los más importantes por AUM) ──
    "💼 ETFs Globales": [
        "VTI","VXUS","VWO","VEA",          # Vanguard broad
        "GLD","SLV","USO",                  # Commodities
        "XLE","XLF","XLK","XLV","XLI",     # Sectores US
        "SOXX","KWEB","ARKK",              # Temáticos
        "TLT","HYG","LQD",                 # Renta fija
    ],
}

# ── TECHNICAL MODEL (faithful Pine translation) ─────────────────
def compute_score(df):
    """Compute technical score 0-100. Anti-repaint: uses closed candles only."""
    c, h, v = df["Close"], df["High"], df["Volume"]
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # ADX
    tr = pd.concat([h - df["Low"], (h - c.shift()).abs(), (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    up = h.diff(); dn = -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    # Bollinger
    bb_basis = c.rolling(20).mean()
    bb_dev = c.rolling(20).std() * 2
    bb_w = (bb_basis + bb_dev - (bb_basis - bb_dev)) / bb_basis
    bb_w_low = bb_w.rolling(50).min()
    # Volume
    vol_ma = v.rolling(20).mean()
    rel_vol = v / vol_ma.replace(0, np.nan)
    # OBV
    obv = (np.sign(c.diff()) * v).cumsum()
    obv_sma = obv.rolling(10).mean()
    # Highest
    h20 = h.rolling(20).max()
    # Score components
    s_trend = ((ema50 > ema200) & (c > ema200)).astype(int) * 30
    s_mom = ((rsi > 50) & (rsi < 70)).astype(int) * 15
    s_adx = ((adx > 18) & (adx < 35)).astype(int) * 15
    s_comp = (bb_w < bb_w_low * 1.2).astype(int) * 15
    s_accum = ((obv > obv_sma) & (rel_vol > 1)).astype(int) * 15
    s_brk = (c > h20 * 0.97).astype(int) * 10
    score = s_trend + s_mom + s_adx + s_comp + s_accum + s_brk
    ext = (c - ema50) / ema50 * 100
    # AI probability (sigmoid)
    mz = (rsi - 50) / 10
    ts = (ema50 - ema200) / ema200.replace(0, np.nan)
    vr = bb_w / bb_w.rolling(50).mean().replace(0, np.nan)
    raw = mz * 0.8 + ts * 5 + (rel_vol - 1) * 1.2 + adx / 25
    ai = (100 / (1 + np.exp(-raw))) * (1 + (vr - 1) * 0.5)
    ai = ai.clip(5, 95)
    # ABC grade
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    sma50 = c.rolling(50).mean()
    return {
        "score": score, "ai": ai, "rsi": rsi, "adx": adx,
        "ext": ext, "rel_vol": rel_vol, "ema50": ema50, "ema200": ema200,
        "ema10": ema10, "ema20": ema20, "sma50": sma50, "atr": atr,
    }

def abc_grade(ema10, ema20, sma50):
    if pd.isna(ema10) or pd.isna(ema20) or pd.isna(sma50): return None
    if ema10 > ema20 > sma50: return "A"
    if ema10 < ema20 < sma50: return "C"
    return "B"

def compute_state(score, ai, ext, fund_score=None):
    if score >= 75 and ai > 70 and ext < 12:
        if fund_score and fund_score >= 60: return "ENTRY+"
        return "ENTRY"
    if score >= 60 and ai > 60: return "ACCUM"
    return "WAIT"

# ── FUNDAMENTAL GRADES ───────────────────────────────────────────
def grade_pe(v):
    if v is None or np.isnan(v): return "N/A", None
    if v < 0: return "Loss", 0
    if v < 15: return "Cheap", 3
    if v < 25: return "Fair", 2
    if v < 40: return "Pricey", 1
    return "Overval", 0

def grade_roe(v):
    if v is None or np.isnan(v): return "N/A", None
    if v > 20: return "Excel", 3
    if v > 15: return "Good", 2
    if v > 10: return "Med", 1
    return "Weak", 0

def grade_roa(v):
    if v is None or np.isnan(v): return "N/A", None
    if v > 10: return "Excel", 3
    if v > 5: return "Good", 2
    if v > 3: return "Med", 1
    return "Weak", 0

def grade_eps(v):
    if v is None or np.isnan(v): return "N/A", None
    if v > 25: return "Strong", 3
    if v > 10: return "Solid", 2
    if v > 0: return "Mod", 1
    return "Decl", 0

def safe_float(info, key):
    try:
        v = info.get(key)
        if v is None or v == "N/A": return None
        f = float(v)
        return f if np.isfinite(f) else None
    except: return None

def safe_pct(info, key):
    v = safe_float(info, key)
    if v is not None and abs(v) < 10: return round(v * 100, 2)
    return v

# ── PRICE PROJECTION ─────────────────────────────────────────────
def compute_projection(close, ema50, ema200, atr, rsi, score, ai, pe, sector_pe=20):
    """Quantitative price projection for LONG bias."""
    if close is None or np.isnan(close) or atr is None or np.isnan(atr):
        return None, None, None
    # Technical target: based on ATR extension and trend
    tech_upside = atr * 3  # 3 ATR upside target
    tech_target = close + tech_upside
    # Momentum adjustment
    if score >= 75 and ai > 70:
        tech_target *= 1.05  # Strong momentum bonus
    elif score < 40:
        tech_target *= 0.95  # Weak momentum penalty
    # Fundamental adjustment via P/E
    if pe and pe > 0 and sector_pe and sector_pe > 0:
        pe_ratio = sector_pe / pe
        if pe_ratio > 1.3:  # Undervalued
            tech_target *= 1.03
        elif pe_ratio < 0.6:  # Very overvalued
            tech_target *= 0.97
    # Confidence based on RSI + Score alignment
    if score >= 75 and 45 < rsi < 65:
        confidence = "High"
    elif score >= 60:
        confidence = "Med"
    else:
        confidence = "Low"
    pct_upside = ((tech_target / close) - 1) * 100
    return round(tech_target, 2), round(pct_upside, 1), confidence

# ── MAIN BUILD ───────────────────────────────────────────────────
def get_stock_data(ticker, spy_close=None):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", auto_adjust=True)
        if hist is None or len(hist) < 60:
            return None
        # Anti-repaint: drop today's potentially incomplete candle
        now = datetime.now()
        if hasattr(hist.index[-1], 'date') and hist.index[-1].date() == now.date():
            hist = hist.iloc[:-1]
        if len(hist) < 60:
            return None
        ind = compute_score(hist)
        last = -1  # last closed candle
        close = float(hist["Close"].iloc[last])
        sc = float(ind["score"].iloc[last])
        ai_p = float(ind["ai"].iloc[last])
        rsi_v = float(ind["rsi"].iloc[last]) if not pd.isna(ind["rsi"].iloc[last]) else None
        adx_v = float(ind["adx"].iloc[last]) if not pd.isna(ind["adx"].iloc[last]) else None
        ext_v = float(ind["ext"].iloc[last]) if not pd.isna(ind["ext"].iloc[last]) else None
        rv = float(ind["rel_vol"].iloc[last]) if not pd.isna(ind["rel_vol"].iloc[last]) else None
        abc = abc_grade(
            float(ind["ema10"].iloc[last]) if not pd.isna(ind["ema10"].iloc[last]) else None,
            float(ind["ema20"].iloc[last]) if not pd.isna(ind["ema20"].iloc[last]) else None,
            float(ind["sma50"].iloc[last]) if not pd.isna(ind["sma50"].iloc[last]) else None,
        )
        atr_v = float(ind["atr"].iloc[last]) if not pd.isna(ind["atr"].iloc[last]) else None
        # Performance
        daily = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100 if len(hist) >= 2 else None
        five_d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-6] - 1) * 100 if len(hist) >= 6 else None
        twenty_d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1) * 100 if len(hist) >= 21 else None
        # Fundamentals
        pe, roe, roa, eps_g = None, None, None, None
        name, sector, mktcap = ticker, "N/A", None
        try:
            info = stock.info
            pe = safe_float(info, "trailingPE")
            roe = safe_pct(info, "returnOnEquity")
            roa = safe_pct(info, "returnOnAssets")
            eps_g = safe_pct(info, "earningsQuarterlyGrowth")
            name = info.get("shortName", ticker)
            sector = info.get("sector", "N/A")
            mktcap = safe_float(info, "marketCap")
        except: pass
        pe_gr, pe_pts = grade_pe(pe)
        roe_gr, roe_pts = grade_roe(roe)
        roa_gr, roa_pts = grade_roa(roa)
        eps_gr, eps_pts = grade_eps(eps_g)
        pts = [p for p in [pe_pts, roe_pts, roa_pts, eps_pts] if p is not None]
        fund_score = round(sum(pts) / (len(pts) * 3) * 100, 0) if pts else None
        state = compute_state(sc, ai_p, ext_v or 0, fund_score)
        # Projection
        tgt, upside, conf = compute_projection(close, 
            float(ind["ema50"].iloc[last]) if not pd.isna(ind["ema50"].iloc[last]) else close,
            float(ind["ema200"].iloc[last]) if not pd.isna(ind["ema200"].iloc[last]) else close,
            atr_v, rsi_v or 50, sc, ai_p, pe)
        # Market cap format
        mc_str = "N/A"
        if mktcap:
            if mktcap >= 1e12: mc_str = f"${mktcap/1e12:.1f}T"
            elif mktcap >= 1e9: mc_str = f"${mktcap/1e9:.0f}B"
            elif mktcap >= 1e6: mc_str = f"${mktcap/1e6:.0f}M"
        return {
            "ticker": ticker, "name": name[:22], "close": round(close, 2),
            "score": int(sc), "ai": round(ai_p, 1),
            "state": state, "abc": abc,
            "rsi": round(rsi_v, 1) if rsi_v else None,
            "adx": round(adx_v, 1) if adx_v else None,
            "ext": round(ext_v, 1) if ext_v else None,
            "rel_vol": round(rv, 2) if rv else None,
            "daily": round(daily, 2) if daily else None,
            "5d": round(five_d, 2) if five_d else None,
            "20d": round(twenty_d, 2) if twenty_d else None,
            "pe": round(pe, 1) if pe else None, "pe_gr": pe_gr,
            "roe": round(roe, 1) if roe else None, "roe_gr": roe_gr,
            "roa": round(roa, 1) if roa else None, "roa_gr": roa_gr,
            "eps_g": round(eps_g, 1) if eps_g else None, "eps_gr": eps_gr,
            "fund": fund_score, "sector": sector, "mktcap": mc_str,
            "target": tgt, "upside": upside, "proj_conf": conf,
        }
    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"ULTRA SCANNER PRO — Building data at {datetime.now()}")
    groups_data = {}
    for group_name, tickers in GROUPS.items():
        rows = []
        seen = set()
        for i, t in enumerate(tickers):
            if t in seen: continue
            seen.add(t)
            print(f"  [{group_name}] {i+1}/{len(tickers)} {t}")
            row = get_stock_data(t)
            if row: rows.append(row)
            time.sleep(0.2)
        groups_data[group_name] = rows
    # Column ranges for visual bars
    col_ranges = {}
    for gn, rows in groups_data.items():
        vals = lambda k: [r[k] for r in rows if r.get(k) is not None]
        for k in ["daily", "5d", "20d"]:
            v = vals(k)
            col_ranges.setdefault(gn, {})[k] = [min(v) if v else -5, max(v) if v else 5]
    snapshot = {
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "groups": groups_data,
        "column_ranges": col_ranges,
    }
    path = os.path.join(args.out_dir, "snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {path} — {sum(len(r) for r in groups_data.values())} tickers processed")

if __name__ == "__main__":
    main()
