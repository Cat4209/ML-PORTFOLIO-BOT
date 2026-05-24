import collections
if not hasattr(collections, 'MutableMapping'):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping

import os
import sys
import json
import csv
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import requests
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================
# LIVE BOT V45 "MULTI-ASSET PORTFOLIO"
#
# Базиран на V45 (BTC+ETH, +19.8%/год, MaxDD -15.8%, Sharpe 2.24)
# Архитектура: V44 × 2 актива
#   - LONG + SHORT + Sniper Entry + Soft Macro Scaling
#   - Portfolio Kelly Penalty (намалява позиция при корелации)
#   - Portfolio Hard Stop (-15% → пауза 30 дни)
#   - Walk-forward преобучение на всеки 90 дни (автоматично)
#
# Режими на стартиране:
#   python live_bot_v45.py          → еднократен сигнал + запазване
#   python live_bot_v45.py --listen → слушател на команди Telegram
#   python live_bot_v45.py --retrain → принудително преобучение
# ==============================================================

# ── Telegram ──────────────────────────────────────────────────
TG_TOKEN   = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TG_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

# ── Файлове ─────────────────────────────────────────────────────
MODELS_DIR    = "models_v45"
STATE_FILE    = "bot_state_v45.json"
TRADES_FILE   = "trades_journal_v45.csv"
PORTFOLIO_FILE = "portfolio_state_v45.json"
FUNDING_FILE  = "funding_rate.csv"
CACHE_DIR     = "onchain_cache"

# ── Активи портфейла ───────────────────────────────────────────
ASSETS = {
    "BTC-USD": {"tc": 0.0002, "label": "BTC", "emoji": "₿"},
    "ETH-USD": {"tc": 0.0002, "label": "ETH", "emoji": "Ξ"},
}

# ── Портфейлни параметри ─────────────────────────────────────
PORTFOLIO_RISK      = 0.05
PORTFOLIO_HARD_STOP = 0.15   # -15% от общия портфейл → пауза 30 дни
ASSET_CORR_BTC_ETH  = 0.88   # средна корелация за Kelly penalty
SNIPER_PCT          = 0.004  # 0.4% под Close за лимитен ордер
RETRAIN_DAYS        = 90

# ── Матрица корелации ────────────────────────────────────────
ASSET_CORR = {("BTC-USD","ETH-USD"): 0.88}


# ==============================================================
# УТИЛИТИ
# ==============================================================
def send_telegram(text: str):
    if not text: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TG_CHAT_ID, "text": text,
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception as e:
        print(f"  TG error: {e}")


def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"
    try:
        return requests.get(url, params={
            "offset": offset, "timeout": 20
        }, timeout=25).json().get("result", [])
    except:
        return []


def load_json(path: str, default=None):
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except:
            pass
    return default or {}


def save_json(path: str, data: dict):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def save_trade(trade: dict):
    exists = Path(TRADES_FILE).exists()
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=trade.keys())
        if not exists:
            w.writeheader()
        w.writerow(trade)


def safe_transform(sc, df, feats):
    X = sc.transform(df[feats].copy())
    return np.where(np.isfinite(X), X, 0.0)


# ==============================================================
# ЗАРЕЖДАНЕ НА МАКРО ДАННИ
# ==============================================================
def load_macro() -> dict:
    """Зарежда SPX, DXY, VIX, ETH веднъж за всички активи."""
    macro = {}

    def _get(ticker, name):
        try:
            df = yf.download(ticker, start="2018-01-01",
                              progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            return df["Close"].rename(name)
        except:
            return None

    macro["spx"] = _get("^GSPC",    "SPX")
    macro["dxy"] = _get("DX-Y.NYB", "DXY")
    macro["vix"] = _get("^VIX",     "VIX")
    macro["eth"] = _get("ETH-USD",  "ETH_macro")

    # Fear & Greed
    fg_path = f"{CACHE_DIR}/fear_greed.csv"
    if os.path.exists(fg_path):
        try:
            fg = pd.read_csv(fg_path, index_col=0, parse_dates=True)
            fg.index = pd.to_datetime(fg.index).normalize()
            col = "fng_value" if "fng_value" in fg.columns else fg.columns[0]
            macro["fg"] = fg[col].rename("FNG")
        except:
            macro["fg"] = None
    else:
        macro["fg"] = None

    return macro


# ==============================================================
# ЗАРЕЖДАНЕ НА ДАННИ ЗА ЕДИН АКТИВ
# ==============================================================
def load_asset_data(ticker: str, macro: dict,
                    start: str = "2019-01-01") -> pd.DataFrame:
    """Зарежда OHLCV + всички характеристики V45 за един актив."""
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()

    d   = df.copy()
    ret = d["Close"].pct_change()

    # BTC технически индикатори
    for p in [3, 7, 14, 30]:
        d[f"mom_{p}"] = d["Close"].pct_change(p)
        d[f"vol_{p}"] = ret.rolling(p).std()

    delta = d["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["RSI"]     = 100 - (100 / (1 + gain / (loss + 1e-9)))

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["ATR"]     = tr.rolling(14).mean()
    d["ATR_pct"] = d["ATR"] / d["Close"]

    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bb_pct"]   = (d["Close"] - sma20) / (2 * std20 + 1e-9)
    d["bb_width"] = (4 * std20) / (sma20 + 1e-9)

    ema21  = d["Close"].ewm(span=21).mean()
    ema55  = d["Close"].ewm(span=55).mean()
    ema200 = d["Close"].ewm(span=200).mean()
    d["ema_cross"]    = (ema21 - ema55) / d["Close"]
    d["above_ema55"]  = (d["Close"] > ema55).astype(float)
    d["above_ema200"] = (d["Close"] > ema200).astype(float)

    d["range"]      = (d["High"] - d["Low"]) / d["Close"]
    d["skew_14"]    = ret.rolling(14).skew()
    d["kurt_14"]    = ret.rolling(14).kurt()
    d["vol_ratio"]  = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)
    d["autocorr_5"] = ret.rolling(20).apply(
        lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
    ).fillna(0)

    # Funding Rate (само BTC)
    if ticker == "BTC-USD" and os.path.exists(FUNDING_FILE):
        try:
            fr = pd.read_csv(FUNDING_FILE, parse_dates=["Date"])
            fr["Date"] = pd.to_datetime(fr["Date"]).dt.normalize()
            fr = fr.groupby("Date")["FundingRate"].sum().to_frame()
            d  = d.join(fr, how="left")
            d["FundingRate"]     = d["FundingRate"].fillna(0)
            d["FundingRate_MA"]  = d["FundingRate"].rolling(7).mean().fillna(0)
            d["FundingRate_chg"] = d["FundingRate"].diff().fillna(0)
        except:
            d["FundingRate"] = d["FundingRate_MA"] = d["FundingRate_chg"] = 0.0
    else:
        d["FundingRate"] = d["FundingRate_MA"] = d["FundingRate_chg"] = 0.0

    # Fear & Greed
    if macro.get("fg") is not None:
        fg = macro["fg"].reindex(d.index).ffill()
        d["FNG"]           = fg.fillna(50)
        d["FNG_MA7"]       = d["FNG"].rolling(7).mean().fillna(50)
        d["FNG_chg"]       = d["FNG"].diff().fillna(0)
        d["extreme_fear"]  = (d["FNG"] < 20).astype(float)
        d["extreme_greed"] = (d["FNG"] > 80).astype(float)
    else:
        d["FNG"] = d["FNG_MA7"] = 50.0
        d["FNG_chg"] = d["extreme_fear"] = d["extreme_greed"] = 0.0

    # Макро: SPX
    if macro.get("spx") is not None:
        spx     = macro["spx"].reindex(d.index).ffill()
        spx_ret = spx.pct_change()
        d["spx_mom_7"]       = spx.pct_change(7).fillna(0)
        d["spx_mom_30"]      = spx.pct_change(30).fillna(0)
        d["spx_above_ma50"]  = (spx > spx.ewm(span=50).mean()).astype(float)
        d["btc_spx_corr_14"] = ret.rolling(14).corr(spx_ret).fillna(0)
        d["btc_spx_corr_30"] = ret.rolling(30).corr(spx_ret).fillna(0)
        d["btc_spx_diverg"]  = (ret - spx_ret).rolling(7).mean().fillna(0)
    else:
        for c in ["spx_mom_7","spx_mom_30","spx_above_ma50",
                  "btc_spx_corr_14","btc_spx_corr_30","btc_spx_diverg"]:
            d[c] = 0.0

    # Макро: DXY
    if macro.get("dxy") is not None:
        dxy     = macro["dxy"].reindex(d.index).ffill()
        dxy_ret = dxy.pct_change()
        d["dxy_mom_7"]       = dxy.pct_change(7).fillna(0)
        d["dxy_mom_30"]      = dxy.pct_change(30).fillna(0)
        d["dxy_above_ma50"]  = (dxy > dxy.ewm(span=50).mean()).astype(float)
        d["btc_dxy_corr_14"] = ret.rolling(14).corr(dxy_ret).fillna(0)
        d["btc_dxy_corr_30"] = ret.rolling(30).corr(dxy_ret).fillna(0)
    else:
        for c in ["dxy_mom_7","dxy_mom_30","dxy_above_ma50",
                  "btc_dxy_corr_14","btc_dxy_corr_30"]:
            d[c] = 0.0

    # Макро: VIX
    if macro.get("vix") is not None:
        vix = macro["vix"].reindex(d.index).ffill()
        d["vix_level"]    = vix.fillna(20)
        d["vix_ma_ratio"] = (vix / (vix.rolling(20).mean() + 1e-9)).fillna(1)
        d["vix_spike"]    = (vix > 30).astype(float)
        d["vix_regime"]   = pd.cut(
            vix, bins=[0,15,25,35,999], labels=[0,1,2,3]
        ).astype(float).fillna(1)
    else:
        d["vix_level"] = 20.0
        d["vix_ma_ratio"] = 1.0
        d["vix_spike"] = d["vix_regime"] = 0.0

    # ETH като leading indicator (не за самия ETH)
    if macro.get("eth") is not None and ticker != "ETH-USD":
        eth     = macro["eth"].reindex(d.index).ffill()
        eth_ret = eth.pct_change()
        d["eth_mom_lag1"]   = eth_ret.shift(1).fillna(0)
        d["eth_mom_3"]      = eth.pct_change(3).fillna(0)
        ratio = eth / (d["Close"] + 1e-9)
        d["eth_btc_ratio"]  = (ratio / (ratio.rolling(30).mean() + 1e-9)).fillna(1)
        d["eth_btc_diverg"] = (eth_ret - ret).rolling(5).mean().fillna(0)
        d["btc_eth_corr_7"] = ret.rolling(7).corr(eth_ret).fillna(0)
    else:
        for c in ["eth_mom_lag1","eth_mom_3","eth_btc_ratio",
                  "eth_btc_diverg","btc_eth_corr_7"]:
            d[c] = 0.0

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.ffill(inplace=True)
    d.fillna(0, inplace=True)
    return d.dropna()


# ==============================================================
# МЕК МАКРО-МНОЖИТЕЛ (от V44/V45)
# ==============================================================
def macro_scale(fng: float, spx_mom: float, dxy_mom: float,
                fund: float, funding_lim: float,
                direction: str = "LONG") -> float:
    if direction == "LONG":
        if   25 <= fng <= 70:                   fng_s = 1.0
        elif 18 <= fng < 25 or 70 < fng <= 78: fng_s = 0.6
        elif 10 <= fng < 18 or 78 < fng <= 85: fng_s = 0.3
        else:                                    fng_s = 0.0
        macro_s = 1.0 if (spx_mom>0 and dxy_mom<0) else \
                  0.75 if (spx_mom>0 or dxy_mom<0) else 0.45
    else:
        if   20 <= fng <= 55:                   fng_s = 1.0
        elif 15 <= fng < 20 or 55 < fng <= 65: fng_s = 0.6
        elif 10 <= fng < 15 or 65 < fng <= 75: fng_s = 0.3
        else:                                    fng_s = 0.0
        macro_s = 1.0 if (spx_mom<0 and dxy_mom>0) else \
                  0.75 if (spx_mom<0 or dxy_mom>0) else 0.45

    fund_s = 1.0 if fund < funding_lim*0.4 else \
             0.7 if fund < funding_lim else \
             0.3 if fund < funding_lim*2 else 0.0
    return float(fng_s * macro_s * fund_s)


# ==============================================================
# PORTFOLIO KELLY PENALTY
# ==============================================================
def portfolio_kelly_scale(active_tickers: list) -> dict:
    if len(active_tickers) <= 1:
        return {t: 1.0 for t in active_tickers}
    scales = {}
    for ticker in active_tickers:
        corrs = []
        for other in active_tickers:
            if other == ticker: continue
            pair = tuple(sorted([ticker, other]))
            c = ASSET_CORR.get(pair, 0.5)
            corrs.append(c)
        avg_c = np.mean(corrs) if corrs else 0.0
        n     = len(active_tickers)
        scale = 1.0 / (1.0 + avg_c * (n-1) * 0.5)
        scales[ticker] = float(np.clip(scale, 0.3, 1.0))
    return scales


# ==============================================================
# ГЕНЕРИРАНЕ НА СИГНАЛ ЗА ЕДИН АКТИВ
# ==============================================================
def predict_asset(ticker: str, data: pd.DataFrame,
                  bundle: dict) -> dict:
    """
    Зарежда запазения модел V45 и генерира сигнал за актива.
    """
    sc    = bundle["scaler"]
    xgb   = bundle["xgb"]
    lgb   = bundle["lgb"]
    feats = bundle["feats"]
    p     = bundle["params"]

    last = data.iloc[[-1]].copy()

    # Проверяваме дали всички характеристики са налице
    missing = [f for f in feats if f not in last.columns]
    if missing:
        return {"ticker": ticker, "error": f"Missing feats: {missing[:3]}"}

    # Предсказване
    try:
        X = safe_transform(sc, last, feats)
        px = xgb.predict_proba(X)[0]
        pl = lgb.predict_proba(X)[0]
        p_flat  = (px[0] + pl[0]) / 2
        p_long  = (px[1] + pl[1]) / 2
        p_short = (px[2] + pl[2]) / 2
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

    price   = float(last["Close"].values[0])
    atr     = float(last["ATR"].values[0])
    fng     = float(last.get("FNG_MA7", pd.Series([50])).values[0])
    spx_mom = float(last.get("spx_mom_7", pd.Series([0])).values[0])
    dxy_mom = float(last.get("dxy_mom_7", pd.Series([0])).values[0])
    fund    = float(last.get("FundingRate_MA", pd.Series([0])).values[0])
    ema55   = float(last.get("above_ema55", pd.Series([1])).values[0]) == 1.0
    ema200  = float(last.get("above_ema200", pd.Series([1])).values[0]) == 1.0

    xgb_conf    = p["xgb_conf"]
    funding_lim = p["funding_lim"]
    tp_mult     = p["tp_mult"]
    sl_mult     = p["sl_mult"]
    be_level    = p.get("be_level", 0.65)

    long_scale  = macro_scale(fng, spx_mom, dxy_mom, fund, funding_lim, "LONG")
    short_scale = macro_scale(fng, spx_mom, dxy_mom, fund, funding_lim, "SHORT")

    long_ok  = p_long  >= xgb_conf and long_scale  > 0 and ema55
    short_ok = p_short >= xgb_conf and short_scale > 0 and not ema200

    # Определяме посоката
    if long_ok and short_ok:
        chosen = "LONG" if p_long >= p_short else "SHORT"
    elif long_ok:
        chosen = "LONG"
    elif short_ok:
        chosen = "SHORT"
    else:
        chosen = None

    result = {
        "ticker":   ticker,
        "label":    ASSETS[ticker]["label"],
        "emoji":    ASSETS[ticker]["emoji"],
        "price":    price,
        "atr":      atr,
        "p_long":   p_long,
        "p_short":  p_short,
        "p_flat":   p_flat,
        "fng":      fng,
        "spx_mom":  spx_mom,
        "dxy_mom":  dxy_mom,
        "fund":     fund,
        "long_scale":  long_scale,
        "short_scale": short_scale,
        "ema55":    ema55,
        "ema200":   ema200,
        "chosen":   chosen,
        "is_trade": chosen is not None,
        "tp_mult":  tp_mult,
        "sl_mult":  sl_mult,
        "be_level": be_level,
        "xgb_conf": xgb_conf,
    }

    if chosen:
        scale = long_scale if chosen == "LONG" else short_scale
        avg_p = p_long if chosen == "LONG" else p_short

        # Sniper Entry: лимит -0.4% за LONG, +0.4% за SHORT
        if chosen == "LONG":
            entry_limit = price * (1 - SNIPER_PCT)
            tp_price    = entry_limit + atr * tp_mult
            sl_price    = entry_limit - atr * sl_mult
        else:
            entry_limit = price * (1 + SNIPER_PCT)
            tp_price    = entry_limit - atr * tp_mult
            sl_price    = entry_limit + atr * sl_mult

        sl_pct = abs(entry_limit - sl_price) / (entry_limit + 1e-9)

        # Half-Kelly
        R        = tp_mult / max(sl_mult, 1e-9)
        kelly_f  = (avg_p * R - (1 - avg_p)) / max(R, 1e-9)
        half_k   = max(0.0, kelly_f * 0.5)
        km       = float(np.clip(half_k / 0.20, 0.3, 1.2))
        eff_risk = PORTFOLIO_RISK * km * scale

        pos_frac = float(np.clip(eff_risk / (sl_pct + 1e-9), 0.3, 4.0))

        result.update({
            "entry_limit": entry_limit,
            "tp":          tp_price,
            "sl":          sl_price,
            "pos_frac":    pos_frac,
            "kelly_mult":  km,
            "scale":       scale,
        })

    return result


# ==============================================================
# ПРОВЕРКА НА PORTFOLIO HARD STOP
# ==============================================================
def check_hard_stop() -> tuple[bool, float]:
    """
    Чете portfolio_state и проверява общия PnL.
    Връща (is_stopped, current_dd).
    """
    state = load_json(PORTFOLIO_FILE)

    # Проверяваме активна пауза
    if state.get("hard_stop_until"):
        stop_until = pd.Timestamp(state["hard_stop_until"])
        if pd.Timestamp.now() < stop_until:
            days_left = (stop_until - pd.Timestamp.now()).days
            return True, state.get("portfolio_dd", 0), days_left

    # Изчисляваме текущата обща просадка
    completed_trades = state.get("completed_trades", [])
    if not completed_trades:
        return False, 0.0, 0

    pnls = [t["pnl_pct"] / 100 for t in completed_trades]
    equity = np.cumprod(1 + np.array(pnls))
    peak   = np.maximum.accumulate(equity)
    dd     = float((equity[-1] - peak[-1]) / peak[-1])

    if dd <= -PORTFOLIO_HARD_STOP:
        # Активираме паузата
        stop_until = pd.Timestamp.now() + pd.Timedelta(days=30)
        state["hard_stop_until"] = str(stop_until)
        state["portfolio_dd"]    = dd
        save_json(PORTFOLIO_FILE, state)
        return True, dd, 30

    return False, dd, 0


# ==============================================================
# ОБНОВЯВАНЕ НА ПОРТФЕЙЛНИЯ СТЕЙТ
# ==============================================================
def record_trade_close(ticker: str, pnl_pct: float, side: str):
    """Записва затворената сделка в portfolio_state за Hard Stop."""
    state = load_json(PORTFOLIO_FILE)
    trades = state.get("completed_trades", [])
    trades.append({
        "ticker": ticker,
        "side":   side,
        "pnl_pct": pnl_pct,
        "date":   datetime.now().strftime("%Y-%m-%d"),
    })
    state["completed_trades"] = trades
    save_json(PORTFOLIO_FILE, state)


# ==============================================================
# ОСНОВНА ФУНКЦИЯ: ГЕНЕРИРАНЕ НА СИГНАЛИ ЗА ПОРТФЕЙЛА
# ==============================================================
def run_portfolio_signal() -> dict:
    """
    Зарежда моделите и данните, генерира сигнали за всички активи.
    Прилага Portfolio Kelly penalty.
    """
    print("📡 Зареждане на макро данни...")
    macro = load_macro()

    results = {}
    for ticker in ASSETS:
        model_name = ticker.replace("-","_").lower()
        model_path = f"{MODELS_DIR}/v45_{model_name}.joblib"

        if not os.path.exists(model_path):
            print(f"  ⚠️  Моделът не е намерен: {model_path}")
            results[ticker] = {"ticker": ticker, "error": "Моделът не е намерен"}
            continue

        print(f"  📊 {ticker}...")
        bundle = joblib.load(model_path)
        start  = "2021-01-01" if "SOL" in ticker else "2019-01-01"
        data   = load_asset_data(ticker, macro, start=start)
        result = predict_asset(ticker, data, bundle)
        results[ticker] = result

    # Portfolio Kelly penalty — ако и двата дават сигнал
    active = [t for t, r in results.items()
              if r.get("is_trade") and not r.get("error")]

    if len(active) > 1:
        scales = portfolio_kelly_scale(active)
        for ticker in active:
            old_pos   = results[ticker]["pos_frac"]
            new_pos   = old_pos * scales[ticker]
            results[ticker]["pos_frac"]      = new_pos
            results[ticker]["kelly_penalty"] = scales[ticker]
            print(f"  Kelly penalty {ticker}: {scales[ticker]:.2f}x → поз {new_pos:.2f}x")
    else:
        for ticker in active:
            results[ticker]["kelly_penalty"] = 1.0

    return results


# ==============================================================
# ФОРМАТИРАНЕ НА СЪОБЩЕНИЯ ЗА TELEGRAM
# ==============================================================
def _fng_label(fng: float) -> str:
    if fng < 15:   return "😱 Екстремален страх"
    if fng < 30:   return "😰 Страх"
    if fng < 45:   return "😐 Неутрално-мечи"
    if fng < 55:   return "😐 Неутрален"
    if fng < 70:   return "🙂 Алчност"
    if fng < 85:   return "😤 Силна алчност"
    return              "🤑 Екстремална алчност"


def _bar(value: float, width: int = 10) -> str:
    """Прогрес-бар за вероятности."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _macro_status(spx_mom: float, dxy_mom: float) -> str:
    if spx_mom > 0 and dxy_mom < 0:
        return "🟢 Риск-ВКЛ"
    if spx_mom < 0 and dxy_mom > 0:
        return "🔴 Риск-ИЗКЛ"
    if spx_mom > 0:
        return "🟡 SPX↑ DXY↑"
    if dxy_mom < 0:
        return "🟡 SPX↓ DXY↓"
    return "⚪️ Неутрално"


def format_portfolio_signal(results: dict, hard_stop: bool,
                             portfolio_dd: float,
                             hard_stop_days: int = 0) -> str:
    now = datetime.now().strftime("%d.%m.%Y  %H:%M")

    # ── Hard Stop ──────────────────────────────────────────
    if hard_stop:
        return (
            f"🚨 *PORTFOLIO HARD STOP*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Търговията е спряна\n\n"
            f"📉 Просадка на портфейла: `{portfolio_dd*100:.1f}%`\n"
            f"⏳ До възобновяване: `{hard_stop_days} дни`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Ботът ще възобнови работата автоматично_"
        )

    lines = [
        f"╔══ 🤖 *V45 PORTFOLIO* ══╗",
        f"📅 `{now} UTC`",
        f"",
    ]

    any_trade = False

    for ticker, r in results.items():
        if r.get("error"):
            lines += [
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"{r.get('emoji','•')} *{r.get('label', ticker)}*  ❌ Грешка",
            ]
            continue

        label    = r["label"]
        emoji    = r["emoji"]
        price    = r["price"]
        p_long   = r["p_long"]
        p_short  = r["p_short"]
        p_flat   = r.get("p_flat", 1 - p_long - p_short)
        fng      = r["fng"]
        spx_mom  = r.get("spx_mom", 0)
        dxy_mom  = r.get("dxy_mom", 0)
        chosen   = r.get("chosen")

        macro_st = _macro_status(spx_mom, dxy_mom)
        fng_st   = _fng_label(fng)

        lines += [
            f"━━━━━━━━━━━━━━━━━━━━━━",
            f"{emoji} *{label}*   `${price:,.0f}`",
            f"",
            f"📊 *Вероятности на ML модела:*",
            f"  🟢 Ръст   `{_bar(p_long)}`  `{p_long:.0%}`",
            f"  🔴 Спад   `{_bar(p_short)}` `{p_short:.0%}`",
            f"  ⚪️ Страничен `{_bar(p_flat)}`  `{p_flat:.0%}`",
            f"",
            f"🌍 *Макро-контекст:*",
            f"  Пазар:  {macro_st}",
            f"  Страх:  `{fng:.0f}/100`  {fng_st}",
        ]

        if chosen:
            any_trade = True
            side_em   = "🟢" if chosen == "LONG" else "🔴"
            entry     = r["entry_limit"]
            tp        = r["tp"]
            sl        = r["sl"]
            pos       = r["pos_frac"]
            penalty   = r.get("kelly_penalty", 1.0)
            scale     = r.get("scale", 1.0)
            rr        = r["tp_mult"] / r["sl_mult"]
            be        = r.get("be_level", 0.65)

            lines += [
                f"",
                f"{'⚡️' if pos > 1.0 else '✳️'} *СИГНАЛ: {side_em} {chosen}*",
                f"  🎯 Вход (лимит):  `${entry:,.0f}`",
                f"  ✅ Take Profit:   `${tp:,.0f}`",
                f"  🛑 Stop Loss:     `${sl:,.0f}`",
                f"  📐 R/R:           `1 : {rr:.2f}`",
                f"  💼 Позиция:       `{pos:.2f}x`",
                f"  🔄 BE trailing:   при `{be*100:.0f}%` от пътя до TP",
                f"  📉 Macro scale:   `{scale:.2f}x`",
                f"  🔗 Kelly penalty: `{penalty:.2f}x`",
                f"  ⚠️ _Поставете лимитен ордер, не пазарен_",
            ]
        else:
            long_scale  = r.get("long_scale",  0)
            short_scale = r.get("short_scale", 0)
            reason = "FNG е екстремален" if fng < 18 or fng > 82 else \
                     "Макро не отговаря" if long_scale < 0.4 and short_scale < 0.4 else \
                     "ML увереност е ниска"
            lines += [
                f"",
                f"💤 *Няма сигнал*",
                f"  Причина: _{reason}_",
                f"  L-scale `{long_scale:.2f}` · S-scale `{short_scale:.2f}`",
            ]

    # ── Обобщение за портфейла ──────────────────────────────
    lines.append(f"")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")

    if portfolio_dd <= -0.10:
        dd_line = f"⚠️ Просадка на портфейла: `{portfolio_dd*100:.1f}%` (внимание)"
    elif portfolio_dd < 0:
        dd_line = f"📊 Просадка на портфейла: `{portfolio_dd*100:.1f}%`"
    else:
        dd_line = f"✅ Портфейлът е в норма"

    lines.append(dd_line)

    if any_trade:
        lines.append(f"")
        lines.append(f"⏰ _Сигналът е валиден до края на деня_")
    else:
        lines.append(f"💤 *Изчакване на вход — условията не са изпълнени*")

    lines.append(f"╚══════════════════════╝")

    return "\n".join(lines)


def format_status(results: dict, state: dict,
                  portfolio_dd: float) -> str:
    now = datetime.now().strftime("%d.%m.%Y  %H:%M")
    lines = [
        f"╔══ 📊 *СТАТУС НА ПОЗИЦИИТЕ* ══╗",
        f"`{now} UTC`",
        f"",
    ]

    has_positions = False
    for ticker, r in results.items():
        if r.get("error"): continue
        label = r["label"]
        emoji = r["emoji"]
        price = r["price"]

        asset_state = state.get(ticker, {})
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"{emoji} *{label}*  `${price:,.0f}`")

        if asset_state.get("in_trade"):
            has_positions = True
            entry = asset_state["entry_price"]
            side  = asset_state["side"]
            date  = asset_state.get("entry_date", "?")
            tp    = asset_state["tp"]
            sl    = asset_state["sl"]
            pos   = asset_state.get("pos_frac", 0)

            pnl = ((price - entry) / entry * 100
                   if side == "LONG"
                   else (entry - price) / entry * 100)

            pnl_em  = "🟢" if pnl > 0 else "🔴"
            side_em = "↗️" if side == "LONG" else "↘️"
            dist_tp = abs(tp - price) / price * 100
            dist_sl = abs(sl - price) / price * 100

            lines += [
                f"  {side_em} Позиция: *{side}* от `{date}`",
                f"  📥 Вход:   `${entry:,.0f}`",
                f"  📌 Сега:   `${price:,.0f}`",
                f"  {pnl_em} P&L:    `{pnl:+.2f}%`",
                f"  ✅ TP: `${tp:,.0f}` ({dist_tp:.1f}% до целта)",
                f"  🛑 SL: `${sl:,.0f}` ({dist_sl:.1f}% до стопа)",
                f"  💼 Размер: `{pos:.2f}x`",
            ]
        else:
            lines.append(f"  💤 Няма позиции")

    lines += [
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if portfolio_dd <= -0.10:
        lines.append(f"⚠️ Просадка: `{portfolio_dd*100:.1f}%`")
    elif portfolio_dd < 0:
        lines.append(f"📉 Просадка: `{portfolio_dd*100:.1f}%`")
    else:
        lines.append(f"✅ Портфейлът е в норма")

    if not has_positions:
        lines.append(f"💤 _Няма отворени позиции_")

    lines.append(f"╚══════════════════════════╝")
    return "\n".join(lines)

    return "\n".join(lines)





# ==============================================================
# ЗАПАЗВАНЕ НА СЪСТОЯНИЕТО НА ПОЗИЦИИТЕ
# ==============================================================
def process_signals(results: dict):
    """
    Запазва отворените позиции в STATE_FILE.
    Проверява затварянето на съществуващи позиции.
    """
    state = load_json(STATE_FILE)

    for ticker, r in results.items():
        if r.get("error"): continue

        asset_state = state.get(ticker, {})
        price       = r.get("price", 0)

        # Проверяваме затварянето на съществуваща позиция
        if asset_state.get("in_trade"):
            entry = asset_state["entry_price"]
            side  = asset_state["side"]
            tp    = asset_state["tp"]
            sl    = asset_state["sl"]

            tp_hit = (price >= tp) if side == "LONG" else (price <= tp)
            sl_hit = (price <= sl) if side == "LONG" else (price >= sl)

            if tp_hit or sl_hit:
                exit_price = tp if tp_hit else sl
                pnl = ((exit_price - entry) / entry * 100
                       if side == "LONG"
                       else (entry - exit_price) / entry * 100)
                result_str = "TP ✅" if tp_hit else "SL ❌"

                save_trade({
                    "Date":   datetime.now().strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Side":   side,
                    "Entry":  entry,
                    "Exit":   exit_price,
                    "PnL%":   f"{pnl:+.2f}",
                    "Result": result_str,
                })

                record_trade_close(ticker, pnl, side)
                state[ticker] = {"in_trade": False}

                send_telegram(
                    f"🔔 *{ASSETS[ticker]['emoji']} {ASSETS[ticker]['label']} "
                    f"позицията е затворена*\n"
                    f"{result_str} PnL: `{pnl:+.2f}%`\n"
                    f"Изход при: `${exit_price:,.0f}`"
                )

        # Отваряме нова позиция ако има сигнал и няма текуща
        if r.get("is_trade") and not asset_state.get("in_trade"):
            entry = r.get("entry_limit", price)
            state[ticker] = {
                "in_trade":    True,
                "side":        r["chosen"],
                "entry_price": entry,
                "entry_date":  datetime.now().strftime("%Y-%m-%d"),
                "tp":          r["tp"],
                "sl":          r["sl"],
                "pos_frac":    r["pos_frac"],
            }
            save_trade({
                "Date":   datetime.now().strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Side":   r["chosen"],
                "Entry":  entry,
                "TP":     r["tp"],
                "SL":     r["sl"],
                "Pos":    f"{r['pos_frac']:.2f}x",
                "Status": "OPEN",
            })

    save_json(STATE_FILE, state)
    return state


# ==============================================================
# СЛУШАТЕЛ НА КОМАНДИ TELEGRAM
# ==============================================================
def handle_commands():
    print("🛰 V45 Bot стартира в режим СЛУШАТЕЛ...")
    send_telegram(
        "🤖 *V45 Portfolio Bot стартира*\n"
        "Команди:\n"
        "`/signal` — текущи сигнали\n"
        "`/status` — статус на позициите\n"
        "`/portfolio` — общ P&L\n"
        "`/help` — помощ"
    )

    offset = None
    while True:
        try:
            for upd in get_updates(offset):
                offset = upd["update_id"] + 1
                msg    = upd.get("message", {})
                text   = msg.get("text", "").strip().lower()
                chat   = str(msg.get("chat", {}).get("id", ""))

                if chat != str(TG_CHAT_ID):
                    continue

                if text in ("/signal", "/сигнал"):
                    send_telegram("⏳ Изчислявам сигнали...")
                    results = run_portfolio_signal()
                    hard_stop, dd, days = check_hard_stop()
                    sig = format_portfolio_signal(results, hard_stop, dd, days)
                    send_telegram(sig)

                elif text in ("/status", "/статус"):
                    results = run_portfolio_signal()
                    state   = load_json(STATE_FILE)
                    _, dd, _ = check_hard_stop()
                    send_telegram(format_status(results, state, dd))

                elif text in ("/portfolio", "/портфейл"):
                    port  = load_json(PORTFOLIO_FILE)
                    trades = port.get("completed_trades", [])
                    if not trades:
                        send_telegram("📊 Все още няма завършени сделки.")
                    else:
                        pnls = [t["pnl_pct"] for t in trades[-20:]]
                        total = sum(pnls)
                        wins  = sum(1 for p in pnls if p > 0)
                        msg = (
                            f"📊 *ПОРТФЕЙЛ V45*\n"
                            f"Последни сделки: {len(pnls)}\n"
                            f"Win Rate: `{wins/len(pnls):.0%}`\n"
                            f"Общ P&L: `{total:+.1f}%`"
                        )
                        send_telegram(msg)

                elif text in ("/help", "/помощ"):
                    send_telegram(
                        "📖 *V45 Portfolio Bot*\n\n"
                        "`/signal` — сигнали BTC+ETH\n"
                        "`/status` — отворени позиции\n"
                        "`/portfolio` — последните 20 сделки\n"
                        "`/help` — тази помощ\n\n"
                        "Ботът работи на архитектура V45:\n"
                        "LONG+SHORT | Sniper Entry | Kelly | Hard Stop"
                    )

            time.sleep(2)
        except Exception as e:
            print(f"  Грешка: {e}")
            time.sleep(5)


# ==============================================================
# ТОЧКА НА ВЛИЗАНЕ
# ==============================================================
if __name__ == "__main__":
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(CACHE_DIR).mkdir(exist_ok=True)

    # Проверяваме наличието на модели
    for ticker in ASSETS:
        name = ticker.replace("-","_").lower()
        path = f"{MODELS_DIR}/v45_{name}.joblib"
        if not os.path.exists(path):
            print(f"⚠️  Моделът не е намерен: {path}")
            print(f"   Стартирайте grok_v45.py първо за обучение.")

    # Режим на слушател
    if len(sys.argv) > 1 and sys.argv[1] == "--listen":
        handle_commands()

    # Еднократен сигнал
    else:
        print("\n" + "="*50)
        print("  V45 PORTFOLIO BOT — СИГНАЛ")
        print("="*50)

        # Проверяваме Hard Stop
        hard_stop, portfolio_dd, days_left = check_hard_stop()
        if hard_stop:
            msg = (
                f"🛑 *HARD STOP АКТИВЕН*\n"
                f"Просадка: `{portfolio_dd*100:.1f}%`\n"
                f"Пауза: още `{days_left}` дни"
            )
            print(msg)
            send_telegram(msg)
        else:
            # Генерираме сигнали
            results = run_portfolio_signal()
            sig = format_portfolio_signal(
                results, False, portfolio_dd)

            print("\n" + sig + "\n")
            send_telegram(sig)

            # Запазваме състоянието
            process_signals(results)
            print("✅ Състоянието е запазено")