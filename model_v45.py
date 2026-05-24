import os
import warnings
import numpy as np
import pandas as pd
import optuna
import yfinance as yf
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================
# V45 "MULTI-ASSET PORTFOLIO"
#
# База: V44 (най-добър модел: +13.1%/год, MaxDD -4.2%, Sharpe 5.65)
#
# Ново: търгуваме 3 актива с независими модели V44
#   BTC-USD  — основен актив, пълна история от 2019
#   ETH-USD  — leading indicator, висока ликвидност
#   SOL-USD  — висока волатилност, добър ATR (от 2021)
#
# Управление на портфейла:
#   - Всеки актив: независим модел V44 (LONG+SHORT+Sniper)
#   - Общ капитал: PORTFOLIO_RISK се разпределя по активи
#   - Correlation penalty: при едновременни сигнали на корелирани
#     активи позицията на всеки намалява
#   - Portfolio Kelly: общият риск не надвишава MAX_PORTFOLIO_RISK
#
# Очакван резултат vs V44:
#   Сделки/год: 15 → 40-50 (статистически значимо)
#   Доходност/год:  13% → 15-22% (зависи от корелацията)
#   MaxDD:      -4.2% → -6-10% (корелацията увеличава риска)
# ==============================================================

# ── Портфейлни константи ─────────────────────────────────────
ASSETS = {
    "BTC-USD": {
        "start":        "2019-01-01",
        "funding_file": "funding_rate.csv",
        "cache_dir":    "onchain_cache",
        "tc":           0.0002,
    },
    "ETH-USD": {
        "start":        "2019-01-01",
        "funding_file": None,   # няма отделен файл — използваме 0
        "cache_dir":    "onchain_cache",
        "tc":           0.0002,
    },
    "SOL-USD": {
        "start":        "2021-01-01",   # SOL стана ликвиден от 2021
        "funding_file": None,
        "cache_dir":    "onchain_cache",
        "tc":           0.0003,         # малко по-висок заради спреда
    },
}

PORTFOLIO_RISK      = 0.05   # риск на сделка — НЕ ПРОМЕНЯЙ
MAX_PORTFOLIO_RISK  = 0.12   # общ hard cap
PORTFOLIO_HARD_STOP = 0.15   # спри ВСИЧКИ сделки при -15% просадка на портфейла
#                              При достигане → пауза 30 дни → след това възобновяване
MAX_POS           = 4.0
MIN_POS           = 0.3
RETRAIN_DAYS      = 90
SNIPER_PCT        = 0.004
MODELS_DIR        = "models_v45"
Path(MODELS_DIR).mkdir(exist_ok=True)

# Матрица на корелация (исторически средни, обновяваме ръчно)
# Използва се за Portfolio Kelly penalty
ASSET_CORR = {
    ("BTC-USD", "ETH-USD"): 0.88,
    ("BTC-USD", "SOL-USD"): 0.80,
    ("ETH-USD", "SOL-USD"): 0.82,
}


# ==============================================================
# СТЪПКА 1: ЗАРЕЖДАНЕ НА МАКРО ДАННИ (веднъж за целия портфейл)
# ==============================================================
def load_macro() -> dict:
    """Зарежда SPX, DXY, VIX, ETH веднъж за всички активи."""
    print("  📥 Зареждане на макро данни...")
    macro = {}

    def _get(ticker, name):
        try:
            df = yf.download(ticker, start="2018-01-01",
                              progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            s = df["Close"].rename(name)
            print(f"    ✅ {name}: {len(s)} реда")
            return s
        except Exception as e:
            print(f"    ⚠️  {name}: {e}")
            return None

    macro["spx"] = _get("^GSPC",    "SPX")
    macro["dxy"] = _get("DX-Y.NYB", "DXY")
    macro["vix"] = _get("^VIX",     "VIX")
    macro["eth"] = _get("ETH-USD",  "ETH_macro")
    macro["fg"]  = _load_fear_greed()
    return macro


def _load_fear_greed() -> pd.Series | None:
    fg_path = "onchain_cache/fear_greed.csv"
    if not os.path.exists(fg_path):
        return None
    try:
        fg = pd.read_csv(fg_path, index_col=0, parse_dates=True)
        fg.index = pd.to_datetime(fg.index).normalize()
        col = "fng_value" if "fng_value" in fg.columns else fg.columns[0]
        s = fg[col].rename("FNG")
        print(f"    ✅ Fear&Greed: {len(s)} реда")
        return s
    except Exception as e:
        print(f"    ⚠️  Fear&Greed: {e}")
        return None


# ==============================================================
# СТЪПКА 2: ЗАРЕЖДАНЕ НА ДАННИ ЗА ЕДИН АКТИВ
# ==============================================================
def load_asset(ticker: str, cfg: dict, macro: dict) -> pd.DataFrame:
    """Зарежда OHLCV + on-chain + макро характеристики за един актив."""
    start = cfg["start"]
    tc    = cfg["tc"]

    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()

    d   = df.copy()
    ret = d["Close"].pct_change()

    # ── Технически индикатори ─────────────────────────────
    for p in [3, 7, 14, 30]:
        d[f"mom_{p}"] = d["Close"].pct_change(p)
        d[f"vol_{p}"] = ret.rolling(p).std()

    delta = d["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

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

    # ── Funding Rate (само BTC) ──────────────────────────
    if cfg["funding_file"] and os.path.exists(cfg["funding_file"]):
        try:
            f = pd.read_csv(cfg["funding_file"], parse_dates=["Date"])
            f["Date"] = pd.to_datetime(f["Date"]).dt.normalize()
            fr = f.groupby("Date")["FundingRate"].sum().to_frame()
            d  = d.join(fr, how="left")
            d["FundingRate"]     = d["FundingRate"].fillna(0)
            d["FundingRate_MA"]  = d["FundingRate"].rolling(7).mean().fillna(0)
            d["FundingRate_chg"] = d["FundingRate"].diff().fillna(0)
        except Exception:
            d["FundingRate"] = d["FundingRate_MA"] = d["FundingRate_chg"] = 0.0
    else:
        d["FundingRate"] = d["FundingRate_MA"] = d["FundingRate_chg"] = 0.0

    # ── Fear & Greed ───────────────────────────────────────
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

    # ── Макро: SPX + DXY + VIX ────────────────────────────
    btc_ret = ret.copy()

    if macro.get("spx") is not None:
        spx = macro["spx"].reindex(d.index).ffill()
        spx_ret = spx.pct_change()
        d["spx_mom_7"]       = spx.pct_change(7).fillna(0)
        d["spx_mom_30"]      = spx.pct_change(30).fillna(0)
        d["spx_above_ma50"]  = (spx > spx.ewm(span=50).mean()).astype(float)
        d["btc_spx_corr_14"] = btc_ret.rolling(14).corr(spx_ret).fillna(0)
        d["btc_spx_corr_30"] = btc_ret.rolling(30).corr(spx_ret).fillna(0)
        d["btc_spx_diverg"]  = (btc_ret - spx_ret).rolling(7).mean().fillna(0)
    else:
        for c in ["spx_mom_7","spx_mom_30","spx_above_ma50",
                  "btc_spx_corr_14","btc_spx_corr_30","btc_spx_diverg"]:
            d[c] = 0.0

    if macro.get("dxy") is not None:
        dxy = macro["dxy"].reindex(d.index).ffill()
        dxy_ret = dxy.pct_change()
        d["dxy_mom_7"]       = dxy.pct_change(7).fillna(0)
        d["dxy_mom_30"]      = dxy.pct_change(30).fillna(0)
        d["dxy_above_ma50"]  = (dxy > dxy.ewm(span=50).mean()).astype(float)
        d["btc_dxy_corr_14"] = btc_ret.rolling(14).corr(dxy_ret).fillna(0)
        d["btc_dxy_corr_30"] = btc_ret.rolling(30).corr(dxy_ret).fillna(0)
    else:
        for c in ["dxy_mom_7","dxy_mom_30","dxy_above_ma50",
                  "btc_dxy_corr_14","btc_dxy_corr_30"]:
            d[c] = 0.0

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

    # ── ETH като leading indicator (само за BTC/SOL) ────
    if macro.get("eth") is not None and ticker != "ETH-USD":
        eth = macro["eth"].reindex(d.index).ffill()
        eth_ret = eth.pct_change()
        d["eth_mom_lag1"]   = eth_ret.shift(1).fillna(0)
        d["eth_mom_3"]      = eth.pct_change(3).fillna(0)
        ratio = eth / (d["Close"] + 1e-9)
        d["eth_btc_ratio"]  = (ratio / (ratio.rolling(30).mean() + 1e-9)).fillna(1)
        d["eth_btc_diverg"] = (eth_ret - btc_ret).rolling(5).mean().fillna(0)
        d["btc_eth_corr_7"] = btc_ret.rolling(7).corr(eth_ret).fillna(0)
    else:
        for c in ["eth_mom_lag1","eth_mom_3","eth_btc_ratio",
                  "eth_btc_diverg","btc_eth_corr_7"]:
            d[c] = 0.0

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.ffill(inplace=True)
    d.fillna(0, inplace=True)
    return d.dropna()


# ==============================================================
# СТЪПКА 3: ТАРГЕТ + МАКРО-СКЕЙЛЕР (от V44)
# ==============================================================
def make_target(data: pd.DataFrame, f_days: int,
                threshold: float) -> pd.DataFrame:
    future_ret = data["Close"].shift(-f_days) / data["Close"] - 1
    data = data.copy()
    data["Target"] = 0
    data.loc[future_ret >  threshold, "Target"] = 1
    data.loc[future_ret < -threshold, "Target"] = 2
    return data


def macro_scale(fng: float, spx_mom: float, dxy_mom: float,
                fund: float, funding_lim: float,
                direction: str = "LONG") -> float:
    if direction == "LONG":
        if   25 <= fng <= 70:                      fng_s = 1.0
        elif 18 <= fng < 25 or 70 < fng <= 78:    fng_s = 0.6
        elif 10 <= fng < 18 or 78 < fng <= 85:    fng_s = 0.3
        else:                                       fng_s = 0.0
        macro_s = 1.0 if (spx_mom>0 and dxy_mom<0) else \
                  0.75 if (spx_mom>0 or dxy_mom<0) else 0.45
    else:
        if   20 <= fng <= 55:                      fng_s = 1.0
        elif 15 <= fng < 20 or 55 < fng <= 65:    fng_s = 0.6
        elif 10 <= fng < 15 or 65 < fng <= 75:    fng_s = 0.3
        else:                                       fng_s = 0.0
        macro_s = 1.0 if (spx_mom<0 and dxy_mom>0) else \
                  0.75 if (spx_mom<0 or dxy_mom>0) else 0.45

    fund_s = 1.0 if fund < funding_lim*0.4 else \
             0.7 if fund < funding_lim else \
             0.3 if fund < funding_lim*2 else 0.0
    return float(fng_s * macro_s * fund_s)


# ==============================================================
# СТЪПКА 4: PORTFOLIO KELLY PENALTY
#
# Когато няколко актива дават сигнал едновременно,
# позицията на всеки намалява пропорционално на корелацията.
# Това предотвратява концентрация на риска в един пазарен режим.
# ==============================================================
def portfolio_kelly_scale(active_signals: list[str]) -> dict[str, float]:
    """
    active_signals: списък активи със сигнали в този ден
    Връща: {ticker: scale_factor}

    Логика:
    - 1 актив:   scale = 1.0 (пълен Kelly)
    - 2 актива:  scale = 1 / (1 + corr(A,B))
    - 3 актива:  scale отчита всички двойни корелации
    """
    if len(active_signals) <= 1:
        return {t: 1.0 for t in active_signals}

    n = len(active_signals)
    scales = {}

    for ticker in active_signals:
        # Средна корелация с останалите активни
        corrs = []
        for other in active_signals:
            if other == ticker:
                continue
            pair = (min(ticker, other), max(ticker, other))
            # Проверяваме двата реда
            c = ASSET_CORR.get(pair) or ASSET_CORR.get((pair[1], pair[0])) or 0.5
            corrs.append(c)
        avg_corr = np.mean(corrs) if corrs else 0.0
        # Формула: scale = 1 / (1 + avg_corr * (n-1) * 0.5)
        scale = 1.0 / (1.0 + avg_corr * (n - 1) * 0.5)
        scales[ticker] = float(np.clip(scale, 0.3, 1.0))

    return scales


# ==============================================================
# СТЪПКА 5: СИМУЛАЦИЯ НА ЕДИН АКТИВ (адаптирано от V44)
# ==============================================================
def simulate_asset(test: pd.DataFrame,
                   f_days: int, tp_mult: float, sl_mult: float,
                   xgb_conf: float, funding_lim: float,
                   be_level: float, tc: float,
                   portfolio_scales: dict | None = None) -> tuple:
    """
    Връща (trades_list, dates_list) за портфейлно отчитане.
    portfolio_scales: {date: scale} — множител от Portfolio Kelly
    """
    trades_list = []   # (date, pnl, side, ticker)
    in_t        = False
    side        = "LONG"
    entry_p     = bars_in = tp = sl = pos = 0.0
    sl_be       = False
    loss_streak = 0
    pause_bars  = 0

    for i in range(1, len(test)):
        row  = test.iloc[i]
        prev = test.iloc[i - 1]
        date = test.index[i]

        if in_t:
            bars_in += 1
            hi = float(row["High"]); lo = float(row["Low"])
            if side == "LONG":
                if be_level > 0 and not sl_be:
                    if hi >= entry_p + (tp - entry_p) * be_level:
                        sl = max(sl, entry_p); sl_be = True
                if hi >= tp or lo <= sl or bars_in >= f_days:
                    exit_p = tp if hi >= tp else (sl if lo <= sl else float(row["Close"]))
                    pnl = (exit_p / entry_p - 1 - tc) * pos
                    trades_list.append((date, pnl, "LONG"))
                    in_t = False; sl_be = False
            else:
                if be_level > 0 and not sl_be:
                    if lo <= entry_p - (entry_p - tp) * be_level:
                        sl = min(sl, entry_p); sl_be = True
                if lo <= tp or hi >= sl or bars_in >= f_days:
                    exit_p = tp if lo <= tp else (sl if hi >= sl else float(row["Close"]))
                    pnl = (entry_p / exit_p - 1 - tc) * pos
                    trades_list.append((date, pnl, "SHORT"))
                    in_t = False; sl_be = False

            if not in_t:
                last_pnl = trades_list[-1][1]
                if last_pnl <= 0:
                    loss_streak += 1
                    if loss_streak >= 3:
                        pause_bars = 14; loss_streak = 0
                else:
                    loss_streak = 0
        else:
            if pause_bars > 0:
                pause_bars -= 1; continue

            p_long  = float(prev.get("P_long",  0))
            p_short = float(prev.get("P_short", 0))
            fng     = float(prev.get("FNG_MA7",    50))
            spx_mom = float(prev.get("spx_mom_7",   0))
            dxy_mom = float(prev.get("dxy_mom_7",   0))
            fund    = float(prev.get("FundingRate_MA", 0))
            atr_v   = float(prev["ATR"])
            cv      = float(prev["Close"])
            ema55   = float(prev.get("above_ema55",  1)) == 1.0
            ema200  = float(prev.get("above_ema200", 1)) == 1.0

            ls = macro_scale(fng, spx_mom, dxy_mom, fund, funding_lim, "LONG")
            ss = macro_scale(fng, spx_mom, dxy_mom, fund, funding_lim, "SHORT")

            long_ok  = p_long  >= xgb_conf and ls > 0 and ema55
            short_ok = p_short >= xgb_conf and ss > 0 and not ema200

            chosen = None
            if long_ok and short_ok:
                chosen = "LONG" if p_long >= p_short else "SHORT"
            elif long_ok:  chosen = "LONG"
            elif short_ok: chosen = "SHORT"
            if chosen is None: continue

            scale = ls if chosen == "LONG" else ss
            avg_p = p_long if chosen == "LONG" else p_short

            # Portfolio Kelly penalty
            if portfolio_scales and date in portfolio_scales:
                scale *= portfolio_scales[date]

            lo_d = float(row["Low"]); hi_d = float(row["High"])
            op   = float(row["Open"])
            if chosen == "LONG":
                lim  = cv * (1 - SNIPER_PCT)
                entry_p = lim if lo_d <= lim else op
                tp = entry_p + atr_v * tp_mult
                sl = entry_p - atr_v * sl_mult
            else:
                lim  = cv * (1 + SNIPER_PCT)
                entry_p = lim if hi_d >= lim else op
                tp = entry_p - atr_v * tp_mult
                sl = entry_p + atr_v * sl_mult

            sl_pct = abs(entry_p - sl) / (entry_p + 1e-9)
            R = tp_mult / max(sl_mult, 1e-9)
            kf  = (avg_p * R - (1 - avg_p)) / max(R, 1e-9)
            hk  = max(0.0, kf * 0.5)
            km  = float(np.clip(hk / 0.20, 0.3, 1.2))  # cap 1.2x (беше 1.5x)
            eff = PORTFOLIO_RISK * km * scale

            pos  = float(np.clip(eff / (sl_pct + 1e-9), MIN_POS, MAX_POS))
            side = chosen; in_t = True; bars_in = 0; sl_be = False

    return trades_list


# ==============================================================
# СТЪПКА 6: ML УТИЛИТИ
# ==============================================================
def safe_transform(sc, df, feats):
    X = sc.transform(df[feats].copy())
    return np.where(np.isfinite(X), X, 0.0)


def train_models(train, feats, sc):
    X_tr = safe_transform(sc, train, feats)
    y_tr = train["Target"].values
    xgb  = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, verbosity=0, random_state=42,
        eval_metric="mlogloss", objective="multi:softprob", num_class=3,
    )
    lgb = LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", verbose=-1, random_state=42,
        objective="multiclass", num_class=3,
    )
    xgb.fit(X_tr, y_tr); lgb.fit(X_tr, y_tr)
    return xgb, lgb


def get_feats(data: pd.DataFrame) -> list:
    return [c for c in data.columns if c not in
            {"Open","High","Low","Close","Volume","Target",
             "P_long","P_short"}]


# ==============================================================
# СТЪПКА 7: WALK-FORWARD ЗА ЕДИН АКТИВ
# ==============================================================
def wf_predict(data: pd.DataFrame, feats: list,
               f_days: int, threshold: float,
               train_window: int = 700) -> pd.DataFrame:
    """
    Стартира WF преобучение и попълва P_long/P_short за целия data.
    Връща data с попълнени колони за прогнози.
    """
    preds_long  = {}
    preds_short = {}
    n           = len(data)
    start       = train_window
    last_r      = start
    xgb = lgb = sc = vf = None

    for i in range(start, n - 1):
        if xgb is None or (i - last_r) >= RETRAIN_DAYS:
            tr  = data.iloc[i - train_window: i].copy()
            lb  = make_target(tr, f_days, threshold)
            lb  = lb.dropna(subset=["Target"] + [f for f in feats if f in lb.columns])
            if len(lb) < 80: continue
            pos_r = (lb["Target"] == 1).mean()
            if pos_r < 0.05 or pos_r > 0.70: continue
            vf2 = [f for f in feats if f in lb.columns and lb[f].std() > 1e-10]
            if len(vf2) < 5: continue
            try:
                sc2 = StandardScaler(); sc2.fit(lb[vf2])
                x2, l2 = train_models(lb, vf2, sc2)
                xgb, lgb, sc, vf = x2, l2, sc2, vf2
                last_r = i
            except Exception:
                continue

        if xgb is None: continue
        try:
            cf = list(sc.feature_names_in_) if hasattr(sc,"feature_names_in_") else vf
            X  = safe_transform(sc, data.iloc[[i]], cf)
            px = xgb.predict_proba(X)[0]
            pl = lgb.predict_proba(X)[0]
            preds_long[data.index[i]]  = float((px[1] + pl[1]) / 2)
            preds_short[data.index[i]] = float((px[2] + pl[2]) / 2)
        except Exception:
            continue

    data = data.copy()
    data["P_long"]  = pd.Series(preds_long)
    data["P_short"] = pd.Series(preds_short)
    return data


# ==============================================================
# СТЪПКА 8: OPTUNA ЗА ЕДИН АКТИВ
# ==============================================================
def optimize_asset(ticker: str, train_data: pd.DataFrame,
                   feats: list, tc: float, n_trials: int = 300) -> dict:
    """Оптимизира хиперпараметрите V44 за един актив."""
    print(f"\n  🔧 Optuna за {ticker} ({n_trials} трайла)...")

    # Фиксираме seed отделно за всеки актив — възпроизводимост
    asset_seeds = {"BTC-USD": 42, "ETH-USD": 43, "SOL-USD": 44}
    seed = asset_seeds.get(ticker, 42)

    def objective(trial):
        f_days      = trial.suggest_int(  "f_days",      4,   10)
        tp_mult     = trial.suggest_float("tp_mult",     1.5,  6.0)
        sl_mult     = trial.suggest_float("sl_mult",     1.0,  4.0)
        threshold   = trial.suggest_float("threshold",   0.01, 0.05)
        xgb_conf    = trial.suggest_float("xgb_conf",   0.40, 0.62)
        funding_lim = trial.suggest_float("funding_lim", 0.0001, 0.002)
        be_level    = trial.suggest_float("be_level",    0.50,  0.85)

        labeled = make_target(train_data, f_days, threshold).dropna()
        if len(labeled) < 100: raise optuna.exceptions.TrialPruned()
        pos_r = (labeled["Target"] == 1).mean()
        if pos_r < 0.05 or pos_r > 0.65: raise optuna.exceptions.TrialPruned()

        tscv = TimeSeriesSplit(n_splits=5, gap=f_days)
        fold_sharpes = []

        for tr_i, te_i in tscv.split(labeled):
            train = labeled.iloc[tr_i].copy()
            test  = labeled.iloc[te_i].copy()
            if len(train) < 80 or len(test) < 20: continue

            sc = StandardScaler(); sc.fit(train[feats])
            xgb, lgb = train_models(train, feats, sc)
            test = test.copy()
            px = xgb.predict_proba(safe_transform(sc, test, feats))
            pl = lgb.predict_proba(safe_transform(sc, test, feats))
            test["P_long"]  = (px[:, 1] + pl[:, 1]) / 2
            test["P_short"] = (px[:, 2] + pl[:, 2]) / 2

            trades_raw = simulate_asset(test, f_days, tp_mult, sl_mult,
                                         xgb_conf, funding_lim, be_level, tc)
            trades = np.array([t[1] for t in trades_raw])
            if len(trades) < 5: raise optuna.exceptions.TrialPruned()

            n = len(trades)
            s = np.mean(trades) / (np.std(trades) + 1e-9) * np.sqrt(252)
            s *= min(1.0, np.sqrt(n / 15))
            fold_sharpes.append(s)

        if len(fold_sharpes) < 3: raise optuna.exceptions.TrialPruned()
        return float(np.min(fold_sharpes))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=30, n_ei_candidates=48),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=25),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    try:
        bp = study.best_params
        bp["tc"] = tc
        print(f"    Най-добър Sharpe: {study.best_value:.3f} | "
              f"f={bp['f_days']} | R/R={bp['tp_mult']/bp['sl_mult']:.2f}")
        return bp
    except ValueError:
        print(f"    ⚠️  Всички трайли pruned за {ticker}")
        return None


# ==============================================================
# СТЪПКА 9: ПОРТФЕЙЛЕН WALK-FORWARD ТЕСТ с Hard Stop
# ==============================================================
def _build_preds(asset_data, asset_params):
    """Изгражда WF прогнози за всички активи."""
    asset_preds = {}
    for ticker, data in asset_data.items():
        if ticker not in asset_params or asset_params[ticker] is None:
            continue
        p     = asset_params[ticker]
        feats = get_feats(data)
        print(f"\n  📊 WF прогнози: {ticker}...")
        data_wf = wf_predict(data, feats, p["f_days"], p["threshold"])
        print(f"     {data_wf['P_long'].notna().sum()} прогнози")
        asset_preds[ticker] = data_wf
    return asset_preds


def _build_kelly_scales(asset_preds, asset_params):
    """Изгражда Portfolio Kelly penalty по дни."""
    signal_cache = {}
    for ticker, data in asset_preds.items():
        conf = asset_params[ticker]["xgb_conf"]
        for idx in data.index:
            if pd.isna(data.at[idx, "P_long"]): continue
            if data.at[idx,"P_long"] >= conf or data.at[idx,"P_short"] >= conf:
                signal_cache.setdefault(idx, []).append(ticker)

    scales_by_asset = {t: {} for t in asset_preds}
    for date, tickers in signal_cache.items():
        for t, s in portfolio_kelly_scale(tickers).items():
            scales_by_asset[t][date] = s
    return scales_by_asset


def apply_portfolio_hard_stop(all_trades_by_ticker: dict,
                               hard_stop: float = PORTFOLIO_HARD_STOP,
                               pause_days: int = 30) -> dict:
    """
    Прилага Portfolio Hard Stop към общия портфейл.
    При просадка >= hard_stop всички активи отиват на пауза от pause_days дни.
    
    Алгоритъм:
    1. Обединяваме всички сделки в единен времеви ред
    2. Изчисляваме кумулативния PnL на портфейла
    3. При достигане на hard_stop — премахваме следващите pause_days сделки
    """
    # Събираме всички сделки с дати в един списък
    all_trades_flat = []
    for ticker, trades in all_trades_by_ticker.items():
        for date, pnl, side in trades:
            all_trades_flat.append((date, pnl, side, ticker))
    
    if not all_trades_flat:
        return all_trades_by_ticker
    
    # Сортираме по дата
    all_trades_flat.sort(key=lambda x: x[0])
    
    # Прилагаме hard stop
    equity      = 1.0
    peak        = 1.0
    pause_until = None
    filtered    = []
    stopped_n   = 0

    for date, pnl, side, ticker in all_trades_flat:
        # Проверяваме паузата
        if pause_until is not None and date <= pause_until:
            stopped_n += 1
            continue

        # Добавяме сделката
        filtered.append((date, pnl, side, ticker))
        equity *= (1 + pnl)
        peak    = max(peak, equity)
        dd      = (equity - peak) / peak

        # Проверяваме hard stop
        if dd <= -hard_stop:
            pause_until = pd.Timestamp(date) + pd.Timedelta(days=pause_days)
            equity      = equity  # не нулираме, продължаваме от текущото ниво
            peak        = equity  # нулираме peak след паузата

    if stopped_n > 0:
        print(f"    🛑 Hard Stop се активира: {stopped_n} сделки пропуснати")

    # Възстановяваме структурата по активи
    result = {t: [] for t in all_trades_by_ticker}
    for date, pnl, side, ticker in filtered:
        result[ticker].append((date, pnl, side))
    return result


def portfolio_wf_test(asset_data: dict, asset_params: dict,
                      combo_label: str = "BTC+ETH+SOL") -> dict:
    """
    Walk-forward тест с Portfolio Hard Stop и Portfolio Kelly.
    combo_label — наименование на комбинацията за извеждане.
    """
    print("\n" + "="*62)
    print(f"  ПОРТФЕЙЛЕН WF ТЕСТ | {combo_label}")
    print(f"  Hard Stop: -{PORTFOLIO_HARD_STOP*100:.0f}% → пауза 30 дни")
    print("="*62)

    asset_preds  = _build_preds(asset_data, asset_params)
    kelly_scales = _build_kelly_scales(asset_preds, asset_params)

    # Симулираме всеки актив
    raw_trades = {}
    for ticker, data in asset_preds.items():
        p     = asset_params[ticker]
        split = int(len(data) * 0.70)
        oos   = data.iloc[split - 700:].copy()
        trades = simulate_asset(
            oos, p["f_days"], p["tp_mult"], p["sl_mult"],
            p["xgb_conf"], p["funding_lim"], p["be_level"], p["tc"],
            portfolio_scales=kelly_scales.get(ticker, {})
        )
        raw_trades[ticker] = trades
        print(f"  {ticker}: {len(trades)} сделки (до hard stop)")

    # Прилагаме Portfolio Hard Stop
    print("\n  Прилагаме Portfolio Hard Stop...")
    all_trades = apply_portfolio_hard_stop(raw_trades)

    # Стъпка 4: обединяваме в портфейл
    _print_portfolio_stats(all_trades, asset_data)
    return all_trades


def _print_portfolio_stats(all_trades: dict, asset_data: dict):
    """Извежда статистика по портфейла."""
    print("\n" + "="*62)
    print("  СТАТИСТИКА ПО АКТИВИ")
    print("="*62)

    combined_pnl = []

    for ticker, trades_raw in all_trades.items():
        if not trades_raw:
            print(f"\n  [{ticker}] ❌ Няма сделки")
            continue

        trades = np.array([t[1] for t in trades_raw])
        sides  = [t[2] for t in trades_raw]
        n_long  = sides.count("LONG")
        n_short = sides.count("SHORT")

        data   = asset_data[ticker]
        n_days = (data.index[-1] - data.index[int(len(data)*0.70)]).days
        if n_days < 1: n_days = 365

        wins   = trades[trades > 0]
        losses = trades[trades <= 0]
        wr     = len(wins) / len(trades) * 100
        pf     = abs(np.sum(wins) / (np.sum(losses) + 1e-9))
        total  = np.sum(trades) * 100
        avg    = np.mean(trades) * 100
        sharpe = np.mean(trades) / (np.std(trades) + 1e-9) * np.sqrt(252)
        cum    = np.cumprod(1 + trades)
        max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1)) * 100
        ann    = (1 + total/100) ** (365 / n_days) - 1

        combined_pnl.extend(trades.tolist())

        print(f"\n  ── {ticker} ──────────────────────────────")
        print(f"  Сделки: {len(trades)} ({n_long}L/{n_short}S) | {len(trades)/(n_days/365):.1f}/год")
        print(f"  WR: {wr:.1f}% | PF: {pf:.2f} | Avg: {avg:+.3f}%")
        print(f"  Доходност/год: {ann*100:+.1f}% | MaxDD: {max_dd:.1f}%")
        print(f"  Sharpe: {sharpe:.3f}")

    # Портфейлна статистика
    if not combined_pnl:
        return
    print("\n" + "="*62)
    print("  ПОРТФЕЙЛ ОБЩО")
    print("="*62)

    p_trades = np.array(combined_pnl)
    p_wins   = p_trades[p_trades > 0]
    p_losses = p_trades[p_trades <= 0]
    p_wr     = len(p_wins) / len(p_trades) * 100
    p_pf     = abs(np.sum(p_wins) / (np.sum(p_losses) + 1e-9))
    p_total  = np.sum(p_trades) * 100
    p_avg    = np.mean(p_trades) * 100
    p_sharpe = np.mean(p_trades) / (np.std(p_trades) + 1e-9) * np.sqrt(252)
    p_cum    = np.cumprod(1 + p_trades)
    p_dd     = float(np.min(p_cum / np.maximum.accumulate(p_cum) - 1)) * 100

    # Вземаме среден период по активи
    n_assets = len([t for t in all_trades if all_trades[t]])
    avg_days = 796  # ~2.18 години OOS
    p_ann    = (1 + p_total/100) ** (365 / avg_days) - 1

    print(f"  Сделки общо:       {len(p_trades)}")
    print(f"  Сделки/год:         {len(p_trades)/(avg_days/365):.1f}")
    print(f"  Win Rate:           {p_wr:.1f}%")
    print(f"  Profit Factor:      {p_pf:.2f}")
    print(f"  Чиста печалба:     {p_total:+.2f}%")
    print(f"  Годишна доходност: {p_ann*100:+.1f}%")
    print(f"  Средна сделка:    {p_avg:+.3f}%")
    print(f"  Portfolio Sharpe:   {p_sharpe:.3f}")
    print(f"  Portfolio MaxDD:    {p_dd:.1f}%")

    # Equity chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                                  gridspec_kw={"height_ratios":[3,2,1]})
        fig.suptitle("V45 Portfolio OOS | BTC + ETH + SOL", fontsize=12)

        colors = {"BTC-USD":"steelblue","ETH-USD":"darkorange","SOL-USD":"seagreen"}
        for ticker, trades_raw in all_trades.items():
            if not trades_raw: continue
            t_arr = np.array([t[1] for t in trades_raw])
            cum   = np.cumprod(1 + t_arr)
            ann_r = (cum[-1] ** (365/avg_days) - 1) * 100
            axes[0].plot(cum, color=colors.get(ticker,"gray"),
                         lw=1.5, label=f"{ticker} ({ann_r:+.1f}%/год)")

        axes[0].axhline(1.0, color="gray", lw=0.8, ls="--")
        axes[0].legend(fontsize=8); axes[0].set_ylabel("Equity")
        axes[0].set_title("Отделни активи", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        p_cum_plot = np.cumprod(1 + p_trades)
        axes[1].plot(p_cum_plot, color="purple", lw=2,
                     label=f"Portfolio ({p_ann*100:+.1f}%/год)")
        axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
        axes[1].legend(fontsize=8); axes[1].set_ylabel("Equity")
        axes[1].set_title("Портфейл общо", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        p_dd_arr = (p_cum_plot / np.maximum.accumulate(p_cum_plot) - 1) * 100
        axes[2].fill_between(range(len(p_dd_arr)), p_dd_arr, 0,
                              color="crimson", alpha=0.4)
        axes[2].set_ylabel("DD%"); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("v45_portfolio_equity.png", dpi=150)
        plt.close()
        print("\n  📊 → v45_portfolio_equity.png")
    except Exception as e:
        print(f"  (chart: {e})")


# ==============================================================
# СТАРТОВА ТОЧКА
# ==============================================================
if __name__ == "__main__":
    print("\n" + "="*62)
    print("  V45 MULTI-ASSET PORTFOLIO")
    print("  BTC + ETH + SOL | V44 Architecture | Portfolio Kelly")
    print("="*62)

    # 1. Зареждаме макро данни веднъж
    macro = load_macro()

    # 2. Зареждаме данни за всеки актив
    print("\n  📥 Зареждане на активи...")
    asset_data = {}
    for ticker, cfg in ASSETS.items():
        print(f"\n  ── {ticker} ──────────────────────────────")
        data = load_asset(ticker, cfg, macro)
        asset_data[ticker] = data
        split = int(len(data) * 0.70)
        feats = get_feats(data)
        print(f"  Данни: {len(data)} | Обучение: {split} | "
              f"Тест: {len(data)-split} | Характеристики: {len(feats)}")

    # 3. Оптимизираме всеки актив
    print("\n" + "="*62)
    print("  ОПТИМИЗАЦИЯ НА ПАРАМЕТРИ (Optuna)")
    print("="*62)
    asset_params = {}
    for ticker, cfg in ASSETS.items():
        data  = asset_data[ticker]
        split = int(len(data) * 0.70)
        train = data.iloc[:split].copy()
        feats = get_feats(train)
        params = optimize_asset(ticker, train, feats, cfg["tc"], n_trials=150)
        asset_params[ticker] = params

    # 4. Извеждаме най-добрите параметри
    print("\n" + "="*62)
    print("  НАЙ-ДОБРИ ПАРАМЕТРИ ПО АКТИВИ:")
    print("="*62)
    for ticker, p in asset_params.items():
        if p:
            print(f"\n  {ticker}:")
            for k, v in p.items():
                if k == "tc": continue
                print(f"    {k:<18} {v:.6f}" if isinstance(v,float)
                      else f"    {k:<18} {v}")

    # 5. Тестваме всички комбинации активи
    print("\n" + "="*62)
    print("  ТЕСТВАНЕ НА КОМБИНАЦИИ АКТИВИ")
    print("="*62)

    combos = [
        (["BTC-USD"],                    "BTC only"),
        (["ETH-USD"],                    "ETH only"),
        (["SOL-USD"],                    "SOL only"),
        (["BTC-USD", "ETH-USD"],         "BTC+ETH"),
        (["BTC-USD", "SOL-USD"],         "BTC+SOL"),
        (["ETH-USD", "SOL-USD"],         "ETH+SOL"),
        (["BTC-USD", "ETH-USD","SOL-USD"],"BTC+ETH+SOL"),
    ]

    combo_results = {}
    for tickers, label in combos:
        # Филтрираме данните и параметрите за тази комбинация
        sub_data   = {t: asset_data[t]   for t in tickers if t in asset_data}
        sub_params = {t: asset_params[t] for t in tickers if t in asset_params
                      and asset_params[t] is not None}
        if not sub_params:
            print(f"  ⚠️  {label}: няма параметри")
            continue

        trades = portfolio_wf_test(sub_data, sub_params, combo_label=label)
        combo_results[label] = trades
        _print_portfolio_stats(trades, sub_data)

    # Сборна таблица по комбинации
    print("\n" + "="*62)
    print("  СБОРНА ТАБЛИЦА КОМБИНАЦИИ")
    print("="*62)
    print(f"  {'Комбо':<18} {'$/год':>8} {'MaxDD':>8} {'Sharpe':>8} {'Сдел/год':>9}")
    print(f"  {'-'*55}")

    avg_days = 796
    for label, trades_by_t in combo_results.items():
        all_t = [t[1] for trades in trades_by_t.values() for t in trades]
        if not all_t: continue
        arr    = np.array(all_t)
        total  = np.sum(arr) * 100
        ann    = (1 + total/100) ** (365/avg_days) - 1
        cum    = np.cumprod(1 + arr)
        dd     = float(np.min(cum / np.maximum.accumulate(cum) - 1)) * 100
        sh     = np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(252)
        tpy    = len(arr) / (avg_days / 365)
        flag   = "✅" if ann > 0.10 and dd > -20 else ("⚠️" if ann > 0 else "❌")
        print(f"  {flag} {label:<16} {ann*100:>+7.1f}% {dd:>7.1f}% "
              f"{sh:>8.2f} {tpy:>8.1f}")

    print()
    # Намираме най-добрата комбинация (Sharpe-коригирана доходност)
    best_label = max(
        combo_results,
        key=lambda lb: (
            lambda arr: np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(252)
        )(np.array([t[1] for trades in combo_results[lb].values() for t in trades]))
        if any(combo_results[lb].values()) else -999
    )
    print(f"  🏆 Най-добра комбинация по Sharpe: {best_label}")

    # 6. Финален тест на най-добрата комбинация + equity chart
    best_tickers = next(t for t, l in combos if l == best_label)
    if isinstance(best_tickers, str):
        best_tickers = [best_tickers]
    best_data   = {t: asset_data[t]   for t in best_tickers}
    best_params = {t: asset_params[t] for t in best_tickers if asset_params.get(t)}

    # 7. Запазваме моделите на победилата комбинация
    print("\n  Запазване на финалните модели...")
    for ticker, p in asset_params.items():
        if p is None: continue
        data  = asset_data[ticker]
        feats = get_feats(data)
        lb    = make_target(data, p["f_days"], p["threshold"]).dropna()
        vf    = [f for f in feats if f in lb.columns and lb[f].std() > 1e-10]
        sc    = StandardScaler(); sc.fit(lb[vf])
        xgb, lgb = train_models(lb, vf, sc)
        bundle = {
            "xgb": xgb, "lgb": lgb, "scaler": sc,
            "feats": vf, "params": p, "ticker": ticker,
            "trained_on": datetime.now().strftime("%Y-%m-%d"),
        }
        name = ticker.replace("-","_").lower()
        path = f"{MODELS_DIR}/v45_{name}.joblib"
        joblib.dump(bundle, path)
        print(f"  ✅ {path}")

    print("\n" + "="*62)
    print(f"  V45 ЗАВЪРШЕН | Най-добра комбо: {best_label}")
    print("="*62)
