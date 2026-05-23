"""
feature_engine.py
Technical + Orderflow + Alpha Factor Engine
100+ features: Price Action, Volume, Volatility, Smart Money signals
"""
import pandas as pd
import numpy as np


# ================================================================
# TECHNICAL FEATURES
# ================================================================

def add_moving_averages(df):
    c = df['close'].astype(float)
    for n in [5, 10, 20, 50, 100, 200]:
        df[f'ma{n}']  = c.rolling(n).mean()
        df[f'ema{n}'] = c.ewm(span=n).mean()
    df['ma_trend'] = np.where(df['ma20'] > df['ma50'], 1, -1)
    return df


def add_momentum(df):
    c = df['close'].astype(float)
    for n in [5, 10, 20, 60, 120]:
        df[f'ret_{n}d'] = c.pct_change(n)
    df['rsi14'] = _rsi(c, 14)
    df['rsi7']  = _rsi(c, 7)
    # Relative Strength vs MA50
    df['rs_ma50'] = c / df['ma50'] - 1
    # Rate of Change
    df['roc20'] = c.pct_change(20)
    return df


def add_macd(df):
    c = df['close'].astype(float)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_cross']  = np.sign(df['macd_hist']).diff().fillna(0)
    return df


def add_bollinger(df, window=20, k=2):
    c = df['close'].astype(float)
    mid = c.rolling(window).mean()
    std = c.rolling(window).std()
    df['bb_mid']   = mid
    df['bb_upper'] = mid + k * std
    df['bb_lower'] = mid - k * std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / mid
    df['bb_pos']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)
    return df


def add_atr(df, window=14):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window).mean()
    df['atr_pct'] = df['atr'] / c
    return df


def add_volume_features(df):
    v = df['volume'].astype(float)
    c = df['close'].astype(float)

    df['vol_ma20']   = v.rolling(20).mean()
    df['vol_ma5']    = v.rolling(5).mean()
    df['vol_ratio']  = v / df['vol_ma20']            # volume spike ratio
    df['vol_trend']  = v.rolling(5).mean() / v.rolling(20).mean()

    # Accumulation / Dry-up
    df['vol_spike']  = (df['vol_ratio'] > 2.0).astype(int)
    df['vol_dryup']  = (df['vol_ratio'] < 0.4).astype(int)

    # Price × Volume (money flow proxy)
    df['turnover']   = c * v
    df['turnover_ma20'] = df['turnover'].rolling(20).mean()

    # On-Balance Volume
    direction = np.sign(c.diff())
    df['obv'] = (direction * v).cumsum()
    df['obv_trend'] = df['obv'].rolling(10).mean() - df['obv'].rolling(30).mean()

    # Volume-weighted price deviation
    df['vwap_dev'] = c / (df['turnover'].rolling(20).sum() / v.rolling(20).sum()) - 1

    return df


def add_price_action(df):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    o = df['open'].astype(float)

    # Candle body & shadow
    df['body']      = (c - o).abs()
    df['upper_wick']= h - pd.concat([c, o], axis=1).max(axis=1)
    df['lower_wick']= pd.concat([c, o], axis=1).min(axis=1) - l
    df['body_ratio']= df['body'] / (h - l + 1e-9)

    # Breakout detection
    df['hi20']      = h.rolling(20).max()
    df['lo20']      = l.rolling(20).min()
    df['hi52w']     = h.rolling(252).max()
    df['lo52w']     = l.rolling(252).min()
    df['breakout20']= (c > df['hi20'].shift(1)).astype(int)
    df['breakdown20']=(c < df['lo20'].shift(1)).astype(int)

    # Compression (tight range = potential breakout)
    range_5 = (h.rolling(5).max() - l.rolling(5).min()) / c
    range_20= (h.rolling(20).max() - l.rolling(20).min()) / c
    df['compression'] = (range_5 / (range_20 + 1e-9))

    return df


# ================================================================
# ORDERFLOW FEATURES (từ foreign_flow / proprietary_flow)
# ================================================================

def add_orderflow_features(df, foreign_df=None, prop_df=None, block_df=None):
    """
    Tích hợp foreign flow, proprietary flow, block trades vào feature set.
    """
    if foreign_df is not None and not foreign_df.empty:
        # Tìm cột buy/sell
        buy_col  = next((c for c in foreign_df.columns if 'buy'  in c.lower()), None)
        sell_col = next((c for c in foreign_df.columns if 'sell' in c.lower()), None)
        if buy_col and sell_col:
            ff = foreign_df[[buy_col, sell_col]].copy()
            ff.columns = ['ff_buy', 'ff_sell']
            ff['ff_net']     = ff['ff_buy'] - ff['ff_sell']
            ff['ff_net_cum'] = ff['ff_net'].rolling(20).sum()
            ff['ff_net_5d']  = ff['ff_net'].rolling(5).sum()
            df = df.join(ff[['ff_net', 'ff_net_cum', 'ff_net_5d']], how='left')

    if prop_df is not None and not prop_df.empty:
        buy_col  = next((c for c in prop_df.columns if 'buy'  in c.lower()), None)
        sell_col = next((c for c in prop_df.columns if 'sell' in c.lower()), None)
        if buy_col and sell_col:
            pp = prop_df[[buy_col, sell_col]].copy()
            pp.columns = ['prop_buy', 'prop_sell']
            pp['prop_net']    = pp['prop_buy'] - pp['prop_sell']
            pp['prop_net_5d'] = pp['prop_net'].rolling(5).sum()
            df = df.join(pp[['prop_net', 'prop_net_5d']], how='left')

    if block_df is not None and not block_df.empty:
        val_col = next((c for c in block_df.columns if 'value' in c.lower() or 'val' in c.lower()), None)
        if val_col:
            bt = block_df[[val_col]].rename(columns={val_col: 'block_value'})
            bt['block_value_5d'] = bt['block_value'].rolling(5).sum()
            df = df.join(bt[['block_value_5d']], how='left')

    return df.ffill()


# ================================================================
# ALPHA FACTORS
# ================================================================

def compute_alpha_factors(df):
    """
    Tính 30+ alpha factors từ feature set.
    Returns dict với score cho từng nhóm.
    """
    factors = {}
    last = df.iloc[-1]

    def safe(val, default=0.0):
        try:
            v = float(val)
            return default if (v != v) else v  # NaN check
        except Exception:
            return default

    # -- Momentum group --
    factors['mom_1m']  = safe(last.get('ret_20d', 0))
    factors['mom_3m']  = safe(last.get('ret_60d', 0))
    factors['mom_6m']  = safe(last.get('ret_120d', 0))
    factors['rsi']     = safe(last.get('rsi14', 50))
    factors['rs_ma50'] = safe(last.get('rs_ma50', 0))

    # -- Volume / Liquidity group --
    factors['vol_ratio']  = safe(last.get('vol_ratio', 1))
    factors['vol_trend']  = safe(last.get('vol_trend', 1))
    factors['obv_trend']  = safe(last.get('obv_trend', 0))
    factors['turnover_ratio'] = float(
        last.get('turnover', 0) / (last.get('turnover_ma20', 1) + 1e-9)
    )

    # -- Smart Money group --
    factors['ff_net_20d']  = safe(last.get('ff_net_cum', 0))
    factors['ff_net_5d']   = safe(last.get('ff_net_5d', 0))
    factors['prop_net_5d'] = safe(last.get('prop_net_5d', 0))

    # -- Volatility / Squeeze --
    factors['atr_pct']    = safe(last.get('atr_pct', 0))
    factors['bb_squeeze'] = safe(last.get('bb_squeeze', 0))
    factors['bb_width']   = safe(last.get('bb_width', 0))

    # -- Price Action --
    factors['breakout20']  = safe(last.get('breakout20', 0))
    factors['compression'] = safe(last.get('compression', 1))

    # -- Trend --
    factors['ma_trend']   = safe(last.get('ma_trend', 0))
    factors['macd_hist']  = safe(last.get('macd_hist', 0))
    factors['macd_cross'] = safe(last.get('macd_cross', 0))

    return factors


def compute_alpha_score(factors):
    """
    Tính composite alpha score (-1 → +1) từ các nhóm factor.
    """
    scores = {}

    # Momentum score
    mom_raw = (
        np.sign(factors.get('mom_1m', 0)) * 0.3 +
        np.sign(factors.get('mom_3m', 0)) * 0.3 +
        np.sign(factors.get('rs_ma50', 0)) * 0.2 +
        (1 if factors.get('rsi', 50) > 50 else -1) * 0.2
    )
    scores['momentum'] = float(np.clip(mom_raw, -1, 1))

    # Volume score
    vol_raw = (
        min(factors.get('vol_ratio', 1) - 1, 1) * 0.4 +
        np.sign(factors.get('vol_trend', 1) - 1) * 0.3 +
        np.sign(factors.get('obv_trend', 0)) * 0.3
    )
    scores['volume'] = float(np.clip(vol_raw, -1, 1))

    # Smart money score
    sm_raw = (
        np.sign(factors.get('ff_net_5d', 0)) * 0.4 +
        np.sign(factors.get('ff_net_20d', 0)) * 0.3 +
        np.sign(factors.get('prop_net_5d', 0)) * 0.3
    )
    scores['smart_money'] = float(np.clip(sm_raw, -1, 1))

    # Trend score
    trend_raw = (
        factors.get('ma_trend', 0) * 0.4 +
        np.sign(factors.get('macd_hist', 0)) * 0.4 +
        factors.get('breakout20', 0) * 0.2
    )
    scores['trend'] = float(np.clip(trend_raw, -1, 1))

    # Composite (weights theo chiến lược Level 6)
    composite = (
        scores['smart_money'] * 0.35 +
        scores['momentum']    * 0.25 +
        scores['volume']      * 0.25 +
        scores['trend']       * 0.15
    )
    scores['composite'] = float(np.clip(composite, -1, 1))

    return scores


# ================================================================
# HELPERS
# ================================================================

def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(ohlcv_df, foreign_df=None, prop_df=None, block_df=None):
    """Pipeline đầy đủ: OHLCV → tất cả features."""
    df = ohlcv_df.copy()
    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_price_action(df)
    df = add_orderflow_features(df, foreign_df, prop_df, block_df)
    return df
