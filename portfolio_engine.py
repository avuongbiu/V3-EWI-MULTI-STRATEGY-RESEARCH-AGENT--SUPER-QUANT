"""
portfolio_engine.py
Portfolio Engine: Dynamic Allocation + Risk Management + Position Sizing
"""
import numpy as np
import pandas as pd

# ── Position tiers theo chiến lược Level 6 ───────────────────────────
POSITION_TIERS = {
    'core':        {'max_pct': 0.10, 'desc': 'Core – nắm dài hạn'},
    'swing':       {'max_pct': 0.07, 'desc': 'Swing – 2-8 tuần'},
    'speculative': {'max_pct': 0.03, 'desc': 'Speculative – ngắn hạn'},
}

REGIME_ALLOCATION = {
    'BULL':       {'equity': 0.90, 'cash': 0.10},
    'BULL_WEAK':  {'equity': 0.70, 'cash': 0.30},
    'SIDEWAY':    {'equity': 0.50, 'cash': 0.50},
    'BEAR':       {'equity': 0.25, 'cash': 0.75},
    'CRASH_RISK': {'equity': 0.00, 'cash': 1.00},
}


def compute_position_size(signal_result, regime_result, portfolio_value=1_000_000_000,
                           atr=None, close_price=None):
    """
    Tính position size theo Kelly-fraction + ATR stop loss.

    Returns dict: size_pct, size_vnd, tier, stop_loss, risk_vnd
    """
    signal  = signal_result.get('signal', 'HOLD')
    score   = signal_result.get('score', 0)
    regime  = regime_result.get('regime', 'SIDEWAY')
    alloc   = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION['SIDEWAY'])
    equity_pct = alloc['equity']

    # Phân loại tier
    if signal in ('STRONG_BUY', 'BREAKOUT'):
        tier = 'swing'
    elif signal in ('BUY',):
        tier = 'swing'
    elif signal == 'ACCUMULATION':
        tier = 'core'
    else:
        return {'size_pct': 0, 'size_vnd': 0, 'tier': 'none',
                'stop_loss': None, 'risk_vnd': 0}

    max_size = POSITION_TIERS[tier]['max_pct']

    # Điều chỉnh theo score và regime
    confidence = min(abs(score), 1.0)
    adj_size   = max_size * confidence * equity_pct

    # Kelly fraction đơn giản (win_rate giả định từ score)
    win_rate     = 0.5 + score * 0.3      # score 1.0 → win_rate 0.8
    reward_risk  = 2.5                     # target R:R = 2.5:1
    kelly        = (win_rate - (1 - win_rate) / reward_risk)
    kelly_frac   = max(0, min(kelly * 0.25, max_size))  # Quarter Kelly

    final_size = min(adj_size, kelly_frac) * portfolio_value

    # Stop loss từ ATR
    stop_loss = None
    risk_vnd  = 0
    if atr and close_price:
        stop_loss = round(close_price - 2 * atr, 0)
        risk_per_share = close_price - stop_loss
        shares = int(final_size / close_price / 100) * 100  # làm tròn lô 100
        risk_vnd = shares * risk_per_share

    return {
        'tier':       tier,
        'tier_desc':  POSITION_TIERS[tier]['desc'],
        'size_pct':   round(adj_size, 4),
        'size_vnd':   round(final_size, 0),
        'kelly_frac': round(kelly_frac, 4),
        'stop_loss':  stop_loss,
        'risk_vnd':   round(risk_vnd, 0),
        'regime_equity_alloc': equity_pct,
    }


def compute_portfolio_allocation(signals_dict, regime_result,
                                  portfolio_value=1_000_000_000):
    """
    Phân bổ portfolio cho nhiều mã.
    signals_dict: {symbol: signal_result}
    Returns: DataFrame với allocation cho từng mã
    """
    regime = regime_result.get('regime', 'SIDEWAY')
    alloc  = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION['SIDEWAY'])
    total_equity = portfolio_value * alloc['equity']

    rows = []
    buy_signals = {
        sym: res for sym, res in signals_dict.items()
        if res.get('signal') in ('STRONG_BUY', 'BREAKOUT', 'BUY', 'ACCUMULATION')
    }

    if not buy_signals:
        return pd.DataFrame(), alloc['cash'] * 100

    # Score-weighted allocation
    total_score = sum(max(r.get('score', 0), 0) for r in buy_signals.values())

    for sym, res in sorted(buy_signals.items(),
                            key=lambda x: x[1].get('score', 0), reverse=True):
        score  = max(res.get('score', 0), 0)
        weight = score / total_score if total_score > 0 else 1 / len(buy_signals)

        # Giới hạn tối đa 10% / mã
        weight = min(weight, 0.10)
        alloc_vnd = total_equity * weight

        rows.append({
            'Mã':           str(sym),
            'Tín hiệu':     str(res.get('label', '-')),
            'Score':        f"{float(res.get('score', 0)):+.3f}",
            'Wyckoff':      str(res.get('wyckoff_phase', '-')),
            'Tỷ trọng':     f"{weight*100:.1f}%",
            'Giá trị (tỷ)': f"{alloc_vnd/1e9:.2f}",
        })

    return pd.DataFrame(rows), alloc['cash'] * 100


def compute_portfolio_risk(returns_df, weights=None):
    """
    Tính portfolio risk metrics: volatility, max drawdown, Sharpe, correlation.
    """
    if returns_df is None or returns_df.empty:
        return {}

    if weights is None:
        weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]

    port_returns = (returns_df * weights).sum(axis=1)

    # Annualized metrics
    ann_ret  = float(port_returns.mean() * 252)
    ann_vol  = float(port_returns.std() * np.sqrt(252))
    sharpe   = ann_ret / (ann_vol + 1e-9)

    # Max Drawdown
    cum = (1 + port_returns).cumprod()
    dd  = (cum / cum.cummax() - 1).min()

    # Correlation matrix
    corr = returns_df.corr()
    avg_corr = float(corr.values[np.triu_indices_from(corr.values, k=1)].mean())

    return {
        'annual_return':  round(ann_ret * 100, 2),
        'annual_vol':     round(ann_vol * 100, 2),
        'sharpe_ratio':   round(sharpe, 2),
        'max_drawdown':   round(float(dd) * 100, 2),
        'avg_correlation':round(avg_corr, 3),
    }


def format_portfolio_report(allocation_df, cash_pct, risk_metrics, regime_label):
    lines = [
        "=== PORTFOLIO ENGINE ===",
        f"Regime: {regime_label}",
        f"Cash: {cash_pct:.0f}% | Equity: {100-cash_pct:.0f}%",
    ]
    if not allocation_df.empty:
        lines.append("\nPhân bổ:")
        lines.append(allocation_df.to_string(index=False))
    if risk_metrics:
        lines.append(f"\nRisk Metrics:")
        lines.append(f"  Sharpe: {risk_metrics.get('sharpe_ratio','N/A')}")
        lines.append(f"  Max DD: {risk_metrics.get('max_drawdown','N/A')}%")
        lines.append(f"  Avg Corr: {risk_metrics.get('avg_correlation','N/A')}")
    return "\n".join(lines)
