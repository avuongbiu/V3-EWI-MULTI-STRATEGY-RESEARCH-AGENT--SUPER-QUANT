"""
valuation_engine.py
DCF Valuation + Comparable Company Analysis (Comps)
Inspired by anthropics/financial-services valuation-reviewer + model-builder
Adapted for Vietnam market (VND, HOSE/HNX, vnstock_data)
"""
import numpy as np
import pandas as pd

def safe_fmt(val, fmt=".2f", default="-"):
    """Format số an toàn, trả về default nếu val là None/NaN"""
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return default
    try:
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return default


# ════════════════════════════════════════════════════════════════════════
# DCF VALUATION
# ════════════════════════════════════════════════════════════════════════

SECTOR_WACC = {
    'Ngân hàng':        0.12,
    'Bất động sản':     0.14,
    'Thép':             0.13,
    'Công nghệ':        0.15,
    'Tiêu dùng':        0.11,
    'Điện':             0.10,
    'Dầu khí':          0.13,
    'default':          0.13,
}

SECTOR_GROWTH = {
    'Ngân hàng':        0.12,
    'Bất động sản':     0.10,
    'Thép':             0.07,
    'Công nghệ':        0.18,
    'Tiêu dùng':        0.10,
    'Điện':             0.08,
    'default':          0.10,
}


def run_dcf(symbol: str, fundamental: dict, income_df=None,
            cashflow_df=None, sector: str = 'default',
            forecast_years: int = 5, terminal_growth: float = 0.05) -> dict:
    """
    3-Stage DCF Model cho cổ phiếu VN.
    Stage 1: Forecast 5 năm từ BCTC
    Stage 2: Transition 3 năm
    Stage 3: Terminal value (Gordon Growth)
    """
    # ── Lấy base metrics ─────────────────────────────────────────────
    wacc = SECTOR_WACC.get(sector, SECTOR_WACC['default'])

    # Dùng net_profit_latest nếu có, fallback sang EPS × shares
    base_profit = fundamental.get('net_profit_latest', None)
    eps         = float(fundamental.get('eps', 0) or 0)

    if base_profit is None:
        # Ước lượng từ EPS (đơn vị VND/cổ phiếu)
        # Giả sử 1B cổ phiếu nếu không có số liệu
        base_profit = eps * 1_000_000_000 / 1_000  # triệu VND

    # Revenue growth từ fundamental
    rev_growth = float(fundamental.get('revenue_growth', SECTOR_GROWTH.get(sector, 0.10) * 100) or 10) / 100
    roe        = float(fundamental.get('roe', 15) or 15) / 100
    pe         = float(fundamental.get('pe', 15) or 15)

    # ── Stage 1: Free Cash Flow forecast ─────────────────────────────
    fcf_base = base_profit * 0.7  # FCF ≈ 70% net profit (conservative)
    if cashflow_df is not None and not cashflow_df.empty:
        cfo_col = next((c for c in cashflow_df.columns
                        if 'operating' in c.lower() or 'cfo' in c.lower()), None)
        capex_col = next((c for c in cashflow_df.columns
                          if 'capex' in c.lower() or 'invest' in c.lower()), None)
        if cfo_col:
            fcf_base = float(cashflow_df[cfo_col].iloc[-1])
            if capex_col:
                fcf_base += float(cashflow_df[capex_col].iloc[-1])  # capex thường âm

    # Tăng trưởng FCF theo phase
    growth_stage1 = min(rev_growth, 0.25)       # Capped at 25% Stage 1
    growth_stage2 = growth_stage1 * 0.6          # Decay to moderate
    growth_stage3 = terminal_growth               # Long-term = 5%

    fcf_forecasts = []
    fcf = fcf_base
    for y in range(1, forecast_years + 1):
        fcf = fcf * (1 + growth_stage1)
        pv  = fcf / (1 + wacc) ** y
        fcf_forecasts.append({'year': y, 'fcf': round(fcf, 0), 'pv_fcf': round(pv, 0)})

    # Stage 2: 3 years transition
    for y in range(1, 4):
        fcf = fcf * (1 + growth_stage2)
        pv  = fcf / (1 + wacc) ** (forecast_years + y)
        fcf_forecasts.append({'year': forecast_years + y, 'fcf': round(fcf, 0), 'pv_fcf': round(pv, 0)})

    # Terminal Value (Gordon Growth)
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_val = terminal_fcf / (wacc - terminal_growth)
    pv_terminal  = terminal_val / (1 + wacc) ** (forecast_years + 3)

    pv_fcf_sum  = sum(f['pv_fcf'] for f in fcf_forecasts)
    enterprise_value = pv_fcf_sum + pv_terminal

    # ── Intrinsic value per share ─────────────────────────────────────
    # Ước lượng số cổ phiếu từ EPS và net profit
    shares_est = 1_000_000_000  # Default 1B
    if eps > 0 and base_profit > 0:
        shares_est = int(base_profit / eps * 1_000)  # Convert đơn vị

    shares_est = max(shares_est, 1)
    intrinsic_per_share = enterprise_value / shares_est * 1_000_000  # VND/cp

    # ── Sensitivity table (WACC × Growth) ────────────────────────────
    sensitivity = {}
    for w in [wacc - 0.02, wacc, wacc + 0.02]:
        row = {}
        for g in [growth_stage1 - 0.05, growth_stage1, growth_stage1 + 0.05]:
            fcf_tmp = fcf_base
            ev_tmp  = 0
            for y in range(1, forecast_years + 4):
                gr = g if y <= forecast_years else g * 0.6
                fcf_tmp = fcf_tmp * (1 + gr)
                ev_tmp += fcf_tmp / (1 + w) ** y
            tv_tmp  = fcf_tmp * (1 + terminal_growth) / (w - terminal_growth)
            ev_tmp += tv_tmp / (1 + w) ** (forecast_years + 3)
            row[f'g={g:.0%}'] = round(ev_tmp / shares_est * 1_000_000, 0)
        sensitivity[f'wacc={w:.0%}'] = row

    return {
        'symbol':              symbol,
        'method':              'DCF 3-Stage',
        'wacc':                round(wacc, 3),
        'growth_stage1':       round(growth_stage1, 3),
        'terminal_growth':     terminal_growth,
        'base_fcf':            round(fcf_base, 0),
        'pv_fcf_sum':          round(pv_fcf_sum, 0),
        'pv_terminal':         round(pv_terminal, 0),
        'enterprise_value':    round(enterprise_value, 0),
        'intrinsic_per_share': round(intrinsic_per_share, 0),
        'forecasts':           fcf_forecasts,
        'sensitivity':         sensitivity,
    }


# ════════════════════════════════════════════════════════════════════════
# COMPARABLE COMPANY ANALYSIS (/comps)
# ════════════════════════════════════════════════════════════════════════

VN_SECTOR_PEERS = {
    'VCB': ['BID','CTG','ACB','TCB','MBB'],
    'BID': ['VCB','CTG','ACB','TCB','STB'],
    'HPG': ['HSG','NKG','TLH','SMC','POM'],
    'VNM': ['MCM','QNS','KDC','MSN','SAB'],
    'FPT': ['CMG','ELC','VGI','FOX','VTC'],
    'MWG': ['PNJ','DGW','FRT','VRE','MPC'],
    'VIC': ['VHM','NLG','KDH','DXG','NVL'],
    'VHM': ['VIC','NLG','KDH','DXG','PDR'],
    'TCB': ['VCB','ACB','MBB','VPB','HDB'],
    'MSN': ['VNM','SAB','KDC','MCM','THD'],
}

SECTOR_MULTIPLE_BENCHMARKS = {
    'Ngân hàng':        {'pe': 9,  'pb': 1.5, 'roe_min': 15},
    'Bất động sản':     {'pe': 15, 'pb': 1.8, 'roe_min': 12},
    'Thép':             {'pe': 8,  'pb': 1.0, 'roe_min': 10},
    'Công nghệ':        {'pe': 20, 'pb': 3.0, 'roe_min': 18},
    'Tiêu dùng':        {'pe': 18, 'pb': 2.5, 'roe_min': 15},
    'Điện':             {'pe': 12, 'pb': 1.5, 'roe_min': 10},
    'default':          {'pe': 12, 'pb': 1.5, 'roe_min': 12},
}


def run_comps(symbol: str, target_fund: dict,
              peer_fundamentals: dict, sector: str = 'default') -> dict:
    """
    Comparable Company Analysis.
    target_fund: fundamental dict of target company
    peer_fundamentals: {sym: fundamental_dict} của peers
    """
    benchmark = SECTOR_MULTIPLE_BENCHMARKS.get(sector, SECTOR_MULTIPLE_BENCHMARKS['default'])

    # Target metrics
    tgt_pe  = float(target_fund.get('pe',  0) or 0)
    tgt_pb  = float(target_fund.get('pb',  0) or 0)
    tgt_roe = float(target_fund.get('roe', 0) or 0)
    tgt_eps = float(target_fund.get('eps', 0) or 0)

    # Peer metrics
    peer_rows = []
    peer_pes, peer_pbs, peer_roes = [], [], []

    for peer_sym, peer_fund in peer_fundamentals.items():
        pe  = float(peer_fund.get('pe',  0) or 0)
        pb  = float(peer_fund.get('pb',  0) or 0)
        roe = float(peer_fund.get('roe', 0) or 0)
        eps = float(peer_fund.get('eps', 0) or 0)

        if pe  > 0: peer_pes.append(pe)
        if pb  > 0: peer_pbs.append(pb)
        if roe > 0: peer_roes.append(roe)

        premium_pe  = ((tgt_pe - pe) / pe * 100) if pe > 0 else None
        peer_rows.append({
            'Symbol':   peer_sym,
            'PE':       f"{pe:.1f}x" if pe > 0 else 'N/A',
            'PB':       f"{pb:.2f}x" if pb > 0 else 'N/A',
            'ROE':      f"{roe:.1f}%" if roe > 0 else 'N/A',
            'EPS':      safe_fmt(eps, ",.0f"),
            'PE vs Target': f"{premium_pe:+.1f}%" if premium_pe is not None else 'N/A',
        })

    # Median peer multiples
    med_pe  = float(np.median(peer_pes))  if peer_pes  else benchmark['pe']
    med_pb  = float(np.median(peer_pbs))  if peer_pbs  else benchmark['pb']
    med_roe = float(np.median(peer_roes)) if peer_roes else benchmark['roe_min']

    # Implied prices từ peer multiples
    implied_pe = tgt_eps * med_pe if tgt_eps > 0 else None
    implied_pb = None  # Cần book value

    # PE discount/premium vs peers
    pe_premium = ((tgt_pe - med_pe) / med_pe * 100) if med_pe > 0 and tgt_pe > 0 else None
    roe_vs_peers= tgt_roe - med_roe

    # Verdict: justified premium?
    verdict = 'FAIRLY_VALUED'
    if pe_premium is not None:
        if pe_premium > 20 and roe_vs_peers < 2:
            verdict = 'EXPENSIVE'
        elif pe_premium < -15 and roe_vs_peers > -2:
            verdict = 'CHEAP'
        elif pe_premium > 20 and roe_vs_peers > 5:
            verdict = 'PREMIUM_JUSTIFIED'

    return {
        'symbol':         symbol,
        'method':         'Comparable Company Analysis',
        'sector':         sector,
        'target': {
            'pe':  tgt_pe, 'pb': tgt_pb, 'roe': tgt_roe, 'eps': tgt_eps
        },
        'peer_median': {
            'pe':  round(med_pe, 2), 'pb': round(med_pb, 2), 'roe': round(med_roe, 2)
        },
        'benchmark':      benchmark,
        'pe_premium':     round(pe_premium, 1) if pe_premium is not None else None,
        'roe_vs_peers':   round(roe_vs_peers, 1),
        'implied_pe_price':round(implied_pe, 0) if implied_pe else None,
        'verdict':        verdict,
        'peer_table':     pd.DataFrame(peer_rows),
    }


# ════════════════════════════════════════════════════════════════════════
# EARNINGS ANALYSIS (/earnings)
# ════════════════════════════════════════════════════════════════════════

def analyze_earnings(symbol: str, income_df, balance_df=None,
                     cashflow_df=None, periods: int = 4) -> dict:
    """
    Phân tích kết quả BCTC qua các quý/năm.
    Inspired by earnings-reviewer agent.
    """
    if income_df is None or income_df.empty:
        return {'symbol': symbol, 'error': 'Không có dữ liệu BCTC'}

    df = income_df.tail(periods).copy()

    # Detect revenue / profit columns
    rev_col = next((c for c in df.columns
                    if any(k in c.lower() for k in ['revenue','doanh thu','net_sale'])), None)
    profit_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ['net_profit','loi nhuan','after_tax'])), None)

    results = {}

    if rev_col and profit_col:
        rev_series    = pd.to_numeric(df[rev_col], errors='coerce')
        profit_series = pd.to_numeric(df[profit_col], errors='coerce')

        # YoY growth
        rev_yoy    = rev_series.pct_change(1).iloc[-1] if len(rev_series) >= 2 else None
        profit_yoy = profit_series.pct_change(1).iloc[-1] if len(profit_series) >= 2 else None

        # Net margin
        net_margin = (profit_series / rev_series).iloc[-1] if rev_series.iloc[-1] > 0 else None

        # Earnings quality
        results = {
            'revenue_latest':     float(rev_series.iloc[-1]),
            'profit_latest':      float(profit_series.iloc[-1]),
            'revenue_yoy':        round(float(rev_yoy) * 100, 2) if rev_yoy is not None else None,
            'profit_yoy':         round(float(profit_yoy) * 100, 2) if profit_yoy is not None else None,
            'net_margin':         round(float(net_margin) * 100, 2) if net_margin is not None else None,
            'trend_revenue':      'GROWING' if (rev_yoy or 0) > 0.05 else ('DECLINING' if (rev_yoy or 0) < -0.05 else 'STABLE'),
            'trend_profit':       'GROWING' if (profit_yoy or 0) > 0.05 else ('DECLINING' if (profit_yoy or 0) < -0.05 else 'STABLE'),
        }

        # Cash flow quality (nếu có)
        if cashflow_df is not None and not cashflow_df.empty:
            cfo_col = next((c for c in cashflow_df.columns
                            if 'operating' in c.lower() or 'cfo' in c.lower()), None)
            if cfo_col:
                cfo = float(pd.to_numeric(cashflow_df[cfo_col], errors='coerce').iloc[-1])
                results['cfo_latest'] = cfo
                results['cfo_to_profit'] = round(cfo / (results['profit_latest'] + 1e-9), 2)
                results['earnings_quality'] = 'HIGH' if results['cfo_to_profit'] > 0.8 else (
                    'MEDIUM' if results['cfo_to_profit'] > 0.5 else 'LOW')

        # Signals
        signals = []
        if results.get('profit_yoy', 0) and results['profit_yoy'] > 20:
            signals.append(f"✅ Lợi nhuận tăng mạnh +{results['profit_yoy']:.1f}% YoY")
        if results.get('profit_yoy', 0) and results['profit_yoy'] < -20:
            signals.append(f"⚠ Lợi nhuận giảm {results['profit_yoy']:.1f}% YoY")
        if results.get('net_margin', 0) and results['net_margin'] > 20:
            signals.append(f"✅ Biên lợi nhuận cao: {results['net_margin']:.1f}%")
        if results.get('cfo_to_profit', 1) and results.get('cfo_to_profit', 1) < 0.5:
            signals.append("⚠ FCF thấp hơn lợi nhuận kế toán → kiểm tra chất lượng earnings")

        results['signals'] = signals

    return {'symbol': symbol, **results}


# ════════════════════════════════════════════════════════════════════════
# INVESTMENT COMMITTEE MEMO (/ic-memo)
# ════════════════════════════════════════════════════════════════════════

def generate_ic_memo(symbol: str, dcf: dict, comps: dict,
                     earnings: dict, signal_result: dict,
                     ers_result: dict, wyckoff: dict,
                     regime_result: dict) -> str:
    """
    Tạo Investment Committee Memo (IC Memo) tổng hợp.
    Inspired by anthropics/financial-services /ic-memo command.
    """
    action = signal_result.get('signal', 'HOLD')
    score  = signal_result.get('score', 0)
    ers_score = ers_result.get('ers_score', 0)

    lines = [
        "=" * 70,
        f"  INVESTMENT COMMITTEE MEMO — {symbol}",
        f"  Ngày: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}",
        "=" * 70,
        "",
        "━━━ I. EXECUTIVE SUMMARY ━━━",
        f"Khuyến nghị: {action}  |  Score: {safe_fmt(score, '+.3f')}  |  ERS: {ers_score}/30 {ers_result.get('color','')}",
        f"Market Regime: {regime_result.get('label','')}",
        f"Wyckoff Phase: Phase {wyckoff.get('phase','?')} — {wyckoff.get('description','')}",
        "",
        "━━━ II. INVESTMENT THESIS ━━━",
    ]

    # Lý do
    reasons = signal_result.get('reasons', [])
    for r in reasons[:4]:
        lines.append(f"  {r}")

    lines += ["", "━━━ III. VALUATION ━━━"]

    # DCF
    if dcf and 'intrinsic_per_share' in dcf:
        lines += [
            f"DCF Intrinsic Value:  {dcf['intrinsic_per_share']:,.0f} VND/cp",
            f"WACC: {safe_fmt(dcf.get('wacc'), '.1%')}  |  Growth Stage1: {safe_fmt(dcf.get('growth_stage1'), '.1%')}",
            f"Terminal Growth: {safe_fmt(dcf.get('terminal_growth'), '.1%')}",
        ]

    # Comps
    if comps and 'pe_premium' in comps:
        verdict_map = {
            'EXPENSIVE': '⚠ ĐANG ĐẮTHƠN PEERS',
            'CHEAP': '✅ ĐANG RẺ HƠN PEERS',
            'PREMIUM_JUSTIFIED': '✅ PREMIUM HỢP LÝ (ROE CAO HƠN)',
            'FAIRLY_VALUED': '→ ĐỊNH GIÁ HỢP LÝ',
        }
        lines += [
            f"\nPeer Comps ({comps['sector']}):",
            f"  PE: {comps['target']['pe']:.1f}x vs Peers median {comps['peer_median']['pe']:.1f}x "
            f"(Premium: {comps['pe_premium']:+.1f}%)",
            f"  ROE: {comps['target']['roe']:.1f}% vs Peers {comps['peer_median']['roe']:.1f}%",
            f"  Verdict: {verdict_map.get(comps['verdict'], comps['verdict'])}",
        ]

    # Earnings
    lines += ["", "━━━ IV. EARNINGS QUALITY ━━━"]
    if earnings and 'profit_yoy' in earnings:
        lines += [
            f"  Doanh thu YoY: {earnings.get('revenue_yoy','N/A')}%",
            f"  LNST YoY:      {earnings.get('profit_yoy','N/A')}%",
            f"  Net Margin:    {earnings.get('net_margin','N/A')}%",
            f"  Earnings Quality: {earnings.get('earnings_quality','N/A')}",
        ]
        for s in earnings.get('signals', []):
            lines.append(f"  {s}")

    # ERS
    lines += ["", "━━━ V. RISK ASSESSMENT ━━━",
              f"Event Risk Score: {ers_score}/30  {ers_result.get('color','')}",
              f"Action: {ers_result.get('action','')}",
              f"Max Position: {safe_fmt(ers_result.get('max_position'), '.0%')}",
              f"Stop Loss: {safe_fmt(ers_result.get('stoploss'), ',.0f')}",
    ]
    for g in ers_result.get('groups', []):
        if g['score'] > 0:
            lines.append(f"  Nhóm {g['group']}: {g['name']} = {g['score']}/{g['max']}")

    # Scenario
    sc = ers_result.get('scenario', {})
    if sc:
        lines += [
            "", "━━━ VI. SCENARIO ANALYSIS ━━━",
            f"  🟢 Bull Case:  {safe_fmt(sc.get('bull_case'), '.0%')}",
            f"  🟡 Base Case:  {safe_fmt(sc.get('base_case'), '.0%')}",
            f"  🔴 Bear Case:  {safe_fmt(sc.get('bear_case'), '.0%')}",
        ]

    lines += [
        "",
        "━━━ VII. ENTRY STRATEGY (Wyckoff) ━━━",
        f"  Phase hiện tại: {wyckoff.get('phase','?')} — {wyckoff.get('description','')}",
        "  Điều kiện entry: Spring test thành công + SOS confirmation + LPS pullback" if wyckoff.get('is_buy_zone') else "  Chờ Phase C/D để entry",
        "",
        "=" * 70,
        "  Báo cáo này chỉ mang tính tham khảo. Không phải lời khuyên đầu tư.",
        "=" * 70,
    ]

    return "\n".join(lines)


def format_dcf_report(dcf: dict) -> str:
    if not dcf or 'error' in dcf:
        return "DCF: Không đủ dữ liệu"
    lines = [
        f"=== DCF VALUATION: {dcf['symbol']} ===",
        f"Intrinsic Value: {dcf['intrinsic_per_share']:,.0f} VND/cp",
        f"WACC: {safe_fmt(dcf.get('wacc'), '.1%')} | Growth: {safe_fmt(dcf.get('growth_stage1'), '.1%')} | Terminal: {safe_fmt(dcf.get('terminal_growth'), '.1%')}",
        f"Enterprise Value: {dcf['enterprise_value']:,.0f} (triệu VND)",
        f"PV FCF: {safe_fmt(dcf.get('pv_fcf_sum'), ',.0f')} | PV Terminal: {safe_fmt(dcf.get('pv_terminal'), ',.0f')}",
    ]
    return "\n".join(lines)


def format_comps_report(comps: dict) -> str:
    if not comps or 'error' in comps:
        return "Comps: Không đủ dữ liệu"
    lines = [
        f"=== COMPARABLE ANALYSIS: {comps['symbol']} ===",
        f"Sector: {comps['sector']}",
        f"PE: {comps['target']['pe']:.1f}x vs Peers {comps['peer_median']['pe']:.1f}x (Premium: {comps.get('pe_premium',0):+.1f}%)",
        f"ROE: {comps['target']['roe']:.1f}% vs Peers {comps['peer_median']['roe']:.1f}%",
        f"Verdict: {comps['verdict']}",
    ]
    if comps.get('implied_pe_price'):
        lines.append(f"Implied Price (peer PE): {comps['implied_pe_price']:,.0f} VND")
    return "\n".join(lines)
