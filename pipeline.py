# pipeline.py
# Super AI Quant VN – Level 6 Pipeline (V2 Stable)
# MARKET = DONG TIEN + THANH KHOAN + TAM LY + VI MO

import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from data_fetcher import DataFetcher
from feature_engine import build_features, compute_alpha_factors, compute_alpha_score
from wyckoff import detect_wyckoff_phase, format_wyckoff_report
from market_regime import classify_market_regime, format_regime_report, get_sector_rotation_signal
from signal_engine import generate_signal, format_signal_report
from portfolio_engine import (compute_position_size, compute_portfolio_allocation,
                               compute_portfolio_risk, format_portfolio_report)
from causal_discovery import run_pcmci, run_lingam, generate_causal_report
from event_risk_engine import compute_ers, format_ers_report, get_ers_action
from volume_profile import (compute_volume_profile, compute_cumulative_delta,
                             detect_absorption, smart_money_checklist,
                             format_vp_report, format_smc_report)
from ai_agents import AIAgent, extract_json, meta_ensemble, build_agents_from_config
from valuation_engine import (run_dcf, run_comps, analyze_earnings, generate_ic_memo,
                               format_dcf_report, format_comps_report, VN_SECTOR_PEERS)
from research_agent import build_research_orchestrator

# ── Config ─────────────────────────────────────────────────────────────
def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ── Agent registry ─────────────────────────────────────────────────────
AGENT_REGISTRY = {
    'DeepSeek': ('deepseek',  'deepseek-chat'),
    'ChatGPT':  ('openai',    'gpt-4o-mini'),
    'Claude':   ('anthropic', 'claude-sonnet-4-5'),
    'Grok':     ('groq',      'llama-3.3-70b-versatile'),
    'Qwen':     ('qwen',      'qwen-max'),
}

SYSTEM_PROMPT = """Ban la EWI Research Agent – chuyen gia phan tich chung khoan Viet Nam cap do dinh cao &Chuyên gia Định lượng & Quản lý Danh mục Cấp cao (C-Level Quant).
Nhiệm vụ: Đánh giá mã cổ phiếu dựa trên Dữ liệu Đầu vào (Input Data)
Tiep can: EWI Event-Driven Framework + Wyckoff + Volume Profile + Smart Money Concepts.
Triet ly: Thi truong = Dong tien + Thanh khoan + Tam ly + Vi mo.
Gia chi la ket qua cuoi cung cua dong tien. Smart money di truoc, retail theo sau.
QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG hallucinate (bịa) dữ liệu. Chỉ dùng số liệu có trong Input.
2. Nếu dữ liệu thiếu (N/A), ghi rõ "Thiếu dữ liệu", không được tự suy diễn.
3. Logic ra quyết định:
   - Nếu Score > 5 và Wyckoff Phase B/C/D: BUY/ACCUMULATE.
   - Nếu EWI Risk ≥ 9 hoặc Wyckoff Distribution: SELL/AVOID.
   - Nếu mâu thuẫn giữa Kỹ thuật và Cơ bản: Ưu tiên Cơ bản & Dòng tiền.
RULE ENTRY: Chi mua tai Spring test thanh cong + SOS confirmation + LPS pullback.
RULE EXIT: Phan phoi (distribution) + Volume climax + EWI Score >= 7.
Tra loi JSON thuan tuy, khong them text ngoai JSON."""

def build_level6_prompt(symbol, tech_summary, wyckoff_report, signal_report,
                         regime_report, fundamental, causal_report, flow_summary,
                         ers_report='', ewi_report='', vp_report='', smc_report='', scenario=None):
    ewi_report = ewi_report or ers_report
    pe  = fundamental.get('pe',  'N/A')
    pb  = fundamental.get('pb',  'N/A')
    roe = fundamental.get('roe', 'N/A')
    eps = fundamental.get('eps', 'N/A')
    de  = fundamental.get('debt_to_equity', 'N/A')
    rg  = fundamental.get('revenue_growth', 'N/A')

    ff_net  = flow_summary.get('foreign_net_30d', 'N/A')
    prop_net= flow_summary.get('prop_net_30d', 'N/A')

    scenario_str = ""
    if scenario:
        scenario_str = f"Base={scenario.get('base_case',0):.0%} | Bull={scenario.get('bull_case',0):.0%} | Bear={scenario.get('bear_case',0):.0%}"

    return f"""Phan tich {symbol} theo EWI Framework Level 6.

=== I. EWI EVENT RISK ===
{ewi_report}

=== II. MARKET REGIME ===
{regime_report}

=== III. VOLUME PROFILE ===
{vp_report}

=== IV. SMART MONEY (Delta + Absorption) ===
{smc_report}

=== V. WYCKOFF ANALYSIS ===
{wyckoff_report}

=== VI. SIGNAL ENGINE ===
{signal_report}

=== VII. KY THUAT & DONG TIEN ===
{tech_summary}
Foreign net 30 ngay: {ff_net} | Prop net: {prop_net}

=== VIII. CO BAN ===
PE={pe} | PB={pb} | ROE={roe}% | EPS={eps} | No/Von={de}

=== IX. CAUSAL DISCOVERY ===
{causal_report}

=== X. SCENARIO PROBABILITY ===
{scenario_str}

RULE: Neu EWI >= 9 -> bat buoc SELL. EWI 7-8 -> khong mua moi.
Chi tra loi dung JSON sau, KHONG them text ngoai:
{{
  "action": "BUY|HOLD|SELL",
  "confidence": <0.0-1.0>,
  "target_price": <gia muc tieu>,
  "stop_loss": <gia cat lo>,
  "position_size": "core|swing|speculative",
  "wyckoff_phase": "<Phase A/B/C/D/E>",
  "ewi_concern": "<rui ro EWI chinh neu co>",
  "vp_key_level": "<POC/VAH/VAL quan trong nhat>",
  "smc_signal": "<tin hieu smart money>",
  "key_risk": "<rui ro chinh can theo doi>",
  "reasoning": "<ly do ngan gon duoi 80 tu>"
}}"""

# ── Main Pipeline ───────────────────────────────────────────────────────
def run_full_pipeline(config_path='config.yaml'):
    cfg     = load_config(config_path)
    symbols = cfg['universe']['symbols']
    source  = cfg['vnstock']['source']
    start   = cfg['vnstock'].get('start_date', '2023-01-01')
    end     = cfg['vnstock'].get('end_date', datetime.today().strftime('%Y-%m-%d'))

    macro_indicators = cfg.get('macro_indicators', ['cpi', 'gdp', 'interest_rate', 'usd_vnd'])
    fetcher = DataFetcher(source=source, start_date=start)
    print(f"[Pipeline] Universe: {symbols}")

    # ═════════════════════════════════════════════════════════════════════
    # 1. MARKET REGIME (VNINDEX)
    # ═════════════════════════════════════════════════════════════════════
    print("[Pipeline] === Step 1: Market Regime ===")
    try:
        vnindex_df = fetcher.get_index_ohlcv('VNINDEX')
        # V2 FIX: Dedup index ngay khi lay du lieu
        if vnindex_df.index.duplicated().any():
            vnindex_df = vnindex_df[~vnindex_df.index.duplicated(keep='last')]
        
        regime_result = classify_market_regime(vnindex_df)
    except Exception as e:
        print(f"[Regime] Loi: {e}")
        from market_regime import REGIMES
        regime_result = {'regime': 'SIDEWAY', 'label': '🟡 Sideway',
                         'strategy': 'swing', 'weight': 0.8,
                         'confidence': 0.5, 'signals': []}
    regime_report = format_regime_report(regime_result)
    print(f"  → {regime_result['label']}")

    macro_context = cfg.get('macro_context', {})
    sector_signals = get_sector_rotation_signal(macro_context)

    # ═════════════════════════════════════════════════════════════════════
    # 2. DATA LAYER – price, flow, macro
    # ═════════════════════════════════════════════════════════════════════
    print("[Pipeline] === Step 2: Data Layer ===")
    price_df = fetcher.get_multi_close(symbols)
    price_df = price_df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    
    # V2 FIX: Dedup price_df
    if price_df.index.duplicated().any():
        price_df = price_df[~price_df.index.duplicated(keep='last')]

    returns_df = price_df.pct_change().dropna()
    # V2 FIX: Dedup returns_df
    if returns_df.index.duplicated().any():
        returns_df = returns_df[~returns_df.index.duplicated(keep='last')]

    # --- normalize macro data ---
    try:
        macro_raw = fetcher.get_macro_data(indicators=macro_indicators, start=start, end=end)
    except Exception:
        macro_raw = {}

    macro_df = None
    try:
        if hasattr(fetcher, "get_macro_df"):
            macro_df = fetcher.get_macro_df(indicators=macro_indicators, start=start, end=end)
    except Exception:
        macro_df = None

    if macro_df is None or not isinstance(macro_df, pd.DataFrame):
        macro_df = pd.DataFrame()
        if isinstance(macro_raw, dict):
            parts = []
            for k, v in macro_raw.items():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    df = v.copy()
                    # V2 FIX: Dedup tung phan truoc khi concat
                    if df.index.duplicated().any():
                        df = df[~df.index.duplicated(keep='last')]
                        
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    col = numeric_cols[0] if numeric_cols else (df.columns[0] if len(df.columns) > 0 else None)
                    if col is None: continue
                    s = pd.to_numeric(df[col], errors="coerce").rename(k)
                    if not isinstance(s.index, pd.DatetimeIndex):
                        try: s.index = pd.to_datetime(df.index)
                        except Exception: pass
                    parts.append(s)
            if parts:
                # V2 FIX: Concat voi join='outer' nhung check duplicate index cuoi cung
                macro_df = pd.concat(parts, axis=1, join="outer")
                if macro_df.index.duplicated().any():
                    macro_df = macro_df[~macro_df.index.duplicated(keep='last')]
                macro_df = macro_df.sort_index()
        elif isinstance(macro_raw, pd.DataFrame):
            macro_df = macro_raw

    if macro_df is not None and not macro_df.empty:
        print(f"[Pipeline] Macro data loaded with columns: {macro_df.columns.tolist()}")
    else:
        print("[Pipeline] No macro data available or macro_df is empty")

    # Combine returns and macro for causal discovery
    try:
        if macro_df is not None and not macro_df.empty:
            # V2 FIX: Reindex macro_df theo index cua returns_df de dam bao k trung lap
            # Su dung ffill de dien du lieu macro cho nhung ngay khong co giao dich
            macro_aligned = macro_df.reindex(returns_df.index, method='ffill')
            combined_df = pd.concat([returns_df, macro_aligned], axis=1, join='inner')
        else:
            combined_df = returns_df.copy()
            
        combined_df = combined_df.dropna(how='all')
        # V2 FIX: Dedup final combined_df
        if combined_df.index.duplicated().any():
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
        var_names = combined_df.columns.tolist()
    except Exception as e:
        print(f"[Pipeline] Warning combining data for causal discovery: {e}")
        combined_df = returns_df.copy()
        var_names = combined_df.columns.tolist()

    # ═════════════════════════════════════════════════════════════════════
    # 3. CAUSAL DISCOVERY
    # ═════════════════════════════════════════════════════════════════════
    print("[Pipeline] === Step 3: Causal Discovery ===")
    pcmci_cfg = cfg.get('pcmci', {})
    try:
        links = run_pcmci(combined_df, var_names,
                          tau_max=pcmci_cfg.get('tau_max', 4),
                          alpha=pcmci_cfg.get('alpha', 0.05))
    except Exception as e:
        print(f"[PCMCI] Loi: {e}")
        links = []
    try:
        links += run_lingam(combined_df, var_names)
    except Exception as e:
        print(f"[LiNGAM] Loi: {e}")
    causal_report = generate_causal_report(links)

    # ═════════════════════════════════════════════════════════════════════
    # 4. AI AGENTS
    # ═════════════════════════════════════════════════════════════════════
    enabled_names = set(cfg['ai_agents'].get('enabled', []))
    agent_cfgs    = [a for a in cfg['ai_agents'].get('agents', []) if a.get('name') in enabled_names]

    if not agent_cfgs:
        agent_cfgs = [
            {'name': n, 'provider': p, 'model': m}
            for n, (p, m) in AGENT_REGISTRY.items()
            if n in enabled_names
        ]
    agents = build_agents_from_config(agent_cfgs)

    # ═════════════════════════════════════════════════════════════════════
    # 5. PER-SYMBOL ANALYSIS
    # ═════════════════════════════════════════════════════════════════════
    results = {}
    port_value = cfg.get('portfolio', {}).get('portfolio_value', None)

    for symbol in symbols:
        print(f"\n[Pipeline] ── {symbol} ──")

        # 5a. OHLCV + Features
        try:
            ohlcv = fetcher.get_ohlcv(symbol)
            # V2 FIX: Dedup OHLCV ngay khi lay
            if ohlcv.index.duplicated().any():
                ohlcv = ohlcv[~ohlcv.index.duplicated(keep='last')]
                
            ff    = fetcher.get_foreign_flow(symbol)
            pp    = fetcher.get_proprietary_flow(symbol)
            try:
                bt = fetcher.get_block_trades(symbol)
            except Exception:
                bt = pd.DataFrame()
            feat_df = build_features(ohlcv, ff, pp, bt)
        except Exception as e:
            print(f"  [Feature] Loi: {e}")
            feat_df = pd.DataFrame()

        # 5b. Alpha Factors
        if not feat_df.empty:
            try:
                alpha_factors = compute_alpha_factors(feat_df)
                alpha_scores  = compute_alpha_score(alpha_factors)
            except Exception as e:
                print(f"  [Alpha] Loi: {e}")
                alpha_scores = {'composite': 0, 'momentum': 0, 'volume': 0,
                                'smart_money': 0, 'trend': 0}
        else:
            alpha_scores = {'composite': 0, 'momentum': 0, 'volume': 0,
                            'smart_money': 0, 'trend': 0}

        # 5c. Wyckoff
        try:
            wyckoff_result = detect_wyckoff_phase(feat_df if not feat_df.empty else ohlcv)
            wyckoff_report = format_wyckoff_report(wyckoff_result)
        except Exception as e:
            print(f"  [Wyckoff] Loi: {e}")
            wyckoff_result = {'phase': 'UNKNOWN', 'confidence': 0,
                              'signals': [], 'is_buy_zone': False, 'is_sell_zone': False}
            wyckoff_report = "Wyckoff: Khong xac dinh duoc"

        # 5d. Signal Engine
        try:
            fundamental   = fetcher.get_fundamental(symbol)
            signal_result = generate_signal(alpha_scores, wyckoff_result,
                                            regime_result, fundamental)
            signal_report = format_signal_report(signal_result)
        except Exception as e:
            print(f"  [Signal] Loi: {e}")
            signal_result = {'signal': 'HOLD', 'label': '🟡 HOLD', 'score': 0,
                             'reasons': [], 'wyckoff_phase': 'UNKNOWN',
                             'regime': regime_result.get('regime','SIDEWAY')}
            signal_report = "Signal: HOLD"
            fundamental   = {}

        # 5e. Flow Summary
        try:
            flow_summary = fetcher.get_flow_summary(symbol)
        except Exception:
            flow_summary = {}

        # 5f. Position Sizing
        try:
            last_close = float(ohlcv['close'].iloc[-1]) if not ohlcv.empty and 'close' in ohlcv.columns else None
            atr_val    = float(feat_df['atr'].iloc[-1]) if not feat_df.empty and 'atr' in feat_df.columns else None
            position   = compute_position_size(signal_result, regime_result,
                                               atr=atr_val, close_price=last_close)
        except Exception as e:
            print(f"  [Position] Loi: {e}")
            position = {}

        # 5f-2. Volume Profile + Cumulative Delta + Absorption
        try:
            vp_result    = compute_volume_profile(ohlcv, lookback=60)
            cum_delta    = compute_cumulative_delta(ohlcv)
            absorption   = detect_absorption(ohlcv)
            smc_result   = smart_money_checklist(ohlcv, vp_result, cum_delta,
                                                  absorption, flow_summary)
            vp_report    = format_vp_report(vp_result)
            smc_report   = format_smc_report(smc_result, cum_delta, absorption)
        except Exception as e:
            print(f"  [VP/SMC] Loi: {e}")
            vp_result = cum_delta = absorption = smc_result = {}
            vp_report = smc_report = "Khong tinh duoc"

        # 5f-3. Event Risk Score
        news_text = ""
        try:
            news_df = fetcher.get_company_news(symbol)
            if news_df is not None and not news_df.empty:
                title_col = next((c for c in news_df.columns
                                  if 'title' in c.lower() or 'head' in c.lower()), None)
                if title_col:
                    news_text = " ".join(news_df[title_col].astype(str).tolist()[:20])
        except Exception:
            pass

        try:
            ers_result = compute_ers(symbol, fundamental, ohlcv,
                                     flow_summary, news_text,
                                     cfg.get('macro_context', {}))
            ers_report = format_ers_report(ers_result)
            if ers_result.get('is_blocked'):
                position = {'tier':'blocked','tier_desc':'ERS≥9 – THOAT TOAN BO',
                            'size_pct':0, 'size_vnd':0, 'stop_loss':None, 'risk_vnd':0}
                print(f"  [ERS] {symbol}: BLOCKED (score={ers_result.get('ers_score')})")
            elif ers_result.get('is_caution'):
                if position.get('size_pct', 0) > ers_result.get('max_position', 0):
                    position['size_pct'] = ers_result.get('max_position', 0)
                    position['size_vnd'] = position.get('size_pct', 0) * (port_value if port_value is not None else 0)
        except Exception as e:
            print(f"  [ERS] Loi: {e}")
            ers_result = {'ers_score':0,'color':'🟢','action':'N/A','max_position':0.2,
                          'is_blocked':False,'is_caution':False,'scenario':{}}
            ers_report = "ERS: Khong tinh duoc"

        # 5g. Tech Summary
        if not feat_df.empty:
            try:
                last = feat_df.iloc[-1]
                close_val = float(ohlcv['close'].iloc[-1]) if not ohlcv.empty and 'close' in ohlcv.columns else 0.0
                tech_summary = (
                    f"Gia dong cua: {close_val:,.0f}\n"
                    f"MA20={float(last.get('ma20',0)):,.0f} | MA50={float(last.get('ma50',0)):,.0f}\n"
                    f"RSI={float(last.get('rsi14',50)):.1f} | MACD hist={float(last.get('macd_hist',0)):+.2f}\n"
                    f"BB pos={float(last.get('bb_pos',0.5)):.0%} | Squeeze={int(last.get('bb_squeeze',0))}\n"
                    f"Volume ratio={float(last.get('vol_ratio',1)):.1f}x | OBV trend={float(last.get('obv_trend',0)):+.0f}\n"
                    f"Breakout20={int(last.get('breakout20',0))} | Compression={float(last.get('compression',1)):.2f}\n"
                    f"Ret 1M={float(last.get('ret_20d',0)):+.1%} | 3M={float(last.get('ret_60d',0)):+.1%}"
                )
            except Exception:
                tech_summary = "Khong tinh duoc tech summary"
        else:
            tech_summary = "Khong co du lieu ky thuat"

        # 5g-2. Valuation Engine
        try:
            income_df = fetcher.get_income_statement(symbol)
            cf_df     = fetcher.get_cash_flow(symbol)
            earnings  = analyze_earnings(symbol, income_df, cashflow_df=cf_df)
            dcf_result= run_dcf(symbol, fundamental, income_df, cf_df)
            dcf_report= format_dcf_report(dcf_result)

            peers     = VN_SECTOR_PEERS.get(symbol, [])[:4]
            peer_funds= {}
            for ps in peers:
                try:
                    peer_funds[ps] = fetcher.get_fundamental(ps)
                except Exception:
                    pass
            comps_result = run_comps(symbol, fundamental, peer_funds) if peer_funds else {}
            comps_report = format_comps_report(comps_result) if comps_result else "Comps: Không có peers"

            ic_memo = generate_ic_memo(symbol, dcf_result, comps_result, earnings,
                                       signal_result, ers_result, wyckoff_result, regime_result)
        except Exception as e:
            print(f"  [Valuation] Loi: {e}")
            earnings = dcf_result = comps_result = {}
            dcf_report = comps_report = ic_memo = "Khong tinh duoc"

        # 5h. AI Ensemble
        prompt    = build_level6_prompt(
            symbol, tech_summary, wyckoff_report,
            signal_report, regime_report, fundamental,
            causal_report, flow_summary,
            ers_report=ers_report,
            vp_report=vp_report,
            smc_report=smc_report,
        )
        decisions = {}
        for name, agent in agents.items():
            try:
                resp = agent.ask(SYSTEM_PROMPT, prompt)
            except Exception:
                resp = None
            if resp:
                dec = extract_json(resp)
                decisions[name] = dec if dec else {'action': 'HOLD', 'confidence': 0.0}
            else:
                decisions[name] = {'action': 'HOLD', 'confidence': 0.0}
            print(f"  [{name}] {decisions[name].get('action','?')} "
                  f"({decisions[name].get('confidence',0):.0%})")
            time.sleep(0.5)

        ai_action, ai_score = meta_ensemble(decisions)
        engine_score = signal_result.get('score', 0)
        final_score  = engine_score * 0.4 + ai_score * 0.6
        final_action = _score_to_action(final_score)

        print(f"  [Ensemble] {symbol}: {final_action} "
              f"(engine={engine_score:+.2f} ai={ai_score:+.2f} final={final_score:+.2f})")

        results[symbol] = {
            'action':        final_action,
            'score':         round(final_score, 4),
            'engine_signal': signal_result,
            'wyckoff':       wyckoff_result,
            'alpha_scores':  alpha_scores,
            'position':      position,
            'details':       decisions,
            'tech_summary':  tech_summary,
            'fundamental':   fundamental,
            'flow_summary':  flow_summary,
            'causal_report': causal_report,
            'regime':        regime_result,
            'ers':           ers_result,
            'ers_report':    ers_report,
            'vp':            vp_result,
            'vp_report':     vp_report,
            'smc':           smc_result,
            'smc_report':    smc_report,
            'cum_delta':     cum_delta,
            'earnings':      earnings,
            'dcf':           dcf_result,
            'comps':         comps_result,
            'dcf_report':    dcf_report,
            'comps_report':  comps_report,
            'ic_memo':       ic_memo,
        }

    # ═════════════════════════════════════════════════════════════════════
    # 6. PORTFOLIO ENGINE
    # ═════════════════════════════════════════════════════════════════════
    print("\n[Pipeline] === Step 6: Portfolio Engine ===")
    signal_map = {sym: res['engine_signal'] for sym, res in results.items()}
    alloc_df, cash_pct = compute_portfolio_allocation(signal_map, regime_result)
    risk_metrics = compute_portfolio_risk(returns_df)
    portfolio_report = format_portfolio_report(
        alloc_df, cash_pct, risk_metrics, regime_result.get('label','')
    )
    print(portfolio_report)

    return {
        'symbols':          results,
        'regime':           regime_result,
        'causal_report':    causal_report,
        'portfolio':        {'allocation': alloc_df, 'cash_pct': cash_pct,
                             'risk': risk_metrics, 'report': portfolio_report},
        'sector_rotation':  sector_signals,
    }

def _score_to_action(score):
    if score >= 0.6:   return 'STRONG_BUY'
    elif score >= 0.4: return 'BUY'
    elif score >= 0.15:return 'ACCUMULATION'
    elif score >= -0.3:return 'HOLD'
    elif score >= -0.6:return 'REDUCE'
    else:              return 'SELL'