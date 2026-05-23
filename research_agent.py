"""
research_agent.py
Multi-Agent Research Orchestration
Inspired by anthropics/financial-services market-researcher + earnings-reviewer + valuation-reviewer
Architecture: Orchestrator → Sub-agents (1 level delegation)
"""
import time
from ai_agents import AIAgent, extract_json
from valuation_engine import (run_dcf, run_comps, analyze_earnings,
                               generate_ic_memo, format_dcf_report,
                               format_comps_report, VN_SECTOR_PEERS)


# ════════════════════════════════════════════════════════════════════════
# SPECIALIZED SYSTEM PROMPTS (Skills)
# Inspired by financial-services/plugins/vertical-plugins/*/skills/
# ════════════════════════════════════════════════════════════════════════

SKILLS = {
    'market_researcher': """Bạn là Market Researcher chuyên sâu về thị trường chứng khoán Việt Nam.
Nhiệm vụ: Phân tích ngành, xu hướng sector, cơ hội investment trong bối cảnh macro VN.
Framework: Top-down (Macro VN → Sector → Stock).
Output: Luôn kết thúc bằng JSON với: sector_outlook, key_catalysts, key_risks, recommended_stocks.""",

    'earnings_analyst': """Bạn là Earnings Analyst chuyên phân tích BCTC doanh nghiệp Việt Nam.
Nhiệm vụ: Đánh giá chất lượng lợi nhuận, xu hướng tăng trưởng, earnings quality.
Kiểm tra: Revenue, LNST, biên lợi nhuận, CFO/Net Profit ratio, working capital.
Output: Luôn kết thúc bằng JSON với: earnings_quality (HIGH/MEDIUM/LOW), growth_trend, key_concern, eps_forecast.""",

    'valuation_agent': """Bạn là Valuation Specialist với chuyên môn về thị trường VN.
Nhiệm vụ: DCF + Peer Comps + Asset-based valuation cho cổ phiếu Việt Nam.
Điều chỉnh cho VN market: discount premium 15-20% cho liquidity risk, governance risk.
Output: Luôn kết thúc bằng JSON với: fair_value (VND/cp), upside_pct, valuation_verdict, key_assumption.""",

    'technical_analyst': """Bạn là Technical Analyst chuyên Wyckoff + Volume Profile + Smart Money VN.
Nhiệm vụ: Xác định Phase Wyckoff, POC/VAL/VAH, tín hiệu Smart Money.
Entry discipline: CHỈ mua tại Spring test thành công + SOS + LPS pullback.
Output: Luôn kết thúc bằng JSON với: phase, entry_trigger, stop_loss, target_1, target_2, confidence.""",

    'risk_analyst': """Bạn là Risk Analyst cho quỹ đầu tư tại Việt Nam.
Nhiệm vụ: Đánh giá Event Risk Score (ERS), rủi ro pháp lý, thanh khoản, pha loãng.
Framework: ERS 4 nhóm (0-30 điểm) + Macro Adjustment.
Output: Luôn kết thúc bằng JSON với: ers_score, max_position_pct, stop_loss_pct, key_risks (list), verdict.""",
}


# ════════════════════════════════════════════════════════════════════════
# SUB-AGENTS
# ════════════════════════════════════════════════════════════════════════

class SubAgent:
    """Specialized sub-agent với skill-based system prompt."""

    def __init__(self, role: str, provider: str, model: str):
        self.role     = role
        self.skill    = SKILLS.get(role, '')
        self._agent   = AIAgent(provider=provider, model=model,
                                temperature=0.2, max_tokens=1200)

    def analyze(self, user_prompt: str) -> dict | None:
        resp = self._agent.ask(self.skill, user_prompt)
        if not resp:
            return None
        dec = extract_json(resp)
        return {'raw': resp, 'parsed': dec, 'role': self.role}


# ════════════════════════════════════════════════════════════════════════
# RESEARCH ORCHESTRATOR
# Inspired by market-researcher agent architecture
# ════════════════════════════════════════════════════════════════════════

class ResearchOrchestrator:
    """
    Điều phối nhiều sub-agents để tạo research report đầy đủ.
    Pattern: Orchestrator → sector-reader + earnings-analyst + valuation-agent + risk-analyst
    """

    def __init__(self, agent_configs: list[dict]):
        """
        agent_configs: [{'name':..., 'provider':..., 'model':...}]
        Tự động gán role theo thứ tự hoặc theo config.
        """
        self.sub_agents: dict[str, SubAgent] = {}
        role_order = ['market_researcher', 'earnings_analyst',
                      'valuation_agent', 'technical_analyst', 'risk_analyst']

        for i, cfg in enumerate(agent_configs[:5]):  # max 5 sub-agents
            role     = cfg.get('role', role_order[i % len(role_order)])
            provider = cfg.get('provider', 'groq')
            model    = cfg.get('model', 'llama-3.3-70b-versatile')
            self.sub_agents[role] = SubAgent(role, provider, model)
            print(f"[ResearchOrchestrator] {role} → {provider}/{model}")

    def run_market_research(self, sector: str, symbols: list[str],
                            regime_report: str, macro_context: dict) -> dict:
        """Sub-agent: Market Researcher → sector overview + catalysts."""
        agent = self.sub_agents.get('market_researcher')
        if not agent:
            return {}

        prompt = f"""Phân tích ngành/sector: {sector}
Các mã trong universe: {', '.join(symbols)}
Market Regime: {regime_report}
Macro context: {macro_context}

Cho tôi:
1. Sector outlook (3-6 tháng)
2. Key catalysts (tăng giá)
3. Key risks
4. Top 3 mã đáng theo dõi nhất và lý do

Kết thúc bằng JSON:
{{"sector_outlook": "POSITIVE|NEUTRAL|NEGATIVE",
  "key_catalysts": ["...", "..."],
  "key_risks": ["...", "..."],
  "recommended_stocks": ["SYM1", "SYM2"],
  "conviction": 0.0-1.0}}"""

        result = agent.analyze(prompt)
        time.sleep(0.5)
        return result or {}

    def run_earnings_review(self, symbol: str, earnings: dict,
                            fundamental: dict) -> dict:
        """Sub-agent: Earnings Analyst → chất lượng lợi nhuận."""
        agent = self.sub_agents.get('earnings_analyst')
        if not agent:
            return {}

        prompt = f"""Phân tích BCTC {symbol}:
Doanh thu YoY: {earnings.get('revenue_yoy','N/A')}%
LNST YoY: {earnings.get('profit_yoy','N/A')}%
Net Margin: {earnings.get('net_margin','N/A')}%
Earnings Quality: {earnings.get('earnings_quality','N/A')}
PE={fundamental.get('pe','N/A')} | ROE={fundamental.get('roe','N/A')}%
CFO/LNST: {earnings.get('cfo_to_profit','N/A')}

Đánh giá chất lượng lợi nhuận và xu hướng. Kết thúc bằng JSON:
{{"earnings_quality": "HIGH|MEDIUM|LOW",
  "growth_trend": "ACCELERATING|STABLE|DECELERATING|DECLINING",
  "key_concern": "...",
  "eps_forecast_yoy": <pct>,
  "surprise_risk": "HIGH|MEDIUM|LOW"}}"""

        result = agent.analyze(prompt)
        time.sleep(0.5)
        return result or {}

    def run_valuation_review(self, symbol: str, dcf: dict,
                              comps: dict, current_price: float) -> dict:
        """Sub-agent: Valuation Agent → fair value + upside."""
        agent = self.sub_agents.get('valuation_agent')
        if not agent:
            return {}

        dcf_val   = dcf.get('intrinsic_per_share', 0) if dcf else 0
        comps_val = comps.get('implied_pe_price', 0) if comps else 0
        verdict   = comps.get('verdict', 'N/A') if comps else 'N/A'

        prompt = f"""Đánh giá định giá {symbol}:
Giá hiện tại: {current_price:,.0f} VND
DCF Intrinsic Value: {dcf_val:,.0f} VND
Comps Implied Value (PE): {comps_val:,.0f} VND
Comps Verdict: {verdict}
PE vs Peers: {comps.get('pe_premium','N/A')}% premium
ROE vs Peers: {comps.get('roe_vs_peers','N/A')}%

Tính upside/downside và cho khuyến nghị. Kết thúc bằng JSON:
{{"fair_value": <VND>, "upside_pct": <pct>, "valuation_verdict": "CHEAP|FAIR|EXPENSIVE",
  "key_assumption": "...", "blended_target": <VND>}}"""

        result = agent.analyze(prompt)
        time.sleep(0.5)
        return result or {}

    def run_technical_review(self, symbol: str, tech_summary: str,
                              wyckoff_report: str, vp_report: str,
                              smc_report: str) -> dict:
        """Sub-agent: Technical Analyst → entry/exit points."""
        agent = self.sub_agents.get('technical_analyst')
        if not agent:
            return {}

        prompt = f"""Phân tích kỹ thuật {symbol}:
{tech_summary}

{wyckoff_report}

{vp_report}

{smc_report}

Cho tôi entry strategy cụ thể. Kết thúc bằng JSON:
{{"phase": "A|B|C|D|E|DIST",
  "entry_trigger": "...",
  "stop_loss": <VND>,
  "target_1": <VND>,
  "target_2": <VND>,
  "confidence": 0.0-1.0,
  "action": "BUY|HOLD|SELL|WAIT"}}"""

        result = agent.analyze(prompt)
        time.sleep(0.5)
        return result or {}

    def run_risk_review(self, symbol: str, ers_report: str,
                         flow_summary: dict) -> dict:
        """Sub-agent: Risk Analyst → ERS + position sizing."""
        agent = self.sub_agents.get('risk_analyst')
        if not agent:
            return {}

        ff_net = flow_summary.get('foreign_net_30d', 0) or 0

        prompt = f"""Đánh giá rủi ro {symbol}:
{ers_report}

Foreign net 30 ngày: {ff_net:,.0f}
Tự doanh net: {flow_summary.get('prop_net_30d',0):,.0f}

Cho tôi risk assessment và position sizing. Kết thúc bằng JSON:
{{"ers_score": <int>, "max_position_pct": <pct>, "stop_loss_pct": <pct>,
  "key_risks": ["...", "..."],
  "verdict": "GREEN|YELLOW|ORANGE|RED"}}"""

        result = agent.analyze(prompt)
        time.sleep(0.5)
        return result or {}

    # ── Master Research Report ─────────────────────────────────────────
    def run_full_research(self, symbol: str, context: dict) -> dict:
        """
        Chạy toàn bộ research pipeline cho 1 mã.
        context: dict với tất cả data đã fetch (ohlcv, fundamental, vp, ers, etc.)
        """
        print(f"  [Research] Bắt đầu research {symbol}...")
        research = {'symbol': symbol}

        # 1. Earnings review
        if 'earnings' in context and 'fundamental' in context:
            research['earnings_review'] = self.run_earnings_review(
                symbol, context['earnings'], context['fundamental'])

        # 2. Valuation review
        if 'dcf' in context and 'comps' in context:
            research['valuation_review'] = self.run_valuation_review(
                symbol, context['dcf'], context['comps'],
                context.get('current_price', 0))

        # 3. Technical review
        if all(k in context for k in ['tech_summary', 'wyckoff_report']):
            research['technical_review'] = self.run_technical_review(
                symbol, context['tech_summary'],
                context.get('wyckoff_report', ''),
                context.get('vp_report', ''),
                context.get('smc_report', ''))

        # 4. Risk review
        if 'ers_report' in context:
            research['risk_review'] = self.run_risk_review(
                symbol, context['ers_report'],
                context.get('flow_summary', {}))

        return research


def build_research_orchestrator(agent_configs: list[dict]) -> ResearchOrchestrator | None:
    """Factory function."""
    if not agent_configs:
        return None
    try:
        return ResearchOrchestrator(agent_configs)
    except Exception as e:
        print(f"[ResearchOrchestrator] Khởi tạo lỗi: {e}")
        return None
