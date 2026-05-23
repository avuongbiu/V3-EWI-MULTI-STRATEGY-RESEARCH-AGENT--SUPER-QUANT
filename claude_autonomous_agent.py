"""
claude_autonomous_agent.py
Claude tự suy luận qua Tool Use (MCP pattern)
Thay vì nhận data pre-fetched, Claude chủ động gọi vnstock tools để research

Pattern: anthropics/financial-services managed-agent-cookbooks
Adapted for VN market với vnstock_data
"""
import json
import time
import os
from dotenv import load_dotenv
from vn_mcp_server import get_anthropic_tools, process_tool_call

load_dotenv()

# ── VN-Specific Skills (Markdown prompts) ──────────────────────────────

VN_SKILLS = {
    "market_analyst": """
# VN Market Analyst Skill

## Vai trò
Chuyên gia phân tích thị trường chứng khoán Việt Nam (HOSE/HNX/UPCoM).
Sử dụng framework: Macro VN → Sector → Stock (Top-down).

## Ngữ cảnh VN
- VNIndex phản ánh tâm lý + dòng tiền ngoại mạnh hơn nền tảng
- Thanh khoản thị trường ~15,000-25,000 tỷ VND/ngày
- Ngoại bán ròng = tín hiệu cảnh báo quan trọng
- Room ngoại (foreign room) ảnh hưởng định giá premium
- T+2.5 settlement → margin call risk khi thị trường volatile

## Cách tiếp cận
1. Kiểm tra VNIndex trend + market breadth
2. Xác định sector đang dẫn dắt dòng tiền
3. Tìm mã có setup kỹ thuật + cơ bản tốt nhất trong sector đó
4. Đánh giá ERS (Event Risk Score) trước khi khuyến nghị
""",

    "equity_researcher": """
# VN Equity Research Skill

## Phương pháp phân tích
1. **Business Quality**: Lợi thế cạnh tranh, thị phần, ROE bền vững
2. **Financial Health**: Nợ/Vốn <1.5x, FCF dương, Earnings quality cao
3. **Growth**: Tăng trưởng doanh thu >10% YoY, expand margin
4. **Valuation**: DCF + Peer comps, không trả premium >20% vs peers nếu không có catalyst

## VN-Specific Adjustments
- Discount 15-20% cho governance risk (quản trị)
- Kiểm tra cổ đông lớn bán (insider selling)
- Xem xét room ngoại còn lại
- BCTC VN theo VAS (khác IFRS) → điều chỉnh cho goodwill, deferral

## Output chuẩn
Luôn kết thúc bằng: BUY/HOLD/SELL + price target + stop loss + key catalyst + key risk
""",

    "risk_manager": """
# VN Risk Management Skill

## ERS Framework (0-30)
- 0-4: Giao dịch bình thường, max position 20%
- 5-6: Giảm tỷ trọng, stop loss 3.5%
- 7-8: Không mở mới, reduce 50-70%
- ≥9: Thoát toàn bộ

## VN-Specific Risks
1. Audit exception (ý kiến ngoại trừ) → +3 điểm ngay
2. Cổ đông lớn đăng ký bán → +3 điểm
3. Trái phiếu doanh nghiệp sắp đáo hạn → +3 điểm
4. Room ngoại = 0% → thanh khoản rủi ro
5. Vốn hóa < 500 tỷ → liquidity risk cao
"""
}

# ── Autonomous Research Agent ───────────────────────────────────────────

class ClaudeAutonomousAgent:
    """
    Claude tự chủ động gọi vnstock tools để research (không pre-fetch).
    Pattern: Tool Use loop như anthropics managed agents.
    """

    def __init__(self, model='claude-sonnet-4-5', max_iterations=10,
                 source='VCI', start_date='2023-01-01'):
        self.model          = model
        self.max_iterations = max_iterations
        self.source         = source
        self.start_date     = start_date
        self.tools          = get_anthropic_tools()
        self._client        = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY', ''))
        return self._client

    def run(self, task: str, skill: str = 'equity_researcher',
            verbose: bool = True) -> dict:
        """
        Chạy autonomous research loop.
        Claude tự quyết định gọi tools nào, bao nhiêu lần, theo thứ tự nào.
        """
        client   = self._get_client()
        skill_md = VN_SKILLS.get(skill, VN_SKILLS['equity_researcher'])

        system = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam đẳng cấp quốc tế.
Bạn có quyền truy cập vào dữ liệu thị trường VN real-time qua các tools.

{skill_md}

## Quy trình làm việc
1. Tự thu thập dữ liệu cần thiết bằng tools (đừng đoán mò)
2. Phân tích toàn diện dựa trên dữ liệu thực
3. Đưa ra kết luận có căn cứ, trích dẫn số liệu cụ thể
4. Kết thúc bằng Summary với Action + Target + Stop Loss + Key Risk

## Nguyên tắc
- Luôn kiểm tra cả kỹ thuật VÀ cơ bản
- Không đưa ra khuyến nghị nếu chưa đủ dữ liệu
- Cảnh báo rõ nếu có Event Risk cao (ERS ≥ 5)
"""

        messages = [{"role": "user", "content": task}]
        tool_calls_made = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            if verbose: print(f"  [AutoAgent] Iteration {iterations}...")

            resp = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                tools=self.tools,
                messages=messages,
            )

            # Xử lý response
            tool_uses    = [b for b in resp.content if b.type == 'tool_use']
            text_blocks  = [b for b in resp.content if b.type == 'text']

            # Nếu Claude dừng (end_turn hoặc không còn tools)
            if resp.stop_reason == 'end_turn' or not tool_uses:
                final_text = "\n".join(b.text for b in text_blocks)
                if verbose: print(f"  [AutoAgent] Hoàn tất sau {iterations} iterations, {len(tool_calls_made)} tool calls")
                return {
                    'result':       final_text,
                    'tool_calls':   tool_calls_made,
                    'iterations':   iterations,
                    'model':        self.model,
                }

            # Thêm assistant message
            messages.append({"role": "assistant", "content": resp.content})

            # Thực thi tool calls
            tool_results = []
            for tool_use in tool_uses:
                if verbose: print(f"    → Calling: {tool_use.name}({json.dumps(tool_use.input)[:80]}...)")
                result = process_tool_call(
                    tool_use.name, tool_use.input,
                    source=self.source,
                    start_date=self.start_date
                )
                tool_calls_made.append({'tool': tool_use.name, 'input': tool_use.input})
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use.id,
                    "content":     result[:8000],  # Truncate large results
                })
                time.sleep(0.3)

            messages.append({"role": "user", "content": tool_results})

        return {
            'result':     'Max iterations reached',
            'tool_calls': tool_calls_made,
            'iterations': iterations,
        }

    # ── Slash Commands ──────────────────────────────────────────────────

    def cmd_comps(self, symbol: str) -> dict:
        """/comps — Comparable Company Analysis VN"""
        return self.run(
            f"""Thực hiện Comparable Company Analysis cho {symbol}:
1. Dùng get_fundamental để lấy PE, PB, ROE, EPS của {symbol}
2. Xác định peer group phù hợp (cùng ngành)
3. Lấy fundamental của từng peer
4. So sánh multiples và tính implied value
5. Đánh giá premium/discount vs peers có hợp lý không (dựa vào ROE diff)
6. Kết luận: CHEAP / FAIR / EXPENSIVE + implied fair value (VND/cp)""",
            skill='equity_researcher'
        )

    def cmd_dcf(self, symbol: str) -> dict:
        """/dcf — DCF Valuation VN"""
        return self.run(
            f"""Xây dựng DCF Model cho {symbol}:
1. Lấy BCTC 3-5 năm qua (get_income_statement, get_cash_flow)
2. Tính FCF lịch sử và margin
3. Dự báo FCF 5 năm tới (WACC phù hợp ngành VN: 11-15%)
4. Terminal value với growth = 5%
5. Sensitivity table (WACC ±2%, Growth ±5%)
6. Kết luận: Intrinsic Value (VND/cp) + Upside/Downside vs giá hiện tại""",
            skill='equity_researcher'
        )

    def cmd_earnings(self, symbol: str) -> dict:
        """/earnings — Earnings Quality Review VN"""
        return self.run(
            f"""Phân tích chất lượng lợi nhuận {symbol}:
1. Lấy BCTC 4 quý gần nhất (get_income_statement period=quarter)
2. Phân tích: Revenue trend, Gross/Net margin trend
3. Kiểm tra earnings quality: CFO/Net Profit ratio
4. Xem news gần nhất có rủi ro kiểm toán/pháp lý không
5. So sánh với kỳ trước và sector median
6. Kết luận: HIGH/MEDIUM/LOW quality + EPS forecast + Surprise risk""",
            skill='equity_researcher'
        )

    def cmd_screen(self, criteria: str) -> dict:
        """/screen — Stock Screening VN"""
        return self.run(
            f"""Tìm kiếm cổ phiếu VN theo tiêu chí: {criteria}
1. Dùng screen_stocks với các filter phù hợp
2. Lấy fundamental của top 5 kết quả
3. Kiểm tra kỹ thuật nhanh (get_stock_price để xem trend)
4. Đánh giá nhanh từng mã
5. Rank top 3 tốt nhất với lý do cụ thể""",
            skill='market_analyst'
        )

    def cmd_ic_memo(self, symbol: str) -> dict:
        """/ic-memo — Investment Committee Memo VN"""
        return self.run(
            f"""Viết Investment Committee Memo đầy đủ cho {symbol}:
1. Thu thập: giá, fundamental, BCTC, news, foreign flow
2. Phân tích: Business model, competitive moat, growth drivers
3. Valuation: DCF + Peer comps
4. Risk: ERS assessment, key risks
5. Scenario: Bull (P=25%) / Base (P=55%) / Bear (P=20%)
6. Khuyến nghị: Action + Entry zone + Target + Stop Loss + Holding period
Format: Executive Summary → Business → Financials → Valuation → Risk → Recommendation""",
            skill='equity_researcher'
        )

    def cmd_morning_note(self, symbols: list) -> dict:
        """/morning-note — VN Morning Briefing"""
        syms_str = ", ".join(symbols[:5])
        return self.run(
            f"""Viết VN Morning Note cho ngày hôm nay:
1. Market Overview: VNIndex trend, breadth, foreign flow (get_market_overview)
2. Sector rotation: ngành nào đang lead/lag
3. Quick update các mã: {syms_str}
4. Top 2-3 trading ideas cho hôm nay
5. Key risks cần theo dõi
Format ngắn gọn, phù hợp đọc buổi sáng (< 500 từ)""",
            skill='market_analyst'
        )


# ── Factory ─────────────────────────────────────────────────────────────

def create_autonomous_agent(provider='anthropic', model=None,
                             source='VCI', start_date='2023-01-01') -> ClaudeAutonomousAgent:
    """Tạo agent. Hiện tại chỉ Claude hỗ trợ tool_use natively."""
    if model is None:
        # Ưu tiên model có tool_use tốt
        model = 'claude-sonnet-4-5'

    return ClaudeAutonomousAgent(
        model=model, source=source, start_date=start_date
    )
