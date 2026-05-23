"""
vn_mcp_server.py
Custom MCP Server wrapping vnstock_data Silver Sponsor
Thay thế Bloomberg/Morningstar/FactSet bằng dữ liệu VN bản địa

Transport: stdio (Claude Code) hoặc HTTP (Streamlit)
"""
import json
import sys
import asyncio
import logging
from typing import Any
from datetime import datetime, timedelta

# Thử import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# vnstock imports
try:
    from data_fetcher import DataFetcher
    VNSTOCK_AVAILABLE = True
except Exception:
    VNSTOCK_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Tool definitions ────────────────────────────────────────────────────
VN_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Lấy dữ liệu giá OHLCV của cổ phiếu VN (HOSE/HNX). Trả về historical price data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Mã cổ phiếu VN (VCB, TCB, HPG...)"},
                "start_date": {"type": "string", "description": "Ngày bắt đầu YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "Ngày kết thúc YYYY-MM-DD"},
                "source": {"type": "string", "default": "VCI"},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_fundamental",
        "description": "Lấy chỉ số tài chính cơ bản: PE, PB, ROE, EPS, Debt/Equity, Revenue Growth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Mã cổ phiếu VN"},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_income_statement",
        "description": "Lấy báo cáo kết quả kinh doanh (BCTC): doanh thu, lợi nhuận, biên lợi nhuận.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string", "enum": ["year", "quarter"], "default": "year"},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_balance_sheet",
        "description": "Lấy bảng cân đối kế toán: tài sản, nợ, vốn chủ sở hữu.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string", "enum": ["year", "quarter"], "default": "year"},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_cash_flow",
        "description": "Lấy báo cáo lưu chuyển tiền tệ: CFO, CFI, CFF.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string", "enum": ["year", "quarter"], "default": "year"},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_foreign_flow",
        "description": "Lấy dữ liệu giao dịch khối ngoại (mua/bán ròng theo ngày).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days": {"type": "integer", "default": 30},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_company_news",
        "description": "Lấy tin tức mới nhất của công ty từ VCI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_company_info",
        "description": "Lấy thông tin tổng quan công ty: ngành, mô tả, ban lãnh đạo.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_market_overview",
        "description": "Lấy tổng quan thị trường VN: VNIndex, VN30, top gainers/losers, foreign flow.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "screen_stocks",
        "description": "Lọc cổ phiếu theo tiêu chí: PE, PB, ROE, market cap, volume...",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_roe": {"type": "number", "description": "ROE tối thiểu (%)"},
                "max_pe":  {"type": "number", "description": "PE tối đa"},
                "min_volume": {"type": "number", "description": "Volume trung bình tối thiểu"},
                "sector":  {"type": "string", "description": "Ngành (Ngân hàng, Bất động sản...)"},
            }
        }
    },
    {
        "name": "get_sector_data",
        "description": "Lấy danh sách cổ phiếu theo ngành ICB, thống kê ngành.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sector_name": {"type": "string", "description": "Tên ngành ICB"},
            }
        }
    },
    {
        "name": "get_macro_data",
        "description": "Lấy dữ liệu vĩ mô VN: CPI, GDP, lãi suất, tỷ giá USD/VND.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "indicators": {"type": "array", "items": {"type": "string"},
                               "description": "CPI, GDP, interest_rate, exchange_rate, FDI"},
            }
        }
    },
    {
        "name": "get_derivatives",
        "description": "Lấy dữ liệu hợp đồng tương lai VN30F: giá, open interest, basis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "default": "VN30F1M"},
            }
        }
    },
]


# ── Tool executor ───────────────────────────────────────────────────────

class VNStockMCPHandler:
    """Thực thi MCP tool calls bằng vnstock_data."""

    def __init__(self, source='VCI', start_date=None):
        self.source     = source
        self.start_date = start_date or (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        self._fetcher   = None

    def _get_fetcher(self, start_date=None):
        sd = start_date or self.start_date
        if self._fetcher is None or sd != self.start_date:
            try:
                self._fetcher = DataFetcher(source=self.source, start_date=sd)
            except Exception as e:
                logger.error(f"DataFetcher init error: {e}")
                return None
        return self._fetcher

    def execute(self, tool_name: str, args: dict) -> str:
        """Thực thi tool và trả về JSON string."""
        f = self._get_fetcher(args.get('start_date'))
        if not f:
            return json.dumps({"error": "vnstock_data không khởi tạo được"})

        try:
            if tool_name == "get_stock_price":
                days = args.get('days', 365)
                sd   = args.get('start_date', (datetime.today()-timedelta(days=days)).strftime('%Y-%m-%d'))
                df   = DataFetcher(source=args.get('source','VCI'), start_date=sd).get_ohlcv(args['symbol'])
                return df.tail(60).to_json(orient='records', date_format='iso')

            elif tool_name == "get_fundamental":
                return json.dumps(f.get_fundamental(args['symbol']))

            elif tool_name == "get_income_statement":
                df = f.get_income_statement(args['symbol'], period=args.get('period','year'))
                return df.tail(8).to_json(orient='records') if df is not None and not df.empty else "{}"

            elif tool_name == "get_balance_sheet":
                df = f.get_balance_sheet(args['symbol'], period=args.get('period','year'))
                return df.tail(8).to_json(orient='records') if df is not None and not df.empty else "{}"

            elif tool_name == "get_cash_flow":
                df = f.get_cash_flow(args['symbol'], period=args.get('period','year'))
                return df.tail(8).to_json(orient='records') if df is not None and not df.empty else "{}"

            elif tool_name == "get_foreign_flow":
                days = args.get('days', 30)
                sd   = (datetime.today()-timedelta(days=days)).strftime('%Y-%m-%d')
                df   = DataFetcher(source=self.source, start_date=sd).get_foreign_flow(args['symbol'])
                return df.tail(30).to_json(orient='records', date_format='iso') if df is not None and not df.empty else "{}"

            elif tool_name == "get_company_news":
                df = f.get_company_news(args['symbol'])
                limit = args.get('limit', 10)
                return df.head(limit).to_json(orient='records') if df is not None and not df.empty else "[]"

            elif tool_name == "get_company_info":
                df = f.get_company_info(args['symbol'])
                return df.to_json(orient='records') if df is not None and not df.empty else "{}"

            elif tool_name == "get_market_overview":
                ov = f.get_market_overview()
                result = {}
                for k, v in ov.items():
                    try:
                        import pandas as pd
                        result[k] = v.head(10).to_dict() if isinstance(v, pd.DataFrame) and not v.empty else str(v)
                    except: result[k] = str(v)
                return json.dumps(result, ensure_ascii=False, default=str)

            elif tool_name == "screen_stocks":
                df = f.get_screener(**{k:v for k,v in args.items()})
                return df.head(20).to_json(orient='records') if df is not None and not df.empty else "[]"

            elif tool_name == "get_sector_data":
                sectors = f.get_symbols_by_sector()
                sname   = args.get('sector_name', '')
                if sname and sname in sectors:
                    return json.dumps({sname: sectors[sname][:30]}, ensure_ascii=False)
                return json.dumps({k: v[:10] for k,v in list(sectors.items())[:5]}, ensure_ascii=False)

            elif tool_name == "get_macro_data":
                df = f.get_macro_data()
                return df.tail(24).to_json(orient='records', date_format='iso') if df is not None and not df.empty else "{}"

            elif tool_name == "get_derivatives":
                sym = args.get('symbol', 'VN30F1M')
                df  = f.get_futures_ohlcv(sym)
                return df.tail(30).to_json(orient='records', date_format='iso') if df is not None and not df.empty else "{}"

            else:
                return json.dumps({"error": f"Tool '{tool_name}' không tồn tại"})

        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name})


# ── MCP Server (stdio) ──────────────────────────────────────────────────

async def run_mcp_server():
    """Chạy MCP server qua stdio transport (Claude Code / Claude Desktop)."""
    if not MCP_AVAILABLE:
        print("Cài mcp SDK: pip install mcp", file=sys.stderr)
        return

    server  = Server("vnstock-vn")
    handler = VNStockMCPHandler()

    @server.list_tools()
    async def list_tools():
        return [types.Tool(**t) for t in VN_TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = handler.execute(name, arguments or {})
        return [types.TextContent(type="text", text=result)]

    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


# ── Anthropic API tools format (for direct API calls) ──────────────────

def get_anthropic_tools() -> list:
    """Convert VN_TOOLS → Anthropic tool_use format."""
    tools = []
    for t in VN_TOOLS:
        tools.append({
            "name":         t["name"],
            "description":  t["description"],
            "input_schema": t["inputSchema"],
        })
    return tools


def process_tool_call(tool_name: str, tool_input: dict,
                      source='VCI', start_date=None) -> str:
    """Gọi tool và trả về kết quả."""
    handler = VNStockMCPHandler(source=source, start_date=start_date)
    return handler.execute(tool_name, tool_input)


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
