"""
ai_agents.py
Multi-provider AI Agent wrapper:
  openai, deepseek, anthropic, groq, qwen, openrouter
✅ Đã cập nhật: Claude 4.5/4.6 + Credit Fallback + Beta Context Support
"""
import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

# ── Provider config ─────────────────────────────────────────────────────
PROVIDER_CONFIG = {
    'openai': {
        'env_key':  'OPENAI_API_KEY',
        'base_url': None,
        'sdk':      'openai',
    },
    'deepseek': {
        'env_key':  'DEEPSEEK_API_KEY',
        'base_url': 'https://api.deepseek.com',
        'sdk':      'openai',
    },
    'anthropic': {
        'env_key':  'ANTHROPIC_API_KEY',
        'base_url': None,
        'sdk':      'anthropic',
        # Beta header cho context window lớn (1M tokens)
        'beta_header': 'max-tokens-3-5-sonnet-2024-07-15',
    },
    'groq': {
        'env_key':  'GROQ_API_KEY',
        'base_url': None,
        'sdk':      'groq',
    },
    'qwen': {
        'env_key':  'DASHSCOPE_API_KEY',
        'base_url': None,
        'sdk':      'qwen',
    },
    'openrouter': {
        'env_key':  'OPENROUTER_API_KEY',
        'base_url': 'https://openrouter.ai/api/v1',
        'sdk':      'openai',
    },
        'grok': {
        'env_key':  'XAI_API_KEY',
        'base_url': 'https://api.x.ai',
        'sdk':      'openai',
    },

}

# ── Model alias → API ID mapping (Anthropic) ───────────────────────────
ANTHROPIC_API_MAP = {
    # Model alias trong config → API ID thực tế
    "claude-opus-4.7":  "claude-3-opus-20240229",      # Fallback sang Opus 3
    "claude-sonnet-4-5": "claude-3-5-sonnet-20241022",
    "claude-haiku-4-5":  "claude-3-5-haiku-20241022",
}

# ── Model catalogue ─────────────────────────────────────────────────────
MODEL_CATALOGUE = {
    # OpenAI
    'gpt-4o':             {'provider':'openai',     'ctx':128_000, 'tier':'premium'},
    'gpt-4o-mini':        {'provider':'openai',     'ctx':128_000, 'tier':'standard'},
    'o1-mini':            {'provider':'openai',     'ctx':128_000, 'tier':'reasoning'},

    # DeepSeek
    'deepseek-chat':      {'provider':'deepseek',   'ctx':64_000,  'tier':'standard'},
    'deepseek-reasoner':  {'provider':'deepseek',   'ctx':64_000,  'tier':'reasoning'},

    # Anthropic ✅ ĐÃ CẬP NHẬT
    'claude-opus-4.7': {'provider': 'anthropic', 'ctx': 200_000, 'tier': 'premium'},
    'claude-sonnet-4-5':  {'provider':'anthropic',  'ctx':200_000, 'tier':'premium'},
    'claude-haiku-4-5':   {'provider':'anthropic',  'ctx':200_000, 'tier':'standard'},
    # Legacy models (vẫn hỗ trợ)
    'claude-3-opus-20240229':   {'provider':'anthropic', 'ctx':200_000, 'tier':'premium'},
    'claude-3-5-sonnet-20241022': {'provider':'anthropic', 'ctx':200_000, 'tier':'premium'},
    'claude-3-5-haiku-20241022':  {'provider':'anthropic', 'ctx':200_000, 'tier':'standard'},

    # Groq (cực nhanh)
    'llama-3.3-70b-versatile':  {'provider':'groq', 'ctx':128_000, 'tier':'standard'},
    'llama-3.1-8b-instant':     {'provider':'groq', 'ctx':128_000, 'tier':'fast'},
    'mixtral-8x7b-32768':       {'provider':'groq', 'ctx':32_000,  'tier':'standard'},
    'gemma2-9b-it':             {'provider':'groq', 'ctx':8_192,   'tier':'fast'},
    'deepseek-r1-distill-qwen-32b': {'provider':'groq','ctx':128_000,'tier':'reasoning'},
    'deepseek-r1-distill-qwen-14b': {'provider':'groq','ctx':128_000,'tier':'reasoning'},
    'qwen-qwq-32b':             {'provider':'groq', 'ctx':32_000,  'tier':'reasoning'},

    # Qwen / DashScope
    'qwen-max':           {'provider':'qwen',       'ctx':32_000,  'tier':'premium'},
    'qwen-plus':          {'provider':'qwen',       'ctx':131_072, 'tier':'standard'},
    'qwen-turbo':         {'provider':'qwen',       'ctx':1_000_000,'tier':'fast'},

    # OpenRouter (truy cập nhiều model)
    'openrouter/google/gemini-2.5-flash-preview':
                          {'provider':'openrouter', 'ctx':1_000_000,'tier':'premium'},
    'openrouter/google/gemini-2.0-flash-001':
                          {'provider':'openrouter', 'ctx':1_000_000,'tier':'standard'},
    'openrouter/meta-llama/llama-4-maverick':
                          {'provider':'openrouter', 'ctx':128_000, 'tier':'standard'},
    'openrouter/mistralai/mistral-large-2411':
                          {'provider':'openrouter', 'ctx':128_000, 'tier':'standard'},
    'openrouter/x-ai/grok-3-mini-beta':
                          {'provider':'openrouter', 'ctx':131_072, 'tier':'standard'},
    'openrouter/deepseek/deepseek-r1':
                          {'provider':'openrouter', 'ctx':64_000,  'tier':'reasoning'},
    'openrouter/anthropic/claude-sonnet-4-5':
                          {'provider':'openrouter', 'ctx':200_000, 'tier':'premium'},
    
}
MODEL_CATALOGUE.update({
    'gpt-5.5':                  {'provider':'openai', 'ctx':1_050_000, 'tier':'premium'},
    'claude-opus-4.7':          {'provider':'anthropic','ctx':1_000_000,'tier':'premium'},
    'gemini-3.1':               {'provider':'openrouter','ctx':1_000_000,'tier':'premium'},
    'grok-4.3':                 {'provider':'grok',     'ctx':1_000_000,'tier':'premium'},
    'deepseek-v4':              {'provider':'deepseek', 'ctx':1_000_000,'tier':'reasoning'},
    'qwen-3.6-plus':            {'provider':'qwen',     'ctx':1_000_000,'tier':'premium'},
    'kimi-2.6':                 {'provider':'groq',     'ctx':256_000,  'tier':'standard'},
})

# Gợi ý preset agent
AGENT_PRESETS = {
    'Fast (free tier)': [
        ('Groq-Llama', 'groq', 'llama-3.3-70b-versatile'),
        ('Groq-Mixtral','groq','mixtral-8x7b-32768'),
        ('Groq-DeepSeekR1','groq','deepseek-r1-distill-llama-70b'),
    ],
    'Balanced': [
        ('DeepSeek',   'deepseek',    'deepseek-chat'),
        ('ChatGPT',    'openai',      'gpt-4o-mini'),
        ('Claude',     'anthropic',   'claude-sonnet-4-5'),
        ('Grok-Llama', 'groq',        'llama-3.3-70b-versatile'),
        ('Qwen',       'qwen',        'qwen-max'),
    ],
    'Premium': [
        ('GPT-4o',       'openai',      'gpt-4o'),
        ('Claude-Sonnet','anthropic',   'claude-sonnet-4-5'),
        ('DeepSeek-R1',  'deepseek',    'deepseek-reasoner'),
        ('Gemini-Flash', 'openrouter',  'openrouter/google/gemini-2.5-flash-preview'),
        ('Grok-3-Mini',  'openrouter',  'openrouter/x-ai/grok-3-mini-beta'),
        ('Qwen-Max',     'qwen',        'qwen-max'),
    ],
    'OpenRouter Only': [
        ('Gemini-2.5',  'openrouter', 'openrouter/google/gemini-2.5-flash-preview'),
        ('Gemini-2.0',  'openrouter', 'openrouter/google/gemini-2.0-flash-001'),
        ('Llama-4',     'openrouter', 'openrouter/meta-llama/llama-4-maverick'),
        ('Mistral-L',   'openrouter', 'openrouter/mistralai/mistral-large-2411'),
        ('Grok-3-Mini', 'openrouter', 'openrouter/x-ai/grok-3-mini-beta'),
        ('DSR1',        'openrouter', 'openrouter/deepseek/deepseek-r1'),
    ],
}


# ════════════════════════════════════════════════════════════════════════
class AIAgent:
    """
    Unified LLM agent: openai / deepseek / anthropic / groq / qwen / openrouter
    ✅ Hỗ trợ: Credit fallback, Beta context header, Auto-retry
    """
    def __init__(self, provider: str, model: str, max_retries: int = 2,
                 temperature: float = 0.3, max_tokens: int = 1000,
                 fallback_model: str = None):
        self.provider       = provider
        self.model          = model
        self.max_retries    = max_retries
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.fallback_model = fallback_model  # Model dự phòng nếu hết credit
        self._client        = None
        self._init_client()

    def _init_client(self):
        cfg = PROVIDER_CONFIG.get(self.provider)
        if not cfg:
            raise ValueError(f"Provider '{self.provider}' chưa hỗ trợ.")

        api_key = os.getenv(cfg['env_key'], '')
        if not api_key:
            print(f"[AIAgent] WARNING: {cfg['env_key']} chưa set trong .env")

        sdk = cfg['sdk']

        if sdk == 'openai':
            from openai import OpenAI
            kwargs = {'api_key': api_key}
            if cfg.get('base_url'):
                kwargs['base_url'] = cfg['base_url']
            if self.provider == 'openrouter':
                kwargs['default_headers'] = {
                    'HTTP-Referer': 'https://superaiquant.vn',
                    'X-Title':      'Super AI Quant VN',
                }
            self._client = OpenAI(**kwargs)

        elif sdk == 'anthropic':
            import anthropic
            # Chuẩn bị headers cho beta context (nếu cần)
            headers = {}
            if cfg.get('beta_header'):
                headers['anthropic-beta'] = cfg['beta_header']
            self._client = anthropic.Anthropic(api_key=api_key, default_headers=headers)

        elif sdk == 'groq':
            from groq import Groq
            self._client = Groq(api_key=api_key)

        elif sdk == 'qwen':
            import dashscope
            dashscope.api_key = api_key
            self._client = None

        elif sdk == 'openai' and self.provider == 'grok':
            try:
                from openai import OpenAI
                kwargs = {'api_key': api_key}
                if cfg.get('base_url'):
                    kwargs['base_url'] = cfg['base_url']
                self._client = OpenAI(**kwargs)
            except Exception:
                try:
                    from xai_sdk import Client as XAIClient
                    self._client = XAIClient(api_key=api_key)
                except Exception:
                    self._client = None


    def _get_api_model_id(self) -> str:
        """Map alias model sang API ID thực tế (đặc biệt cho Anthropic)."""
        if self.provider == 'anthropic' and self.model in ANTHROPIC_API_MAP:
            return ANTHROPIC_API_MAP[self.model]
        # OpenRouter: strip prefix
        if self.provider == 'openrouter' and self.model.startswith('openrouter/'):
            return self.model[len('openrouter/'):]
        return self.model

    # ── Public ──────────────────────────────────────────────────────
    def ask(self, system_prompt: str, user_prompt: str) -> str | None:
        for attempt in range(self.max_retries + 1):
            try:
                return self._call(system_prompt, user_prompt)
            except Exception as e:
                err_msg = str(e).lower()
                # ✅ Xử lý lỗi hết credit → fallback sang model khác
                if 'credit' in err_msg or 'quota' in err_msg or 'insufficient' in err_msg:
                    if self.fallback_model:
                        print(f"[AIAgent] ⚠️ Hết credit {self.model}, fallback sang {self.fallback_model}")
                        self.model = self.fallback_model
                        self._init_client()  # Re-init client với model mới
                        continue
                # Retry logic thông thường
                wait = 2 ** attempt
                print(f"[{self.provider}/{self.model}] Lần {attempt+1} lỗi: {e} | retry sau {wait}s")
                if attempt < self.max_retries:
                    time.sleep(wait)
        return None

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        sdk = PROVIDER_CONFIG[self.provider]['sdk']
        model_id = self._get_api_model_id()

        if sdk in ('openai', 'groq'):
            resp = self._client.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content

        elif sdk == 'anthropic':
            # ✅ Anthropic: max_tokens mặc định 4096, có thể tăng lên 8192/16384
            resp = self._client.messages.create(
                model=model_id,
                max_tokens=min(self.max_tokens, 16384),  # Giới hạn an toàn
                system=system_prompt,
                messages=[{'role': 'user', 'content': user_prompt}],
            )
            return resp.content[0].text

        elif sdk == 'qwen':
            from dashscope import Generation
            resp = Generation.call(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                result_format='message',
            )
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            raise RuntimeError(f"Qwen API lỗi: {resp.code} – {resp.message}")

    @property
    def display_name(self):
        return f"{self.provider}/{self.model.split('/')[-1]}"


# ════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════

def extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    try:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass
    return None


def meta_ensemble(decisions: dict) -> tuple[str, float]:
    """
    Weighted voting từ nhiều agent.
    Returns: (action, score) với score ∈ [-1, 1]
    """
    weight_map = {'BUY': 1.0, 'STRONG_BUY': 1.0, 'ACCUMULATION': 0.5,
                  'HOLD': 0.0, 'REDUCE': -0.5, 'SELL': -1.0}
    total_score = 0.0
    total_weight = 0.0

    for name, dec in decisions.items():
        action = str(dec.get('action', 'HOLD')).upper()
        try:
            confidence = float(dec.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.5
        total_score  += weight_map.get(action, 0.0) * confidence
        total_weight += confidence

    if total_weight == 0:
        return 'HOLD', 0.0

    avg = total_score / total_weight

    if avg > 0.6:   action = 'STRONG_BUY'
    elif avg > 0.3: action = 'BUY'
    elif avg > 0.1: action = 'ACCUMULATION'
    elif avg > -0.3: action = 'HOLD'
    elif avg > -0.6: action = 'REDUCE'
    else:            action = 'SELL'

    return action, round(avg, 4)


def build_agents_from_config(agent_list: list[dict]) -> dict:
    """
    Khởi tạo agents từ list config.
    agent_list: [{'name': 'MyAgent', 'provider': 'groq', 'model': 'llama-3.3-70b-versatile'}, ...]
    ✅ Hỗ trợ fallback_model để tự động chuyển khi hết credit
    """
    agents = {}
    for cfg in agent_list:
        name     = cfg.get('name', cfg.get('model', 'unknown'))
        provider = cfg.get('provider')
        model    = cfg.get('model')
        if not provider or not model:
            continue
        try:
            agents[name] = AIAgent(
                provider=provider,
                model=model,
                temperature=cfg.get('temperature', 0.3),
                max_tokens=cfg.get('max_tokens', 1000),
                fallback_model=cfg.get('fallback_model'),  # ✅ Mới
            )
            print(f"[AIAgent] {name} ({provider}/{model}) ✓")
        except Exception as e:
            print(f"[AIAgent] {name} lỗi: {e}")
    return agents

# Failover / circuit-breaker state & helpers
PROVIDER_UNAVAILABLE: dict = {}

def get_fallback_list(provider: str, cfg: dict) -> list:
    try:
        fb = cfg.get('ai_agents', {}).get('fallback', {})
        explicit = fb.get(provider, [])
        if explicit:
            return explicit
    except Exception:
        explicit = []

    try:
        tuning = cfg.get('ai_agents', {}).get('tuning', {}) if isinstance(cfg, dict) else {}
        provider_priority = tuning.get('provider_priority', [])
        allow_same = tuning.get('allow_same_provider_fallback', False)

        candidates = []
        for model_name, meta in MODEL_CATALOGUE.items():
            p = meta.get('provider')
            candidates.append({
                'provider': p,
                'model': model_name,
                'tier': meta.get('tier', 'standard'),
                'ctx': meta.get('ctx', 0)
            })

        tier_rank = {'premium': 0, 'reasoning': 1, 'standard': 2, 'fast': 3}
        def rank_key(c):
            try:
                pr = provider_priority.index(c['provider']) if c['provider'] in provider_priority else len(provider_priority)
            except Exception:
                pr = len(provider_priority)
            return (pr, tier_rank.get(c.get('tier','standard'), 2), -int(c.get('ctx',0)))

        candidates.sort(key=rank_key)

        fallback_list = []
        for c in candidates:
            if not allow_same and c['provider'] == provider:
                continue
            fallback_list.append({'provider': c['provider'], 'model': c['model']})

        if not fallback_list and not allow_same:
            for c in candidates:
                if c['provider'] == provider:
                    fallback_list.append({'provider': c['provider'], 'model': c['model']})

        return fallback_list
    except Exception:
        return explicit

def estimate_request_cost(provider: str, cfg: dict, tokens: int) -> float:
    try:
        cost_map = cfg.get('cost_per_1k_tokens', {}) if isinstance(cfg, dict) else {}
        unit = float(cost_map.get(provider, 0.0))
        return (tokens / 1000.0) * unit
    except Exception:
        return 0.0

def _is_sensitive_request(system_prompt: str, user_prompt: str) -> bool:
    s = (system_prompt or "") + " " + (user_prompt or "")
    keywords = ['personal', 'ssn', 'social security', 'bank account', 'investment committee', 'legal', 'medical']
    return any(k in s.lower() for k in keywords)

def _attempt_failover(self, system_prompt: str, user_prompt: str, cfg: dict) -> str | None:
    fb_list = get_fallback_list(self.provider, cfg)
    if not fb_list:
        print(f"[Failover] No fallback configured for provider {self.provider}")
        return None

    tuning = cfg.get('ai_agents', {}).get('tuning', {}) if isinstance(cfg, dict) else {}
    max_attempts = int(tuning.get('max_failover_attempts', 3))
    cb_seconds = int(tuning.get('circuit_breaker_seconds', 300))
    max_cost = float(tuning.get('max_fallback_cost_usd', 0.5))
    log_events = bool(tuning.get('log_failover_events', True))
    sensitive_policy = tuning.get('sensitive_request_policy', 'relaxed')

    attempts = 0
    for fb in fb_list:
        if attempts >= max_attempts:
            break
        fb_provider = fb.get('provider')
        fb_model = fb.get('model')

        est_cost = estimate_request_cost(fb_provider, cfg, max(256, int(self.max_tokens)))

        if max_cost and est_cost > max_cost:
            if log_events:
                print(f"[Failover] skipping {fb_provider}/{fb_model} due to est_cost {est_cost:.4f} > max {max_cost}")
            continue

        if sensitive_policy == 'strict' and _is_sensitive_request(system_prompt, user_prompt):
            tier = MODEL_CATALOGUE.get(fb_model, {}).get('tier', 'standard')
            if tier != 'premium':
                if log_events:
                    print(f"[Failover] skipping {fb_provider}/{fb_model} for sensitive request (tier={tier})")
                continue

        try:
            if log_events:
                print(f"[Failover] trying {fb_provider}/{fb_model} (est_cost={est_cost:.4f})")
            alt = AIAgent(provider=fb_provider, model=fb_model,
                          temperature=self.temperature, max_tokens=self.max_tokens)
            res = alt._call(system_prompt, user_prompt)
            if log_events:
                print(f"[Failover] success with {fb_provider}/{fb_model}")
            return res
        except Exception as e:
            attempts += 1
            msg = str(e).lower()
            if log_events:
                print(f"[Failover] {fb_provider}/{fb_model} failed: {e}")
            if any(k in msg for k in ('credit', 'quota', '401', '403', 'insufficient')):
                PROVIDER_UNAVAILABLE[fb_provider] = time.time() + cb_seconds
            continue

    if log_events:
        print("[Failover] all fallbacks exhausted or skipped")
    return None

setattr(AIAgent, "_attempt_failover", _attempt_failover)
