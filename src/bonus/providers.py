"""Explicit bonus provider selection; no credential or model fallback inside runs."""
from __future__ import annotations

import copy
import os
from dataclasses import replace

from ..config import Settings

ENDPOINTS={'openrouter':'https://openrouter.ai/api/v1',
           'openai':'https://api.openai.com/v1',
           'gemini':'https://generativelanguage.googleapis.com/v1beta/openai'}
KEY_NAMES={'openrouter':'OPENROUTER_API_KEY','openai':'OPENAI_API_KEY','gemini':'GEMINI_API_KEY'}


def settings_for(config: dict) -> Settings:
    base=Settings.from_env()
    provider=config.get('provider','openrouter')
    if provider not in ENDPOINTS:
        raise ValueError('unsupported bonus provider')
    endpoint=config.get('base_url',ENDPOINTS[provider]).rstrip('/')
    if endpoint!=ENDPOINTS[provider]:
        raise ValueError('provider endpoint differs from approved destination')
    model=config.get('model',base.chat_model if provider=='openrouter' else '')
    if not model:
        raise ValueError('explicit model is required for direct provider')
    key=base.chat_key if provider=='openrouter' else os.getenv(KEY_NAMES[provider],'')
    if not key:
        raise RuntimeError('selected provider key is missing locally; no silent key fallback')
    return replace(base,chat_base_url=endpoint,chat_model=model,chat_key=key)


def adapt_payload(payload: dict, config: dict) -> dict:
    result=copy.deepcopy(payload)
    provider=config.get('provider','openrouter')
    if provider not in ENDPOINTS:
        raise ValueError('unsupported provider')
    if provider!='openrouter':
        result.pop('provider',None)
        result['parallel_tool_calls']=False
    if provider=='gemini':
        def compatible(value):
            if isinstance(value,list):
                return [compatible(v) for v in value]
            if isinstance(value,dict):
                return {k:compatible(v) for k,v in value.items() if k not in ('strict','minLength','minItems','maxItems')}
            return value
        result['tools']=compatible(result['tools'])
        result['reasoning_effort']=config.get('reasoning_effort','minimal')
    return result


def assistant_input(message: dict, config: dict) -> dict:
    provider=config.get('provider','openrouter')
    if provider=='openrouter':
        return message
    # tool_calls are copied intact: Gemini's per-call thought_signature must survive.
    keys=('role','content','tool_calls','refusal','extra_content') if provider=='gemini' else ('role','content','tool_calls','refusal')
    return {k:v for k,v in message.items() if k in keys}
