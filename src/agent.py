"""Bounded real-model loop. Offline doubles are used only by tests."""
from __future__ import annotations

import json
import re
import time

from .context import pack_context
from .contracts import nonempty, runtime_query
from .controller import SYSTEM_PROMPT


def run_agent(query: dict, controller, retriever, config: dict) -> dict:
    runtime = runtime_query(query)
    events, searches, model_calls = [], [], []
    context = pack_context([], config['budget_chars'])
    last_search_message, last_call = None, None
    started = time.perf_counter()

    def finish(status, answer='', citations=None, stop_reason='', error=None):
        return {'status': status, 'answer': answer, 'citations': citations or [],
                'original_question': runtime['question'], 'filter': runtime['filters'],
                'events': events, 'context': context, 'model_calls': model_calls,
                'search_count': len(searches), 'stop_reason': stop_reason, 'error': error,
                'elapsed_ms': (time.perf_counter() - started) * 1000}

    try:
        for _ in range(config['max_searches'] + 1):
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': runtime['question']}]
            if last_search_message is not None:
                # Only the current cumulative context is sent; prior full tool
                # contents are not repeated outside the configured budget.
                messages.extend([last_search_message, {'role': 'tool', 'tool_call_id': last_call['id'],
                    'content': json.dumps({'queries_performed': [s['query'] for s in searches],
                                           'context': context['text'], 'context_ids': context['ids'],
                                           'budget_chars': config['budget_chars']}, ensure_ascii=False)}])
            response = controller.decide(messages, first=not searches,
                                         can_search=len(searches) < config['max_searches'])
            model_calls.append({**response, 'messages': messages})
            message = response['message']
            calls = message.get('tool_calls')
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError('model must emit exactly one tool call per turn')
            call = calls[0]
            if call.get('type') != 'function' or not isinstance(call.get('id'), str):
                raise ValueError('invalid tool call identity')
            function = call.get('function', {})
            args = json.loads(function.get('arguments', ''))
            if not isinstance(args, dict):
                raise ValueError('tool arguments must be an object')
            name = function.get('name')
            if name == 'search_corpus':
                if len(searches) >= config['max_searches']:
                    raise ValueError('model exceeded configured search limit')
                if set(args) != {'query'}:
                    raise ValueError('search tool only accepts query; model cannot infer filters from labels')
                text = nonempty(args['query'], 'tool query')
                # Save attempt before invoking an external dependency, including failures.
                event = {'event': 'search', 'tool': name, 'arguments': args, 'query': text,
                         'filter': runtime['filters'], 'status': 'started'}
                events.append(event)
                result = retriever.search(text, filters=runtime['filters'])
                searches.append(result)
                context = pack_context(searches, config['budget_chars'])
                event.update(result)
                event.update({'status': 'success', 'context': context['text'], 'context_ids': context['ids']})
                # Preserve provider-specific reasoning details required for tool signatures.
                last_search_message = {k: v for k, v in message.items()
                                       if k in ('role', 'content', 'tool_calls', 'reasoning_details')}
                last_search_message['content'] = None
                last_call = call
            elif name in ('answer_with_citations', 'abstain'):
                if not searches:
                    raise ValueError('model attempted to finish without retrieval')
                if name == 'abstain':
                    if set(args) != {'reason'}:
                        raise ValueError('invalid abstain arguments')
                    nonempty(args['reason'], 'abstention reason')
                    events.append({'event': 'abstain', 'tool': name, 'reason': args['reason']})
                    return finish('abstain', 'Không đủ bằng chứng trong corpus để trả lời câu hỏi này.',
                                  stop_reason='model_abstained')
                if set(args) != {'answer', 'citations'}:
                    raise ValueError('invalid answer arguments')
                answer = nonempty(args['answer'], 'answer')
                citations = args['citations']
                bracket_ids = re.findall(r'\[([A-Za-z0-9:_-]+)\]', answer)
                if (not isinstance(citations, list) or not citations
                        or any(not isinstance(cid, str) for cid in citations)
                        or len(set(citations)) != len(citations)
                        or set(bracket_ids) != set(citations)
                        or any(cid not in context['ids'] or f'[{cid}]' not in answer for cid in citations)):
                    events.append({'event': 'invalid_citations', 'attempted_answer': args})
                    return finish('abstain', 'Không đủ bằng chứng có citation hợp lệ để trả lời.',
                                  stop_reason='invalid_citations')
                events.append({'event': 'answer', 'tool': name})
                return finish('answered', answer, citations, stop_reason='model_answered')
            else:
                raise ValueError('model requested an unknown tool')
        return finish('error', stop_reason='loop_limit', error={'type': 'LoopLimit', 'message': 'no final action'})
    except Exception as exc:
        # Never turn transport/DB/parser errors into successful abstention.
        if events and events[-1].get('status') == 'started':
            events[-1]['status'] = 'error'
        return finish('error', stop_reason='dependency_or_protocol_error',
                      error={'type': type(exc).__name__, 'message': str(exc)})
