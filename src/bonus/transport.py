"""Bonus-only request pacing and safe provider error diagnostics; no retries."""
from __future__ import annotations

import json
import re
import threading
import time
from urllib.error import HTTPError


class RequestPacer:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self, interval: float) -> float:
        if not isinstance(interval,(int,float)) or not 0 <= interval <= 60:
            raise ValueError('request interval must be between 0 and 60 seconds')
        with self.lock:
            delay = max(0.0,self.next_start-self.clock())
            remaining = delay
            while remaining > 0:
                part = min(remaining,30.0)
                self.sleep(part)
                remaining -= part
            self.next_start = self.clock()+interval
            return delay

    def defer(self, seconds: float) -> None:
        if not isinstance(seconds,(int,float)) or not 0 <= seconds <= 120:
            raise ValueError('cooldown must be between 0 and 120 seconds')
        with self.lock:
            self.next_start=max(self.next_start,self.clock()+seconds)


PACER = RequestPacer()


def provider_error_details(exc: Exception, key: str) -> dict:
    """Shared HTTP adapter suppresses display of its HTTPError, but keeps context."""
    context = exc.__context__
    if not isinstance(context,HTTPError):
        return {}
    try:
        error = json.load(context).get('error',{})
    except (ValueError,UnicodeError,OSError,AttributeError):
        error = {}
    message = str(error.get('message',''))
    if key:
        message = message.replace(key,'[redacted key]')
    message = re.sub(r'https?://\S+','[provider link omitted]',message)
    message = re.sub(r'\b[0-9a-fA-F]{24,}\b','[identifier omitted]',message)
    source = error.get('metadata',{}).get('limit_source')
    if source is not None and not re.fullmatch(r'[A-Za-z0-9_/-]{1,100}',str(source)):
        source = '[invalid metadata omitted]'
    return {'http_status':context.code,'provider_message':message[:1200],
            'limit_source':source,'retry_after':context.headers.get('Retry-After')}
