"""Small API client: bounded timeout, no retry, sanitized errors."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict, key: str, *, timeout: int, allow_external: bool) -> dict:
    endpoint = urlsplit(url)
    local = endpoint.hostname in ('localhost', '127.0.0.1', '::1')
    if endpoint.scheme != 'https' and not (local and endpoint.scheme == 'http'):
        raise ValueError('API endpoint must use HTTPS, except local model endpoints')
    if not local and not allow_external:
        raise RuntimeError('external processing is disabled; check dataset permission and local settings')
    if not key and not local:
        raise RuntimeError('API key is missing in local environment')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'provider HTTP {exc.code}; no automatic retry') from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError('provider network/timeout error; no automatic retry') from None
    except (ValueError, UnicodeError):
        raise RuntimeError('provider returned invalid JSON') from None
    if not isinstance(data, dict) or data.get('error'):
        raise RuntimeError('provider returned an API error or invalid object')
    return data
