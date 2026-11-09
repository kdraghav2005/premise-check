"""One function, `complete`, is the only place a model is called. Swap the model by editing
config.json (or the file named by $PITCHCHECK_CONFIG). Standard library only.

Providers:
  gemini         Google Generative Language REST API (free tier available)
  openai_compat  any OpenAI-style /chat/completions endpoint (Groq, OpenRouter, ...)
  mock, mock_faulty   offline stand-ins for testing the plumbing (see mock.py)

Failures are typed so callers can tell them apart:
  ProviderError     the provider could not be used (no key, HTTP error, timeout, network, bad reply shape)
  ModelOutputError  the provider answered, but the text was not the JSON we asked for
There is never a silent fallback to the mock provider.
"""
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request

from .mock import mock_complete

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRY_SLEEP = (2, 4)  # seconds before retry 1 and 2 (tests set this to zeros)


class LLMError(RuntimeError):
    pass


class ProviderError(LLMError):
    def __init__(self, kind, message, hint=""):
        super().__init__(message)
        self.kind, self.message, self.hint = kind, message, hint

    def to_dict(self):
        return {"kind": self.kind, "message": self.message, "hint": self.hint}


class ModelOutputError(LLMError):
    pass


def load_config(path=None):
    path = path or os.environ.get("PITCHCHECK_CONFIG") or os.path.join(ROOT, "config.json")
    with open(path, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    cfg["_path"] = path
    return cfg


def is_mock(cfg):
    return str(cfg.get("provider", "")).startswith("mock")


def public_config(cfg):
    """The part of the configuration that is safe to store with a run (never any key)."""
    out = {k: cfg.get(k) for k in ("provider", "model", "base_url", "temperature", "timeout_s")}
    out["retrieval"] = cfg.get("retrieval", {})
    return out


def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        pass
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except ValueError:
            pass
    raise ModelOutputError("the model's reply was not valid JSON")


def _hint_for_http(code, cfg):
    env = cfg.get("api_key_env", "PITCHCHECK_API_KEY")
    if code == 403:
        return f"HTTP 403 means the provider (or a firewall in front of it) refused the request. Possible causes: a blocked client signature, a wrong key in ${env}, a model not enabled for your account, or a network/VPN block. Test the same key with a plain browser or PowerShell request to tell these apart."
    if code == 401:
        return f"The provider rejected the credentials. Check the key in ${env}, that the provider/model in config.json match that key, and that the model is enabled for your account."
    if code == 404:
        return "Model or URL not found. Check 'model' and 'base_url' in config.json."
    if code == 429:
        return "Free-tier rate limit reached. Wait a minute and rerun; catalog cards already extracted are cached."
    if code == 400:
        return "The provider rejected the request. Check 'model' in config.json and the detail above."
    if code >= 500:
        return "The provider is temporarily unavailable. Retry in a few minutes."
    return "Check provider, model and base_url in config.json."


MAX_GEN_RETRIES = 2       # sampling failures the provider reports as HTTP 400 (reasoning used up the output budget)


def _is_generation_failure(code, detail):
    d = (detail or "").lower()
    return code == 400 and ("json_validate_failed" in d or "max completion tokens" in d or "failed_generation" in d)


MAX_RATE_RETRIES = 8      # rate-limit (429) waits per request
MAX_RATE_WAIT = 65.0      # never wait longer than this for one retry (seconds)
RATE_WAIT_SCALE = 1.0     # tests set this to 0


def _rate_wait_seconds(detail, retry_after, n):
    """Seconds to wait after HTTP 429: the provider's own hint if present, else a growing default."""
    import re
    wait = None
    try:
        if retry_after:
            wait = float(retry_after)
    except ValueError:
        wait = None
    if wait is None:
        m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)\s*s", detail or "")
        if m:
            wait = float(m.group(1) or 0) * 60 + float(m.group(2))
    if wait is None:
        wait = min(5.0 * n, 30.0)
    return min(wait + 1.0, MAX_RATE_WAIT) * RATE_WAIT_SCALE


USER_AGENT = "kuku-pitchcheck/2.0 (+local research tool)"


def _post(url, headers, body, cfg, secret=""):
    timeout = cfg.get("timeout_s", 60)
    data = json.dumps(body).encode("utf-8")
    # Python's default "Python-urllib" agent is blocked by Cloudflare in front of some providers (Groq: HTTP 403, error 1010).
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **headers}

    def scrub(s):
        return s.replace(secret, "<key>") if secret else s

    attempts = len(RETRY_SLEEP) + 1
    rate_waits = 0
    gen_retries = 0
    attempt = -1
    while attempt < attempts - 1:
        attempt += 1
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", "replace")
            try:
                return json.loads(raw)
            except ValueError:
                raise ProviderError("bad_response", f"the provider's reply was not JSON: {scrub(raw[:200])}", "Check base_url: it should be the API endpoint, not a web page.")
        except urllib.error.HTTPError as e:
            detail = scrub(e.read()[:300].decode("utf-8", "replace"))
            if e.code == 429 and rate_waits < MAX_RATE_RETRIES:
                # Free tiers limit tokens per minute. The provider says how long to wait: do that, then retry.
                rate_waits += 1
                attempt -= 1
                time.sleep(_rate_wait_seconds(detail, e.headers.get("Retry-After"), attempt + 1))
                continue
            if _is_generation_failure(e.code, detail) and gen_retries < MAX_GEN_RETRIES:
                # Seen live (Groq, gpt-oss-120b): "max completion tokens reached before generating a valid document".
                # The model's own reasoning used the output budget on that sample; the same request usually succeeds on a retry.
                gen_retries += 1
                attempt -= 1
                time.sleep(RETRY_SLEEP[0] if RETRY_SLEEP else 0)
                continue
            if e.code in (500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(RETRY_SLEEP[attempt])
                continue
            hint = _hint_for_http(e.code, cfg)
            if _is_generation_failure(e.code, detail):
                hint = ("The model ran out of output budget while writing its reply (it was retried automatically). "
                        "This is a provider-side failure, not a finding about the pitch. Click Compare again; it usually passes.")
            raise ProviderError(f"http_{e.code}", f"HTTP {e.code} from the provider: {detail}", hint)
        except (socket.timeout, TimeoutError):
            if attempt < attempts - 1:
                time.sleep(RETRY_SLEEP[attempt])
                continue
            raise ProviderError("timeout", f"no reply from the provider within {timeout} s (after {attempts} tries)", "Raise 'timeout_s' in config.json, or retry; free tiers can be slow.")
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                if attempt < attempts - 1:
                    time.sleep(RETRY_SLEEP[attempt])
                    continue
                raise ProviderError("timeout", f"no reply from the provider within {timeout} s", "Raise 'timeout_s' in config.json, or retry.")
            if attempt < attempts - 1:
                time.sleep(RETRY_SLEEP[attempt])
                continue
            raise ProviderError("network", f"could not reach the provider: {scrub(str(e.reason))}", "Check your internet connection, proxy settings and 'base_url' in config.json.")
        except (OSError, http.client.HTTPException) as e:
            if attempt < attempts - 1:
                time.sleep(RETRY_SLEEP[attempt])
                continue
            raise ProviderError("network", f"connection problem: {scrub(str(e))}", "Check your internet connection and retry.")
    raise ProviderError("network", "request failed")  # pragma: no cover


def complete(prompt, cfg, task=None, context=None):
    """Return the model's raw text for `prompt`. `task`/`context` are used only by mock providers."""
    prov = cfg.get("provider", "")
    if prov.startswith("mock"):
        return mock_complete(prov, task, context)

    env = cfg.get("api_key_env", "PITCHCHECK_API_KEY")
    key = os.environ.get(env, "").strip()
    if not key:
        raise ProviderError(
            "missing_key",
            f"no API key found in the environment variable {env}",
            f'In PowerShell run:  $env:{env} = "your-key"   then rerun in the same window. (macOS/Linux: export {env}=your-key)',
        )
    model = cfg.get("model")
    if not model:
        raise ProviderError("config", "config.json has no 'model'", "Set 'model' in config.json.")
    temp = cfg.get("temperature", 0)

    if prov == "gemini":
        base = cfg.get("base_url") or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base.rstrip('/')}/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "responseMimeType": "application/json"},
        }
        data = _post(url, headers, body, cfg, key)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            why = ""
            if isinstance(data, dict) and data.get("promptFeedback"):
                why = f" (promptFeedback: {str(data['promptFeedback'])[:120]})"
            raise ProviderError("bad_response", f"unexpected Gemini reply shape{why}: {str(data)[:160]}", "The request may have been blocked or the model name may be wrong. Check 'model' in config.json.")

    if prov == "openai_compat":
        base = cfg.get("base_url")
        if not base:
            raise ProviderError("config", "provider 'openai_compat' needs 'base_url' in config.json", "For Groq use https://api.groq.com/openai/v1")
        url = f"{base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "response_format": {"type": "json_object"},
        }
        data = _post(url, headers, body, cfg, key)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("bad_response", f"unexpected reply shape: {str(data)[:160]}", "Check 'model' and 'base_url' in config.json.")

    raise ProviderError("config", f"unknown provider {prov!r}", "Use 'gemini', 'openai_compat' or 'mock' in config.json.")
