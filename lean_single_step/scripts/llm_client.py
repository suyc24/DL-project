#!/usr/bin/env python3
"""
llm_client.py

Simplified OpenAI‑compatible client wrapper with logging, reasoning control,
and concurrent batch calling.

Features:
- Synchronous call_llm(...) with retries, logging, reasoning toggling.
- call_llm_batch(tasks, max_workers=5) for concurrent execution of multiple
  calls.  Each task is a dict of the same keyword arguments accepted by
  call_llm.  Results are returned in the same order as tasks.
- All calls log to the shared logger with a `call_label` to distinguish
  concurrent requests.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# .env loading (unchanged)
# ---------------------------------------------------------------------------
def _load_env_from_tree(filename: str = ".env", max_levels: int = 6) -> None:
    p = Path(__file__).resolve().parent
    for _ in range(max_levels):
        env_file = p / filename
        if env_file.is_file():
            _parse_dotenv(env_file)
            return
        p = p.parent


def _parse_dotenv(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_from_tree()


# ---------------------------------------------------------------------------
# logger setup (unchanged)
# ---------------------------------------------------------------------------
logger = logging.getLogger("llm_client")

if not logger.handlers:
    log_file = os.environ.get("LLM_LOG_FILE", "llm_calls.log")
    log_level = os.environ.get("LLM_LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, log_level)
    except AttributeError:
        level = logging.INFO

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

logger.info("LLM client module loaded")


def _truncate(s: str, max_len: int = 200) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


# ---------------------------------------------------------------------------
# OpenAI client (import)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Synchronous call (modified with call_label)
# ---------------------------------------------------------------------------
def call_llm(
    system_prompt: str = "",
    user_prompt: str = "",
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    mock: bool = False,
    retries: int = 2,
    backoff: float = 1.0,
    timeout: int = 120,
    reasoning: Optional[bool] = None,
    call_label: str = "",
) -> str:
    """Call OpenAI chat completions and return the text content.

    See the original docstring.  The new `call_label` parameter is added
    purely for logging purposes; it does not affect the API call.
    """
    t_start = time.time()
    label_info = f"[{call_label}] " if call_label else ""
    logger.info(
        "%sLLM call initiated | model=%s | temperature=%.2f | max_tokens=%d | timeout=%d | mock=%s | reasoning=%s",
        label_info, model or "default", temperature, max_tokens, timeout, mock, reasoning,
    )

    # -- mock mode -----------------------------------------------------------
    if mock or os.environ.get("LLM_MOCK") in ("1", "true", "True"):
        result = (
            "Chain (mock)\n"
            "[Step 1] Mock step 1.\n"
            "[Step 2] Mock step 2.\n"
            "[Final Answer] mock answer\n"
        )
        logger.info("%sMock response returned (length=%d) in %.2fs", label_info, len(result), time.time() - t_start)
        return result

    # -- prerequisites --------------------------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Place it in a .env file or export it.")
    if OpenAI is None:
        raise RuntimeError("OpenAI library not installed. Run: pip install openai")

    model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o"

    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"), timeout=timeout)

    if messages is not None:
        msgs = messages
        logger.info("%sUsing multi-turn conversation (%d messages)", label_info, len(messages))
    else:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.debug("%sSystem prompt (%d chars): %s", label_info, len(system_prompt), _truncate(system_prompt))
        logger.debug("%sUser prompt (%d chars): %s", label_info, len(user_prompt), _truncate(user_prompt))

    # -- reasoning mode control -----------------------------------------------
    extra_body = None
    if reasoning is not None:
        if reasoning:
            extra_body = {"thinking": {"type": "enabled"}}
            logger.info("%sReasoning mode explicitly enabled", label_info)
        else:
            extra_body = {"thinking": {"type": "disabled"}}
            logger.info("%sReasoning mode explicitly disabled", label_info)

    last_error = None
    for attempt in range(retries + 1):
        t_attempt = time.time()
        try:
            kwargs = dict(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**kwargs)

            content = None
            try:
                if response.choices:
                    content = response.choices[0].message.content
            except Exception:
                pass

            elapsed = time.time() - t_attempt

            # Check for reasoning_content when content is empty
            reasoning_content = None
            try:
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
            except Exception:
                pass

            if reasoning_content and not (content and content.strip()):
                logger.warning("%sContent empty but reasoning_content present (%d chars); retrying",
                               label_info, len(reasoning_content) if reasoning_content else 0)
                last_error = RuntimeError("Content empty; only reasoning_content returned")
            elif content and content.strip():
                logger.info(
                    "%sLLM call succeeded on attempt %d | response length=%d | elapsed=%.2fs",
                    label_info, attempt + 1, len(content), elapsed,
                )
                logger.debug("%sResponse: %s", label_info, _truncate(content))
                return content
            else:
                last_error = RuntimeError("Empty response from API")
                logger.warning("%sLLM attempt %d returned empty content after %.2fs",
                               label_info, attempt + 1, elapsed)
        except Exception as e:
            elapsed = time.time() - t_attempt
            last_error = e
            logger.error("%sLLM attempt %d failed after %.2fs: %s",
                         label_info, attempt + 1, elapsed, e)

        if attempt < retries:
            sleep_sec = backoff * (2 ** attempt)
            logger.info("%sRetrying in %.2f seconds...", label_info, sleep_sec)
            time.sleep(sleep_sec)

    total_elapsed = time.time() - t_start
    logger.error(
        "%sAll %d retries exhausted (total %.2fs). Last error: %s",
        label_info, retries + 1, total_elapsed, last_error,
    )
    return (
        "```lean\n"
        "-- LLM returned empty or failed after retries\n"
        f"-- Last error: {last_error}\n"
        "```"
    )


# ---------------------------------------------------------------------------
# Concurrent batch calling
# ---------------------------------------------------------------------------
def call_llm_batch(
    tasks: List[Dict[str, Any]],
    max_workers: int = 5,
) -> List[str]:
    """
    Execute multiple LLM calls concurrently.

    Parameters
    ----------
    tasks : list of dict
        Each dict must contain keyword arguments accepted by `call_llm`,
        e.g. {"system_prompt": ..., "user_prompt": ..., "model": ...}.
        You may optionally include "call_label" to identify the task in logs.
    max_workers : int
        Maximum number of threads to use.

    Returns
    -------
    list of str
        Responses in the same order as `tasks`.  A call that fails after all
        retries returns a placeholder string (as in synchronous `call_llm`).
    """
    results: List[Optional[str]] = [None] * len(tasks)

    def _worker(idx: int, kwargs: Dict[str, Any]) -> None:
        label = kwargs.pop("call_label", f"batch-{idx}")
        kwargs["call_label"] = label
        try:
            results[idx] = call_llm(**kwargs)
        except Exception as e:
            # This should not happen because call_llm catches its own errors,
            # but we guard anyway.
            logger.error("[%s] Unhandled exception in batch worker: %s", label, e)
            results[idx] = f"```lean\n-- Batch worker error: {e}\n```"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, task_kwargs in enumerate(tasks):
            # Make a copy to avoid mutating the original dict
            kwargs = dict(task_kwargs)
            futures.append(executor.submit(_worker, i, kwargs))
        # Wait for all to complete
        concurrent.futures.wait(futures)

    return [r or "```lean\n-- No response\n```" for r in results]


__all__ = ["call_llm", "call_llm_batch"]