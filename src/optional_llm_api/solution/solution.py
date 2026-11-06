"""
Day 1 — LLM API Foundation (side lab tuy chon)
Bai lam: implement moi TODO cua template.py, giu nguyen chu ky ham.

Model (11/2026): gpt-6.1-terra (lon) / gpt-6-luna (nho).
Ten khoa "gpt4o_*" / "mini_*" la hop dong cua test, giu nguyen: doc la "model lon" / "model nho".
"""

import os
import time
from typing import Any, Callable

# Gia uoc tinh moi 1K token OUTPUT (USD) - 11/2026
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


# ---------------------------------------------------------------------------
# Task 1 — Goi model lon
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Goi Chat Completions, tra ve (noi dung tra loi, thoi gian cho bang giay)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",          # GPT-6 chi nhan temperature/top_p khi effort = "none"
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,  # GPT-6 khong nhan max_tokens
    )
    latency = max(time.perf_counter() - start, 1e-9)   # luon la so duong
    return resp.choices[0].message.content or "", latency


# ---------------------------------------------------------------------------
# Task 2 — Goi model nho
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Goi model nho bang cach dung lai call_openai."""
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — So sanh model lon va model nho
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """Goi ca hai model voi cung prompt, tra ve dict so sanh."""
    big_text, big_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)
    # 0.75 tu ~ 1 token (uoc luong tho)
    est_tokens = len(big_text.split()) / 0.75
    cost = est_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
    return {
        "gpt4o_response": big_text,
        "mini_response": mini_text,
        "gpt4o_latency": big_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }


# ---------------------------------------------------------------------------
# Task 4 — Chatbot streaming, nho 3 luot hoi dap gan nhat
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """Chatbot trong terminal: in token ngay khi nhan duoc, giu 3 luot gan nhat, 'quit'/'exit' de thoat."""
    history: list[dict] = []
    client = None
    while True:
        try:
            user_input = input("\nBan: ").strip()
        except EOFError:
            break
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        if client is None:                       # tao client khi that su can goi API
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        history.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            reasoning_effort="none",
            max_completion_tokens=512,
            stream=True,
        )
        print("Bot: ", end="", flush=True)
        reply = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]                   # 3 luot = 6 tin nhan


# ---------------------------------------------------------------------------
# Bonus A — Thu lai voi do tre tang theo cap so nhan
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Goi fn(). Neu loi thi thu lai toi da max_retries lan, cho base_delay * 2^attempt giua cac lan."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise                              # het luot thu -> nem loi cuoi cung
            time.sleep(base_delay * 2 ** attempt)


# ---------------------------------------------------------------------------
# Bonus B — So sanh theo lo
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """Chay compare_models cho tung prompt, them khoa 'prompt' vao ket qua."""
    results = []
    for p in prompts:
        r = dict(compare_models(p))
        r["prompt"] = p
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Bonus C — Dinh dang bang so sanh
# ---------------------------------------------------------------------------
def _trunc(text: str, width: int = 40) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def format_comparison_table(results: list[dict]) -> str:
    """Bang chu: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency."""
    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    rows = [
        [
            _trunc(r.get("prompt", "")),
            _trunc(r.get("gpt4o_response", "")),
            _trunc(r.get("mini_response", "")),
            f"{r.get('gpt4o_latency', 0.0):.2f}s",
            f"{r.get('mini_latency', 0.0):.2f}s",
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(row) for row in rows)])


if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
