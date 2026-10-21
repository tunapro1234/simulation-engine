# `simengine.llm`: LLM backends

The module has three backends that implement the `simengine.types.LLM` protocol
(`complete(system, user, max_tokens=..., **kw)` and `map(prompts, parallel=...)`), plus the pieces they share.

| component | file | what it does |
|---|---|---|
| `Claude` | `claude.py` | Claude API through the official `anthropic` SDK, imported lazily (`pip install simengine[llm]`). |
| `ClaudeCode` | `claude_code.py` | `claude -p` subprocess. Runs on the subscription with no API bill. Meant for long-lived or low-volume agents. |
| `FakeLLM` | `fake.py` | Deterministic and offline, for tests and dry runs. Unchanged. |
| `DiskCache`, `cache_key` | `cache.py` | One JSON file per sha256(model + system + user + params). Writes are atomic, so threads and processes can share one cache. |
| `CostLedger` | `ledger.py` | JSONL ledger with one line per real call, written under flock. `total()` reads incrementally. |
| `PRICES`, `cost_usd`, `usage_cost` | `pricing.py` | Price per model in $/MTok and the token-to-USD formula. |
| `json_extract`, `complete_json` | `jsonx.py` | Robust JSON extraction from LLM text. `complete_json` re-asks once if parsing fails. |
| `parallel_map` | `parallel.py` | Thread pool that keeps input order. A `StopError` cancels the pending items. |
| errors | `errors.py` | `LLMError` > `StopError` > `AuthError`, `SpendLimitError`, `BudgetExceeded`; plus `RefusalError`. |

Ported from (read-only research code, `uretim/<branch>/`):
- `talp73/kod/llm/claude_api.py`: effort, cache_control, the price table, the cost formula, the budget gate, and the 401 / limit stop.
- `talp73/kod/llm/openrouter.py`: the cache layout, the ledger and its total, the parallel map, and `parse_json`.
- `talp73/kod/llm/kos.py`: how the backends are driven, including `sample` (the research code's `ornek`).
- `sahne/kod/sahne/kaynak.py`: `Onbellek`, `Gunluk`, `Sayac` (call budget), `json_coz`, and the `ClaudeKaynak` command line.
- `makro/llmabm/kod/llm.py`: the `claude -p` empty working directory, the call cap, and the `sor_json` re-ask.
- `dongu/kod/gozlemci.sh`: the observer command line.

## Minimal usage

```python
from simengine.llm import Claude, ClaudeCode, json_extract

llm = Claude(model="claude-haiku-5-5", effort="low", cache_dir=".cache", budget_usd=5)
r = llm.complete("You are a survey respondent.", "Answer A or B and explain in one line.", max_tokens=200)
print(r.text, r.cost_usd, r.cached)

rs = llm.map([("You are a survey respondent.", q) for q in questions], max_tokens=200, parallel=4)
data = llm.complete_json("Return JSON only.", 'Give {"p": <probability>} for ...')
print(llm.stats)            # calls, cache_hits, retries, errors, cost_usd, ledger_total_usd

obs = ClaudeCode(model="claude-haiku-5-5", cache_dir=".cache", max_calls=50, timeout=300)
proposal = json_extract(obs.complete("You review experiment logs.", log_text).text)
```

Key: the `api_key=` argument, else `ANTHROPIC_API_KEY`, else the contents of the file named in `SIMENGINE_KEY_FILE`.
The key value is never written to the cache, the ledger, an exception or `repr()`. Without a key, answers already in
the cache are still served; a new call raises `AuthError`.

## `Claude` behaviour

- **Request.** The request carries `model`, `max_tokens` (default 256), `messages=[user]` and
  `output_config={"effort": effort}` (default `"low"`). The system prompt is sent as one text block with
  `cache_control: {"type": "ephemeral"}`. No temperature is sent. Other `messages.create` keywords (`stop_sequences`,
  `output_config={"format": ...}`, ...) pass through.
- **Thinking.** Omitted by default, which means adaptive thinking on the 5.x models. `thinking="off"` maps to
  `{"type": "disabled"}` on Haiku and `{"type": "between_tools"}` on Sonnet 5.5. On Opus 5.5 and Fable it raises
  `ValueError`, because thinking cannot be turned off there; lower the effort instead.
- **Cache key.** sha256 over model, system, user and every request parameter except `cache_control`, plus
  `sample` (an integer that gives independent draws of the same prompt). Only answers that pass `validate` are cached.
  Refusals and errors are never cached.
- **Ledger and cost.** Every real call adds one line: `model` (`anthropic/<id>`), input / output / cache-read /
  cache-write tokens, `cost_usd`, `tag`, `served_model`, `attempt`. The cost is
  `((input + 1.25·cache_write + 0.1·cache_read)·p_in + output·p_out) / 1e6`.

  | model | input $/MTok | output $/MTok |
  |---|---|---|
  | `claude-haiku-5-5` | 0.10 | 0.50 (0.50 / 2.50 above 100K prompt tokens) |
  | `claude-sonnet-5-5` | 2 | 10 |
  | `claude-opus-5-5` | 4 | 20 |
  | `claude-fable-5-1` | 10 | 50 |

  Cache reads are counted at 0.1x for every model. Opus 5.5's list cache-read price is 0.05x; the research code
  kept 0.1x on purpose as a conservative estimate (`talp73/kod/llm/claude_api.py`), and so does this module. A model
  that is not in the table raises `KeyError` when the backend is constructed, because an unpriced model would make
  the budget guard blind. Pass `prices=` to add one.
- **Budget guard.** Before every real call (retries included), `BudgetExceeded` is raised if `ledger.total() >= budget_usd`.
  The total covers the whole ledger file, so every process that writes to the same ledger shares one budget, as the
  research code's ledger did. The guard checks spend that has already happened. One call in flight can still push
  the total past the budget by the cost of that single call.
- **Errors.** The SDK's own retries are turned off (`max_retries=0`) so that a spend-limit 429 is never retried.
  - HTTP 401 raises `AuthError` at once.
  - HTTP 400, 402, 403 or 429 whose message mentions a usage limit, spend limit, credit balance or billing raises
    `SpendLimitError` at once. There is no retry and no fallback key.
  - A plain 429, a 529 or "overloaded", 5xx, 408 / 409 and connection or timeout errors are retried up to
    `max_attempts` (4). The wait is the `retry-after` header if present, otherwise `rate_wait·attempt`
    (15 s, 30 s, ...) for 429/529 and `error_wait·attempt` (3 s, ...) for the rest.
  - Any other 4xx raises `LLMError` without a retry.
  - `stop_reason == "refusal"` raises `RefusalError`. It is not cached and not retried.
- **Validation.** A `validate(text)` hook that raises `ValueError`, `KeyError` or `TypeError` triggers a retry, which
  costs another call and is logged in the ledger.
- **map and stops.** `map` uses a thread pool (`parallel`, default 4). A `StopError` cancels the items that have not
  started. Finished answers are already in the cache, so re-running the same code resumes where it stopped.
- **Fallbacks.** `fallbacks=None` by default. Setting it (for example `"default"`) uses the server-side-fallback beta,
  which can let a different model answer. That breaks a measured model comparison, so it is opt-in. The model that
  actually answered is recorded as `served_model`.

## `ClaudeCode` behaviour

- **Command.** Each call runs `claude -p --model <m> --output-format json --tools "" --no-session-persistence
  --strict-mcp-config --setting-sources "" --settings '{"language":"English"}' [--system-prompt <system>]`.
  The prompt goes on stdin. The command runs in an empty working directory (`<cache_dir>/claude-cwd`, or a temporary
  directory) so that no project `CLAUDE.md` is loaded, with a timeout (default 300 s). `CLAUDE_BIN` overrides the
  binary, `extra_args` appends flags (for example `--safe-mode`, as in `dongu/kod/gozlemci.sh`), and
  `use_system_prompt=False` puts the system text in front of the prompt instead of in the flag.
- **Language.** The language setting is passed on every call and is part of the cache key.
- **Cost.** The text comes from the envelope's `result`. `total_cost_usd` is a list-price equivalent, not a bill. It is
  stored as `list_price_usd` with `cost_usd = 0`, so it does not count toward `budget_usd`. Use `max_calls` to cap
  new calls per instance, as the research code's 200-call and 500-call caps did.
- **Errors.** A timeout, unparseable stdout or `is_error` is retried up to `max_attempts` (3). An envelope or stderr
  that mentions a usage limit raises `SpendLimitError`; not-logged-in or invalid-key text raises `AuthError`; neither
  is retried. These marker strings are guesses: they have not been checked against real CLI failure output.
- **Arguments.** `max_tokens` is accepted for protocol compatibility but is not passed on; the CLI has no such flag.
  Other keyword arguments raise `TypeError`.

## When it works and when it does not (measured)

Every number below was measured in the research runs, not in this library. This port has made **no live call**;
its tests are offline (a mocked client, the real SDK over a mocked HTTP transport, and a mocked subprocess).

**Claude API cost per call.** At effort low with a short system prompt, Haiku 5.5 is very cheap:

| run | measured | source |
|---|---|---|
| Haiku L1, dry run (20 calls) | $0.000265 per call | `talp73/PROTOKOL.md` (v1.4 addendum) |
| Haiku L2, dry run (20 calls) | $0.000517 per call | `talp73/PROTOKOL.md` (v1.4 addendum) |
| Sonnet 5.5 L1, dry run (20 calls) | $0.00507 per call; 0 errors in all three dry runs | `talp73/PROTOKOL.md` (v1.4 addendum) |
| L1-haiku55, full run (971 calls) | $0.2577 | `talp73/kod/llm/LLM-TABLO.md` |
| L2-haiku55, full run (971 calls) | $0.5315 | `talp73/kod/llm/LLM-TABLO.md` |
| L1-sonnet55, full run (976 calls) | $4.9455 | `talp73/kod/llm/LLM-TABLO.md` |
| Haiku L0 tokens | 577 input / 322 output per call | `talp73/kod/llm/PROTOKOL-v1.7c-L0sonnet.md` |
| Haiku SSR arm, dry run (10 calls) | 1,069 input / 1,159 output tokens, $0.000686 per call | `talp73/kod/llm/PROTOKOL-v1.7d-ssr.md` |

**cache_control does nothing on short prompts.** With a system prefix of about 50 tokens, below the minimum cacheable
length, a 10-call test showed `cache_creation_input_tokens = 0` (`talp73/kod/llm/PROTOKOL-v1.5-ek.md`). Caching only
pays off when a long, stable system prompt is shared across calls. Check `raw["usage"]["cache_read_input_tokens"]`.

**Temperature.** Haiku 5.5 and Sonnet 5.5 reject a non-default temperature with HTTP 400 (`talp73/PROTOKOL.md`,
v1.4 addendum). The backend never sends one. A `temperature=` you pass yourself goes through and will fail on those
models.

**Thinking at effort low.** Haiku with adaptive thinking and effort low used 0 thinking tokens on L1. Its answers
correlated r = 0.90 with thinking turned off (`talp73/PROTOKOL.md`, v1.4 addendum). The research runs used
`max_tokens=2000` because thinking tokens count toward it. With the default of 256, a high effort can be cut off
(`stop_reason == "max_tokens"`).

**Why 401 stops the run.** In one run the key returned 401 partway through. The two remaining arms (L2-sonnet55 and
L1x3-haiku55) were not run (`talp73/kod/llm/SONUC-v1.6-L1sonnet.md`). The
stop / limit / 429 logic was first tested with fake HTTP responses (`talp73/kod/llm/PROTOKOL-v1.6-ek.md`). This port
re-tests the same rules offline, including through the real SDK.

**Throughput.** The research runs used `parallel=8`. They planned it against a limit of 35M tokens per hour, at about
2.5K tokens per L2-Sonnet call (`talp73/kod/llm/PROTOKOL-v1.6-ek.md`). The library defaults to 4, the cap in
`sahne/kod/sahne/kaynak.py` (at most 4 threads) and in `makro/onkayit/yayilim-24s.md` (at most 4 concurrent requests).

**`claude -p` works for small and medium batches:**
- `sahne` clean clone with an empty cache: 103 calls, 0 errors, 0 missing answers, $0.155 list price, 257 s for the
  OE rehearsal, 236 s of it in LLM arms (`sahne/RAPOR.md`).
- `deney` kit (Opus 5.5): 48 calls, 20 of them from cache, in 148 s (`deney/RAPOR.md`).
- `intent`: an LLM step of 20 calls took about 1 minute and about $0.02 (`intent/RAPOR.md`).

**`claude -p` is the wrong tool for thousands of short calls:**
- Each `claude -p` call wrote about 2.4K input tokens to the 1-hour cache, and they were never read back. Opus over
  300 tests cost $0.50 to $0.79 for 30 calls, and the persona panel $2.54 to $3.89 for 150 calls, at list price
  (`deney/RAPOR.md`, cost table).
- The `makro` pre-registration chose the API over `claude -p` for 2,000+ calls. Its reasons: the large tool context
  carried by every call, the hourly token limit and the subscription quota. This is a stated rationale, not a
  measurement (`makro/onkayit/yayilim-24s.md`).

**Language leak in `claude -p`.** `claude -p` can inherit the user's `language` setting (`deney/RAPOR.md`, 21 Oct).
Run with `--setting-sources ""`, 2,362 cached answers contained no Turkish characters (`deney/RAPOR.md`), and 184
scanned answers were all English (`sahne/RAPOR.md`). An explicit Turkish prompt overrides the English setting
(`sahne/RAPOR.md`). That is why `language` is always passed and is part of the cache key.

## Not ported / limits

- The Batch API (`claude_api.toplu`) is not ported. The research program banned Batch for its runs.
- OpenRouter, the local Qwen backend and the TALP API stub (`sahne` `QwenKaynak` / `TalpKaynak`) are not ported.
- Server-side fallbacks and the real `claude -p` error texts have not been exercised live.
- The budget guard counts only calls logged to the ledger. Spend from other tools on the same key is invisible to it.
