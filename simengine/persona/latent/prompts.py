"""Prompt templates of the latent-persona tasks, split into a person part (prefix) and a cell part (suffix).

The prefix is cached once per person (prompt arms) or once per category / product x price (short prompts of the
injection arms); the suffix runs per cell. `split` renders the chat template with a sentinel and cuts there, so
prefix + suffix is exactly the rendered prompt. Thinking is disabled (Qwen enable_thinking=False).

    pre, suf = split(tok, g1_messages("L2", profile, history, aralik, window="test"), cell=" Books")
    p = readout.yes_prob(ld, [pre], [suf])
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

SYSTEM = "You predict the behaviour of one real Amazon customer."
SYSTEM_TWIN = "You predict how one real US adult answers a survey."
SYSTEM_SHOP = "You predict the shopping choices of one real US adult."
SENTINEL = "§§CELL§§"
SOFT_MARK = "§§SOFT§§"
WINDOWS = {"test": "July-December 2022", "train": "January-June 2022"}
CF_LINES = {"price_up_50": "Prices in this category have risen by 50%.",
            "discount_30": "This category is on a 30% discount."}
YES_NO = ("Yes", "No")


def windows() -> dict[str, str]:
    return dict(WINDOWS)


def cf_line(kind: str) -> str:
    return CF_LINES[kind]


def g1_question(window: str, category: str = SENTINEL, cf: str | None = None) -> str:
    w = WINDOWS.get(window, window)
    q = (f"Question: Will this customer buy at least one item in the category {category} between {w}? "
         f"Answer Yes or No.")
    return (cf_line(cf) + " " + q) if cf else q


def g1_messages(level: str, profile: str | None = None, history: str | None = None, aralik: str | None = None,
                *, window: str = "test", cf: str | None = None) -> list[dict]:
    """G1 chat messages with the category left as SENTINEL. level: L0 (no persona), L1 (survey profile), L2
    (profile + purchase history block), short (no persona; the injection arms)."""
    if level == "L0":
        head = "Consider a randomly chosen US Amazon customer.\n\n"
    elif level == "L1":
        head = f"Survey answers of this customer about themselves and their Amazon account:\n{profile}\n\n"
    elif level == "L2":
        head = (f"Survey answers of this customer about themselves and their Amazon account:\n{profile}\n\n"
                f"Their Amazon purchase history, {aralik}:\n{history}\n\n")
    elif level == "short":
        head = ""
    else:
        raise ValueError("level must be L0, L1, L2 or short")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + g1_question(window, cf=cf)}]


def render(tok, messages: list[dict], *, add_generation_prompt: bool = True) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


def split(tok, messages: list[dict], cell: str = "") -> tuple[str, str]:
    """(prefix, suffix) of the rendered prompt cut at SENTINEL; `cell` replaces the sentinel at the head of the
    suffix (e.g. ' Books'; a leading space keeps BPE word boundaries the same as in the joint text)."""
    s = render(tok, messages)
    if s.count(SENTINEL) != 1:
        raise ValueError("the rendered prompt must hold the sentinel exactly once")
    pre, post = s.split(SENTINEL)
    if pre.endswith(" "):                       # the word boundary space goes with the cell, as BPE joins it
        pre, cell = pre[:-1], " " + cell.lstrip(" ")
    return pre, cell + post


def fill(messages: list[dict], cell: str) -> list[dict]:
    """Messages with SENTINEL replaced by `cell` (the short prompts carry their cell in the prefix)."""
    return [dict(m, content=m["content"].replace(SENTINEL, cell)) for m in messages]


def short_split(tok, messages: list[dict], *, soft_k: int = 0, soft_token: str = "") -> tuple[str, str]:
    """Short prompt of the injection arms: the question (with its cell text already filled) is the prefix; the
    suffix is the chat-template tail, preceded by k soft placeholders when soft_k > 0 ("Customer: <s>...<s>")."""
    msgs = [dict(m) for m in messages]
    msgs[-1]["content"] = msgs[-1]["content"] + SENTINEL
    pre, post = render(tok, msgs).split(SENTINEL)
    if soft_k:
        post = "\nCustomer: " + soft_token * soft_k + post
    return pre, post


def answer_token_ids(tok, words: Sequence[str] = YES_NO) -> dict[str, list[int]]:
    """First-token ids of each word's variants (w, ' w', lower, ' lower'). An id shared by two words (e.g. a bare
    space token) is dropped from both; ValueError if a word is left without an id."""
    ids: dict[str, set] = {}
    for w in words:
        s = set()
        for v in {w, " " + w, w.lower(), " " + w.lower()}:
            t = tok.encode(v, add_special_tokens=False)
            if t:
                s.add(int(t[0]))
        ids[w] = s
    out = {}
    for w, s in ids.items():
        others = set().union(*(ids[o] for o in ids if o != w)) if len(ids) > 1 else set()
        keep = sorted(s - others)
        if not keep:
            raise ValueError(f"no first token of {w!r} is free of the other answers' first tokens")
        out[w] = keep
    return out


def option_labels(k: int) -> list[str]:
    """'1'..'k' for k <= 9, else 'A'.. (two-digit labels share their first token with '1')."""
    return [str(i + 1) for i in range(k)] if k <= 9 else [chr(65 + i) for i in range(k)]


# ------------------------------------------------------------------------------------------------- Twin (G2 / G3)
def twin_profile(row: Mapping) -> str:
    """Compact Twin persona: the demographic block plus the construct scores (skor_*), ~550 tokens."""
    lines = [str(row.get("metin_demografi") or "").strip()]
    sc = [(k[5:], v) for k, v in row.items() if str(k).startswith("skor_") and v is not None and v == v]
    if sc:
        lines.append("Scores on psychological and cognitive measures (waves 1-3):")
        lines += [f"- {k}: {round(float(v), 3)}" for k, v in sc]
    return "\n".join(l for l in lines if l)


def twin_item(item: Mapping, profile: str | None = None) -> tuple[list[dict], list[str]]:
    """G2 option item: chat messages (persona in the user turn when given) and its option labels."""
    opts = item.get("secenekler")
    opts = json.loads(opts) if isinstance(opts, str) else list(opts or [])
    labels = option_labels(len(opts))
    body = str(item.get("soru") or "")
    if item.get("alt"):
        body += f"\n{item['alt']}"
    body += "\n" + "\n".join(f"{l}. {o}" for l, o in zip(labels, opts))
    body += f"\nAnswer with the option label only ({labels[0]}-{labels[-1]})."
    head = f"About this person:\n{profile}\n\n" if profile else "Consider a randomly chosen US adult.\n\n"
    return [{"role": "system", "content": SYSTEM_TWIN}, {"role": "user", "content": head + body}], labels


def tf_history_lines(history_rows: Sequence[Mapping], names: Mapping[str, str]) -> str:
    return "\n".join(f"- {names.get(r['oge'], r['oge'])} at ${float(r['fiyat']):.2f}: "
                     f"{'would buy' if int(r['al']) else 'would not buy'}" for r in history_rows)


def tf_question(product: str = SENTINEL, price: float | None = None) -> str:
    p = "" if price is None else f" at ${price:.2f}"
    return f"Question: Would you buy {product}{p}? Answer Yes or No."


def tf_messages(profile: str | None, history: str | None) -> list[dict]:
    """G3 messages with the product x price cell left as SENTINEL. profile None and history None = no persona
    (prm-<m>-tf0)."""
    if profile is None and history is None:
        head = "Consider a randomly chosen US adult shopper.\n\n"
    else:
        head = ""
        if profile:
            head += f"About this person:\n{profile}\n\n"
        if history:
            head += f"Their earlier answers to the same kind of question:\n{history}\n\n"
    return [{"role": "system", "content": SYSTEM_SHOP}, {"role": "user", "content": head + tf_question()}]


def tf_cell(product: str, price: float) -> str:
    return f" {product} at ${price:.2f}"


__all__ = ["SYSTEM", "SYSTEM_TWIN", "SYSTEM_SHOP", "SENTINEL", "WINDOWS", "CF_LINES", "windows", "cf_line",
           "g1_question", "g1_messages", "render", "split", "fill", "short_split", "answer_token_ids", "option_labels",
           "twin_profile", "twin_item", "tf_history_lines", "tf_question", "tf_messages", "tf_cell"]
