"""Local open-model loading for the latent-persona work: registry, precision / quantisation, decoder-layer lookup.

torch / transformers are imported inside the functions, so `import simengine.persona.latent` works without them.

    ld = load("q38", "bf16")                  # Qwen3.8-27B as Qwen3_5ForCausalLM (text only), cuda
    ld = load("q38f", "fp8")                  # official FP8 checkpoint (compute capability >= 8.9)
    ld = load("g31", "nf4")                   # Gemma 4 31B, bitsandbytes 4-bit
    ld = tiny("qwen3_5")                      # tiny random hybrid model + byte tokenizer, no download (tests, --kucuk)
    layers(ld.model)[ld.layer_index(0.5)]     # decoder layer at half depth
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODELS: dict[str, dict] = {
    "q38": {"hf_id": "Qwen/Qwen3.8-27B", "family": "qwen3_5", "hybrid": True},
    "q38f": {"hf_id": "Qwen/Qwen3.8-27B-FP8", "family": "qwen3_5", "hybrid": True},
    "g31": {"hf_id": "google/gemma-4-31B-it", "family": "gemma4", "hybrid": False},
    "g26": {"hf_id": "google/gemma-4-26B-A4B-it", "family": "gemma4", "hybrid": False, "moe": True},
    "q35": {"hf_id": "Qwen/Qwen3.5-27B", "family": "qwen3_5", "hybrid": True},
    "q3": {"hf_id": "Qwen/Qwen3-32B", "family": "qwen3", "hybrid": False},
    "q25": {"hf_id": "Qwen/Qwen2.5-32B-Instruct", "family": "qwen2", "hybrid": False},
    "q35s": {"hf_id": "Qwen/Qwen3.5-2B", "family": "qwen3_5", "hybrid": True},
}
PRECISIONS = ("bf16", "fp16", "fp32", "fp8", "int8", "nf4")
LAYER_PATHS = ("model.language_model.layers", "model.layers", "language_model.layers", "base_model.model.model.layers",
               "transformer.h", "gpt_neox.layers", "layers")
EMBED_PATHS = ("model.language_model.embed_tokens", "model.embed_tokens", "language_model.embed_tokens",
               "transformer.wte", "embed_tokens")


@dataclass
class Loaded:
    """A loaded local model. `precision` is the one its vectors/probes/projectors must be applied in."""
    model: Any
    tok: Any
    code: str
    hf_id: str
    revision: str
    precision: str
    n_layers: int
    hidden_size: int
    full_attention: list = field(default_factory=list)
    embed_scale: float = 1.0
    family: str = ""

    @property
    def device(self):
        return next(self.model.parameters()).device

    def layer_index(self, frac: float) -> int:
        return layer_index(self, frac)

    def meta(self) -> dict:
        return {"model": self.code, "hf_id": self.hf_id, "revision": self.revision, "precision": self.precision,
                "n_layers": self.n_layers, "hidden_size": self.hidden_size, "family": self.family}


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def layers(model):
    """The decoder layer list: model.language_model.layers (qwen3_5 / gemma4 multimodal wrappers), model.layers
    (qwen2 / qwen3 / llama / mistral / *ForCausalLM), PEFT-wrapped, gpt2 / neox. ValueError if none is found."""
    import torch
    for p in LAYER_PATHS:
        try:
            v = _get(model, p)
        except AttributeError:
            continue
        if isinstance(v, torch.nn.ModuleList) and len(v):
            return v
    raise ValueError(f"no decoder layer list on {type(model).__name__} (tried {', '.join(LAYER_PATHS)})")


def input_embeddings(model):
    """The token embedding module (the one whose output soft tokens replace)."""
    try:
        return model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        pass
    for p in EMBED_PATHS:
        try:
            return _get(model, p)
        except AttributeError:
            continue
    raise ValueError(f"no input embedding on {type(model).__name__}")


def text_config(model_or_config):
    cfg = getattr(model_or_config, "config", model_or_config)
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def full_attention_layers(cfg) -> list[int]:
    """Indices of full (global) attention layers: from layer_types when present (qwen3_5: i % 4 == 3, gemma4:
    i % 6 == 5), else every layer."""
    tc = text_config(cfg)
    types = getattr(tc, "layer_types", None)
    n = tc.num_hidden_layers
    if not types:
        return list(range(n))
    return [i for i, t in enumerate(types) if t == "full_attention"]


def _embed_scale(model) -> float:
    emb = input_embeddings(model)
    s = getattr(emb, "embed_scale", None)
    if s is None:
        return 1.0
    return float(s.item() if hasattr(s, "item") else s)


def _wrap(model, tok, code, hf_id, revision, precision) -> Loaded:
    tc = text_config(model)
    return Loaded(model=model, tok=tok, code=code, hf_id=hf_id, revision=revision or "", precision=precision,
                  n_layers=len(layers(model)), hidden_size=int(tc.hidden_size),
                  full_attention=full_attention_layers(model), embed_scale=_embed_scale(model),
                  family=getattr(tc, "model_type", "").replace("_text", ""))


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _dtype(precision: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.bfloat16)


def quant_config(precision: str, family: str = "", *, moe: bool = False):
    """transformers quantization_config for fp8 / int8 / nf4, None for bf16 / fp16 / fp32. fp8 needs compute
    capability >= 8.9 (no silent bf16 fallback); nf4/int8 refuse MoE models whose experts are 3-D tensors."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision in ("bf16", "fp16", "fp32"):
        return None
    if precision == "fp8":
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            raise ValueError("fp8 needs a GPU with compute capability >= 8.9 (Ada / Hopper); use nf4 or bf16")
        from transformers import FineGrainedFP8Config
        return FineGrainedFP8Config()
    if moe:
        raise ValueError("bitsandbytes does not quantise 3-D expert tensors; load this MoE model in bf16")
    from transformers import BitsAndBytesConfig      # needs bitsandbytes
    import torch
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def _snapshot(hf_id: str, revision: str | None, cache_dir: str | None) -> str | None:
    """Commit hash of the cached snapshot (transformers 5 no longer sets config._commit_hash)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(hf_id, "config.json", revision=revision, cache_dir=cache_dir)
    except Exception:                                                # noqa: BLE001 - meta only
        return None
    return Path(p).parent.name if isinstance(p, str) else None


def load(code: str, precision: str = "bf16", *, device: str = "cuda", text_only: bool = True,
         revision: str | None = None, cache_dir: str | None = None) -> Loaded:
    """Load a registry model (or any HF id) frozen in eval mode.

    qwen3_5 with text_only: AutoModelForCausalLM -> Qwen3_5ForCausalLM (no vision tower; the multimodal checkpoint
    loads without missing keys). gemma4: always Gemma4ForConditionalGeneration (Gemma4ForCausalLM.from_pretrained on
    the multimodal checkpoint silently gives random weights). The official FP8 checkpoint (q38f) carries its own
    quantization config: pass precision "fp8"."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if code.startswith("tiny-"):
        return tiny(code[5:], precision=precision, device=device)
    spec = MODELS.get(code, {"hf_id": code, "family": "", "hybrid": False})
    hf_id = spec["hf_id"]
    kw: dict = {"revision": revision, "cache_dir": cache_dir, "dtype": _dtype(precision)}
    prequant = code == "q38f"
    if prequant and precision != "fp8":
        raise ValueError("q38f is the FP8 checkpoint: load it with precision='fp8' (or use q38 for bf16 / nf4)")
    qc = None if prequant else quant_config(precision, spec["family"], moe=spec.get("moe", False))
    if prequant:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            raise ValueError("fp8 needs a GPU with compute capability >= 8.9 (Ada / Hopper); use q38 in nf4 or bf16")
    if qc is not None:
        kw["quantization_config"] = qc
    if device != "cpu":
        kw["device_map"] = device if device != "cuda" else "auto"
    if spec["family"] == "gemma4":
        cls = transformers.Gemma4ForConditionalGeneration
    else:
        cls = AutoModelForCausalLM
        if not text_only and spec["family"] == "qwen3_5":
            cls = transformers.Qwen3_5ForConditionalGeneration
    model = cls.from_pretrained(hf_id, **kw)
    if device == "cpu":
        model = model.to("cpu")
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision, cache_dir=cache_dir)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rev = getattr(model.config, "_commit_hash", None) or _snapshot(hf_id, revision, cache_dir) or revision or ""
    return _wrap(_freeze(model), tok, code, hf_id, rev, precision)


def from_config(config, *, tok=None, seed: int = 0, code: str = "tiny", precision: str = "fp32",
                device: str = "cpu") -> Loaded:
    """Random-weight model from a config (tests; no download). The class is the config's architecture
    (qwen3_5 / gemma4 multimodal wrappers included)."""
    import torch
    import transformers
    torch.manual_seed(seed)
    arch = (getattr(config, "architectures", None) or [None])[0]
    if arch:
        model = getattr(transformers, arch)(config)
    else:
        model = transformers.AutoModelForCausalLM.from_config(config)
    model = model.to(dtype=_dtype(precision), device=device)
    return _wrap(_freeze(model), tok if tok is not None else byte_tokenizer(), code, code, "random", precision)


def layer_index(ld: Loaded, frac: float) -> int:
    """round(frac * n_layers) clipped to [0, n_layers - 1]."""
    return int(min(max(round(frac * ld.n_layers), 0), ld.n_layers - 1))


def check_backward(ld: Loaded) -> bool:
    """One forward + backward to a trainable residual vector through the frozen model (fp8 / nf4 calibration step).
    True when a finite, non-zero gradient reaches the vector and no model parameter gets a gradient."""
    import torch
    from .hooks import add_to_residual
    ids = ld.tok("Will this customer buy books? Answer Yes or No.", return_tensors="pt")["input_ids"].to(ld.device)
    v = torch.zeros(ld.hidden_size, device=ld.device, dtype=torch.float32, requires_grad=True)
    try:
        with add_to_residual(ld.model, layer_index(ld, 0.5), lambda h: v.to(h.dtype).expand(h.shape[0], -1)):
            out = ld.model(input_ids=ids).logits[0, -1].float()
        out.logsumexp(-1).backward()
    except Exception:                                                  # noqa: BLE001 - reported as False
        return False
    ok = v.grad is not None and bool(torch.isfinite(v.grad).all()) and float(v.grad.abs().sum()) > 0
    return ok and all(p.grad is None for p in ld.model.parameters())


# ------------------------------------------------------------------------------------------------- tiny models
def byte_tokenizer(extra: tuple = ("Yes", "No", "yes", "no")):
    """A byte-level tokenizer with a ChatML template, built in memory (no download). Words in `extra` are single
    tokens; a leading space is a separate token, so ' Yes' and ' No' share their first token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token=None, eos_token="<|im_end|>", pad_token="<|pad|>",
                                  additional_special_tokens=["<|im_start|>", "<|soft|>"])
    tok.add_tokens(list(extra))
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    tok.padding_side = "left"
    return tok


def tiny_config(family: str, vocab_size: int = 320):
    """Small config with the real layer layout of a target family: qwen3_5 (L L L F x2, multimodal wrapper),
    qwen3_5_causal, gemma4 (S x5 F x2), gemma4_moe, qwen3, qwen2, llama, mistral, qwen3_moe."""
    import transformers as tf
    if family in ("qwen3_5", "qwen3_5_causal"):
        text = dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                    layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
                    + ["full_attention"], num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                    linear_value_head_dim=16, linear_conv_kernel_dim=4, attn_output_gate=True,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                     "mrope_section": [2, 1, 1], "mrope_interleaved": True},
                    pad_token_id=None, tie_word_embeddings=False)
        if family == "qwen3_5_causal":
            c = tf.Qwen3_5TextConfig(**text)
            c.architectures = ["Qwen3_5ForCausalLM"]
            return c
        vision = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64)
        V = vocab_size
        c = tf.Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False, image_token_id=V - 1,
                             video_token_id=V - 2, vision_start_token_id=V - 3, vision_end_token_id=V - 4)
        c.architectures = ["Qwen3_5ForConditionalGeneration"]
        return c
    if family in ("gemma4", "gemma4_moe"):
        text = dict(vocab_size=vocab_size, vocab_size_per_layer_input=vocab_size, hidden_size=64,
                    intermediate_size=128, num_hidden_layers=12,
                    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 2, num_attention_heads=2,
                    num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16, global_head_dim=32,
                    sliding_window=4, hidden_size_per_layer_input=0, final_logit_softcapping=30.0,
                    attention_k_eq_v=True, pad_token_id=0, tie_word_embeddings=True,
                    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                                        "rope_type": "proportional"},
                                     "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
        if family == "gemma4_moe":
            text.update(enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=32)
        vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=16, global_head_dim=16)
        V = vocab_size                          # multimodal special ids inside the tiny vocabulary
        c = tf.Gemma4Config(text_config=text, vision_config=vision, audio_config=None, image_token_id=V - 1,
                            video_token_id=V - 2, audio_token_id=V - 3, boi_token_id=V - 4, eoi_token_id=V - 5,
                            boa_token_id=V - 6, eoa_token_id=V - 7)
        c.architectures = ["Gemma4ForConditionalGeneration"]
        return c
    dense = dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                 num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    if family == "qwen3":
        c = tf.Qwen3Config(head_dim=16, **dense)
        c.architectures = ["Qwen3ForCausalLM"]
    elif family == "qwen2":
        c = tf.Qwen2Config(**dense)
        c.architectures = ["Qwen2ForCausalLM"]
    elif family == "llama":
        c = tf.LlamaConfig(**dense)
        c.architectures = ["LlamaForCausalLM"]
    elif family == "mistral":
        c = tf.MistralConfig(**dense)
        c.architectures = ["MistralForCausalLM"]
    elif family == "qwen3_moe":
        c = tf.Qwen3MoeConfig(head_dim=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                              decoder_sparse_step=1, **dense)
        c.architectures = ["Qwen3MoeForCausalLM"]
    else:
        raise KeyError(f"unknown tiny family {family!r}")
    return c


TINY_FAMILIES = ("qwen3_5", "qwen3_5_causal", "gemma4", "gemma4_moe", "qwen3", "qwen2", "llama", "mistral",
                 "qwen3_moe")


def tiny(family: str, *, seed: int = 0, precision: str = "fp32", device: str = "cpu") -> Loaded:
    """Tiny random model of a family with the byte tokenizer (no download). Outputs are meaningless."""
    tok = byte_tokenizer()
    return from_config(tiny_config(family, vocab_size=int(math.ceil(len(tok) / 64) * 64)), tok=tok, seed=seed,
                       code=f"tiny-{family}", precision=precision, device=device)


__all__ = ["MODELS", "PRECISIONS", "TINY_FAMILIES", "Loaded", "load", "from_config", "tiny", "tiny_config",
           "byte_tokenizer", "layers", "input_embeddings", "text_config", "full_attention_layers", "layer_index",
           "quant_config", "check_backward"]
