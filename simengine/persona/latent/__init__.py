"""simengine.persona.latent: persona as a vector inside a local open model instead of in the prompt.

Extract a person vector (survey one-hot / autoencoder, history features + ALS / bag-SVD, text embedding, the
model's own hidden state, contrastive trait directions, SAE codes), then read it out (probe on hidden states,
P(yes) from logits) or put it inside (CAA residual steering, a learned residual projector, soft tokens), next to the
LLM-free encoder-only controls. Pre-registration: docs/prereg/latent-persona-2026-10-23.md; runner
runs/latent/kos.py; usage docs/modules/latent.md. torch / transformers are imported lazily.

    from simengine.persona import latent as lt
    ld = lt.load("q38", "fp8")                                    # or lt.tiny("qwen3_5") without a download
    p, mass = lt.yes_prob(ld, prefixes, suffixes)                 # prefix-cached P(Yes) per (person, category)
    with lt.add_to_residual(ld.model, 30, vecs): ...              # steering hook, removed on exit
"""
from __future__ import annotations

from .encoders import (TRAIT_EDITS_G1, TRAITS_G1, BagSVD, ImplicitALS, SequenceEncoder, Standardizer,
                       SurveyAutoencoder, SurveyOneHot, behaviour_directions, count_matrix, edit_profile,
                       history_features, llm_states, load_sae, sae_codes, text_embedding, trait_directions,
                       trait_person_vector, trait_values)
from .eval import decide, mad_accuracy, score_g1, score_g2, score_g3
from .heads import EncoderHead, Probe, platt, platt_scores, stack_lgbm
from .hooks import add_to_residual, capture, embedding_norm, replace_embeddings, residual_norm
from .models import (MODELS, PRECISIONS, TINY_FAMILIES, Loaded, byte_tokenizer, check_backward, from_config,
                     full_attention_layers, input_embeddings, layer_index, layers, load, tiny, tiny_config)
from .prompts import answer_token_ids, g1_messages, render, split, tf_messages, twin_item, twin_profile
from .readout import Injection, expand_cache, option_probs, prefix_states, select_cache, yes_prob
from .select import acc_1a, choose, grid, kisi_auc, val_metrics_g1
from .steer import (CAA, derange, mean_diff_direction, pca_direction, predict_injector, probe_direction,
                    residual_projector, soft_placeholder, soft_token_projector, train_injector)
from .tasks import G1, G2, G3, LeakError, TestGuard, link_baselines, load_g1, load_g2, load_g3, write_pred

__all__ = [
    # models / hooks
    "MODELS", "PRECISIONS", "TINY_FAMILIES", "Loaded", "load", "tiny", "tiny_config", "from_config",
    "byte_tokenizer", "layers", "input_embeddings", "full_attention_layers", "layer_index", "check_backward",
    "add_to_residual", "capture", "replace_embeddings", "residual_norm", "embedding_norm",
    # prompts / readout
    "g1_messages", "render", "split", "answer_token_ids", "twin_item", "twin_profile", "tf_messages",
    "Injection", "yes_prob", "option_probs", "prefix_states", "select_cache", "expand_cache",
    # extraction
    "SurveyOneHot", "SurveyAutoencoder", "history_features", "count_matrix", "ImplicitALS", "BagSVD",
    "SequenceEncoder", "Standardizer", "text_embedding", "llm_states", "trait_directions", "trait_values",
    "trait_person_vector", "edit_profile", "behaviour_directions", "load_sae", "sae_codes", "TRAIT_EDITS_G1",
    "TRAITS_G1",
    # heads / steering
    "EncoderHead", "Probe", "platt", "platt_scores", "stack_lgbm", "CAA", "mean_diff_direction", "pca_direction",
    "probe_direction", "residual_projector", "soft_token_projector", "soft_placeholder", "train_injector",
    "predict_injector", "derange",
    # tasks / selection / scoring
    "TestGuard", "LeakError", "G1", "G2", "G3", "load_g1", "load_g2", "load_g3", "write_pred", "link_baselines",
    "grid", "choose", "kisi_auc", "acc_1a", "val_metrics_g1", "score_g1", "score_g2", "score_g3", "mad_accuracy",
    "decide",
]
