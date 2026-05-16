"""
Ad Matcher (Part 2 of the AD recommender pipeline)

Given a list of CIResult objects from the constrained decoder and the
vanilla LLM response text, computes semantic similarity between each
commercial intent and the response, then returns ads for CIs that meet
the match threshold.

Matching uses cosine similarity between mean-pooled embeddings from the
same Qwen model that is already loaded — no additional model required.
"""

import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchedAd:
    """An ad that passed the match threshold."""
    product_id: str
    title: str
    brand: str
    category: str
    description: str
    url: str
    price: Optional[float]
    ci_text: str
    match_score: float  # cosine similarity in [0, 1]


def _mean_pool(model, tokenizer, device, text: str) -> torch.Tensor:
    """
    Encode `text` into a single fixed-size vector via mean pooling of the
    last hidden layer of the causal LM.

    Returns a 1-D tensor of shape [hidden_size], on CPU for easy comparison.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]          # [1, seq_len, H]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.squeeze(0).cpu()                   # [H]


def match_ads(
    model,
    tokenizer,
    device,
    ci_results: list,
    response_text: str,
    ci_ad_index_path: str,
    match_threshold: float = 0.70,
    max_ads: int = 2,
) -> list[MatchedAd]:
    """
    Return ads whose commercial intent is semantically close to the response.

    Steps:
      1. Embed the full response once.
      2. For each CI, embed the CI phrase and compute cosine similarity.
      3. If similarity >= match_threshold, include ads for that CI.
      4. Stop after max_ads unique products are collected.

    Args:
        model:              Loaded AutoModelForCausalLM (shared with decoder)
        tokenizer:          Matching AutoTokenizer
        device:             torch.device
        ci_results:         list[CIResult] from ConstrainedBeamSearchDecoder.generate()
        response_text:      The vanilla LLM response (Part 1 output)
        ci_ad_index_path:   Path to data/ci_ad_index.json
        match_threshold:    Minimum cosine similarity to include an ad (default 0.70)
        max_ads:            Maximum number of ads to return (default 2)

    Returns:
        List of MatchedAd, ordered by match_score descending.
    """
    with open(ci_ad_index_path) as f:
        ci_ad_index = json.load(f)

    response_emb = _mean_pool(model, tokenizer, device, response_text)

    candidates: list[tuple[float, dict, str]] = []  # (score, ad_record, ci_text)

    for ci in ci_results:
        ci_emb = _mean_pool(model, tokenizer, device, ci.ci_text)
        score = F.cosine_similarity(
            response_emb.unsqueeze(0),
            ci_emb.unsqueeze(0),
        ).item()

        if score >= match_threshold:
            for ad in ci_ad_index.get(ci.ci_text, []):
                candidates.append((score, ad, ci.ci_text))

    # Sort by score, deduplicate by product_id, cap at max_ads
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    matched: list[MatchedAd] = []

    for score, ad, ci_text in candidates:
        if len(matched) >= max_ads:
            break
        if ad["product_id"] in seen:
            continue
        seen.add(ad["product_id"])
        matched.append(
            MatchedAd(
                product_id=ad["product_id"],
                title=ad["title"],
                brand=ad["brand"],
                category=ad["category"],
                description=ad["description"],
                url=ad["url"],
                price=ad.get("price"),
                ci_text=ci_text,
                match_score=score,
            )
        )

    return matched
