"""Semantic ad matching using cosine similarity (Part 2 of the AD recommender pipeline)."""

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class MatchedAd:
    """An ad that matched the response above the similarity threshold."""
    product_id: str
    title: str
    brand: str
    category: str
    description: str
    url: str
    price: Optional[float]
    ci_text: str
    match_score: float


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings weighted by the attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed(text: str, model, tokenizer, device: torch.device) -> torch.Tensor:
    """Encode a string into a single embedding vector via mean-pooled hidden states."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]
    return _mean_pool(last_hidden, attention_mask)


def match_ads(
    response: str,
    ci_results,
    ci_ad_index_path: str,
    model,
    tokenizer,
    device: torch.device,
    threshold: float = 0.70,
    max_ads: int = 2,
) -> list[MatchedAd]:
    """
    Match ads to the LLM response via semantic cosine similarity.

    For each CI produced by constrained decoding:
    1. Compute cosine similarity between the CI embedding and the response embedding
    2. Keep CIs above `threshold`
    3. Look up ads for matching CIs from the CI-Ad index
    4. Deduplicate by product_id, cap results at `max_ads`

    Args:
        response: The vanilla LLM response text (Part 1 output)
        ci_results: List of CIResult objects from constrained decoding (Part 2)
        ci_ad_index_path: Path to ci_ad_index.json
        model: The loaded language model
        tokenizer: The corresponding tokenizer
        device: Torch device
        threshold: Minimum cosine similarity [0, 1] to include an ad (default 0.70)
        max_ads: Maximum number of ads to return (default 2)

    Returns:
        List of MatchedAd objects sorted by match_score descending
    """
    with open(ci_ad_index_path) as f:
        ci_ad_index = json.load(f)

    response_emb = _embed(response, model, tokenizer, device)

    scored_cis: list[tuple[str, float]] = []
    for ci in ci_results:
        ci_emb = _embed(ci.ci_text, model, tokenizer, device)
        score = F.cosine_similarity(response_emb, ci_emb, dim=1).item()
        if score >= threshold:
            scored_cis.append((ci.ci_text, score))

    scored_cis.sort(key=lambda x: x[1], reverse=True)

    seen_product_ids: set[str] = set()
    matched_ads: list[MatchedAd] = []

    for ci_text, score in scored_cis:
        for ad in ci_ad_index.get(ci_text, []):
            pid = ad["product_id"]
            if pid in seen_product_ids:
                continue
            seen_product_ids.add(pid)
            matched_ads.append(
                MatchedAd(
                    product_id=pid,
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
            if len(matched_ads) >= max_ads:
                return matched_ads

    return matched_ads
