"""Unit tests for src/ad_matcher.py"""

import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from src.ad_matcher import match_ads, MatchedAd, _mean_pool


# ─── Helpers ─────────────────────────────────────────────────────────────────


@dataclass
class FakeCIResult:
    ci_text: str
    confidence: float = 0.9


def _write_index(tmp_path, index: dict) -> str:
    p = tmp_path / "ci_ad_index.json"
    p.write_text(json.dumps(index))
    return str(p)


def _make_ads(product_ids: list[str]) -> list[dict]:
    return [
        {
            "product_id": pid,
            "title": f"Title {pid}",
            "brand": "Brand",
            "category": "cat",
            "description": "Desc",
            "url": f"http://{pid}.com",
            "price": None,
        }
        for pid in product_ids
    ]


def _high_sim_embeddings(n_cis: int) -> torch.Tensor:
    """Response + n_cis CI embeddings all pointing in the same direction (sim ≈ 1)."""
    embs = torch.ones(1 + n_cis, 4)
    return F.normalize(embs, p=2, dim=-1)


def _low_sim_embeddings() -> torch.Tensor:
    """Response orthogonal to a single CI (sim = 0)."""
    embs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    return F.normalize(embs, p=2, dim=-1)


# ─── _mean_pool ───────────────────────────────────────────────────────────────


def test_mean_pool_basic():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[1, 1]])
    result = _mean_pool(hidden, mask)
    assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])  # third token is padding
    result = _mean_pool(hidden, mask)
    assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))


def test_mean_pool_single_token():
    hidden = torch.tensor([[[5.0, 7.0]]])
    mask = torch.tensor([[1]])
    result = _mean_pool(hidden, mask)
    assert torch.allclose(result, torch.tensor([[5.0, 7.0]]))


# ─── MatchedAd dataclass ──────────────────────────────────────────────────────


def test_matched_ad_fields():
    ad = MatchedAd(
        product_id="X1",
        title="Boot",
        brand="Nike",
        category="outdoor",
        description="Great boots",
        url="http://example.com",
        price=99.99,
        ci_text="waterproof hiking boots",
        match_score=0.85,
    )
    assert ad.product_id == "X1"
    assert ad.match_score == 0.85
    assert ad.price == 99.99


def test_matched_ad_nullable_price():
    ad = MatchedAd(
        product_id="X2", title="T", brand="B", category="c",
        description="d", url="u", price=None, ci_text="ci", match_score=0.5,
    )
    assert ad.price is None


# ─── match_ads — threshold filtering ─────────────────────────────────────────


def test_match_ads_below_threshold_returns_empty(tmp_path):
    index = {"waterproof boots": _make_ads(["P1"])}
    idx_path = _write_index(tmp_path, index)

    ci_results = [FakeCIResult("waterproof boots")]
    model, tokenizer = MagicMock(), MagicMock()

    import src.ad_matcher as m
    with patch.object(m, "_embed", return_value=_low_sim_embeddings()):
        result = match_ads(model, tokenizer, "cpu", "response", ci_results, idx_path, threshold=0.70)

    assert result == []


def test_match_ads_above_threshold_returns_ad(tmp_path):
    index = {"waterproof boots": _make_ads(["P1"])}
    idx_path = _write_index(tmp_path, index)

    ci_results = [FakeCIResult("waterproof boots")]
    model, tokenizer = MagicMock(), MagicMock()

    import src.ad_matcher as m
    with patch.object(m, "_embed", return_value=_high_sim_embeddings(1)):
        result = match_ads(model, tokenizer, "cpu", "response", ci_results, idx_path, threshold=0.70)

    assert len(result) == 1
    assert result[0].product_id == "P1"


# ─── match_ads — deduplication ────────────────────────────────────────────────


def test_match_ads_deduplicates_by_product_id(tmp_path):
    # Two CIs both resolve to the same product
    index = {
        "hiking boots": _make_ads(["P1"]),
        "trail boots": _make_ads(["P1"]),
    }
    idx_path = _write_index(tmp_path, index)

    ci_results = [FakeCIResult("hiking boots"), FakeCIResult("trail boots")]
    model, tokenizer = MagicMock(), MagicMock()

    import src.ad_matcher as m
    with patch.object(m, "_embed", return_value=_high_sim_embeddings(2)):
        result = match_ads(model, tokenizer, "cpu", "response", ci_results, idx_path,
                           threshold=0.50, max_ads=5)

    product_ids = [ad.product_id for ad in result]
    assert len(product_ids) == len(set(product_ids))  # no duplicates


# ─── match_ads — max_ads cap ──────────────────────────────────────────────────


def test_match_ads_respects_max_ads(tmp_path):
    n = 5
    index = {f"intent {i}": _make_ads([f"P{i}"]) for i in range(n)}
    idx_path = _write_index(tmp_path, index)

    ci_results = [FakeCIResult(f"intent {i}") for i in range(n)]
    model, tokenizer = MagicMock(), MagicMock()

    import src.ad_matcher as m
    with patch.object(m, "_embed", return_value=_high_sim_embeddings(n)):
        result = match_ads(model, tokenizer, "cpu", "response", ci_results, idx_path,
                           threshold=0.50, max_ads=2)

    assert len(result) <= 2


# ─── match_ads — edge cases ───────────────────────────────────────────────────


def test_match_ads_empty_ci_list_returns_empty(tmp_path):
    idx_path = _write_index(tmp_path, {})
    model, tokenizer = MagicMock(), MagicMock()
    result = match_ads(model, tokenizer, "cpu", "response", [], idx_path)
    assert result == []


def test_match_ads_empty_response_returns_empty(tmp_path):
    index = {"hiking boots": _make_ads(["P1"])}
    idx_path = _write_index(tmp_path, index)
    ci_results = [FakeCIResult("hiking boots")]
    model, tokenizer = MagicMock(), MagicMock()
    result = match_ads(model, tokenizer, "cpu", "   ", ci_results, idx_path)
    assert result == []


def test_match_ads_ci_not_in_index_skipped(tmp_path):
    # CI exists in ci_results but not in ci_ad_index
    idx_path = _write_index(tmp_path, {})
    ci_results = [FakeCIResult("unknown intent")]
    model, tokenizer = MagicMock(), MagicMock()

    import src.ad_matcher as m
    with patch.object(m, "_embed", return_value=_high_sim_embeddings(1)):
        result = match_ads(model, tokenizer, "cpu", "response", ci_results, idx_path,
                           threshold=0.50)

    assert result == []


def test_match_ads_returns_matched_ad_type(tmp_path):
    index = {"running shoes": _make_ads(["S1"])}
    idx_path = _write_index(tmp_path, index)
    ci_results = [FakeCIResult("running shoes")]
    model, tokenizer = MagicMock(), MagicMock()

    import src.ad_matcher as m
    with patch.object(m, "_embed", return_value=_high_sim_embeddings(1)):
        result = match_ads(model, tokenizer, "cpu", "response", ci_results, idx_path,
                           threshold=0.50)

    assert len(result) == 1
    assert isinstance(result[0], MatchedAd)
    assert result[0].ci_text == "running shoes"
    assert 0.0 <= result[0].match_score <= 1.0
