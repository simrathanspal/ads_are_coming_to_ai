"""Tests for src/ad_matcher.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.ad_matcher import match_ads, MatchedAd
from src.constrained_decoder import CIResult


def _make_ci(text: str, confidence: float = 0.8) -> CIResult:
    return CIResult(
        ci_text=text,
        score=-1.0,
        normalized_score=-0.5,
        confidence=confidence,
        token_ids=[1, 2],
        token_count=2,
    )


def _write_index(tmp_path, entries: list) -> str:
    """
    Build a ci_ad_index.json from (ci_text, product_id, title) tuples.
    Returns the path to the written file.
    """
    index = {}
    for ci_text, product_id, title in entries:
        index[ci_text] = [
            {
                "product_id": product_id,
                "title": title,
                "brand": "TestBrand",
                "category": "test",
                "description": "A test product.",
                "url": "https://example.com",
                "price": None,
            }
        ]
    path = tmp_path / "ci_ad_index.json"
    path.write_text(json.dumps(index))
    return str(path)


class TestMatchAds:
    def test_empty_ci_results_returns_empty(self, tmp_path):
        path = _write_index(tmp_path, [])
        result = match_ads("some response", [], path, MagicMock(), MagicMock(), torch.device("cpu"))
        assert result == []

    def test_empty_response_returns_empty(self, tmp_path):
        path = _write_index(tmp_path, [("running shoes", "PROD-1", "Shoe")])
        result = match_ads("", [_make_ci("running shoes")], path, MagicMock(), MagicMock(), torch.device("cpu"))
        assert result == []

    def test_whitespace_only_response_returns_empty(self, tmp_path):
        path = _write_index(tmp_path, [("running shoes", "PROD-1", "Shoe")])
        result = match_ads("   ", [_make_ci("running shoes")], path, MagicMock(), MagicMock(), torch.device("cpu"))
        assert result == []

    @patch("src.ad_matcher._embed")
    def test_below_threshold_excluded(self, mock_embed, tmp_path):
        # Orthogonal vectors → cosine similarity = 0.0 < 0.70 → excluded
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),  # response
            torch.tensor([[0.0, 1.0]]),  # CI (orthogonal)
        ]
        path = _write_index(tmp_path, [("running shoes", "PROD-1", "Shoe")])
        result = match_ads(
            "test response",
            [_make_ci("running shoes")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            threshold=0.70,
        )
        assert result == []

    @patch("src.ad_matcher._embed")
    def test_above_threshold_included(self, mock_embed, tmp_path):
        # Identical vectors → cosine similarity = 1.0 > 0.70 → included
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),  # response
            torch.tensor([[1.0, 0.0]]),  # CI (identical)
        ]
        path = _write_index(tmp_path, [("running shoes", "PROD-1", "Running Shoe")])
        result = match_ads(
            "test response",
            [_make_ci("running shoes")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            threshold=0.70,
        )
        assert len(result) == 1
        assert result[0].product_id == "PROD-1"
        assert result[0].match_score == pytest.approx(1.0, abs=1e-5)

    @patch("src.ad_matcher._embed")
    def test_deduplication_keeps_highest_score(self, mock_embed, tmp_path):
        # Two CIs map to the same product — only the highest-scoring one survives
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),     # response
            torch.tensor([[1.0, 0.0]]),     # "running shoes" CI → sim 1.0
            torch.tensor([[0.9, 0.436]]),   # "marathon shoes" CI → sim ≈ 0.9
        ]
        path = _write_index(tmp_path, [
            ("running shoes", "PROD-1", "Running Shoe"),
            ("marathon shoes", "PROD-1", "Running Shoe"),
        ])
        result = match_ads(
            "test",
            [_make_ci("running shoes"), _make_ci("marathon shoes")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            threshold=0.70,
        )
        assert len(result) == 1
        assert result[0].product_id == "PROD-1"
        assert result[0].match_score == pytest.approx(1.0, abs=1e-5)

    @patch("src.ad_matcher._embed")
    def test_max_ads_cap(self, mock_embed, tmp_path):
        # Three distinct products all match — cap at max_ads=2
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),  # response
            torch.tensor([[1.0, 0.0]]),  # shoes CI
            torch.tensor([[1.0, 0.0]]),  # boots CI
            torch.tensor([[1.0, 0.0]]),  # sandals CI
        ]
        path = _write_index(tmp_path, [
            ("shoes", "PROD-1", "Shoe"),
            ("boots", "PROD-2", "Boot"),
            ("sandals", "PROD-3", "Sandal"),
        ])
        result = match_ads(
            "test",
            [_make_ci("shoes"), _make_ci("boots"), _make_ci("sandals")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            threshold=0.70,
            max_ads=2,
        )
        assert len(result) == 2

    @patch("src.ad_matcher._embed")
    def test_ci_not_in_index_skipped(self, mock_embed, tmp_path):
        # CI exists in results but not in the ad index — skip without error
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),  # response only — no CI embed needed
        ]
        path = _write_index(tmp_path, [])  # empty index
        result = match_ads(
            "test",
            [_make_ci("unknown commercial intent")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
        )
        assert result == []

    @patch("src.ad_matcher._embed")
    def test_results_sorted_by_match_score_descending(self, mock_embed, tmp_path):
        # Two products: boots has sim=1.0, shoes has sim≈0.9
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),    # response
            torch.tensor([[0.9, 0.436]]),  # shoes → sim ≈ 0.9
            torch.tensor([[1.0, 0.0]]),    # boots → sim = 1.0
        ]
        path = _write_index(tmp_path, [
            ("shoes", "PROD-1", "Shoe"),
            ("boots", "PROD-2", "Boot"),
        ])
        result = match_ads(
            "test",
            [_make_ci("shoes"), _make_ci("boots")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            threshold=0.70,
            max_ads=5,
        )
        assert len(result) == 2
        assert result[0].match_score >= result[1].match_score
        assert result[0].product_id == "PROD-2"  # boots first (higher score)

    @patch("src.ad_matcher._embed")
    def test_returns_matched_ad_dataclass(self, mock_embed, tmp_path):
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ]
        path = _write_index(tmp_path, [("shoes", "PROD-1", "Great Shoe")])
        result = match_ads(
            "test",
            [_make_ci("shoes")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
        )
        assert isinstance(result[0], MatchedAd)
        assert result[0].title == "Great Shoe"
        assert result[0].ci_text == "shoes"

    @patch("src.ad_matcher._embed")
    def test_exact_threshold_boundary_included(self, mock_embed, tmp_path):
        # Similarity exactly at threshold should be included (>= not >).
        # Use identical unit vectors (sim=1.0) with threshold=1.0 to avoid float32 issues.
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),  # response
            torch.tensor([[1.0, 0.0]]),  # CI (identical → sim = 1.0 = threshold)
        ]
        path = _write_index(tmp_path, [("shoes", "PROD-1", "Shoe")])
        result = match_ads(
            "test",
            [_make_ci("shoes")],
            path,
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            threshold=1.0,
        )
        assert len(result) == 1

    @patch("src.ad_matcher._embed")
    def test_multiple_ads_per_ci_all_considered(self, mock_embed, tmp_path):
        # One CI maps to two different products — both should appear (below max_ads)
        mock_embed.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ]
        index = {
            "shoes": [
                {"product_id": "PROD-1", "title": "Shoe A", "brand": "B1",
                 "category": "c", "description": "d", "url": "u", "price": None},
                {"product_id": "PROD-2", "title": "Shoe B", "brand": "B2",
                 "category": "c", "description": "d", "url": "u", "price": None},
            ]
        }
        path = tmp_path / "ci_ad_index.json"
        path.write_text(json.dumps(index))
        result = match_ads(
            "test",
            [_make_ci("shoes")],
            str(path),
            MagicMock(),
            MagicMock(),
            torch.device("cpu"),
            max_ads=5,
        )
        assert len(result) == 2
        product_ids = {ad.product_id for ad in result}
        assert "PROD-1" in product_ids
        assert "PROD-2" in product_ids
