"""
Constrained Beam Search Decoder for Commercial Intent Generation

This is the core engine of the RARE framework implementation.

Given a user query, this module generates Commercial Intents (CIs) using an LLM
with constrained beam search. The trie constraint guarantees that every generated
CI exists in the pre-built CI-Ad index.

Three key mechanisms:
1. TRIE CONSTRAINT — At each decoding step, mask all tokens not in trie to -inf
2. BEAM SEARCH — Explore top-K partial hypotheses in parallel
3. TRUNCATION — Discard valid-but-low-probability tokens before beam expansion

Reference: RARE paper (arXiv:2504.01304), Section 3.4
  "We employ a constrained beam search algorithm for generating commercial
   intentions (CIs), ensuring that the model's outputs are confined to a
   predefined CIs set."
"""

import json
import logging
import time
import torch
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.trie import TokenTrie, TrieNode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    """
    Hyperparameters for constrained beam search.

    Defaults are tuned for our capstone (189 CIs, Qwen2.5-0.5B).
    RARE paper values noted where applicable.
    """
    # Beam search parameters
    beam_width: int = 10          # RARE online uses 50, we use 10 (189 CIs not millions)
    top_n: int = 5                # Number of CIs to return
    max_length: int = 7           # Max trie depth is 7 tokens

    # Decoding parameters
    temperature: float = 0.7      # RARE online uses 0.7
    truncation_threshold: float = 0.01   # Discard tokens with < 1% probability

    # Length normalization: score = log_prob_sum / (length ^ length_penalty)
    # Prevents bias toward shorter CIs
    length_penalty: float = 1.0

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    @classmethod
    def from_yaml(cls, path: str) -> "DecoderConfig":
        """
        Load config from a YAML file.

        Args:
            path: Path to a YAML config file (e.g., configs/default.yaml)

        Returns:
            DecoderConfig populated from the YAML values
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Beam Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class BeamHypothesis:
    """
    One active beam during search.

    Tracks the generated token sequence, current trie position,
    and cumulative log-probability score.
    """
    token_ids: list[int] = field(default_factory=list)
    trie_node: TrieNode = None
    score: float = 0.0   # Cumulative log-probability

    def normalized_score(self, length_penalty: float = 1.0) -> float:
        """Length-normalized score to avoid short-CI bias."""
        length = max(len(self.token_ids), 1)
        return self.score / (length ** length_penalty)


# ---------------------------------------------------------------------------
# Completed CI result
# ---------------------------------------------------------------------------

@dataclass
class CIResult:
    """A completed Commercial Intent with its score."""
    ci_text: str
    score: float                  # Raw cumulative log-probability
    normalized_score: float       # Length-normalized score
    confidence: float             # Converted to 0-1 scale
    token_ids: list[int]
    token_count: int


# ---------------------------------------------------------------------------
# Constrained Beam Search Decoder
# ---------------------------------------------------------------------------

class ConstrainedBeamSearchDecoder:
    """
    Generates Commercial Intents using constrained beam search over a token trie.

    Usage:
        decoder = ConstrainedBeamSearchDecoder(config)
        decoder.load_model()
        decoder.build_trie("data/ci_list.json")
        results = decoder.generate("hiking boots for wet weather")
        for r in results:
            print(f"  {r.ci_text} (confidence={r.confidence:.2%})")
    """

    def __init__(self, config: DecoderConfig = None):
        self.config = config or DecoderConfig()
        self.model = None
        self.tokenizer = None
        self.trie = None
        self.device = None

    def load_model(self):
        """Load the LLM and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        print(f"Model loaded on {self.device} ({self._get_model_memory_mb():.0f} MB)")

    def build_trie(self, ci_list_path: str):
        """Build the token trie from ci_list.json."""
        assert self.tokenizer is not None, "Load model first (need tokenizer for trie)"

        self.trie = TokenTrie(self.tokenizer)
        tokenizations = self.trie.build_from_file(ci_list_path)

        stats = self.trie.stats()
        print(f"Trie built: {stats['total_cis']} CIs, "
              f"{stats['total_nodes']} nodes, "
              f"max_depth={stats['max_depth']}")
        return tokenizations

    # -------------------------------------------------------------------
    # Main generation method
    # -------------------------------------------------------------------

    def generate(self, query: str, top_n: int = None, verbose: bool = False) -> list[CIResult]:
        """
        Generate top-N Commercial Intents for a user query.

        Args:
            query: The user's search query / question
            top_n: Number of CIs to return (default: config.top_n)
            verbose: Print step-by-step decoding details

        Returns:
            List of CIResult objects, sorted by confidence (highest first)
        """
        assert self.model is not None, "Call load_model() first"
        assert self.trie is not None, "Call build_trie() first"

        top_n = top_n or self.config.top_n
        start_time = time.time()

        # 1. Build and tokenize prompt
        prompt_ids = self._build_prompt_ids(query)
        if verbose:
            print(f"\nQuery: '{query}'")
            print(f"Prompt tokens: {len(prompt_ids)}")

        # 2. Initialize with a single beam at the trie root
        initial_beam = BeamHypothesis(
            token_ids=[],
            trie_node=self.trie.root,
            score=0.0
        )
        active_beams = [initial_beam]
        completed_cis: list[tuple[str, float, list[int]]] = []  # (ci_text, score, token_ids)

        # 3. Decode step by step
        for step in range(self.config.max_length):
            if not active_beams:
                break

            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(f"  Active beams: {len(active_beams)}")

            # Build batch input: prompt + beam tokens for each beam
            batch_input_ids = []
            valid_beam_indices = []

            for i, beam in enumerate(active_beams):
                # Check if this beam has valid continuations in the trie
                valid_tokens = self.trie.get_valid_tokens(beam.trie_node)
                if not valid_tokens:
                    continue  # Dead beam — no continuations
                full_ids = prompt_ids + beam.token_ids
                batch_input_ids.append(full_ids)
                valid_beam_indices.append(i)

            if not batch_input_ids:
                break  # All beams are dead

            # Forward pass — batch all beams together
            batch_tensor = torch.tensor(batch_input_ids, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=batch_tensor)

            # Get logits for the next token (last position)
            # Shape: [batch_size, vocab_size]
            next_logits = outputs.logits[:, -1, :].float()  # float32 for numerical stability

            # Collect all candidates across all beams
            all_candidates = []

            for batch_idx, beam_idx in enumerate(valid_beam_indices):
                beam = active_beams[beam_idx]
                logits = next_logits[batch_idx]  # [vocab_size]

                # STEP A: Get valid tokens from trie
                valid_token_ids = self.trie.get_valid_tokens(beam.trie_node)

                # STEP B: Apply trie constraint — mask everything else to -inf
                constrained_logits = torch.full_like(logits, float('-inf'))
                constrained_logits[valid_token_ids] = logits[valid_token_ids]

                # STEP C: Apply temperature
                scaled_logits = constrained_logits / self.config.temperature

                # STEP D: Softmax to get probabilities
                probs = torch.softmax(scaled_logits, dim=-1)

                # STEP E: TRUNCATION — discard low-confidence tokens
                # This is RARE's key innovation
                for tid in valid_token_ids:
                    prob_val = probs[tid].item()
                    if prob_val < self.config.truncation_threshold:
                        # Kill this token — too low confidence
                        continue
                    else:
                        # Surviving token — compute beam score
                        log_prob = torch.log(probs[tid]).item()
                        candidate_score = beam.score + log_prob
                        all_candidates.append((beam, tid, candidate_score))

                        if verbose and prob_val > 0.05:
                            token_text = self.tokenizer.decode([tid])
                            print(f"  Beam '{self.tokenizer.decode(beam.token_ids)}' → "
                                  f"'{token_text}' (p={prob_val:.3f})")

            if not all_candidates:
                break  # Nothing survived truncation

            # STEP F: Select top beam_width candidates across ALL beams
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            selected = all_candidates[:self.config.beam_width]

            # STEP G: Create new beams, check for completions
            new_active = []
            for parent_beam, tid, score in selected:
                next_node = self.trie.step(parent_beam.trie_node, tid)
                new_token_ids = parent_beam.token_ids + [tid]

                if self.trie.is_complete(next_node):
                    # This beam produced a complete CI!
                    ci_text = self.trie.get_ci_text(next_node)
                    completed_cis.append((ci_text, score, new_token_ids))

                    if verbose:
                        print(f"  ✓ COMPLETED: '{ci_text}' (score={score:.3f})")

                    # If this node also has children, keep exploring
                    # (a CI might be a prefix of another CI)
                    if self.trie.has_continuations(next_node):
                        new_beam = BeamHypothesis(
                            token_ids=new_token_ids,
                            trie_node=next_node,
                            score=score
                        )
                        new_active.append(new_beam)
                else:
                    # Incomplete — keep going
                    new_beam = BeamHypothesis(
                        token_ids=new_token_ids,
                        trie_node=next_node,
                        score=score
                    )
                    new_active.append(new_beam)

            active_beams = new_active

            # STEP H: Early stopping
            if len(completed_cis) >= top_n and active_beams:
                # Check if any active beam can beat the worst completed CI
                best_active = max(b.score for b in active_beams)
                sorted_completed = sorted(completed_cis, key=lambda x: x[1], reverse=True)
                if len(sorted_completed) >= top_n:
                    worst_in_top_n = sorted_completed[top_n - 1][1]
                    if best_active < worst_in_top_n:
                        if verbose:
                            print(f"\n  Early stop: best active ({best_active:.3f}) < "
                                  f"worst top-{top_n} ({worst_in_top_n:.3f})")
                        break

        # 4. Deduplicate, normalize scores, and sort
        elapsed = time.time() - start_time
        results = self._build_results(completed_cis)

        if verbose:
            print(f"\nCompleted in {elapsed:.2f}s — {len(results)} CIs generated")

        return results[:top_n]

    # -------------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------------

    def _build_prompt_ids(self, query: str) -> list[int]:
        """
        Build the prompt that instructs the LLM to generate a CI.

        Since we don't have a fine-tuned model (unlike RARE's Hunyuan),
        the prompt does the heavy lifting of steering the model toward
        generating commercial-intent-like tokens.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate short commercial intent phrases. "
                    "A commercial intent is a 3-5 word phrase describing "
                    "what product or service a user wants to buy. "
                    "Output only the intent phrase, nothing else."
                )
            },
            {
                "role": "user",
                "content": f"Generate the commercial intent for: {query}"
            }
        ]

        # Use the model's chat template for proper formatting
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_ids

    # -------------------------------------------------------------------
    # Result building
    # -------------------------------------------------------------------

    def _build_results(self, completed_cis: list[tuple[str, float, list[int]]]) -> list[CIResult]:
        """Convert raw completed CIs into sorted, deduplicated CIResult objects."""
        # Deduplicate — keep the highest score for each CI
        best_scores: dict[str, tuple[float, list[int]]] = {}
        for ci_text, score, token_ids in completed_cis:
            if ci_text not in best_scores or score > best_scores[ci_text][0]:
                best_scores[ci_text] = (score, token_ids)

        # Build CIResult objects with normalized scores
        results = []
        for ci_text, (score, token_ids) in best_scores.items():
            length = len(token_ids)
            norm_score = score / (length ** self.config.length_penalty)

            results.append(CIResult(
                ci_text=ci_text,
                score=score,
                normalized_score=norm_score,
                confidence=0.0,  # Will be computed below
                token_ids=token_ids,
                token_count=length
            ))

        # Sort by normalized score (highest first)
        results.sort(key=lambda r: r.normalized_score, reverse=True)

        # Convert scores to confidence (softmax over normalized scores)
        if results:
            scores_tensor = torch.tensor([r.normalized_score for r in results])
            confidences = torch.softmax(scores_tensor, dim=0)
            for i, r in enumerate(results):
                r.confidence = confidences[i].item()

        return results

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    def _get_model_memory_mb(self) -> float:
        """Estimate model memory usage in MB."""
        if self.model is None:
            return 0
        total_params = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_params / (1024 * 1024)

    def lookup_ads(self, ci_results: list[CIResult], ci_ad_index_path: str) -> dict:
        """
        Look up ads for generated CIs using the CI-Ad index.

        Returns a dict mapping CI text → list of matching ads.
        """
        with open(ci_ad_index_path) as f:
            ci_ad_index = json.load(f)

        results = {}
        for ci in ci_results:
            if ci.ci_text in ci_ad_index:
                results[ci.ci_text] = ci_ad_index[ci.ci_text]
            else:
                results[ci.ci_text] = []
        return results
