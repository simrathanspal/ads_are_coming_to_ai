"""
Token-Level Trie for Constrained Beam Search

This is the core data structure that powers constrained decoding in the RARE framework.

Key idea: Every Commercial Intent (CI) is tokenized using the LLM's tokenizer and
inserted into a trie where each node represents a token ID. During beam search,
at each decoding step we look up the current trie node to find which token IDs
are valid continuations — all other tokens are masked to -inf in the logits.

This guarantees that the LLM can ONLY generate CIs that exist in our pre-built index.

Reference: RARE paper, Section 3.4 "Constrained Beam Search"
  "We employ a constrained beam search algorithm for generating commercial
   intentions (CIs), ensuring that the model's outputs are confined to a
   predefined CIs set."
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """A single node in the token-level trie."""
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False
    # Store the original CI text at leaf nodes for easy retrieval
    ci_text: Optional[str] = None
    # Depth in the trie (0 = root)
    depth: int = 0


class TokenTrie:
    """
    Token-level trie over Commercial Intents.

    Each CI string is tokenized into a sequence of token IDs, and the sequence
    is inserted as a path from root to a leaf. During constrained beam search,
    we traverse this trie in lockstep with the decoding process.

    Usage:
        trie = TokenTrie(tokenizer)
        trie.insert("waterproof hiking boots")
        trie.insert("best noise cancelling headphones")

        # During decoding, get valid next tokens:
        node = trie.root
        valid_tokens = trie.get_valid_tokens(node)  # All first tokens
        # After choosing token 'waterproof':
        next_node = trie.step(node, token_id_for_waterproof)
        valid_tokens = trie.get_valid_tokens(next_node)  # tokens after 'waterproof'
    """

    def __init__(self, tokenizer=None):
        """
        Initialize the trie.

        Args:
            tokenizer: A HuggingFace tokenizer instance. If None, must be set
                       before calling insert() or build_from_file().
        """
        self.root = TrieNode(depth=0)
        self.tokenizer = tokenizer
        self.size = 0  # Number of CIs inserted
        self._max_depth = 0  # Longest CI in tokens

    def set_tokenizer(self, tokenizer):
        """Set or replace the tokenizer."""
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> list[int]:
        """
        Tokenize a CI string into token IDs.

        We do NOT add special tokens (BOS/EOS) because these are generated
        mid-sequence during constrained decoding — the CI tokens appear
        after the prompt, not as standalone sequences.
        """
        assert self.tokenizer is not None, "Tokenizer must be set before tokenization"
        return self.tokenizer.encode(text, add_special_tokens=False)

    def insert(self, ci_text: str) -> list[int]:
        """
        Insert a single CI into the trie.

        Args:
            ci_text: The commercial intent text (e.g., "waterproof hiking boots")

        Returns:
            The token IDs that represent this CI
        """
        token_ids = self._tokenize(ci_text)

        if len(token_ids) == 0:
            return token_ids

        node = self.root
        for depth, tid in enumerate(token_ids, start=1):
            if tid not in node.children:
                node.children[tid] = TrieNode(depth=depth)
            node = node.children[tid]

        node.is_end = True
        node.ci_text = ci_text
        self.size += 1
        self._max_depth = max(self._max_depth, len(token_ids))

        return token_ids

    def insert_many(self, ci_texts: list[str]) -> dict[str, list[int]]:
        """
        Insert multiple CIs into the trie.

        Returns:
            Dict mapping CI text → token IDs
        """
        tokenizations = {}
        for ci_text in ci_texts:
            token_ids = self.insert(ci_text)
            tokenizations[ci_text] = token_ids
        return tokenizations

    def get_valid_tokens(self, node: TrieNode = None) -> list[int]:
        """
        Get all valid next token IDs from a given node.

        This is called at each decoding step to build the logit mask.
        Only these token IDs will have non-negative logits; all others
        are set to -inf.

        Args:
            node: Current trie node (default: root)

        Returns:
            List of valid token IDs
        """
        if node is None:
            node = self.root
        return list(node.children.keys())

    def step(self, node: TrieNode, token_id: int) -> Optional[TrieNode]:
        """
        Advance one step in the trie by following a token edge.

        Args:
            node: Current trie node
            token_id: The chosen token ID

        Returns:
            The child node, or None if this token_id is not a valid continuation
        """
        return node.children.get(token_id, None)

    def is_complete(self, node: TrieNode) -> bool:
        """Check if a node marks the end of a complete CI."""
        return node.is_end

    def get_ci_text(self, node: TrieNode) -> Optional[str]:
        """Get the CI text at a leaf node."""
        return node.ci_text if node.is_end else None

    def has_continuations(self, node: TrieNode) -> bool:
        """Check if there are more tokens possible after this node."""
        return len(node.children) > 0

    def search(self, ci_text: str) -> bool:
        """Check if a CI exists in the trie."""
        token_ids = self._tokenize(ci_text)
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                return False
            node = node.children[tid]
        return node.is_end

    def prefix_search(self, partial_text: str) -> list[str]:
        """
        Find all CIs that start with the given partial text.

        Useful for debugging and validation.
        """
        token_ids = self._tokenize(partial_text)
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                return []
            node = node.children[tid]

        # DFS to collect all complete CIs from this node
        results = []
        self._collect_completions(node, results)
        return results

    def _collect_completions(self, node: TrieNode, results: list[str]):
        """Recursively collect all CIs reachable from a node."""
        if node.is_end:
            results.append(node.ci_text)
        for child in node.children.values():
            self._collect_completions(child, results)

    def get_all_cis(self) -> list[str]:
        """Return all CIs stored in the trie."""
        results = []
        self._collect_completions(self.root, results)
        return results

    def build_from_file(self, ci_list_path: str) -> dict[str, list[int]]:
        """
        Build the trie from a ci_list.json file.

        Args:
            ci_list_path: Path to ci_list.json (output of index_builder.py)

        Returns:
            Dict mapping CI text → token IDs
        """
        with open(ci_list_path, "r") as f:
            data = json.load(f)
        ci_texts = data["commercial_intents"]
        return self.insert_many(ci_texts)

    def stats(self) -> dict:
        """Return statistics about the trie."""
        node_count = self._count_nodes(self.root)
        return {
            "total_cis": self.size,
            "total_nodes": node_count,
            "max_depth": self._max_depth,
            "branching_factor_root": len(self.root.children),
        }

    def _count_nodes(self, node: TrieNode) -> int:
        """Count total nodes in the trie."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def visualize(self, max_depth: int = 3, node: TrieNode = None, prefix: str = "", is_last: bool = True) -> str:
        """
        Create a text visualization of the trie (for debugging).

        Args:
            max_depth: Maximum depth to display
            node: Starting node (default: root)
            prefix: Indentation prefix
            is_last: Whether this is the last child

        Returns:
            String representation of the trie
        """
        if node is None:
            node = self.root
            lines = ["[ROOT]"]
        else:
            lines = []

        if node.depth >= max_depth:
            if node.children:
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}... ({len(node.children)} more branches)")
            return "\n".join(lines)

        children = list(node.children.items())
        for i, (token_id, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            connector = "└── " if is_last_child else "├── "

            token_text = self.tokenizer.decode([token_id]) if self.tokenizer else str(token_id)
            end_marker = " ✓" if child.is_end else ""
            ci_marker = f' → "{child.ci_text}"' if child.is_end else ""

            lines.append(f"{prefix}{connector}[{token_id}] '{token_text}'{end_marker}{ci_marker}")

            extension = "    " if is_last_child else "│   "
            subtree = self.visualize(
                max_depth=max_depth,
                node=child,
                prefix=prefix + extension,
                is_last=is_last_child
            )
            if subtree:
                lines.append(subtree)

        return "\n".join(lines)

    def __len__(self):
        return self.size

    def __contains__(self, ci_text: str):
        return self.search(ci_text)

    def __repr__(self):
        return f"TokenTrie(size={self.size}, max_depth={self._max_depth})"
