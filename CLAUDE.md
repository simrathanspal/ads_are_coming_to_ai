# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a capstone implementation of the **RARE framework** (arXiv:2504.01304) — a system that injects ads into AI responses by detecting *Commercial Intents* (CIs) in user queries via constrained beam search decoding.

The pipeline: User query → LLM with constrained decoding → CI (e.g., "waterproof hiking boots") → Ad lookup via CI-Ad index → Ad injected into response.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild all data indexes from source (ad_library.json + commercial_intents.json)
python src/index_builder.py

# Validate Phase 1 outputs (indexes + trie correctness)
python scripts/validate_phase1.py

# Run a full experiment with MLflow tracking (downloads Qwen2.5-0.5B on first run)
python scripts/run_experiment.py
python scripts/run_experiment.py --config configs/aggressive.yaml
python scripts/run_experiment.py --queries "best running shoes" "gift for dad"
python scripts/run_experiment.py --name "my_run" --tag purpose=baseline

# View MLflow experiment results
mlflow ui   # then open http://127.0.0.1:5000
```

## Architecture

### Data flow

```
data/ad_library.json          ← product catalog (products with IDs, titles, descriptions)
data/commercial_intents.json  ← curated CI → product_id mapping
      ↓  (src/index_builder.py)
data/ci_list.json             ← flat list of all CI strings (input to trie)
data/ci_ad_index.json         ← forward index: CI text → list of ad records
data/ad_ci_index.json         ← reverse index: product_id → list of CI texts
```

### Core modules

**`src/trie.py` — `TokenTrie` / `TrieNode`**
Token-level trie built from the LLM's tokenizer. Each CI is tokenized (no special tokens) and inserted as a path of token IDs. During decoding, `get_valid_tokens(node)` returns only the token IDs that are valid next steps — all others are masked to `-inf` in logits.

**`src/constrained_decoder.py` — `ConstrainedBeamSearchDecoder`**
The core RARE decoding loop. Three key mechanisms applied at each step:
1. Trie constraint: mask logits of all tokens not in trie to `-inf`
2. Temperature scaling + truncation: drop tokens with probability < `truncation_threshold` (RARE's key innovation)
3. Beam search: keep top-`beam_width` candidates across all active beams

`DecoderConfig` can be loaded from a YAML file. The model defaults to `Qwen/Qwen2.5-0.5B-Instruct` and auto-selects MPS → CUDA → CPU.

Usage pattern:
```python
decoder = ConstrainedBeamSearchDecoder(DecoderConfig.from_yaml("configs/default.yaml"))
decoder.load_model()
decoder.build_trie("data/ci_list.json")
results = decoder.generate("hiking boots for wet weather")
ads = decoder.lookup_ads(results, "data/ci_ad_index.json")
```

**`src/experiment_tracker.py` — `ExperimentTracker`**
Wraps MLflow. Logs `DecoderConfig` fields as parameters, per-query metrics (latency, num CIs, trie verification), and aggregate metrics on `end_run()`. Results JSON saved as MLflow artifact.

**`src/index_builder.py`**
Builds all indexes from raw data. Run directly (`python src/index_builder.py`) whenever `ad_library.json` or `commercial_intents.json` changes.

### Configs

Three YAML profiles in `configs/`: `default.yaml`, `aggressive.yaml`, `conservative.yaml`. All fields map 1:1 to `DecoderConfig` dataclass fields. The key tuning knobs are `beam_width`, `truncation_threshold`, and `temperature`.

### Important constraints

- The trie must be rebuilt with the **same tokenizer** used during decoding. `validate_phase1.py` uses `gpt2` for fast validation; `run_experiment.py` uses the Qwen2.5 tokenizer. Trie is not serialized — it's rebuilt in memory each run.
- All scripts add the project root to `sys.path`, so `src.*` imports work from any script location.
- MLflow logs to `./mlruns/` by default (gitignored).

## Add unit tests
- Whenever you add any changes add unit tests.
- Always make sure the tests are passing before committing.

## Pull Request Guidelines

Every PR must be written as a **self-contained mini tutorial** for a reviewer who has never seen this codebase. Assume the reviewer is a capable engineer but has zero context on the RARE framework, constrained decoding, or how this repo works. The goal is that after reading the PR, the reviewer fully understands what was changed, why it was changed, and can give meaningful feedback without needing to ask clarifying questions.

### Required PR structure

**1. Background & Motivation**
Open with a plain-English explanation of the relevant concept being touched. Derive the context from the current state of the code and CLAUDE.md — do not hardcode examples or diagrams that can go stale. Do not assume the reviewer knows domain-specific terms — define any non-obvious concept inline.

**2. What problem this PR solves**
One short paragraph: what was broken, missing, or suboptimal before this change, and why it matters.

**3. Concept primer (if introducing new mechanics)**
If the PR touches a non-trivial mechanism, include a short explanation of how that mechanism works. Use a minimal, generic code snippet to illustrate the concept — do not copy snippets directly from the codebase, as those will go stale. The snippet should be the simplest possible illustration of the idea, not a reproduction of the actual implementation.

**4. Changes made**
A bulleted list of every file changed with a one-line description of what changed and why. Include short before/after code snippets for non-trivial logic changes.

**5. How to test / verify**
Exact commands the reviewer can run to verify the change works correctly. Reference the relevant validation scripts:
- `python scripts/validate_phase1.py` — verifies trie and index integrity
- `python scripts/run_experiment.py --queries "..."` — runs the full pipeline end to end
- `mlflow ui` — inspect logged metrics

**6. Reviewer focus areas**
Call out explicitly what you want the reviewer to scrutinize — e.g., "Does the truncation threshold logic in step E of `generate()` handle edge cases correctly?" This helps the reviewer know where to spend attention.

### Tone and style
- Write in plain English. Avoid jargon without definition.
- Prefer concrete examples over abstract descriptions.
- Keep code snippets short and focused — they illustrate a point, not reproduce the full diff.
- The PR body should be long enough that a reviewer can approve confidently without reading the full source diff.