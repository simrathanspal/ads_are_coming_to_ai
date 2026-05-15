"""
Index Builder for CI-Ad System

Builds two core indexes from the curated data:
1. CI → Ad Index (forward): Given a CI, which ads match?
2. Ad → CI Index (reverse): Given an ad, which CIs describe it?

Also generates summary statistics useful for validation.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def load_ad_library(path: str = None) -> dict:
    """Load the product catalog."""
    path = path or str(DATA_DIR / "ad_library.json")
    with open(path, "r") as f:
        data = json.load(f)
    # Build lookup by product_id
    return {p["product_id"]: p for p in data["products"]}


def load_commercial_intents(path: str = None) -> dict:
    """Load the curated CI → product_ids mapping."""
    path = path or str(DATA_DIR / "commercial_intents.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["commercial_intents"]


def build_ci_ad_index(ci_to_products: dict, ad_library: dict) -> dict:
    """
    Build the forward index: CI text → list of full ad records.

    This is the primary index used during ad retrieval:
    after constrained decoding produces CIs, we look up ads here.
    """
    ci_ad_index = {}
    for ci_text, product_ids in ci_to_products.items():
        ads = []
        for pid in product_ids:
            if pid in ad_library:
                ads.append({
                    "product_id": pid,
                    "title": ad_library[pid]["title"],
                    "brand": ad_library[pid]["brand"],
                    "category": ad_library[pid]["category"],
                    "description": ad_library[pid]["description"],
                    "url": ad_library[pid]["url"],
                    "price": ad_library[pid].get("price")
                })
            else:
                print(f"  WARNING: product_id '{pid}' in CI '{ci_text}' not found in ad library")
        ci_ad_index[ci_text] = ads
    return ci_ad_index


def build_ad_ci_index(ci_to_products: dict) -> dict:
    """
    Build the reverse index: product_id → list of CI texts.

    Useful for:
    - Validating coverage (every product should have >= 1 CI)
    - Understanding which intents map to which product
    """
    ad_ci_index = defaultdict(list)
    for ci_text, product_ids in ci_to_products.items():
        for pid in product_ids:
            ad_ci_index[pid].append(ci_text)
    return dict(ad_ci_index)


def compute_statistics(ci_to_products: dict, ad_ci_index: dict, ad_library: dict) -> dict:
    """Compute summary statistics about the index."""
    ci_texts = list(ci_to_products.keys())

    # CI statistics
    ci_word_lengths = [len(ci.split()) for ci in ci_texts]
    ci_char_lengths = [len(ci) for ci in ci_texts]
    ads_per_ci = [len(pids) for pids in ci_to_products.values()]

    # Product coverage
    cis_per_product = [len(cis) for cis in ad_ci_index.values()]
    products_with_cis = set(ad_ci_index.keys())
    all_products = set(ad_library.keys())
    uncovered_products = all_products - products_with_cis

    # Category distribution
    category_ci_count = defaultdict(int)
    for ci_text, product_ids in ci_to_products.items():
        categories = set()
        for pid in product_ids:
            if pid in ad_library:
                categories.add(ad_library[pid]["category"])
        for cat in categories:
            category_ci_count[cat] += 1

    stats = {
        "total_cis": len(ci_texts),
        "total_products": len(all_products),
        "products_with_cis": len(products_with_cis),
        "uncovered_products": sorted(uncovered_products),
        "ci_word_length": {
            "min": min(ci_word_lengths),
            "max": max(ci_word_lengths),
            "mean": round(sum(ci_word_lengths) / len(ci_word_lengths), 2)
        },
        "ci_char_length": {
            "min": min(ci_char_lengths),
            "max": max(ci_char_lengths),
            "mean": round(sum(ci_char_lengths) / len(ci_char_lengths), 2)
        },
        "ads_per_ci": {
            "min": min(ads_per_ci),
            "max": max(ads_per_ci),
            "mean": round(sum(ads_per_ci) / len(ads_per_ci), 2)
        },
        "cis_per_product": {
            "min": min(cis_per_product),
            "max": max(cis_per_product),
            "mean": round(sum(cis_per_product) / len(cis_per_product), 2)
        },
        "category_distribution": dict(sorted(category_ci_count.items(), key=lambda x: -x[1]))
    }
    return stats


def build_all_indexes(output_dir: str = None):
    """Build all indexes and save to disk."""
    output_dir = output_dir or str(DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CI-Ad Index Builder")
    print("=" * 60)

    # Load source data
    print("\n1. Loading ad library...")
    ad_library = load_ad_library()
    print(f"   Loaded {len(ad_library)} products")

    print("\n2. Loading commercial intents...")
    ci_to_products = load_commercial_intents()
    print(f"   Loaded {len(ci_to_products)} commercial intents")

    # Build indexes
    print("\n3. Building CI → Ad index (forward)...")
    ci_ad_index = build_ci_ad_index(ci_to_products, ad_library)
    print(f"   Built index with {len(ci_ad_index)} entries")

    print("\n4. Building Ad → CI index (reverse)...")
    ad_ci_index = build_ad_ci_index(ci_to_products)
    print(f"   Built index with {len(ad_ci_index)} entries")

    # Statistics
    print("\n5. Computing statistics...")
    stats = compute_statistics(ci_to_products, ad_ci_index, ad_library)

    # Save outputs
    ci_ad_output = os.path.join(output_dir, "ci_ad_index.json")
    with open(ci_ad_output, "w") as f:
        json.dump(ci_ad_index, f, indent=2)
    print(f"\n   Saved CI → Ad index to: {ci_ad_output}")

    ad_ci_output = os.path.join(output_dir, "ad_ci_index.json")
    with open(ad_ci_output, "w") as f:
        json.dump(ad_ci_index, f, indent=2)
    print(f"   Saved Ad → CI index to: {ad_ci_output}")

    # Also save just the CI list (needed for trie construction)
    ci_list_output = os.path.join(output_dir, "ci_list.json")
    ci_list = sorted(ci_to_products.keys())
    with open(ci_list_output, "w") as f:
        json.dump({"total": len(ci_list), "commercial_intents": ci_list}, f, indent=2)
    print(f"   Saved CI list to: {ci_list_output}")

    # Print statistics
    print("\n" + "=" * 60)
    print("INDEX STATISTICS")
    print("=" * 60)
    print(f"\n  Total CIs:              {stats['total_cis']}")
    print(f"  Total Products:         {stats['total_products']}")
    print(f"  Products with CIs:      {stats['products_with_cis']}")
    print(f"\n  CI Word Length:         min={stats['ci_word_length']['min']}, "
          f"max={stats['ci_word_length']['max']}, "
          f"mean={stats['ci_word_length']['mean']}")
    print(f"  CI Char Length:         min={stats['ci_char_length']['min']}, "
          f"max={stats['ci_char_length']['max']}, "
          f"mean={stats['ci_char_length']['mean']}")
    print(f"\n  Ads per CI:             min={stats['ads_per_ci']['min']}, "
          f"max={stats['ads_per_ci']['max']}, "
          f"mean={stats['ads_per_ci']['mean']}")
    print(f"  CIs per Product:        min={stats['cis_per_product']['min']}, "
          f"max={stats['cis_per_product']['max']}, "
          f"mean={stats['cis_per_product']['mean']}")

    if stats['uncovered_products']:
        print(f"\n  ⚠ Uncovered products:   {stats['uncovered_products']}")

    print(f"\n  Category Distribution:")
    for cat, count in stats['category_distribution'].items():
        print(f"    {cat:20s} → {count} CIs")

    print("\n" + "=" * 60)
    print("Done! All indexes built successfully.")
    print("=" * 60)

    return ci_ad_index, ad_ci_index, stats


if __name__ == "__main__":
    build_all_indexes()
