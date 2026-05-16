"""
AD Recommender — Streamlit UI

Wires together the two-part pipeline:
  Part 1: Vanilla LLM response via generate_response()
  Part 2: Constrained beam search CI generation + semantic ad matching

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.constrained_decoder import ConstrainedBeamSearchDecoder, DecoderConfig
from src.response_generator import generate_response
from src.ad_matcher import match_ads

CI_LIST_PATH = "data/ci_list.json"
CI_AD_INDEX_PATH = "data/ci_ad_index.json"
PROFILES = ["default", "aggressive", "conservative"]


@st.cache_resource
def load_decoder(profile: str) -> ConstrainedBeamSearchDecoder:
    """Load and cache the decoder (model + trie) for a given config profile."""
    config = DecoderConfig.from_yaml(f"configs/{profile}.yaml")
    decoder = ConstrainedBeamSearchDecoder(config)
    decoder.load_model()
    decoder.build_trie(CI_LIST_PATH)
    return decoder


# ─── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Ad Recommender", layout="wide")
st.title("AI Response with Ad Recommendations")
st.caption("Powered by constrained beam search (RARE framework — arXiv:2504.01304)")

# ─── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    profile = st.selectbox("Decoder Profile", PROFILES, index=0)
    threshold = st.slider(
        "Ad Match Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        format="%.0%%",
        help="Minimum cosine similarity between a commercial intent and the response.",
    )
    max_ads = st.slider(
        "Max Ads per Response",
        min_value=1,
        max_value=5,
        value=2,
        help="Maximum number of sponsored ads shown.",
    )

# ─── Query input ──────────────────────────────────────────────────────────────

query = st.text_input(
    "Ask anything...",
    placeholder="e.g. best hiking boots for wet weather",
)

if not query:
    st.stop()

# ─── Pipeline execution ───────────────────────────────────────────────────────

decoder = load_decoder(profile)
model = decoder.model
tokenizer = decoder.tokenizer
device = decoder.device

with st.spinner("Generating response..."):
    response = generate_response(model, tokenizer, device, query)

with st.spinner("Detecting commercial intents..."):
    ci_results = decoder.generate(query)

with st.spinner("Matching ads..."):
    matched_ads = match_ads(
        model,
        tokenizer,
        device,
        response,
        ci_results,
        CI_AD_INDEX_PATH,
        threshold=threshold,
        max_ads=max_ads,
    )

# ─── Results display ──────────────────────────────────────────────────────────

st.subheader("Response")
st.write(response)

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Commercial Intents Detected")
    if ci_results:
        for ci in ci_results:
            st.markdown(f"- **{ci.ci_text}** — confidence: `{ci.confidence:.1%}`")
    else:
        st.info("No commercial intents detected for this query.")

with col_right:
    st.subheader("Sponsored")
    if matched_ads:
        for ad in matched_ads:
            with st.container(border=True):
                st.markdown(f"**[{ad.title}]({ad.url})**")
                st.caption(f"{ad.brand} · {ad.category} · match: `{ad.match_score:.0%}`")
                st.write(ad.description)
                if ad.price is not None:
                    st.markdown(f"**${ad.price:.2f}**")
    else:
        st.info("No relevant ads found above the match threshold.")
