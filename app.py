"""
AI Ad Recommender — Streamlit Interface

Two-part pipeline per query:
  Part 1 — Vanilla LLM response (Qwen2.5-0.5B, unconstrained generation)
  Part 2 — Commercial intent generation via constrained beam search,
            followed by semantic ad matching against the response

Run:
    streamlit run app.py
"""

import time
from pathlib import Path

import streamlit as st

from src.constrained_decoder import ConstrainedBeamSearchDecoder, DecoderConfig
from src.response_generator import generate_response
from src.ad_matcher import match_ads

# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Ad Recommender",
    page_icon="🎯",
    layout="wide",
)

# ── Sidebar — configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    config_profile = st.selectbox(
        "Decoder Profile",
        options=["default", "conservative", "aggressive"],
        index=0,
        help=(
            "**default** — beam_width=10, temp=0.7  \n"
            "**conservative** — beam_width=5, temp=0.9 (faster)  \n"
            "**aggressive** — beam_width=20, temp=0.5 (best quality, slower)"
        ),
    )

    st.divider()

    match_threshold_pct = st.slider(
        "Ad Match Threshold",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        format="%d%%",
        help=(
            "Minimum semantic similarity (cosine) between a commercial intent "
            "and the response for its ad to be shown. Lower = more ads shown."
        ),
    )
    match_threshold = match_threshold_pct / 100.0

    max_ads = st.number_input(
        "Max Ads per Response",
        min_value=1,
        max_value=5,
        value=2,
        help="Maximum number of sponsored results displayed per query.",
    )

    st.divider()
    st.caption("Model: Qwen/Qwen2.5-0.5B-Instruct")
    st.caption("Framework: RARE (arXiv:2504.01304)")

# ── Model loader (cached across reruns) ───────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_decoder(profile: str) -> ConstrainedBeamSearchDecoder:
    config = DecoderConfig.from_yaml(f"configs/{profile}.yaml")
    decoder = ConstrainedBeamSearchDecoder(config)
    decoder.load_model()
    decoder.build_trie("data/ci_list.json")
    return decoder

# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🎯 AI Response with Ad Recommendations")
st.caption(
    "Ask any question. The system generates a helpful response **(Part 1)** "
    "and surfaces relevant sponsored links via constrained beam search **(Part 2)**."
)

query = st.text_input(
    "Your question:",
    placeholder="e.g., What are the best running shoes for beginners?",
)

run = st.button("Ask", type="primary", disabled=not bool(query.strip()))

if run and query.strip():

    # ── Load model (spinner shown only on first load) ──────────────────────
    with st.spinner("Loading model… (first run may take a minute or two)"):
        decoder = load_decoder(config_profile)

    # ── Part 1: vanilla LLM response ──────────────────────────────────────
    st.subheader("Response")

    with st.spinner("Generating response…"):
        t0 = time.time()
        response_text = generate_response(
            decoder.model, decoder.tokenizer, decoder.device, query
        )
        response_time = time.time() - t0

    st.write(response_text)
    st.caption(f"⏱ Generated in {response_time:.1f}s")

    st.divider()

    # ── Part 2: commercial intents + ad matching ───────────────────────────
    col_ci, col_ads = st.columns(2)

    with col_ci:
        st.subheader("Commercial Intents")
        st.caption("Constrained beam search over 189 known intents")

        with st.spinner("Decoding commercial intents…"):
            t1 = time.time()
            ci_results = decoder.generate(query)
            ci_time = time.time() - t1

        if ci_results:
            for ci in ci_results:
                st.markdown(
                    f"- **{ci.ci_text}** &nbsp; "
                    f"<span style='color:gray'>confidence {ci.confidence:.0%}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No commercial intents generated for this query.")

        st.caption(f"⏱ Decoded in {ci_time:.1f}s")

    with col_ads:
        st.subheader("Sponsored")
        st.caption(
            f"Threshold: {match_threshold_pct}% similarity · Max: {int(max_ads)} ads"
        )

        with st.spinner("Matching ads to response…"):
            matched = match_ads(
                model=decoder.model,
                tokenizer=decoder.tokenizer,
                device=decoder.device,
                ci_results=ci_results,
                response_text=response_text,
                ci_ad_index_path="data/ci_ad_index.json",
                match_threshold=match_threshold,
                max_ads=int(max_ads),
            )

        if matched:
            for ad in matched:
                with st.container(border=True):
                    st.markdown(f"**[{ad.title}]({ad.url})**")
                    st.caption(
                        f"{ad.brand}  ·  "
                        f"{ad.category.replace('_', ' ').title()}  ·  "
                        f"Match: {ad.match_score:.0%}"
                    )
                    st.write(ad.description)
                    if ad.price:
                        st.write(f"💰 From ${ad.price:.2f}")
                    st.caption(f"Intent matched: _{ad.ci_text}_")
        else:
            st.info(
                f"No ads matched above {match_threshold_pct}% similarity.  \n"
                "Try lowering the **Ad Match Threshold** in the sidebar."
            )
