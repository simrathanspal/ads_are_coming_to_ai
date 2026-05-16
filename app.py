"""
Streamlit UI for the RARE Ad Recommender System.

Two-part pipeline per query:
  Part 1 — Vanilla LLM response (generate_response)
  Part 2 — Constrained beam search → CIs → semantic ad matching (match_ads)

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.constrained_decoder import ConstrainedBeamSearchDecoder, DecoderConfig
from src.response_generator import generate_response
from src.ad_matcher import match_ads

CI_AD_INDEX = "data/ci_ad_index.json"
CI_LIST = "data/ci_list.json"
PROFILES = ["default", "aggressive", "conservative"]


@st.cache_resource(show_spinner="Loading model (first time only)...")
def load_decoder(profile: str) -> ConstrainedBeamSearchDecoder:
    config = DecoderConfig.from_yaml(f"configs/{profile}.yaml")
    decoder = ConstrainedBeamSearchDecoder(config)
    decoder.load_model()
    decoder.build_trie(CI_LIST)
    return decoder


st.title("Ads Are Coming to AI")
st.caption("RARE framework demo — constrained decoding meets ad injection")

with st.sidebar:
    st.header("Settings")
    profile = st.selectbox("Decoder Profile", PROFILES, index=0)
    threshold = st.slider(
        "Ad Match Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help="Minimum cosine similarity between CI and response to show an ad",
    )
    max_ads = st.slider(
        "Max Ads per Response",
        min_value=1,
        max_value=5,
        value=2,
        help="Cap on sponsored results shown per query",
    )

query = st.text_input(
    "Ask anything...", placeholder="e.g. best running shoes for a marathon"
)

if st.button("Submit", type="primary") and query.strip():
    decoder = load_decoder(profile)

    with st.spinner("Generating response..."):
        response = generate_response(
            decoder.model, decoder.tokenizer, decoder.device, query
        )

    st.subheader("Response")
    st.write(response)
    st.divider()

    with st.spinner("Detecting commercial intents..."):
        ci_results = decoder.generate(query)

    with st.spinner("Matching ads..."):
        matched_ads = match_ads(
            response_text=response,
            ci_results=ci_results,
            ci_ad_index_path=CI_AD_INDEX,
            model=decoder.model,
            tokenizer=decoder.tokenizer,
            device=decoder.device,
            threshold=threshold,
            max_ads=max_ads,
        )

    col_ci, col_ads = st.columns([1, 1])

    with col_ci:
        st.subheader("Commercial Intents Detected")
        if ci_results:
            for ci in ci_results:
                st.markdown(f"**{ci.ci_text}** — {ci.confidence:.0%} confidence")
        else:
            st.info("No commercial intents detected.")

    with col_ads:
        st.subheader("Sponsored")
        if matched_ads:
            for ad in matched_ads:
                with st.container(border=True):
                    st.markdown(f"**[{ad.title}]({ad.url})**")
                    st.caption(
                        f"{ad.brand} · {ad.category} · match: {ad.match_score:.0%}"
                    )
                    st.write(ad.description)
                    if ad.price:
                        st.markdown(f"**${ad.price:.2f}**")
                    st.link_button("View Deal", ad.url)
        else:
            st.info("No relevant ads found above the match threshold.")
