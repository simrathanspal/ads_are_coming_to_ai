"""Streamlit UI for the AI Ad Recommender System.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.constrained_decoder import ConstrainedBeamSearchDecoder, DecoderConfig
from src.response_generator import generate_response
from src.ad_matcher import match_ads

DATA_DIR = Path("data")
CONFIGS_DIR = Path("configs")


@st.cache_resource(show_spinner="Loading model (first run only — this may take a minute)...")
def load_decoder(profile: str) -> ConstrainedBeamSearchDecoder:
    config = DecoderConfig.from_yaml(str(CONFIGS_DIR / f"{profile}.yaml"))
    decoder = ConstrainedBeamSearchDecoder(config)
    decoder.load_model()
    decoder.build_trie(str(DATA_DIR / "ci_list.json"))
    return decoder


st.set_page_config(page_title="AI Ad Recommender", layout="wide")
st.title("AI Ad Recommender")
st.caption(
    "Powered by constrained beam search + semantic matching "
    "([RARE framework](https://arxiv.org/abs/2504.01304))"
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    profile = st.selectbox(
        "Decoder profile",
        ["default", "aggressive", "conservative"],
        help="Selects configs/{profile}.yaml — controls beam width, temperature, etc.",
    )
    threshold_pct = st.slider(
        "Ad match threshold",
        min_value=0,
        max_value=100,
        value=70,
        format="%d%%",
        help="Minimum cosine similarity between the LLM response and a commercial intent to show its ad.",
    )
    max_ads = st.slider(
        "Max ads per response",
        min_value=1,
        max_value=5,
        value=2,
        help="Maximum number of sponsored results shown.",
    )

threshold = threshold_pct / 100.0

# --- Load model (cached across reruns) ---
decoder = load_decoder(profile)

# --- Query input ---
query = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the best hiking boots for wet weather?",
)

if query:
    col_resp, col_ads = st.columns([3, 2])

    with col_resp:
        with st.spinner("Generating response..."):
            response = generate_response(
                decoder.model, decoder.tokenizer, decoder.device, query
            )
        st.subheader("Response")
        st.write(response)

    with col_ads:
        with st.spinner("Finding commercial intents..."):
            ci_results = decoder.generate(query)

        with st.spinner("Matching ads..."):
            ads = match_ads(
                response=response,
                ci_results=ci_results,
                ci_ad_index_path=str(DATA_DIR / "ci_ad_index.json"),
                model=decoder.model,
                tokenizer=decoder.tokenizer,
                device=decoder.device,
                threshold=threshold,
                max_ads=max_ads,
            )

        st.subheader("Commercial Intents")
        if ci_results:
            for ci in ci_results:
                st.markdown(f"- **{ci.ci_text}** ({ci.confidence:.1%} confidence)")
        else:
            st.info("No commercial intents found.")

        st.subheader("Sponsored")
        if ads:
            for ad in ads:
                with st.container(border=True):
                    st.markdown(f"**[{ad.title}]({ad.url})**")
                    price_str = f" · ${ad.price:.2f}" if ad.price else ""
                    st.caption(f"{ad.brand} · {ad.category}{price_str}")
                    st.write(ad.description)
                    st.progress(ad.match_score, text=f"Match: {ad.match_score:.0%}")
        else:
            st.info("No relevant ads found above the threshold.")
