from __future__ import annotations

from pathlib import Path

import streamlit as st

from llm_from_scratch.runtime import generate_text, load_checkpoint


st.set_page_config(page_title="Tiny LLM", page_icon="🤖", layout="centered")

st.title("Tiny LLM From Scratch")
st.write("A small GPT-style model you can train and test from your browser.")

checkpoint_path = st.text_input(
    "Checkpoint path",
    value=str(Path("checkpoints/model.pt")),
)

prompt = st.text_area("Prompt", value="Language models")
max_new_tokens = st.slider("New tokens", min_value=1, max_value=200, value=80)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
top_k = st.slider("Top-k", min_value=1, max_value=20, value=5)

generate_button = st.button("Generate")

if generate_button:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        st.error("Checkpoint not found. Run `python train.py` first.")
    else:
        with st.spinner("Loading model and generating text..."):
            model, tokenizer, _ = load_checkpoint(ckpt)
            output = generate_text(
                model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        st.subheader("Output")
        st.code(output)

st.caption("Tip: train on your own text file, then come back here to generate samples.")
