from __future__ import annotations

from pathlib import Path

import streamlit as st

from llm_from_scratch.runtime import generate_text, load_checkpoint


@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str):
    return load_checkpoint(Path(ckpt_path))


def main():
    st.set_page_config(page_title="Tiny LLM", page_icon="🤖", layout="centered")

    st.title("Tiny LLM From Scratch")
    st.write("A small GPT-style model you can train and test from your browser.")

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=str(Path("checkpoints/model.pt")),
    )

    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    if "raw_output" not in st.session_state:
        st.session_state.raw_output = ""
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    if "result_ready" not in st.session_state:
        st.session_state.result_ready = False

    with st.form("generate_form"):
        prompt = st.text_area("Prompt", value="Language models")
        max_new_tokens = st.slider("New tokens", min_value=1, max_value=200, value=80)
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
        top_k = st.slider("Top-k", min_value=1, max_value=20, value=3)
        submitted = st.form_submit_button("Generate")

    if submitted:
        st.session_state.last_error = ""
        st.session_state.result_ready = False
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            st.session_state.last_error = "Checkpoint not found. Run `python train.py` first."
            st.session_state.generated_text = ""
            st.session_state.raw_output = ""
        else:
            try:
                with st.spinner("Loading model and generating text..."):
                    model, tokenizer, _ = get_model(str(ckpt))
                    output = generate_text(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                st.session_state.raw_output = output
                visible_output = output.strip()
                if not visible_output:
                    visible_output = "(no visible text generated)"
                    st.session_state.last_error = "The model generated only whitespace. Try a lower temperature or more training."
                st.session_state.generated_text = visible_output
                st.session_state.result_ready = True
            except Exception as exc:
                st.session_state.generated_text = ""
                st.session_state.raw_output = ""
                st.session_state.last_error = f"Generation failed: {exc}"

    if st.session_state.last_error:
        st.warning(st.session_state.last_error)

    if st.session_state.result_ready and st.session_state.generated_text:
        st.subheader("Output")
        st.text_area("Generated text", value=st.session_state.generated_text, height=200)
        with st.expander("Debug output"):
            st.code(repr(st.session_state.raw_output), language="text")

    st.caption("Tip: train on your own text file, then come back here to generate samples.")


if __name__ == "__main__":
    main()
