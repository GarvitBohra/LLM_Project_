# LLM From Scratch, Simplified

This is a tiny beginner project that shows the main pieces of a GPT-style
language model:

1. text -> numbers
2. numbers -> Transformer
3. Transformer training
4. text generation

It is not a large, useful chatbot. It is a learning project that helps you
understand how an LLM works.

## What this project uses

- Python 3.10+
- PyTorch
- Streamlit

It does **not** require `tiktoken` or `datasets`.  
For learning, this version uses a simple **character-level tokenizer** so the
code is easier to understand.

## Files

- `llm_from_scratch/tokenizer.py` - turns text into token IDs
- `llm_from_scratch/model.py` - a tiny GPT-style Transformer
- `llm_from_scratch/data.py` - loads text and makes training batches
- `train.py` - trains the model
- `generate.py` - generates text from a prompt
- `app.py` - Streamlit interface for chatting with the model
- `data/tiny.txt` - a tiny sample text file

## Setup

Install PyTorch first:

```bash
pip install -r requirements.txt
```

If you want, you can also make a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Train

```bash
python train.py
```

This will:

- read `data/tiny.txt`
- build the tokenizer from that text
- train a small model
- save a checkpoint to `checkpoints/model.pt`

## Generate text

After training:

```bash
python generate.py --prompt "hello" --max-new-tokens 100
```

## Streamlit app

Run the browser UI with:

```bash
python3 -m streamlit run app.py
```

If `streamlit` is on your PATH, `streamlit run app.py` also works.

## Easy next steps

1. Replace `data/tiny.txt` with your own text.
2. Increase `steps` in `train.py`.
3. Increase `n_layer` and `n_embd` in `train.py` after it works.
4. Later, switch the tokenizer to BPE or `tiktoken`.
