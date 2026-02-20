"""
FastAPI backend for the Urdu Children's Story Generation System.

Uses:
  - BPETokenizer to encode Urdu text into token IDs and decode IDs back to Urdu text
  - TrigramLanguageModel to generate next token IDs using interpolated trigram probabilities

Usage:
  1. Place your n_grams.pkl file in backend/trigram_language_model/ as ngram_counts.pkl
     (OR keep it as backend/n_grams.pkl — the loader handles both)
  2. Ensure backend/bpe_tokenizer_model/ has vocab.json and merges.txt
  3. pip install -r requirements.txt
  4. uvicorn main:app --reload --port 8000
"""

import pickle
import asyncio
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from bpe_tokenizer import BPETokenizer
from trigram_language_model import TrigramLanguageModel

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Urdu Story Generator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

tokenizer = BPETokenizer(vocab_size=250)
model = TrigramLanguageModel(vocab_size=250)

BACKEND_DIR = Path(__file__).parent
TOKENIZER_DIR = BACKEND_DIR / "bpe_tokenizer_model"
TRIGRAM_DIR = BACKEND_DIR / "trigram_language_model"
LEGACY_PKL = BACKEND_DIR / "n_grams.pkl"


@app.on_event("startup")
def load_models():
    """Load both the BPE tokenizer and the trigram language model."""

    # --- Load BPE Tokenizer ---
    if not TOKENIZER_DIR.exists():
        print(f"[ERROR] Tokenizer directory not found: {TOKENIZER_DIR}")
        return

    tokenizer.load(str(TOKENIZER_DIR))
    print(f"[INFO] BPE Tokenizer loaded — vocab size: {len(tokenizer.vocab)}")

    # Build inverse vocab for decoding
    inv = {v: k for k, v in tokenizer.vocab.items()}
    sample_ids = list(inv.keys())[:10]
    print(f"[INFO] Sample inverse vocab: { {i: inv[i] for i in sample_ids} }")

    # --- Load Trigram Model ---
    # The TrigramLanguageModel.load() expects ngram_counts.pkl + config.json
    # inside a directory.  If the user only has n_grams.pkl at the backend root,
    # copy it into the expected location.
    expected_pkl = TRIGRAM_DIR / "ngram_counts.pkl"
    if not expected_pkl.exists() and LEGACY_PKL.exists():
        TRIGRAM_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_PKL, expected_pkl)
        print(f"[INFO] Copied {LEGACY_PKL} -> {expected_pkl}")

    if not expected_pkl.exists():
        print(f"[ERROR] No trigram model found at {expected_pkl} or {LEGACY_PKL}")
        return

    model.load(str(TRIGRAM_DIR))

    # Also load the vocabulary into the trigram model so id_to_token is available
    vocab_path = TOKENIZER_DIR / "vocab.json"
    if vocab_path.exists():
        model.load_vocabulary(str(vocab_path))
        print(f"[INFO] Vocabulary loaded into trigram model — {len(model.id_to_token)} tokens")

    print(f"[INFO] Trigram model loaded — "
          f"unigrams: {model.total_unigrams}, "
          f"bigram contexts: {len(model.bigram_counts)}, "
          f"trigram contexts: {len(model.trigram_counts)}")
    print(f"[INFO] Lambdas: l1={model.lambda1}, l2={model.lambda2}, l3={model.lambda3}")

    # Debug: show a few actual trigram entries decoded to Urdu
    inv = {v: k for k, v in tokenizer.vocab.items()}
    sample_tri = list(model.trigram_counts.items())[:3]
    for ctx, counter in sample_tri:
        w1_str = inv.get(ctx[0], f"?{ctx[0]}")
        w2_str = inv.get(ctx[1], f"?{ctx[1]}")
        top3 = [(inv.get(tid, f"?{tid}"), cnt) for tid, cnt in counter.most_common(3)]
        print(f"[INFO] Trigram: ({w1_str}, {w2_str}) -> {top3}")

    # Debug: test generation of one token
    test_ctx = list(model.trigram_counts.keys())[0] if model.trigram_counts else (2, 2)
    test_next = model.generate_next_token(test_ctx)
    test_decoded = tokenizer.decode([test_next])
    print(f"[INFO] Test generation: ctx={test_ctx} -> next_id={test_next} -> '{test_decoded}'")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prefix: str
    max_length: int = 200


# ---------------------------------------------------------------------------
# Generation Logic  (encode -> generate -> decode, streamed token-by-token)
# ---------------------------------------------------------------------------

async def generate_tokens(prefix: str, max_length: int):
    """
    1. Encode the Urdu prefix into BPE token IDs.
    2. Use the trigram model to generate next token IDs one at a time.
    3. Decode each new token ID back to Urdu text and stream it.
    """
    if model.total_unigrams == 0:
        yield {
            "event": "error",
            "data": json.dumps(
                {"error": "Model not loaded. Check backend logs."},
                ensure_ascii=False,
            ),
        }
        return

    # Encode the user's prefix into token IDs
    prefix_ids = tokenizer.encode(prefix)
    print(f"[INFO] Encoded prefix '{prefix}' -> {prefix_ids}")

    # Decode them back to verify round-trip
    decoded_prefix = tokenizer.decode(prefix_ids)
    print(f"[INFO] Decoded back: '{decoded_prefix}'")

    # Send the original prefix back as the first token
    yield {
        "event": "token",
        "data": json.dumps({"token": prefix}, ensure_ascii=False),
    }
    await asyncio.sleep(0.05)

    # Build the initial context (last two token IDs for trigram)
    if len(prefix_ids) == 0:
        ctx = [model.eos_id, model.eos_id]
    elif len(prefix_ids) == 1:
        ctx = [model.eos_id, prefix_ids[0]]
    else:
        ctx = list(prefix_ids[-2:])

    generated_ids = list(prefix_ids)
    consecutive_special = 0

    for step in range(max_length):
        # Generate the next token ID using interpolated trigram probabilities
        next_id = model.generate_next_token((ctx[-2], ctx[-1]))

        # Log first 15 steps for debugging
        if step < 15:
            inv = {v: k for k, v in tokenizer.vocab.items()}
            ctx_str = f"({inv.get(ctx[-2], '?')}, {inv.get(ctx[-1], '?')})"
            next_str = inv.get(next_id, '?')
            print(f"[GEN] step={step} ctx={ctx_str} ctx_ids=({ctx[-2]},{ctx[-1]}) -> next_id={next_id} token='{next_str}'")

        # Stop only on end-of-text (EOT = 4), NOT on end-of-sentence (EOS = 2)
        if next_id == model.eot_id:
            print(f"[INFO] Generation stopped: hit EOT at step {step}")
            break

        # Skip PAD and UNK tokens
        if next_id in (model.pad_id, model.unk_id):
            continue

        generated_ids.append(next_id)
        ctx.append(next_id)

        # Decode just this single token to Urdu text
        token_text = tokenizer.decode([next_id])

        # Handle EOS (end of sentence = ۔) — emit it but keep generating
        if next_id == model.eos_id:
            consecutive_special += 1
            # If we get too many EOS in a row, stop
            if consecutive_special >= 3:
                print(f"[INFO] Generation stopped: too many consecutive EOS at step {step}")
                break
            yield {
                "event": "token",
                "data": json.dumps({"token": " ۔"}, ensure_ascii=False),
            }
            await asyncio.sleep(0.08)
            continue

        # Handle EOP (end of paragraph) — emit newline but keep generating
        if next_id == model.eop_id:
            consecutive_special += 1
            if consecutive_special >= 3:
                print(f"[INFO] Generation stopped: too many consecutive special tokens at step {step}")
                break
            yield {
                "event": "token",
                "data": json.dumps({"token": "\n\n"}, ensure_ascii=False),
            }
            await asyncio.sleep(0.08)
            continue

        # Reset consecutive special counter on normal tokens
        consecutive_special = 0

        # Stream the decoded Urdu text
        if token_text.strip():
            yield {
                "event": "token",
                "data": json.dumps({"token": " " + token_text.strip()}, ensure_ascii=False),
            }

        await asyncio.sleep(0.08)

    yield {"event": "done", "data": "{}"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tokenizer_loaded": len(tokenizer.vocab) > 0,
        "model_loaded": model.total_unigrams > 0,
        "vocab_size": len(tokenizer.vocab),
        "unigram_count": model.total_unigrams,
    }


@app.get("/debug-model")
async def debug_model():
    """Debug endpoint: inspect the model + tokenizer state."""
    inv = {v: k for k, v in tokenizer.vocab.items()}

    # Show a few trigram entries decoded
    sample_trigrams = []
    for ctx, counter in list(model.trigram_counts.items())[:5]:
        w1_str = inv.get(ctx[0], f"?{ctx[0]}")
        w2_str = inv.get(ctx[1], f"?{ctx[1]}")
        top_next = counter.most_common(3)
        decoded_next = [(inv.get(tid, f"?{tid}"), cnt) for tid, cnt in top_next]
        sample_trigrams.append({
            "context": f"{w1_str} {w2_str}",
            "context_ids": list(ctx),
            "top_next": decoded_next,
        })

    return {
        "vocab_size": len(tokenizer.vocab),
        "total_unigrams": model.total_unigrams,
        "bigram_contexts": len(model.bigram_counts),
        "trigram_contexts": len(model.trigram_counts),
        "lambdas": {
            "l1": model.lambda1,
            "l2": model.lambda2,
            "l3": model.lambda3,
        },
        "sample_trigrams": sample_trigrams,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate a story continuation from an Urdu prefix.
    Returns Server-Sent Events stream of decoded Urdu tokens.
    """
    if not request.prefix.strip():
        raise HTTPException(status_code=400, detail="Prefix cannot be empty.")

    return EventSourceResponse(
        generate_tokens(request.prefix.strip(), request.max_length)
    )
