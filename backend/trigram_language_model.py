
import json
import pickle
from collections import defaultdict, Counter
import numpy as np

class TrigramLanguageModel:
    def __init__(self, vocab_size: int = 250):
        self.vocab_size = vocab_size

        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)     # w1 -> Counter(w2)
        self.trigram_counts = defaultdict(Counter)    # (w1,w2) -> Counter(w3)

        self.total_unigrams = 0
        self.total_bigrams = defaultdict(int)         # w1 -> total
        self.total_trigrams = defaultdict(int)        # (w1,w2) -> total

        self.lambda1, self.lambda2, self.lambda3 = 0.1, 0.3, 0.6

        self.vocab = {}
        self.id_to_token = {}

        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.eop_id = 3
        self.eot_id = 4

    def load_vocabulary(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def train(self, token_ids):
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.total_unigrams = 0
        self.total_bigrams = defaultdict(int)
        self.total_trigrams = defaultdict(int)

        for t in token_ids:
            self.unigram_counts[t] += 1
            self.total_unigrams += 1

        for i in range(len(token_ids) - 1):
            w1, w2 = token_ids[i], token_ids[i + 1]
            self.bigram_counts[w1][w2] += 1
            self.total_bigrams[w1] += 1

        for i in range(len(token_ids) - 2):
            w1, w2, w3 = token_ids[i], token_ids[i + 1], token_ids[i + 2]
            ctx = (w1, w2)
            self.trigram_counts[ctx][w3] += 1
            self.total_trigrams[ctx] += 1

    def _p1(self, w):  # unigram
        return 0.0 if self.total_unigrams == 0 else self.unigram_counts[w] / self.total_unigrams

    def _p2(self, w1, w2):  # bigram
        denom = self.total_bigrams[w1]
        return 0.0 if denom == 0 else self.bigram_counts[w1][w2] / denom

    def _p3(self, w1, w2, w3):  # trigram
        ctx = (w1, w2)
        denom = self.total_trigrams[ctx]
        return 0.0 if denom == 0 else self.trigram_counts[ctx][w3] / denom

    def get_interpolated_prob(self, w1, w2, w3):
        return (self.lambda1 * self._p1(w3) +
                self.lambda2 * self._p2(w2, w3) +
                self.lambda3 * self._p3(w1, w2, w3))

    def generate_next_token(self, context):
        w1, w2 = context
        candidates = set()

        if context in self.trigram_counts:
            candidates.update(self.trigram_counts[context].keys())
        if w2 in self.bigram_counts:
            candidates.update(self.bigram_counts[w2].keys())

        candidates.update([t for t, _ in self.unigram_counts.most_common(50)])

        candidates.discard(self.pad_id)
        candidates.discard(self.unk_id)

        if not candidates:
            return self.eos_id

        tokens = list(candidates)
        probs = [self.get_interpolated_prob(w1, w2, t) for t in tokens]
        s = sum(probs)
        if s == 0:
            probs = [1.0 / len(tokens)] * len(tokens)
        else:
            probs = [p / s for p in probs]

        return int(np.random.choice(tokens, p=probs))

    def generate_story(self, max_length: int = 500, temperature: float = 1.0):
        # temperature kept in signature (but unused like original)
        generated = [self.eos_id, self.eos_id]
        for _ in range(max_length):
            nxt = self.generate_next_token((generated[-2], generated[-1]))
            generated.append(nxt)
            if nxt == self.eot_id:
                break
        return generated[2:]

    def generate_from_prefix(self, prefix_ids, max_length: int = 300):
        if len(prefix_ids) == 0:
            ctx_seq = [self.eos_id, self.eos_id]
        elif len(prefix_ids) == 1:
            ctx_seq = [self.eos_id, prefix_ids[0]]
        else:
            ctx_seq = prefix_ids[-2:]

        out = list(prefix_ids)
        for _ in range(max_length):
            nxt = self.generate_next_token((ctx_seq[-2], ctx_seq[-1]))
            out.append(nxt)
            ctx_seq.append(nxt)
            if nxt == self.eot_id:
                break
        return out

    def save(self, output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "ngram_counts.pkl"), "wb") as f:
            pickle.dump({
                "unigram_counts": dict(self.unigram_counts),
                "bigram_counts": {k: dict(v) for k, v in self.bigram_counts.items()},
                "trigram_counts": {k: dict(v) for k, v in self.trigram_counts.items()},
                "total_unigrams": self.total_unigrams,
                "total_bigrams": dict(self.total_bigrams),
                "total_trigrams": dict(self.total_trigrams),
            }, f)

        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "lambda3": self.lambda3,
            }, f, indent=2)

    def load(self, model_dir: str):
        import os
        with open(os.path.join(model_dir, "ngram_counts.pkl"), "rb") as f:
            d = pickle.load(f)
        self.unigram_counts = Counter(d["unigram_counts"])
        self.bigram_counts = defaultdict(Counter, {k: Counter(v) for k, v in d["bigram_counts"].items()})
        self.trigram_counts = defaultdict(Counter, {k: Counter(v) for k, v in d["trigram_counts"].items()})
        self.total_unigrams = d["total_unigrams"]
        self.total_bigrams = defaultdict(int, d["total_bigrams"])
        self.total_trigrams = defaultdict(int, d["total_trigrams"])

        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab_size = cfg["vocab_size"]
        self.lambda1, self.lambda2, self.lambda3 = cfg["lambda1"], cfg["lambda2"], cfg["lambda3"]
