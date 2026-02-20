import json
from collections import Counter

class BPETokenizer:
    EOS_CHAR = "\uE000"  
    EOP_CHAR = "\uE001" 
    EOT_CHAR = "\uE002" 

    def __init__(self, vocab_size: int = 250):
        self.vocab_size = vocab_size
        self.vocab = {}          # token -> id
        self.merges = []         # list of (a,b)
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            self.EOS_CHAR: 2,
            self.EOP_CHAR: 3,
            self.EOT_CHAR: 4,
        }
        self.legacy_special_aliases = {
            "<EOS>": self.EOS_CHAR,
            "<EOP>": self.EOP_CHAR,
            "<EOT>": self.EOT_CHAR,
        }

    def _get_pair_stats(self, words: dict) -> Counter:
        pairs = Counter()
        for w, freq in words.items():
            syms = w.split()
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, words: dict) -> dict:
        bigram = " ".join(pair)
        repl = "".join(pair)
        return {w.replace(bigram, repl): freq for w, freq in words.items()}

    def train(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = f.read()

        word_freqs = Counter(corpus.split())

        # word -> "c h a r s </w>"
        vocab_words = {" ".join(list(w)) + " </w>": freq for w, freq in word_freqs.items()}

        # base symbol set
        symbols = set()
        for w in vocab_words:
            symbols.update(w.split())

        available = self.vocab_size - len(self.special_tokens)
        # keep some room for merges (same idea as original)
        if len(symbols) >= available:
            sym_freq = Counter()
            for w, freq in vocab_words.items():
                for s in w.split():
                    sym_freq[s] += freq
            base_vocab_size = min(len(symbols), max(1, available - 50))
            kept = set(s for s, _ in sym_freq.most_common(base_vocab_size))
            vocab_words = {w: freq for w, freq in vocab_words.items()
                           if all(s in kept for s in w.split())}
            symbols = kept

        num_merges = available - len(symbols)
        if num_merges <= 0:
            num_merges = 1

        for _ in range(num_merges):
            pairs = self._get_pair_stats(vocab_words)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab_words = self._merge_vocab(best, vocab_words)
            self.merges.append(best)

        # build final vocab by token frequency (same approach)
        token_freq = Counter()
        for w, freq in vocab_words.items():
            for tok in w.split():
                token_freq[tok] += freq

        self.vocab = dict(self.special_tokens)
        next_id = len(self.special_tokens)
        slots = self.vocab_size - len(self.special_tokens)

        for tok, _ in token_freq.most_common():
            if tok not in self.vocab and slots > 0:
                self.vocab[tok] = next_id
                next_id += 1
                slots -= 1
            if len(self.vocab) >= self.vocab_size:
                break

    def _separate_specials(self, text: str) -> str:
        for legacy, ch in self.legacy_special_aliases.items():
            text = text.replace(legacy, ch)
        for ch in (self.EOS_CHAR, self.EOP_CHAR, self.EOT_CHAR):
            text = text.replace(ch, f" {ch} ")
        return text

    def encode_word(self, word: str):
        w = " ".join(list(word)) + " </w>"
        for a, b in self.merges:
            w = w.replace(f"{a} {b}", f"{a}{b}")
        return w.split()

    def encode(self, text: str):
        text = self._separate_specials(text)
        ids = []
        for w in text.split():
            w = self.legacy_special_aliases.get(w, w)

            if w in self.special_tokens:
                ids.append(self.special_tokens[w])
                continue

            for tok in self.encode_word(w):
                ids.append(self.vocab.get(tok, self.special_tokens["<UNK>"]))
        return ids

    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        toks = [id_to_token.get(i, "") for i in token_ids]
        text = "".join(toks).replace("</w>", " ")

        text = text.replace(self.EOS_CHAR, "۔ ")
        text = text.replace(self.EOP_CHAR, "\n\n")
        text = text.replace(self.EOT_CHAR, "")
        return " ".join(text.split())

    def save(self, out_dir: str):
        import os
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        cfg = {
            "vocab_size": len(self.vocab),
            "target_vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def load(self, tok_dir: str):
        import os

        with open(os.path.join(tok_dir, "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.merges = []
        with open(os.path.join(tok_dir, "merges.txt"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))

        cfg_path = os.path.join(tok_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.vocab_size = cfg.get("target_vocab_size", self.vocab_size)
            self.special_tokens = cfg.get("special_tokens", self.special_tokens)