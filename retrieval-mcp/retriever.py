"""
A dependency-free BM25 retriever — a real ranking function (used in production
hybrid search), with zero model downloads so the demo runs anywhere offline.

Swap path: replace `.query()` with a Chroma / pgvector similarity search and the
rest of the pipeline is unchanged — retrieval is just "given a query, return the
top-k passages."
"""
from __future__ import annotations
import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(s: str):
    return _TOK.findall(s.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def index(self, corpus: list):
        """corpus: list of {"id": str, "text": str}"""
        self.ids = [d["id"] for d in corpus]
        self.texts = {d["id"]: d["text"] for d in corpus}
        self.docs = [tokenize(d["text"]) for d in corpus]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        df = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        return self

    def query(self, q: str, k: int = 3):
        """Return [(id, score), ...] for the top-k passages."""
        qt = tokenize(q)
        scored = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            dl = len(d)
            s = 0.0
            for w in qt:
                if w not in tf:
                    continue
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(w, 0.0) * num / den
            scored.append((self.ids[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
