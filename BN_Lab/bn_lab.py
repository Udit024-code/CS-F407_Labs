"""CS F407 - Bayesian Network / Autoregressive LM lab.
First-order model  P(Xt | Xt-1)        (graph: X1 -> X2 -> X3 ...)
Second-order model P(Xt | Xt-2, Xt-1)  (graph: Xt-2 -> Xt <- Xt-1)
Only ordinary Python data structures and random sampling are used."""
import random
from collections import Counter, defaultdict

START, END = "<START>", "<END>"
SENTENCES = [
    "the cat sat on the mat", "the cat sat on the rug", "the dog sat on the mat",
    "the dog ran to the park", "the cat ran to the park", "the dog sat on the rug",
]
def tokenise(s): return s.lower().split()

class NGramLM:
    """Context of `n` previous tokens. n=1: first-order, n=2: second-order."""
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(Counter)        # counts[context][next_token] = C(context, next)   (Q4)
    def fit(self, sentences):
        for s in sentences:
            toks = [START] * self.n + tokenise(s) + [END]
            for i in range(self.n, len(toks)):
                self.counts[tuple(toks[i - self.n:i])][toks[i]] += 1
        return self
    def dist(self, context):
        """P(next | context) = C(context,next) / sum_k C(context,k)               (Q5)"""
        c = self.counts.get(tuple(context))
        if not c: return {}                        # unseen context -> empty distribution (Q7)
        total = sum(c.values())
        return {w: n / total for w, n in c.items()}
    def predict(self, context):
        """arg max_w P(w | context); ties -> the transition seen first in the training data."""
        d = self.dist(context)
        return max(d, key=d.get) if d else None
    def sample_next(self, context, rng):
        """Sample from P(. | context) (Q6, option 2)."""
        d = self.dist(context)
        if not d: return None
        return rng.choices(list(d), weights=list(d.values()))[0]
    def generate(self, mode="sample", rng=None, max_len=30):
        rng = rng or random
        ctx, out = [START] * self.n, []
        while len(out) < max_len:
            w = self.predict(ctx) if mode == "greedy" else self.sample_next(ctx, rng)
            if w is None or w == END: return out, False
            out.append(w); ctx = ctx[1:] + [w] if self.n > 1 else [w]
        return out, True                           # truncated: no <END> within max_len
    def check_normalisation(self):
        """Every context's distribution must sum to 1."""
        return {ctx: sum(self.dist(ctx).values()) for ctx in self.counts}
    def n_parameters(self): return sum(len(c) for c in self.counts.values())

class FirstOrderLM(NGramLM):
    def __init__(self): super().__init__(1)
class SecondOrderLM(NGramLM):
    def __init__(self): super().__init__(2)

def vocab():
    return sorted({w for s in SENTENCES for w in tokenise(s)})

if __name__ == "__main__":
    m1, m2 = FirstOrderLM().fit(SENTENCES), SecondOrderLM().fit(SENTENCES)
    rng = random.Random(0)
    for name, m in [("first-order", m1), ("second-order", m2)]:
        print(name, "params", m.n_parameters(), "contexts", len(m.counts))
        print(" greedy:", m.generate("greedy")); print(" sampled:", [" ".join(m.generate("sample", rng)[0]) for _ in range(5)])
