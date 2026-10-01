"""All experiments for the BN lab (run: python3 bn_experiments.py). Writes generated_sentences.txt."""
import random
from bn_lab import *

m1, m2 = FirstOrderLM().fit(SENTENCES), SecondOrderLM().fit(SENTENCES)
V = vocab()                              # 10 words
NEXT_V = V + [END]                       # possible next tokens (11)
CTX1 = [START] + V                       # first-order contexts (11)
TRAIN = set(SENTENCES)

def sample_many(m, n, seed):
    rng = random.Random(seed); return [m.generate("sample", rng) for _ in range(n)]

def stats(m, n=5000, seed=1):
    gens = sample_many(m, n, seed)
    sents = [" ".join(g) for g, trunc in gens]
    return dict(n=n, distinct=len(set(sents)), in_train=sum(s in TRAIN for s in sents) / n,
                novel=sum(s not in TRAIN for s in sents) / n, truncated=sum(t for _, t in gens) / n,
                avg_len=sum(len(g) for g, _ in gens) / n)

def table_stats(m, n_ctx_possible):
    params = m.n_parameters(); obs = len(m.counts)
    return dict(params=params, observed_contexts=obs, possible_contexts=n_ctx_possible,
                unseen_contexts=n_ctx_possible - obs, table_cells=n_ctx_possible * len(NEXT_V),
                zero_cells=n_ctx_possible * len(NEXT_V) - params,
                zero_cells_in_observed=obs * len(NEXT_V) - params)

def write_generated():
    out = []
    for name, m, seed in [("FIRST-ORDER (sampling, seed 0)", m1, 0), ("SECOND-ORDER (sampling, seed 0)", m2, 0)]:
        rng = random.Random(seed); out.append(name)
        for i in range(20):
            g, t = m.generate("sample", rng); out.append(f"{i+1:2d}. " + " ".join(g) + ("  [truncated]" if t else ""))
        out.append("")
    out.append("GREEDY (arg max, ties -> first seen in training data), max_len = 30")
    for name, m in [("first-order", m1), ("second-order", m2)]:
        g, t = m.generate("greedy"); out.append(f"{name}: " + " ".join(g) + ("  [truncated at 30 tokens: cycle]" if t else ""))
    open("generated_sentences.txt", "w").write("\n".join(out) + "\n")
    return out

if __name__ == "__main__":
    print("== first-order CPTs ==")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        d = m1.dist([w]); print(f"P(. | {w:7s}) =", {k: round(v, 4) for k, v in d.items()}, "argmax:", m1.predict([w]), "zero:", [x for x in NEXT_V if x not in d][:4], "...")
    print("== normalisation ==")
    for nm, m in [("first", m1), ("second", m2)]:
        t = m.check_normalisation(); print(nm, "min", min(t.values()), "max", max(t.values()), "contexts", len(t))
    print("== second-order selected CPTs ==")
    for c in [(START, START), (START, "the"), ("the", "cat"), ("cat", "sat"), ("sat", "on"), ("on", "the"), ("cat", "ran"), ("to", "the"), ("the", "mat")]:
        print(c, {k: round(v, 4) for k, v in m2.dist(c).items()})
    print("== table stats =="); print(table_stats(m1, len(CTX1))); print(table_stats(m2, len(CTX1) ** 2))
    print("== sampling stats =="); print(stats(m1)); print(stats(m2))
    print("\n".join(write_generated()))
    print("unseen test:", m1.dist(["banana"]), m1.predict(["banana"]), m2.generate("sample"))
