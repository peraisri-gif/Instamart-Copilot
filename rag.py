import os, glob, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

# Task 1: chunk (40 words, overlap 10)
def chunk_docs(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob("data/policies/*.txt")):
        words = open(path, encoding="utf-8").read().split()
        src, start = os.path.basename(path), 0
        while True:
            end = start + size
            chunks.append({"source": src, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += size - overlap
    return chunks

# Task 2: TF-IDF retrieval
def make_retriever(chunks):
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])
    def retrieve(query, k=3):
        sims = cosine_similarity(vec.transform([query]), mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": chunks[i]["source"], "score": float(sims[i]),
                 "text": chunks[i]["text"]} for i in top]
    return retrieve

# Task 3: augment and answer
SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
          "If the context does not contain the information, say you could not find it.")

def make_answer(retrieve):
    def answer(question):
        hits = retrieve(question, k=3)
        ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"}]
        resp = mock_llm.chat(msgs)
        return resp.choices[0].message.content, [h["source"] for h in hits]
    return answer

# Task 4: evaluation
def evaluate(overlap):
    chunks = chunk_docs(40, overlap)
    retrieve = make_retriever(chunks)
    answer = make_answer(retrieve)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    exp_doc = ev["expected_doc"].str.strip()
    unans = exp_doc.isin(["", "none", "None", "NONE"])
    hit = ground = fb = 0
    failed = []
    for i, r in ev.iterrows():
        text, srcs = answer(r["question"])
        if unans[i]:
            fb += text.startswith("I could not find")
        else:
            h = r["expected_doc"] in srcs
            g = r["expected_fact"] in text
            hit += h; ground += g
            if not (h and g):
                failed.append(r["question"])
    print(f"overlap={overlap}: chunks={len(chunks)} "
          f"hit {hit}/{(~unans).sum()} grounded {ground}/{(~unans).sum()} "
          f"fallback {fb}/{unans.sum()}")
    for q in failed:
        print("  FAILED:", q)

if __name__ == "__main__":
    chunks = chunk_docs()
    print("total chunks:", len(chunks))
    print(pd.Series([c["source"] for c in chunks]).value_counts().to_dict())
    retrieve = make_retriever(chunks)
    print([h["source"] for h in retrieve("refund policy late delivery")])
    evaluate(10)
    evaluate(0)