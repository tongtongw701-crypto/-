import argparse
import ast
import os
import sys
from typing import List, Optional

import pandas as pd


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def _parse_contexts(cell) -> List[str]:
    """
    retrieved_contexts 在 CSV 中通常是字符串形式的 Python list。
    尝试安全解析；失败则返回空。
    """
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    s = str(cell).strip()
    if not s or s == "[]":
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        return []
    return []


def main():
    parser = argparse.ArgumentParser(description="Analyze RAGAS evaluation results (rag_eval_results.csv).")
    parser.add_argument("--csv", default="rag_eval_results.csv", help="Path to rag_eval_results.csv")
    parser.add_argument("--topn", type=int, default=10, help="Top N rows to show for worst cases")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"rows: {len(df)}")
    print("columns:", df.columns.tolist())

    # contexts stats
    if "retrieved_contexts" in df.columns:
        ctx_lists = df["retrieved_contexts"].apply(_parse_contexts)
        df["_ctx_items"] = ctx_lists.apply(len)
        df["_ctx_chars"] = ctx_lists.apply(lambda xs: sum(len(x) for x in xs))
        print("\n=== contexts ===")
        print("non-empty:", int((df["_ctx_items"] > 0).sum()), "/", len(df))
        print("avg items:", float(df["_ctx_items"].mean()))
        print("avg chars:", float(df["_ctx_chars"].mean()))

    print("\n=== metrics: mean ===")
    print(df[metric_cols].mean(numeric_only=True))
    print("\n=== metrics: median ===")
    print(df[metric_cols].median(numeric_only=True))
    print("\n=== metrics: null counts ===")
    print(df[metric_cols].isna().sum())
    print("\n=== metrics: zero counts ===")
    print((df[metric_cols] == 0).sum())

    # Suspected LLM failures / empty responses
    if "response" in df.columns:
        resp = df["response"].fillna("").astype(str)
        llm_fail = resp.str.startswith("[LLM调用失败]")
        empty = resp.str.strip().str.len() == 0
        print("\n=== response failures ===")
        print("LLM失败:", int(llm_fail.sum()))
        print("空回答:", int(empty.sum()))
        if int(llm_fail.sum()) > 0:
            print("\n-- sample LLM failures --")
            sample = df[llm_fail][["user_input"] + metric_cols].head(min(args.topn, int(llm_fail.sum())))
            print(sample.to_string(index=False))

    # Retrieval failures
    print("\n=== retrieval failures (precision=0 & recall=0) ===")
    rf = df[(df["context_precision"] == 0) & (df["context_recall"] == 0)]
    print("count:", len(rf))
    if len(rf) > 0:
        show = rf[["user_input"] + metric_cols].head(args.topn)
        print(show.to_string(index=False))

    # Worst by answer relevancy
    print(f"\n=== worst {args.topn} by answer_relevancy ===")
    print(df.sort_values("answer_relevancy")[["user_input"] + metric_cols].head(args.topn).to_string(index=False))

    # Worst by faithfulness (non-null)
    if df["faithfulness"].notna().any():
        print(f"\n=== worst {args.topn} by faithfulness (non-null) ===")
        w = df[df["faithfulness"].notna()].sort_values("faithfulness")[["user_input"] + metric_cols].head(args.topn)
        print(w.to_string(index=False))


if __name__ == "__main__":
    main()


