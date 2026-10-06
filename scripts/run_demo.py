"""Run G-reasoner retrieval on a prebuilt graph.

Run from the repo folder:  python scripts/run_demo.py
Always run this with `python`, never by importing gfmrag in a notebook cell.
"""

import argparse
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch_geometric

# gfmrag 2.0.0 crashes on PyG versions like "2.8.0.post1". Trim it to "2.8.0".
torch_geometric.__version__ = ".".join(re.findall(r"\d+", torch_geometric.__version__)[:3])

from gfmrag import GFMRetriever


class StringMatchNER:
    """Stand-in for gfmrag's LLM entity extractor (which needs an API key).
    Returns the graph entities whose names appear in the question."""

    def __init__(self, entity_names):
        self.entity_names = entity_names

    def __call__(self, text, *args, **kwargs):
        question = text.lower()
        return [name for name in self.entity_names if name.lower() in question]


class ExactMatchEL:
    """Stand-in for gfmrag's entity linker. Links a mention to the graph node
    with exactly the same name, and skips anything it can't find."""

    def __init__(self):
        self.nodes = set()

    def index(self, entity_list):
        self.nodes = set(entity_list)

    def __call__(self, ner_entity_list, topk=1, *args, **kwargs):
        return {
            mention: [{"entity": mention, "score": 1.0}]
            for mention in ner_entity_list
            if mention in self.nodes
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--data_name", default="toy")
    parser.add_argument("--model_path", default="rmanluo/G-reasoner-34M")
    parser.add_argument("--query", default="Who is the president of France?")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    nodes_csv = os.path.join(args.data_dir, args.data_name, "processed", "stage1", "nodes.csv")
    nodes = pd.read_csv(nodes_csv, keep_default_na=False)
    entity_names = nodes.loc[nodes["type"] == "entity", "name"].tolist()
    ner_model = StringMatchNER(entity_names)

    # No embedding patches here on purpose: gfmrag uses its own Qwen3 embedder
    # for both the graph and the question, so the two always match.
    retriever = GFMRetriever.from_index(
        data_dir=args.data_dir,
        data_name=args.data_name,
        model_path=args.model_path,
        ner_model=ner_model,
        el_model=ExactMatchEL(),
    )

    results = retriever.retrieve(args.query, top_k=args.top_k)

    print("\n==========================================")
    print(f"Question: {args.query}")
    print(f"Start entities: {ner_model(args.query)}")
    print(f"Top {args.top_k} documents ({args.model_path}):")
    for rank, item in enumerate(results["document"], start=1):
        print(f"  {rank}. {item['id']}  (score {item['score']:.3f})")
    print("==========================================")


# This guard is required. Without it the embedding model fails to start.
if __name__ == "__main__":
    main()