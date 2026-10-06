"""Build a tiny test graph in the format gfmrag expects.

Run from the repo folder:  python scripts/make_toy_data.py
Writes data/toy/raw/documents.json and data/toy/processed/stage1/*.csv
"""

import json
import os
import shutil

import pandas as pd

ROOT = "data/toy"

# 8 short documents. Only one of them answers the test question.
DOCS = {
    "France": "France is a country in Western Europe. Its capital is Paris.",
    "Paris": "Paris is the capital and most populous city of France. The Seine river runs through it.",
    "Emmanuel Macron": "Emmanuel Macron is a French politician who has served as president of France since 2017.",
    "Germany": "Germany is a country in Central Europe. Its capital is Berlin.",
    "Berlin": "Berlin is the capital and largest city of Germany.",
    "Angela Merkel": "Angela Merkel is a German politician who served as chancellor of Germany from 2005 to 2021.",
    "Eiffel Tower": "The Eiffel Tower is a wrought-iron tower in Paris, completed in 1889.",
    "Seine": "The Seine is a river in northern France that flows through Paris.",
}

# Facts between entities: (head, relation, tail). Entity names are lowercase
# so they never clash with document names.
TRIPLES = [
    ("paris", "capital of", "france"),
    ("emmanuel macron", "president of", "france"),
    ("berlin", "capital of", "germany"),
    ("angela merkel", "chancellor of", "germany"),
    ("eiffel tower", "located in", "paris"),
    ("seine", "flows through", "paris"),
]

# Which entities each document mentions.
MENTIONS = {
    "France": ["france", "paris"],
    "Paris": ["paris", "france", "seine"],
    "Emmanuel Macron": ["emmanuel macron", "france"],
    "Germany": ["germany", "berlin"],
    "Berlin": ["berlin", "germany"],
    "Angela Merkel": ["angela merkel", "germany"],
    "Eiffel Tower": ["eiffel tower", "paris"],
    "Seine": ["seine", "france", "paris"],
}


def main() -> None:
    raw_dir = os.path.join(ROOT, "raw")
    stage1_dir = os.path.join(ROOT, "processed", "stage1")
    # Old cached embeddings no longer match once the graph changes.
    shutil.rmtree(os.path.join(ROOT, "processed", "stage2"), ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(stage1_dir, exist_ok=True)

    with open(os.path.join(raw_dir, "documents.json"), "w") as f:
        json.dump(DOCS, f, indent=2)

    # Document text goes inside `attributes`. gfmrag only embeds
    # name + type + attributes, so a separate `content` column is ignored.
    nodes = [
        {"name": title, "type": "document", "attributes": str({"content": text})}
        for title, text in DOCS.items()
    ]
    entities = sorted({e for h, _, t in TRIPLES for e in (h, t)})
    nodes += [{"name": e, "type": "entity", "attributes": "{}"} for e in entities]

    # Same convention as gfmrag's own graph builder: entity -> is_mentioned_in -> document
    relation_names = sorted({r for _, r, _ in TRIPLES}) + ["is_mentioned_in"]
    relations = [{"name": r, "attributes": "{}"} for r in relation_names]

    edges = [
        {"source": h, "relation": r, "target": t, "attributes": "{}"}
        for h, r, t in TRIPLES
    ]
    edges += [
        {"source": e, "relation": "is_mentioned_in", "target": doc, "attributes": "{}"}
        for doc, ents in MENTIONS.items()
        for e in ents
    ]

    pd.DataFrame(nodes).to_csv(os.path.join(stage1_dir, "nodes.csv"), index=False)
    pd.DataFrame(relations).to_csv(os.path.join(stage1_dir, "relations.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(stage1_dir, "edges.csv"), index=False)

    print(f"Wrote {len(DOCS)} documents, {len(entities)} entities, {len(edges)} edges to {ROOT}/")


if __name__ == "__main__":
    main()