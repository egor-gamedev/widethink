"""Wide thinking with an open model behind an OpenAI-compatible server.

Start a server first, for example with vLLM:

    vllm serve Qwen/Qwen3-32B --port 8000

then:

    pip install "widethink[openai]"
    python examples/openai_compatible.py "Add caching to /products" ./my-project

Everything stays on your machine: the model, the embeddings and the context.
Set ``structured`` to ``json_object`` or ``prompt`` if your server does not
support strict JSON schemas.
"""

from __future__ import annotations

import sys

from widethink import ThinkConfig, Thinker
from widethink.embeddings import SentenceTransformerEmbedder
from widethink.llm import OpenAICompatibleLLM

BASE_URL = "http://localhost:8000/v1"


def main() -> None:
    task, project = sys.argv[1], sys.argv[2]
    llm = OpenAICompatibleLLM(
        "Qwen/Qwen3-32B", base_url=BASE_URL, api_key="not-needed", structured="json_schema"
    )
    thinker = Thinker(
        llm,
        embedder=SentenceTransformerEmbedder(),  # pip install "widethink[local]"
        config=ThinkConfig(max_thoughts=16, parallel=2),
    )
    result = thinker.think(task, project, budget=120_000)
    print(result.answer)
    print(result.render())


if __name__ == "__main__":
    main()
