from typing import List, Dict, Any
import dspy
import json

def _prepare_passages(path: str) -> list[str]:
    """
    Loads and parses a JSONL file located at `path`, returning a list of text passages.
    Each line is a JSON object with keys:
      - 'pid' (int)
      - 'title' (str)
      - 'text' (list[str])  # list of sentences
    """
    passages = []

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "long_text" in obj and isinstance(obj["long_text"], str):
                passage = obj["long_text"].strip()
                if passage:
                    passages.append(passage)
                continue

            text = obj.get("text")
            if not isinstance(text, list) or not text:
                continue

            body = " ".join(s.strip() for s in text if s and s.strip())
            if not body:
                continue

            title = obj.get("title", "").strip()
            if title:
                passage = f"{title}\n{body}"
            else:
                passage = body

            passages.append(passage)

    return passages

    
class HoverRetrieverLocal(dspy.RetrieverLocal):
    def __init__(self, path: str):
        self.path = path
        self.passages = _prepare_passages(path)

        self.config = dspy.ColBERTConfig(
            checkpoint="colbert-ir/colbertv2.0",
            index_name="wiki17_abstracts_hover",
            experiment="hover",
            nranks=1,
        )

        self.rm = dspy.ColBERTv2RetrieverLocal(
            passages=self.passages,
            colbert_config=self.config,
            load_only=False,
        )

    def retrieve(self, query: str, k: int) -> list[str]:
        topK_passages = self.rm(query, k)

        assert isinstance(topK_passages, list)
        assert len(topK_passages) <= k

        return topK_passages

    
