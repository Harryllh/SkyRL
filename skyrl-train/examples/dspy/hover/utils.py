from typing import List, Dict, Any
import dspy
import os
import pickle

def _prepare_passages(path: str) -> list[str]:
    if not os.path.exists(path):
        return None
    else:
        with open(path, "rb") as f:
            return pickle.load(f)

class HoverRetrieverLocal(dspy.RetrieverLocal):
    def __init__(self, corpus_path: str):
        self.passages = _prepare_passages(corpus_path)

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

def hover_query_reward_fn(example, pred):
    gold_titles = example.titles
    retrieved_titles = [doc.split(" | ")[0] for doc in pred.retrieved_docs]
    return sum(x in retrieved_titles for x in set(gold_titles)) / len(gold_titles)

def hover_final_reward_fn(example, pred):
    #TODO
    return 0.0
    
