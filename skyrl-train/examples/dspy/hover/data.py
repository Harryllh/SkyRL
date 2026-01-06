import random
import ujson
import dspy
import os
import pickle
import tqdm
from datasets import load_dataset

def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))

def prepare_corpus(input_path: str, output_path: str) -> None:
    if not os.path.exists(output_path):
        corpus = []
        with open(input_path, "r") as f:
            for line in tqdm.tqdm(f):
                line = ujson.loads(line)
                corpus.append(f"{line['title']} | {' '.join(line['text'])}")
        with open(output_path, "wb") as f:
            pickle.dump(corpus, f)
    
def hover_data():
    print("Loading hover-nlp/hover dataset...")
    dataset = load_dataset("hover", trust_remote_code=True)

    print("Dataset loaded.")
    hf_trainset = dataset["train"]
    hf_testset = dataset[
        "validation"
    ]  # Using validation dataset because test dataset is not labeled

    print("Reformatting dataset...")
    reformatted_hf_trainset = []
    reformatted_hf_testset = []

    for example in tqdm.tqdm(hf_trainset):
        claim = example["claim"]
        supporting_facts = example["supporting_facts"]
        label = example["label"]

        if count_unique_docs(example) == 3:  # Limit to 3 hop examples
            reformatted_hf_trainset.append(
                dict(claim=claim, supporting_facts=supporting_facts, label=label)
            )

    for example in tqdm.tqdm(hf_testset):
        claim = example["claim"]
        supporting_facts = example["supporting_facts"]
        label = example["label"]

        reformatted_hf_testset.append(
            dict(claim=claim, supporting_facts=supporting_facts, label=label)
        )

    print("Shuffling dataset...")
    rng = random.Random()
    rng.seed(0)
    rng.shuffle(reformatted_hf_trainset)
    rng = random.Random()
    rng.seed(1)
    rng.shuffle(reformatted_hf_testset)

    print("Dataset shuffled.")

    print("Dataset reformatted and shuffled.")

    trainset = reformatted_hf_trainset
    testset = reformatted_hf_testset

    trainset = [dspy.Example(**x).with_inputs("claim") for x in trainset]
    testset = [dspy.Example(**x).with_inputs("claim") for x in testset]

    print("Preparing corpus...")
    prepare_corpus("wiki.abstracts.2017.jsonl", "corpus.pkl")
    print("Corpus prepared.")

    return trainset, testset