from datasets import load_dataset

import dspy
from dspy.datasets import DataLoader

CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features["label"].names

def banking77_data():
    kwargs = {"fields": ("text", "label"), "input_keys": ("text",), "split": "train", "trust_remote_code": True}

    trainset = [
        dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
        for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
    ]
    validationset = [
        dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
        for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[2000: 4000]
    ]
    
    return trainset, validationset




    
