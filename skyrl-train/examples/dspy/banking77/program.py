from re import S
import dspy
from typing import List
from dspy.adapters import XMLAdapter
from dspy.dsp.utils import deduplicate
from dspy.adapters.xml_adapter import XMLAdapter
from .data import CLASSES

instr1 = """
Given a claim and some key facts, generate a follow-up search query to find the next most essential clue towards verifying or refuting the claim. The goal ultimately is to find all documents implicated by the claim.
""".strip()

instr2 = """
Given a claim, some key facts, and new search results, identify any new learnings from the new search results, which will extend the key facts known so far about the whether the claim is true or false. The goal is to ultimately collect all facts that would help us find all documents implicated by the claim.
""".strip()


class Banking77(dspy.Module):
    def __init__(self):
        self.intent_classifier = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")
        self.adapter = XMLAdapter()

    def forward(self, text: str) -> str:
        intent = self.intent_classifier(text=text)
        
        return intent
    
class Banking77_intent_classifier(Banking77):
    def __init__(self):
        super().__init__()
        self.intent_classifier_traces = []
        self.intents = []
        
    async def forward(self, example) -> str:
        text = example.get("text")
        intent = await self.intent_classifier.acall(text=text)
        return intent
    
    def append_trace(self, pred, **kwargs):
        finetune_data = self.adapter.format_finetune_data(
            signature=self.intent_classifier.predictors()[0].signature,
            inputs=kwargs,
            outputs=pred,
            demos=[] # TODO: Add support for demos
        )
        
        all_messages = finetune_data.get('messages', [])
        
        self.intent_classifier_traces.extend(all_messages)
        self.intents.append(pred)
    
    def collect_trace(self, example, pred):
        return self.intent_classifier_traces, self.intents