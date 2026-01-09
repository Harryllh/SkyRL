from re import S
import dspy
from typing import List
from dspy.adapters import XMLAdapter
from dspy.dsp.utils import deduplicate
from dspy.adapters.xml_adapter import XMLAdapter
from .data import CLASSES

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