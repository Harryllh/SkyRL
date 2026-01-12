import dspy
from typing import List
from .data import CLASSES
from dspy.adapters import XMLAdapter
from dspy.dsp.utils import deduplicate

class Banking77(dspy.Module):
    def __init__(self):
        self.intent_classifier = dspy.ChainOfThought(dspy.Signature(f"text -> label: Literal{CLASSES}"))
        self.adapter = XMLAdapter()

    def forward(self, text: str) -> str:
        intent = self.intent_classifier(text=text)
        
        return intent
    
class Banking77_intent_classifier(Banking77):
    def __init__(self):
        super().__init__()
        self.intent_classifier_traces = []
        self.intents = []
        
        self.intent_classifier.set_lm(dspy.LM(
            model="openai/Qwen/Qwen2.5-1.5B",
            api_base="http://127.0.0.1:8000/v1",
            api_key="fake-key",
            temperature=1.0,
            model_type="chat",
            max_tokens=4096,
            cache=False,
        ))
        
    async def forward(self, example) -> str:
        text = example.get("text")
        intent = await self.intent_classifier.acall(text=text)
        
        self.append_trace(example, intent)
        return intent
    
    def append_trace(self, kwargs, pred):
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