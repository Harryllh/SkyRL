from .data import CLASSES
# async def banking77_final_reward_fn(example, pred, trace=None):
#     return 1 if pred.get("label") == example.get("label") else 0

async def banking77_final_reward_fn(example, pred, trace=None):
    pred_label = pred.get("label")
    ref_label = example.get("label")
    
    if pred_label not in CLASSES:
        return 0
    if ref_label != pred_label:
        return 0.5
    else:
        return 1
    
async def banking77_local_reward_fn(example, pred):
    assert len(pred) == 1, "Pred should have only one element"
    
    # No local reward for this task
    return 0