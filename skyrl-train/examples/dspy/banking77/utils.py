from .data import CLASSES
# Final task reward (correctness)
async def banking77_final_reward_fn(example, pred, trace=None):
    label = pred.get("label")
    gold = example.get("label")

    if label is None:
        return 0.0

    return 1.0 if label == gold else 0.0


# No local reward needed for this task
async def banking77_local_reward_fn(example, pred):
    return 0.0