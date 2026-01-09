def banking77_final_reward_fn(example, pred):
    return 1 if pred == example.get("label") else 0