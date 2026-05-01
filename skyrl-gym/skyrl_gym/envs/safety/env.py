import re
from typing import Any, Dict, Set
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


def _normalize_category(category: str) -> str:
    return re.sub(r"\s+", " ", category.strip().lower())


class SafetyEnv(BaseTextEnv):
    """
    Single-turn environment for prompt safety classification.

    Expects the model to output a line matching:
        "User Safety: unsafe"  ->  predicted unsafe
        "User Safety: safe"    ->  predicted safe

    reward_spec.ground_truth must be "safe" or "unsafe".
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec"
        reward_spec = extras["reward_spec"]
        self.ground_truth = reward_spec["ground_truth"].lower()
        assert self.ground_truth in ("safe", "unsafe"), f"ground_truth must be 'safe' or 'unsafe', got {self.ground_truth!r}"
        self.ground_truth_categories = self._normalize_categories(reward_spec.get("categories", []))
        self.latest_metrics: Dict[str, Any] = {}

    @staticmethod
    def _normalize_categories(categories) -> Set[str]:
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",")]
        return {_normalize_category(str(c)) for c in categories if str(c).strip()}

    def _parse_categories(self, action: str) -> Set[str]:
        match = re.search(r"Safety\s+Categories:\s*(.+)", action, re.IGNORECASE)
        if match is None:
            return set()
        return self._normalize_categories(match.group(1))

    def _get_category_reward(self, predicted_label: str, action: str) -> tuple[float, Set[str]]:
        if not self.ground_truth_categories:
            return 0.0, set()
        if self.ground_truth != "unsafe" or predicted_label != "unsafe":
            return 0.0, set()

        predicted_categories = self._parse_categories(action)
        if predicted_categories == self.ground_truth_categories:
            category_reward = 0.25
        else:
            category_reward = 0
        return category_reward, predicted_categories

    def _score_action(self, action: str) -> tuple[float, Dict[str, Any]]:
        match = re.search(r"User\s+Safety:\s*(safe|unsafe)\b", action, re.IGNORECASE)
        if match is None:
            final_reward = 0.0  # no format reward - unparseable output
            return final_reward, {
                "safety_reward": 0.0,
                "category_reward": 0.0,
                "final_reward": final_reward,
                "parsed_label": None,
                "parsed_categories": [],
            }
        predicted_label = match.group(1).lower()
        if predicted_label == self.ground_truth:
            safety_reward = 1.0
        elif predicted_label == "unsafe" and self.ground_truth == "safe":
            safety_reward = -1.5
        else:
            safety_reward = -1.0

        category_reward, predicted_categories = self._get_category_reward(predicted_label, action)
        # final_reward = safety_reward + category_reward
        final_reward = safety_reward
        return final_reward, {
            "safety_reward": safety_reward,
            "category_reward": category_reward,
            "final_reward": final_reward,
            "parsed_label": predicted_label,
            "parsed_categories": sorted(predicted_categories),
        }

    def _get_reward(self, action: str) -> float:
        reward, _ = self._score_action(action)
        return reward

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward, metrics = self._score_action(action)
        self.latest_metrics = metrics
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={
                "ground_truth": self.ground_truth,
                "ground_truth_categories": list(self.ground_truth_categories),
                **metrics,
            },
        )

    def get_metrics(self) -> Dict[str, Any]:
        return self.latest_metrics
