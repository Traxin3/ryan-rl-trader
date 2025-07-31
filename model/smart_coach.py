import re
import json
import numpy as np
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SmartCoach:
    def __init__(self, config: Dict):
        self.config = config['coach']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config['llm_model'],
            torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llm_model'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.param_bounds = {
            k: tuple(v) for k, v in self.config['param_bounds'].items()
        }

    def analyze_and_adjust(self, model, recent_rewards: List[float]) -> bool:
        """Analyze rewards and apply safe PPO adjustments if needed."""
        diagnosis = self._diagnose(recent_rewards)
        recommendation = self._get_recommendation(diagnosis)

        if self._validate(recommendation):
            return self._apply(model, recommendation)
        else:
            print("[SmartCoach] ⚠️ LLM recommendation invalid or out of bounds. Ignored.")
        return False


    def _diagnose(self, rewards: List[float]) -> Dict:
        """Simple diagnosis from rewards"""
        return {
            'avg': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'trend': 'up' if rewards[-1] > rewards[0] else 'down'
        }

    def _get_recommendation(self, diagnosis: Dict) -> Dict:
        """Ask the LLM for a single param adjustment in JSON"""
        prompt = f"""Current PPO Performance:
- Avg Reward: {diagnosis['avg']:.2f}
- Std: {diagnosis['std']:.2f}
- Trend: {diagnosis['trend']}

Suggest ONE PPO parameter change as JSON only:
{{
  "parameter": "learning_rate|ent_coef|clip_range|gamma",
  "change_pct": -10 to +10
}}"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3
        )
        raw_text = self.tokenizer.decode(outputs[0])

        return self._parse_output(raw_text)

    def _parse_output(self, text: str) -> Dict:
        """Try to extract JSON from LLM output robustly"""
        try:
            return json.loads(re.search(r'\{.*\}', text).group(0))
        except Exception:
            return {}

    def _validate(self, recommendation: Dict) -> bool:
        """Check that LLM suggestion is safe and makes sense"""
        required = {'parameter', 'change_pct'}
        if not all(k in recommendation for k in required):
            return False

        param = recommendation['parameter']
        if param not in self.param_bounds:
            return False

        try:
            change = float(recommendation['change_pct'])
            return -10 <= change <= 10
        except Exception:
            return False

    def _apply(self, model, adjustment: Dict) -> bool:
        """Apply parameter change to PPO model live"""
        param = adjustment['parameter']
        change_pct = float(adjustment['change_pct'])

        if param == 'learning_rate':
            for g in model.policy.optimizer.param_groups:
                g['lr'] *= (1 + change_pct / 100)
            new_val = model.policy.optimizer.param_groups[0]['lr']

        elif param == 'clip_range':
            old_clip = model.clip_range
            factor = (1 + change_pct / 100)
            model.clip_range = lambda _: old_clip(1) * factor if callable(old_clip) else old_clip * factor
            new_val = old_clip(1) * factor if callable(old_clip) else old_clip * factor

        else:
            current = getattr(model, param)
            new_val = current * (1 + change_pct / 100)
            setattr(model, param, new_val)

        print(f"[SmartCoach] ✅ Adjusted {param}: {change_pct:+.1f}% → new value: {new_val:.6f}")
        return True
