"""Guven-tabanli (entropi) kapali-form Mixture-of-Experts yonlendirme."""
from .utils import _entropy


class MixtureOfExperts:
    def __init__(self):
        self.experts = {}

    def add_expert(self, name, hybrid_lm):
        self.experts[name] = hybrid_lm

    def route(self, context_token_ids, context_words=None, temperature=1.0):
        if not self.experts:
            raise RuntimeError("Hic uzman eklenmedi.")
        scored = {}
        for name, expert in self.experts.items():
            p = expert.probs(context_token_ids, temperature=temperature, context_words=context_words)
            conf = 1.0 / (_entropy(p) + 1e-3)
            scored[name] = (conf, p)
        total_conf = sum(c for c, _ in scored.values())
        mixed = None
        weights = {}
        for name, (conf, p) in scored.items():
            w = conf / total_conf
            weights[name] = w
            mixed = w * p if mixed is None else mixed + w * p
        return mixed, weights

