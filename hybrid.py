"""Tum alt-sistemleri carpimsal-bonus semasiyla birlestiren HibritLM."""
import math

import numpy as np

from .constants import SENTENCE_BOUNDARY_TOKENS, UNK_TOKEN
from .utils import detokenize


class HybridLM:
    def __init__(self, ridm, ngram, alpha=0.5, graph=None, sdm=None, rag=None,
                 graph_weight=0.15, sdm_weight=0.15, rag_weight=0.15,
                 reasoning=None, reasoning_weight=0.20,
                 reservoir=None, reservoir_weight=0.15,
                 relation_attn=None, relation_attn_weight=0.15,
                 calibrator=None, calibrator_weight=0.20,
                 block_stack=None, block_stack_weight=0.15,
                 controller=None):
        self.ridm = ridm
        self.ngram = ngram
        self.alpha = alpha
        self.graph = graph
        self.sdm = sdm
        self.rag = rag
        self.graph_weight = graph_weight
        self.sdm_weight = sdm_weight
        self.rag_weight = rag_weight
        self.reasoning = reasoning
        self.reasoning_weight = reasoning_weight
        self.reservoir = reservoir
        self.reservoir_weight = reservoir_weight
        self.relation_attn = relation_attn
        self.relation_attn_weight = relation_attn_weight
        self.calibrator = calibrator
        self.calibrator_weight = calibrator_weight
        self.block_stack = block_stack
        self.block_stack_weight = block_stack_weight
        self.controller = controller  # verilirse, statik 'reasoning' yerine dinamik hop-planlama kullanilir

    def probs(self, context_token_ids, temperature=1.0, context_words=None):
        p_r = self.ridm.softmax_probs(context_token_ids, temperature=temperature)
        p_n = self.ngram.probs(context_token_ids)
        mixed = (p_r ** (1 - self.alpha)) * (p_n ** self.alpha)

        if self.graph is not None:
            act = self.graph.spreading_activation(context_token_ids, steps=1, decay=0.5)
            if act.max() > 0:
                mixed = mixed * (1.0 + self.graph_weight * act / act.max())

        if self.sdm is not None:
            addr = self.ridm.context_vector_for(
                [self.ridm.idx2word[t] for t in context_token_ids if 0 <= t < self.ridm.V]
            )
            recalled = self.sdm.read(addr)
            sdm_logits = self.ridm.word_emb @ recalled
            sdm_score = np.clip(sdm_logits - sdm_logits.min(), 0, None)
            if sdm_score.max() > 0:
                mixed = mixed * (1.0 + self.sdm_weight * sdm_score / sdm_score.max())

        if self.rag is not None and context_words is not None:
            bonus = self.rag.prior_bonus(context_words)
            if bonus.max() > 0:
                mixed = mixed * (1.0 + self.rag_weight * bonus / bonus.max())

        if self.controller is not None:
            final_vec, _info = self.controller.plan_and_run(context_token_ids)
            r_logits = self.ridm.word_emb @ (self.ridm._Vt_k @ final_vec)
            r_score = np.clip(r_logits - r_logits.min(), 0, None)
            if r_score.max() > 0:
                mixed = mixed * (1.0 + self.reasoning_weight * r_score / r_score.max())
        elif self.reasoning is not None:
            final_vec, _trace = self.reasoning.refine(context_token_ids)
            r_logits = self.ridm.word_emb @ (self.ridm._Vt_k @ final_vec)
            r_score = np.clip(r_logits - r_logits.min(), 0, None)
            if r_score.max() > 0:
                mixed = mixed * (1.0 + self.reasoning_weight * r_score / r_score.max())

        if self.reservoir is not None:
            res_logits = self.reservoir.score(context_token_ids)
            if res_logits is not None:
                res_score = np.clip(res_logits - res_logits.min(), 0, None)
                if res_score.max() > 0:
                    mixed = mixed * (1.0 + self.reservoir_weight * res_score / res_score.max())

        if self.relation_attn is not None:
            rel_logits = self.relation_attn.forward(context_token_ids)
            rel_score = np.clip(rel_logits - rel_logits.min(), 0, None)
            if rel_score.max() > 0:
                mixed = mixed * (1.0 + self.relation_attn_weight * rel_score / rel_score.max())

        if self.block_stack is not None:
            ctx_ids = [t for t in context_token_ids if 0 <= t < self.ridm.V]
            cvecs = self.ridm.context_vecs[ctx_ids] if ctx_ids else np.zeros((0, self.ridm.dim))
            out_vec = self.block_stack.forward(cvecs)
            b_logits = self.ridm.word_emb @ (self.ridm._Vt_k @ out_vec)
            b_score = np.clip(b_logits - b_logits.min(), 0, None)
            if b_score.max() > 0:
                mixed = mixed * (1.0 + self.block_stack_weight * b_score / b_score.max())

        if self.calibrator is not None:
            cal_logits = self.calibrator.calibrated_logits(context_token_ids)
            if cal_logits is not None:
                cal_score = np.clip(cal_logits - cal_logits.min(), 0, None)
                if cal_score.max() > 0:
                    mixed = mixed * (1.0 + self.calibrator_weight * cal_score / cal_score.max())

        s = mixed.sum()
        if s <= 0 or not np.isfinite(s):
            return p_n
        return mixed / s

    def evaluate(self, token_ids, max_samples=1000):
        n = len(token_ids)
        w = self.ridm.window
        if n <= w:
            return {"accuracy": float("nan"), "perplexity": float("nan"), "n_samples": 0}
        correct, nll_sum, count = 0, 0.0, 0
        step = max(1, (n - w) // max_samples) if max_samples else 1
        for i in range(w, n, step):
            ctx = token_ids[i - w : i]
            target = token_ids[i]
            p = self.probs(ctx)
            correct += int(np.argmax(p) == target)
            nll_sum += -math.log(max(p[target], 1e-12))
            count += 1
            if max_samples and count >= max_samples:
                break
        ppl = math.exp(nll_sum / count) if count else float("nan")
        return {"accuracy": correct / count if count else float("nan"), "perplexity": ppl, "n_samples": count}

    @classmethod
    def tune_alpha(cls, ridm, ngram, val_ids, max_samples=500, grid=11):
        best_alpha, best_ppl = 0.0, float("inf")
        for a in np.linspace(0.0, 1.0, grid):
            hyb = cls(ridm, ngram, alpha=a)
            res = hyb.evaluate(val_ids, max_samples=max_samples)
            if res["perplexity"] < best_ppl:
                best_ppl, best_alpha = res["perplexity"], a
        return best_alpha, best_ppl

    def generate(self, seed_words, length=40, temperature=1.0, top_k=10, top_p=None,
                 seed=None, num_sentences=None, repetition_penalty=1.3,
                 avoid_unk=True, max_unk_resample=5):
        """Baglam-penceresi olasiliklarindan orneklem yaparak devam metni uretir.

        Eski surum sadece serbest bir kelime dizisi uretiyordu: hicbir cumle
        sinir kavrami yoktu (SENTENCE_BOUNDARY_TOKENS tanimliydi ama burada
        HIC kullanilmiyordu - bkz. hierarchical.py'deki tek kullanim yeri),
        tekrar cezasi yoktu, ve <unk> serbestce uretilebiliyordu. Bu yuzden
        cikti "kelime kelime" kaliyor, gercek bir CUMLE gibi gorunmuyordu.

        Bu surum:
          - ``num_sentences`` verilirse, o kadar cumle-sonu noktalamasi
            (``. ! ?``) uretilince DURUR (cumle sinirlarini fiilen kullanir).
          - ``repetition_penalty`` (>1) az once uretilen tokenlarin
            olasiligini dusurur (dongusel/tekrarlayan cikti azalir).
          - ``top_p`` (nucleus sampling) istege bagli olarak top_k'ya ek
            filtre olarak uygulanabilir.
          - ``avoid_unk``, mumkun oldugunca <unk> uretmek yerine yeniden
            orneklem yapar (okunabilirligi buyuk olcude artirir).

        Donen deger ham token listesidir; ``generate_text()`` bunu
        noktalamasi/buyuk harfi duzgun tam bir metne cevirir.
        """
        rng = np.random.RandomState(seed)
        w2i = self.ridm.word2idx
        unk = w2i[UNK_TOKEN]
        ids = [w2i.get(w, unk) for w in seed_words]
        out_words = list(seed_words)
        window = self.ridm.window
        sentence_count = 0
        recent_counts = {}

        for _ in range(length):
            ctx = ids[-window:] if ids else []
            probs = self.probs(ctx, temperature=temperature, context_words=out_words[-window:])
            probs = np.asarray(probs, dtype=np.float64)

            if repetition_penalty and repetition_penalty != 1.0 and recent_counts:
                probs = probs.copy()
                for tok_id, cnt in recent_counts.items():
                    if 0 <= tok_id < len(probs):
                        probs[tok_id] /= (repetition_penalty ** cnt)
                total = probs.sum()
                if total > 0:
                    probs = probs / total

            if top_k and 0 < top_k < len(probs):
                top_idx = np.argpartition(probs, -top_k)[-top_k:]
                mask = np.zeros_like(probs)
                mask[top_idx] = probs[top_idx]
                probs = mask / mask.sum()

            if top_p and 0 < top_p < 1.0:
                order = np.argsort(probs)[::-1]
                cum = np.cumsum(probs[order])
                cutoff = int(np.searchsorted(cum, top_p)) + 1
                keep = order[: max(1, cutoff)]
                mask = np.zeros_like(probs)
                mask[keep] = probs[keep]
                s = mask.sum()
                if s > 0:
                    probs = mask / s

            next_id = int(rng.choice(len(probs), p=probs))

            if avoid_unk and next_id == unk and probs[unk] < 1.0:
                probs_wo_unk = probs.copy()
                probs_wo_unk[unk] = 0.0
                tries = 0
                while True:
                    s = probs_wo_unk.sum()
                    if s <= 1e-12 or tries >= max_unk_resample:
                        break
                    probs_wo_unk = probs_wo_unk / s
                    candidate = int(rng.choice(len(probs_wo_unk), p=probs_wo_unk))
                    if candidate != unk:
                        next_id = candidate
                        break
                    probs_wo_unk[candidate] = 0.0
                    tries += 1

            next_word = self.ridm.idx2word.get(next_id, UNK_TOKEN)
            out_words.append(next_word)
            ids.append(next_id)
            recent_counts[next_id] = recent_counts.get(next_id, 0) + 1

            if next_word in SENTENCE_BOUNDARY_TOKENS:
                sentence_count += 1
                if num_sentences is not None and sentence_count >= num_sentences:
                    break

        return out_words

    def generate_text(self, seed_words, **kwargs):
        """``generate()`` cagirip sonucu okunabilir, noktalamasi ve buyuk
        harfi dogru bir cumle/paragraf metnine cevirir."""
        words = self.generate(seed_words, **kwargs)
        return detokenize(words)


# ======================================================
# 7) MIXTURE-OF-EXPERTS
# ======================================================
