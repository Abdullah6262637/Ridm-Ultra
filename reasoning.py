"""Cok-adimli (multi-hop) rafine etme zinciri + entropi-tabanli
kapali-form planlayici/denetleyici (Sinirlama #9: planner/controller).
"""
import numpy as np

from .utils import _entropy


class ReasoningChain:
    """Cok-adimli 'muhakeme' dongusu: tek bir SVD gecisi yerine, baglam
    vektorunu H adim boyunca ATTENTION -> GRAF YAYILIMI -> PKM GERI-CAGIRMA
    modulleri arasinda dolastirarak rafine eder. Her hop, bir onceki hop'un
    ciktisini yeni bir 'sorgu' olarak kullanir (sabit-noktali rafine etme
    dongusu). Bu, Transformer'daki 'katman katman islenen token' fikrinin,
    GRADYANLA OGRENILMEYEN ama YINE DE COK ADIMLI bir yaklasimidir: H arttikca
    cikti yakinsayabilir, ama ara adimlar dolayli iliskileri (A ilgilidir B'ye,
    B ilgilidir C'ye -> A'nin sorgusu C'ye de agirlik kazanir) v3'teki tek
    toplamli modelden daha iyi yakalayabilir."""

    def __init__(self, ridm, attention, graph=None, pkm=None, sdm=None, hops=2, blend=0.5, early_stop_eps=None):
        self.ridm = ridm
        self.attn = attention
        self.graph = graph
        self.pkm = pkm
        self.sdm = sdm
        self.hops = max(1, hops)
        self.blend = blend
        self.early_stop_eps = early_stop_eps  # None -> her zaman tam 'hops' adim calisir

    def refine(self, context_token_ids, hops=None):
        ctx_ids = [t for t in context_token_ids if 0 <= t < self.ridm.V]
        if not ctx_ids:
            return np.zeros(self.ridm.dim), []
        cvecs = self.ridm.context_vecs[ctx_ids]
        query = cvecs.sum(axis=0)
        trace = []
        n_hops = self.hops if hops is None else max(1, hops)
        for _ in range(n_hops):
            attn_out, _weights = self.attn.forward(cvecs, query_vec=query)
            new_query = self.blend * query + (1 - self.blend) * attn_out

            if self.graph is not None:
                act = self.graph.spreading_activation(ctx_ids, steps=1, decay=0.5)
                top = np.argsort(act)[::-1][:5]
                if len(top) > 0 and act[top[0]] > 0:
                    graph_vec = self.ridm.context_vecs[top].mean(axis=0)
                    new_query = 0.8 * new_query + 0.2 * graph_vec

            if self.pkm is not None:
                recalled = self.pkm.read(new_query)
                if np.linalg.norm(recalled) > 0:
                    new_query = 0.85 * new_query + 0.15 * recalled

            if self.sdm is not None:
                recalled_k = self.sdm.read(new_query)  # k-boyutlu (word_emb uzayinda)
                if np.linalg.norm(recalled_k) > 0:
                    back_to_dim = self.ridm._Vt_k.T @ recalled_k
                    new_query = 0.9 * new_query + 0.1 * back_to_dim

            delta = float(np.linalg.norm(new_query - query))
            trace.append(delta)
            query = new_query
            if self.early_stop_eps is not None and delta < self.early_stop_eps:
                break
        return query, trace


# ======================================================
# 13) COOCCURRENCE RELATION BASIS - attention'a VERI-TUREVLI (Wq/Wk RASTGELE
#     DEGIL) Q/K rolleri kazandirir
# ======================================================


class ReasoningController:
    """KAPALI-FORM 'planlayici/denetleyici': RIDM'in ham tahmininin
    ENTROPISINE (belirsizligine) bakarak KAC HOP calisacagina kural-tabanli
    olarak karar verir; ReasoningChain'in erken-durma (early-stop) esigine
    gore de GERCEKTEN erken durur. Bu bir OGRENILEN 'gating network' DEGILDIR
    (ACT/PonderNet gibi egitilen adaptif-hesaplama yontemlerinden farkli
    olarak agirliksizdir); yine de hesaplamayi baglamin GUCLUGUNE gore
    dinamik yonlendiren kural-tabanli bir plancidir - kolay/belirgin
    baglamlarda ucuz (az hop), zor/belirsiz baglamlarda daha derin (cok-hop)
    islem yapar."""

    def __init__(self, ridm, reasoning_chain, max_hops=4, min_hops=1, entropy_high=2.0):
        self.ridm = ridm
        self.chain = reasoning_chain
        self.max_hops = max_hops
        self.min_hops = min_hops
        self.entropy_high = entropy_high

    def plan_and_run(self, context_token_ids):
        base_probs = self.ridm.softmax_probs(context_token_ids)
        ent = _entropy(base_probs)
        planned_hops = int(np.clip(
            round(self.min_hops + (ent / max(self.entropy_high, 1e-6)) * (self.max_hops - self.min_hops)),
            self.min_hops, self.max_hops,
        ))
        query, trace = self.chain.refine(context_token_ids, hops=planned_hops)
        return query, {"entropy": ent, "planned_hops": planned_hops, "effective_hops": len(trace), "trace": trace}


# ======================================================
# HIBRIT MODEL v4
# ======================================================
