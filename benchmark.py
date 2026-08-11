"""Genel kok-secmeli (multiple-choice) benchmark kosum motoru
(Sinirlama #5: 'gercek benchmark eksik' geri bildirimine yanit).

DURUSTLUK: Bu ortamin internet erisimi YOKTUR; dolayisiyla gercek
MMLU / HellaSwag / ARC / GSM8K veri kumelerini bu kod INDIREMEZ ve
'MMLU skoru X%' gibi bir sayi UYDURULAMAZ (uydurmak yanlis bilgi
olurdu). Burada saglanan, bu formatlardaki JSONL dosyalarini KULLANICI
kendi makinesine indirip --benchmark-file ile verdiginde CALISACAK
genel bir degerlendirme motorudur, artı gercek veri olmadan da en
azindan TEMEL yeteneklerin olculebilmesi icin birkac kucuk, YERLESIK
sentetik gorev (bunlar da 'benchmark' degil, sadece akil-saligi
kontrolleridir - istatistiksel guvenilirlikleri yoktur).

Beklenen JSONL formati (MMLU/ARC-benzeri, satir-basi bir JSON nesnesi):
    {"question": "...", "choices": ["secenek A", "secenek B", ...], "answer": 2}
'answer' dogru secenegin (0-tabanli) indeksidir.
"""

import json
import math

import numpy as np

from .constants import UNK_TOKEN


class BenchmarkRunner:
    def __init__(self, hybrid, ridm):
        self.hybrid = hybrid
        self.ridm = ridm

    def _score_choice(self, context_words, choice_text):
        """Bir secenegin, baglam verildiginde KELIME-BASINA ORTALAMA
        log-olasiligini hesaplar - cok-secmeli degerlendirmede standart
        'per-token likelihood' yaklasimidir (MMLU/HellaSwag kosucularinin
        buyuk cogunlugunun kullandigi ilkeyle aynidir)."""
        ctx = list(context_words)
        total_logp = 0.0
        n = 0
        unk_id = self.ridm.word2idx[UNK_TOKEN]
        for w in choice_text.split():
            ctx_ids = [self.ridm.word2idx.get(cw, unk_id) for cw in ctx[-self.ridm.window :]]
            probs = self.hybrid.probs(ctx_ids, context_words=ctx[-self.ridm.window :])
            wid = self.ridm.word2idx.get(w, unk_id)
            total_logp += math.log(max(probs[wid], 1e-12))
            n += 1
            ctx.append(w)
        return total_logp / max(n, 1)

    def run_multiple_choice(self, items):
        """items: [{'question': str, 'choices': [str, ...], 'answer': int}, ...]"""
        correct = 0
        for item in items:
            q_words = item["question"].split()
            scores = [self._score_choice(q_words, c) for c in item["choices"]]
            pred = int(np.argmax(scores))
            correct += int(pred == item["answer"])
        return {"accuracy": correct / max(len(items), 1), "n_items": len(items)}

    def run_jsonl_file(self, path):
        """MMLU/ARC/HellaSwag-formatinda yerel bir JSONL dosyasini calistirir.
        Dosya YOKSA veya BOSSA acikca hata verir (sessizce sahte sonuc
        uretmez)."""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        if not items:
            raise ValueError(f"'{path}' dosyasinda hic ogre (item) bulunamadi.")
        return self.run_multiple_choice(items)

    def synthetic_analogy_task(self, pairs):
        """KUCUK, YERLESIK bir analoji saglik-kontrolu: (a,b,c,d) dortlusunde
        'b-a+c' vektorune en yakin kelimenin d olup olmadigina bakar. GERCEK
        bir benchmark DEGILDIR - sadece RIDM'in dogrusal analoji yapisini
        (word2vec'teki 'king-man+woman=queen' fikrinin kucuk-olcekli hali)
        yakalayip yakalamadigina dair hizli bir gosterge."""
        correct, total = 0, 0
        for a, b, c, d in pairs:
            if not all(w in self.ridm.word2idx for w in (a, b, c, d)):
                continue
            va = self.ridm.context_vecs[self.ridm.word2idx[a]]
            vb = self.ridm.context_vecs[self.ridm.word2idx[b]]
            vc = self.ridm.context_vecs[self.ridm.word2idx[c]]
            target = vb - va + vc
            norms = np.linalg.norm(self.ridm.context_vecs, axis=1) * np.linalg.norm(target) + 1e-8
            sims = (self.ridm.context_vecs @ target) / norms
            pred_id = int(np.argmax(sims))
            total += 1
            correct += int(self.ridm.idx2word[pred_id] == d)
        return {"accuracy": (correct / total) if total else float("nan"), "n_items": total}

    def synthetic_cloze_task(self, token_ids, n_items=50, seed=0):
        """Rastgele bosluk-doldurma saglik-kontrolu: standart evaluate()
        ile ayni ilke, 'benchmark' arayuzuyle standartlastirilmis hali."""
        rng = np.random.RandomState(seed)
        W = self.ridm.window
        n = len(token_ids)
        if n <= W:
            return {"accuracy": float("nan"), "n_items": 0}
        pool = np.arange(W, n)
        idxs = rng.choice(pool, size=min(n_items, len(pool)), replace=False)
        correct = 0
        for i in idxs:
            ctx = token_ids[i - W : i]
            target = token_ids[i]
            probs = self.hybrid.probs(ctx)
            correct += int(np.argmax(probs) == target)
        return {"accuracy": correct / len(idxs), "n_items": int(len(idxs))}
