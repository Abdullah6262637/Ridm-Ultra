import json
import random
from pathlib import Path

random.seed(7)
OUTPUT = Path(__file__).parent / "corpus.jsonl"
topics = [
    "Türkiye'nin coğrafi konumu, tarihi İpek Yolu güzergahları ve iklim çeşitliliği açısından önemli bir kavşak noktasıdır. Anadolu yarımadası, binlerce yıldır farklı medeniyetlere ev sahipliği yapmıştır.",
    "Yapay zeka sistemleri, büyük miktarda metin verisinden örüntüleri öğrenerek dil üretme becerisi kazanır. Bu süreçte tokenizasyon, dikkat mekanizmaları ve optimizasyon algoritmaları kritik rol oynar.",
    "Türk mutfağı, bölgesel çeşitlilik açısından son derece zengindir. Ege bölgesinde zeytinyağlı yemekler öne çıkarken, Güneydoğu Anadolu'da baharatlı ve etli yemekler daha yaygındır.",
    "İstanbul Boğazı, Avrupa ve Asya kıtalarını birbirinden ayıran doğal bir su yoludur. Tarih boyunca ticaret ve askeri açıdan stratejik bir öneme sahip olmuştur.",
    "Bilgisayar bilimlerinde algoritma karmaşıklığı, bir programın çalışma süresinin girdi boyutuna göre nasıl değiştiğini inceler. Büyük O gösterimi bu analizde yaygın olarak kullanılır.",
    "Anadolu'nun tarımsal üretimi, iklim koşullarına bağlı olarak bölgeden bölgeye büyük farklılıklar gösterir. Karadeniz kıyı şeridinde çay ve fındık üretimi öne çıkar.",
    "Doğal dil işleme alanında, bir modelin performansı genellikle şaşkınlık (perplexity) ve doğruluk gibi metriklerle ölçülür. Bu metrikler, modelin metin üzerindeki tahmin gücünü yansıtır.",
    "Osmanlı İmparatorluğu döneminde inşa edilen kervansaraylar, ticaret yollarının güvenliğini sağlamak amacıyla belirli aralıklarla yerleştirilmiş konaklama yapılarıdır.",
    "Enerji verimliliği, hem çevresel sürdürülebilirlik hem de ekonomik tasarruf açısından günümüzde giderek daha fazla önem kazanan bir konu haline gelmiştir.",
    "Türkçe, eklemeli bir dil yapısına sahiptir; bu da kelime köklerine çok sayıda ek getirilerek yeni anlamlar ve gramer ilişkileri oluşturulabildiği anlamına gelir.",
]
with open(OUTPUT, 'w', encoding='utf-8') as f:
    for i in range(400):
        base = random.choice(topics)
        f.write(json.dumps({"text": base + f" (kayıt no {i})"}, ensure_ascii=False) + "\n")
print(f"corpus yazildi: {OUTPUT}")
