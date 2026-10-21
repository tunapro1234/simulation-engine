"""Ezber testi (memorization audit): LLM açık veriyi (ve sonucunu) eğitimde görmüş olabilir mi?

Yöntem (tamamlama farkı): her öğe bir kayıttan kurulmuş istem + beklenen cevap (ör. A/B testinin kazanan başlığı,
anketin gerçek yüzdesi). Aynı biçimde kurulmuş KONTROL öğeleri (bozulmuş/karıştırılmış kayıtlar: cevabı bilmek
mümkün değil) şans düzeyini verir. Ezber skoru = gerçek öğelerde isabet − kontrol öğelerinde isabet; öğe düzeyinde
bootstrap %95 GA. Alt sınır > 0 ise "uyari" (ezber şüphesi): o veride simülasyon başarısı kısmen hatırlama olabilir.

Port note (simengine): the original called `claude -p` through subprocess. Here the caller is any
`simengine.types.LLM` (simengine.llm.Claude / ClaudeCode / FakeLLM) wrapped by `llm_caller`, or any
str -> str callable. Caching, language settings and budget belong to the LLM backend. At most 500 calls per run.
The result dict is what `denetim.ezber` (and `evaluate(..., memorization_json=...)`) reads.
"""
from __future__ import annotations

import re

import numpy as np

EN_FAZLA = 500


def llm_caller(llm, system: str = "Answer briefly and directly.", max_tokens: int = 64):
    """simengine LLM (protocol: .complete(system, user)) -> str -> str caller for `olc`."""
    def cagir(istem: str) -> str:
        return llm.complete(system, istem, max_tokens=max_tokens).text.strip()
    return cagir


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9ğüşıöç%.]+", " ", str(s).lower()).strip()


def isabet(yanit: str, beklenen, tolerans: float = 0.02) -> bool:
    """Sayısal beklenende yanıttaki ilk sayı ± tolerans (yüzde ise 0-100 ölçeği de denenir); metinde içerme."""
    if isinstance(beklenen, (int, float)):
        for m in re.findall(r"-?\d+(?:[.,]\d+)?", str(yanit)):
            v = float(m.replace(",", "."))
            if abs(v - beklenen) <= tolerans or abs(v / 100 - beklenen) <= tolerans:
                return True
        return False
    b = _norm(beklenen)
    return bool(b) and b in _norm(yanit)


def olc(ogeler: list[dict], cagir, en_fazla: int = EN_FAZLA, B: int = 1000, tohum: int = 0) -> dict:
    """ogeler: [{"istem", "beklenen", "kontrol": bool}]. Döner: denetim.ezber'in okuduğu sözlük."""
    if len(ogeler) > en_fazla:
        raise ValueError(f"{len(ogeler)} öğe > çağrı sınırı {en_fazla}; alt örnek al")
    g, k = [], []
    for o in ogeler:
        (k if o.get("kontrol") else g).append(float(isabet(cagir(o["istem"]), o["beklenen"])))
    g, k = np.array(g), np.array(k)
    if not len(g) or not len(k):
        return {"durum": "atlandi", "ayrinti": "gerçek ve kontrol öğeleri birlikte gerekli", "n_gercek": len(g), "n_kontrol": len(k)}
    rng = np.random.default_rng(tohum)
    fark = g.mean() - k.mean()
    fb = np.array([rng.choice(g, len(g)).mean() - rng.choice(k, len(k)).mean() for _ in range(B)])
    alt, ust = np.percentile(fb, [2.5, 97.5])
    durum = "uyari" if alt > 0 else "tamam"
    ayrinti = (f"gerçek öğede isabet {g.mean():.3f} (n={len(g)}), kontrolde {k.mean():.3f} (n={len(k)}); "
               f"fark {fark:+.3f} [{alt:+.3f}, {ust:+.3f}]" + (" · ezber şüphesi" if durum == "uyari" else " · ezber işareti yok"))
    return {"durum": durum, "ayrinti": ayrinti, "deger": float(fark), "alt": float(alt), "ust": float(ust),
            "isabet_gercek": float(g.mean()), "isabet_kontrol": float(k.mean()), "n_gercek": len(g), "n_kontrol": len(k)}
