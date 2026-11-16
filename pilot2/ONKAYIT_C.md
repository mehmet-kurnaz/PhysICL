# Önkayıt C — simülatör v1 (CWRU tipi rejim)

Tarih: 3 Kasım 2026. Tohum 2 epizotları hiçbir yöntemle çalıştırılmadan ÖNCE yazıldı.

## Gerekçe
Doğrulamada (ONKAYIT.md, tohum 1) fizik_v1 CWRU'da en iyi tabanın 2–3 puan gerisinde kaldı (H3 desteklenmedi).
Önizleme, simülatörün CWRU'daki rezonans baskın, düşük frekans içeriği zayıf hasar sinyallerini az ürettiğini
gösteriyordu. Simülatör v1 bu açığı hedefler; tasarımı yalnızca geliştirme verisinin öznitelik kapsamasına bakılarak
yapıldı (CWRU OR/IR "<500 Hz enerji oranı" kapsaması %55/%82 → %66/%100; SEU N basıklık %80 → %100).

## Simülatör v1 değişiklikleri (v0'a ek)
- Makinelerin ~%30'u rezonans baskın rejimde: 2,5–5,4 kHz yapısal mod(lar), geniş bant, mil/dişli/renkli gürültü
  bileşenleri zayıflatılmış; bu makinelerde rulman hasarı SNR'si +5…+25 dB (bilye 0…+15 dB).
- Sağlam sınıfta çeşitlilik: birimlerin %60'ında, birim başına sabit seviyede (−20…0 dB) pürüzlülük darbeleri.
- Diğer her şey (eğitim 2000 adım, ayarlar, görünüm ham+zarf) fizik_v1 ile aynı. Model: fizik_v2 (tohum 0).

## Test
Geliştirme veri setleri (CWRU, SEU, PU), pilot 1 bölmeleri, **30 yeni epizot, tohum 2**; 1/5/10-shot.
Taban çizgileri yeniden çalıştırılır. Wilcoxon iki yönlü.

## Hipotezler
- **HC1:** CWRU 5- ve 10-shot'ta (fizik_v2 − en iyi taban) farkının %95 GA alt sınırı > −3 puan.
- **HC2:** CWRU 5- ve 10-shot'ta fizik_v2 > fizik_v1 (Holm düzeltmeli p < 0,05, iki hücre için).
- **HC3:** PU 10-shot'ta fizik_v2 > en iyi taban (Wilcoxon p < 0,05) — ana bulgunun tekrarı.
- **HC4:** SEU 5- ve 10-shot'ta (fizik_v2 − en iyi taban) alt sınırı > −3 puan.
