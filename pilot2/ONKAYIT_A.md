# Önkayıt A — dışarıda tutulan veri setleri ve eğitim tohumları

Tarih: 3 Kasım 2026. Aşağıdaki veri setlerinde hiçbir model çalıştırılmadan ÖNCE yazıldı.
Yükleyiciler yalnızca yapısal olarak (dosya sayısı, etiketler, uzunluklar) denetlendi; hiçbir yöntemin
bu verilerdeki başarımına bakılmadı. Yöntem, görünüm ve eğitim ayarları ONKAYIT.md'deki ile aynıdır.

## Dışarıda tutulan veri setleri (geliştirmede hiç kullanılmadı)
- **JNU** (Jiangnan Üniv., GitHub: ClarkGableWang/JNU-Bearing-Dataset): 50 kHz, N/IR/OR/B, her sınıf tek rulman,
  600/800/1000 rpm. Bölme: destek bir devirden, sorgular diğer devirlerden (CWRU/SEU gibi).
- **HIT-MFS** ve **HIT-SB** (Harbin Teknoloji Enst., GitHub: hitwzc/Bearing-datasets): iki farklı test düzeneği,
  51,2 kHz, N/IR/OR; IR ve OR için üç hasar boyutu (2°, 5°, 8°), 600/900/1200 rpm.
  Bölme (PU gibi, birim düzeyinde): IR/OR'da destek bir hasar boyutundan, sorgular diğer boyutlardan;
  N tek rulman olduğundan destek bir devirden, sorgular diğer devirlerden.
- Ortak temsil pilot 1 ile aynı: 12 kHz, 1024 örnek, pencere başına z-skor. 30 epizot (tohum 1), 1/5/10-shot.

## Sabitler
- Ana yöntem: TimEE + fizik_v1 · ham+zarf. Üç eğitim tohumu (fizik_v1 = tohum 0, fizik_v1_s1, fizik_v1_s2);
  epizot başına doğruluk üç tohumun ortalaması.
- Kontrol: genel_v1 (tohum 0, 1, 2), aynı şekilde ortalanır.
- Taban çizgileri: TimEE hazır · ham, MantisV2 + LR, zarf spektrumu + prototip; "en iyi taban" her hücrede
  ortalaması en yüksek olan.

## Hipotezler
- **HA1 (ana):** 3 dış veri seti × {5, 10}-shot = 6 hücre. Epizot başına (ana − en iyi taban) farkı 6 hücrede
  ortalanır; bu 30 değerin ortalaması > 0 ve Wilcoxon p < 0,05.
- **HA2 (kontrol):** Aynı birleşik ölçütle ana − kontrol > 0, Wilcoxon p < 0,05.
- **HA3 (zarar yok):** 6 hücrenin her birinde (ana − en iyi taban) farkının %95 GA alt sınırı > −3 puan.
- **HA4 (tohum tutarlılığı):** Doğrulama epizotlarında (tohum 1) PU 10-shot'ta üç eğitim tohumunun her biri
  için ortalama (ana − zarf spektrumu) farkı > 0.
- 1-shot ve TimEE hazır · ham+zarf keşif amaçlı raporlanır.

## Not
B aşaması (ablasyonlar) keşif amaçlıdır, önkayıtlı değildir; sonuçları açıklayıcı olarak raporlanır.
