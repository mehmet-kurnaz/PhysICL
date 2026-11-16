# Önkayıt — v1 doğrulama deneyi

Tarih: 3 Kasım 2026. Bu dosya doğrulama epizotları (tohum 1) çalıştırılmadan ÖNCE yazıldı.
Geliştirme sonuçları (tohum 0, 10 epizot) yöntem ve görünüm seçiminde kullanıldı; doğrulama
epizotları yalnızca aşağıdaki sabit hipotezleri test etmek için kullanılır. Hiçbir ayar doğrulama
sonuçlarına bakılarak değiştirilmeyecek.

## Sabitler
- Ana yöntem: **TimEE + fizik_v1 · ham+zarf** (fizik önseliyle 2000 adım ince ayar; ham pencere ve
  500 Hz üstü Hilbert zarfı görünümlerinin olasılık ortalaması). Ağırlık: `ince_ayar_fizik_v1/son`.
- Kontrol: TimEE + genel_v1 · ham+zarf (zorluğu eşitlenmiş genel önsel, aynı eğitim ayarları).
- Taban çizgileri: TimEE hazır · ham, MantisV2 + LR, zarf spektrumu + prototip.
  "En iyi taban" = her hücrede doğrulama epizotlarında ortalaması en yüksek olan.
- Veri: CWRU, SEU, PU; pilot 1 bölmeleri (sızıntısız); 30 yeni epizot (tohum 1); 1/5/10-shot.
- Test: eşleştirilmiş Wilcoxon işaretli sıra testi, iki yönlü; H1 ve H2 içinde iki shot için Holm düzeltmesi.

## Hipotezler
- **H1 (ana):** PU 5-shot ve 10-shot'ta ana yöntem en iyi taban çizgisinden yüksektir (Holm p < 0,05).
- **H2 (kontrol):** PU 5-shot ve 10-shot'ta ana yöntem kontrolden yüksektir (Holm p < 0,05).
- **H3 (zarar yok):** CWRU ve SEU 5/10-shot'ta ana yöntem − en iyi taban farkının %95 GA alt sınırı > −3 puan.
- 1-shot keşif amaçlı raporlanır; hipotez değildir (geliştirmede 1-shot kazancının fizikten değil
  ince ayarın kendisinden geldiği görüldü).

## Ayrıca raporlanacak (keşif)
TimEE hazır · ham+zarf ve · zarf (ince ayarın katkısını görünümün katkısından ayırmak için),
fizik_v1 · zarf.
