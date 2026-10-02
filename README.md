# TBB · RAG ile Kurumsal Yapay Zeka Uygulamaları — Eğitim Materyali

Türkiye Bankalar Birliği için hazırlanan üç günlük online eğitimin uygulama dosyaları.

> **Erdağı Bank ve tüm dokümanları kurgusaldır.** Kişiler, tutarlar, kurallar ve kimlik/hesap/kart numaraları eğitim amaçlı uydurulmuştur; gerçek bir bankayı veya kişiyi temsil etmez.

## Notebook'lar

| Gün | Konu | Colab'da aç |
| --- | --- | --- |
| 1 | Temeller: embedding, ingestion, vektör arama | [Gun1_RAG_Temeller](https://colab.research.google.com/github/sharpwareco/tbb-rag-egitimi/blob/main/notebooks/Gun1_RAG_Temeller.ipynb) |
| 2 | Cevap üretimi, RAG servisi, retrieval stratejileri, değerlendirme | [Gun2_RAG_Servis_ve_Kalite](https://colab.research.google.com/github/sharpwareco/tbb-rag-egitimi/blob/main/notebooks/Gun2_RAG_Servis_ve_Kalite.ipynb) |
| 3 | Güvenlik ve atölye | [Gun3_Guvenlik_ve_Atolye](https://colab.research.google.com/github/sharpwareco/tbb-rag-egitimi/blob/main/notebooks/Gun3_Guvenlik_ve_Atolye.ipynb) |

Colab'da **Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU** seçin.

## Erdağı Bank doküman seti (`erdagi_bank/`)

| Dosya | Doküman | Not |
| --- | --- | --- |
| `krd-yon-v3.pdf` | Bireysel Kredi Yönergesi, sürüm 3 | Yürürlükte |
| `krd-yon-v2.pdf` | Bireysel Kredi Yönergesi, sürüm 2 | Yürürlükten kalkmış; farklı tutarlar içerir |
| `kart-pro.pdf` | Kredi Kartı İşlemleri Prosedürü | |
| `eft-yon.pdf` | Para Transferi Yönergesi | |
| `mev-hes.pdf` | Mevduat ve Hesap İşlemleri Yönergesi | |
| `tic-teminat.pdf` | Ticari Kredi Teminat Yönergesi | Gizli; yalnızca kredi tahsis ve üst yönetim |
| `bilgi-guv.pdf` | Bilgi Güvenliği Politikası | |
| `ik-izin.pdf` | İnsan Kaynakları İzin Yönetmeliği | |
| `tedarikci-pro.pdf` | Tedarikçi Risk Değerlendirme Prosedürü | |
| `yk-karar-2026-14.pdf` | Yönetim Kurulu Kararı 2026/14 | 3. gün · çok gizli |
| `nova-teklif.pdf` | Tedarikçi teklifi (Nova Bilişim) | 3. gün · içinde görünmez bir talimat gizli (prompt injection demosu) |
| `sikayet-2026-0912.pdf` | Müşteri şikâyet kaydı | 3. gün · sahte kişisel veri içerir (maskeleme demosu) |
| `manifest.json` | Her dokümanın metadata'sı | Yürürlük durumu, gizlilik, izinli gruplar |

Metadata bilerek PDF'lerin içinde değil, `manifest.json` dosyasındadır: gerçek kurumlarda bu bilgiler doküman yönetim sisteminden gelir.

Mevzuat metinleri (6698 sayılı KVKK, 5411 sayılı Bankacılık Kanunu) notebook içinde doğrudan mevzuat.gov.tr'den indirilir.
