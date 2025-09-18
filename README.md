<img width="1536" height="1024" alt="ChatGPT Image 18 Eyl 2025 11_35_31" src="https://github.com/user-attachments/assets/e9f47c6c-122b-44b8-8b49-32f2687d4755" /># Analist AI Agent

Doğal dilde gelen soruları **SQL**’e çeviren, **SQLite** üzerinde güvenli şekilde çalıştıran ve sonucu okunur biçimde özetleyen analist ajan.

---

## İçindekiler

* [Özet](#özet)
* [Öne Çıkanlar](#öne-çıkanlar)
* [Mimari](#mimari)
* [Dizin Yapısı](#dizin-yapısı)
* [Kurulum](#kurulum)
* [Yapılandırma](#yapılandırma)
* [Çalıştırma](#çalıştırma)
* [Araç Envanteri (Tools)](#araç-envanteri-tools)
* [Gözlemlenebilirlik (Telemetry)](#gözlemlenebilirlik-telemetry)
* [Değerlendirme (Eval Suite)](#değerlendirme-eval-suite)
* [Güvenlik ve Politikalar](#güvenlik-ve-politikalar)
* [Örnek Sorular](#örnek-sorular)
* [Sorun Giderme](#sorun-giderme)
* [Geliştirme Notları](#geliştirme-notları)
* [Lisans](#lisans)

---

## Özet

**Analist AI Agent**, LangGraph tabanlı tool-calling yaklaşımıyla; çok tablolı şemalarda planlı, güvenli ve açıklanabilir sorgulama deneyimi sunar. LLM sağlayıcısı olarak **OpenRouter** (tool calling destekli) ve opsiyonel **Hugging Face Inference API** fallback kullanır.

* Veritabanı: `SQLite` (örnek: `app.db`)
* UI: `Streamlit`
* Ajan: `LangChain + LangGraph`

---

## Öne Çıkanlar

* 🧭 **Planner (QuerySpec)**: Intent, tablolar, join path, metrikler, limit…
* 🛡️ **SELECT-only + Hard LIMIT**: DDL/DML engeli; büyük sorgularda üstten limit.
* 🔧 **Self-Healing**: `validate_sql → repair_sql → safe_run_sql` döngüsü.
* 📄 **Paginator**: Büyük sonuçları sayfalar.
* ⚡ **Query Cache (TTL)**: Tekrarlanan sorgularda hızlı yanıt.
* 🗣️ **Summarizer**: Doğal dilde, açıklanabilir özet.
* 🖥️ **Streamlit UI**: Canlı tool günlüğü, Plan/Attempt/Explain panelleri, CSV/JSON indir, SQL kopyala.
* 👁️ **Telemetry**: Her koşu JSONL log; süreler, hatalar, cache hit, onarım denemeleri.
* 🔁 **LLM Sağlayıcıları**: OpenRouter (tool-calling), opsiyonel HF fallback.

---

## Mimari
<img width="1536" height="1024" alt="ChatGPT Image 18 Eyl 2025 11_35_31" src="https://github.com/user-attachments/assets/6e3fc120-1715-4b3b-9d84-febe0afd0fd2" />

> Not: Üretimde `runtime.mode = guarded` önerilir (planla/doğrula/onar/çalıştır/sayfala/özetle rayları).

---

## Dizin Yapısı

```text
.
├─ config.yaml
├─ app.db
├─ graph.py
├─ ui_streamlit.py
├─ main.py
├─ tools/
│  ├─ db.py
│  ├─ planner.py
│  ├─ sql_validator.py
│  ├─ repair_sql.py
│  ├─ safe_sql.py
│  ├─ paginator.py
│  ├─ query_cache.py
│  ├─ summarizer.py
│  └─ result_utils.py
├─ utils/
│  ├─ llm.py
│  └─ telemetry.py
├─ eval/
│  ├─ eval_questions.jsonl
│  └─ eval_report.csv
└─ scripts/
   └─ eval.py
```

---

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U streamlit pyyaml langchain-core langchain-openai openai
```

**Anahtarlar**

* OpenRouter: `OPENROUTER_API_KEY="sk-or-..."`
* (Opsiyonel) Hugging Face: `HUGGINGFACE_API_TOKEN="hf_..."`

> Geliştirme kolaylığı için anahtarları `config.yaml` içine de yazabilirsiniz. UI, yoksa ortam değişkenini bu dosyadan enjekte eder.

---

## Yapılandırma

`config.yaml` örneği:

```yaml
db:
  path: "app.db"

runtime:
  mode: guarded           # guarded | free
  eval_mode: false

llm:
  provider_priority: ["openrouter"]       # istersen ["openrouter","hf"]
  temperature: 0.1

  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    model_name: "openrouter/sonoma-sky-alpha"   # tool-calling destekli
    api_key: "sk-or-XXX"                        # veya ortam: OPENROUTER_API_KEY

  hf:
    model_id: "meta-llama/Meta-Llama-3-8B-Instruct"
    token: ""                                   # veya ortam: HUGGINGFACE_API_TOKEN

policy:
  select_only: true
  allow_tables: []       # boş = hepsi; üretimde allow-list önerilir
  deny_tables: []
  deny_columns: []
```

---

## Çalıştırma

### CLI

```bash
python main.py
```

### Streamlit UI

```bash
streamlit run ui_streamlit.py
```

* Tool çağrıları canlı görünür.
* Plan/Attempts/Explain panelleri ile karar süreci izlenir.
* Sonuçlar tablo; **CSV/JSON indir**, **SQL kopyala**.
* Büyük sonuçlarda **Sayfalama**.
* “Run History” expander’ında son çalıştırmalar.

---

## Araç Envanteri (Tools)

| Tool                  | Amaç                                                                |
| --------------------- | ------------------------------------------------------------------- |
| `plan_query`          | NL → **QuerySpec** (intent, tablolar, join path, metrikler, limit…) |
| `validate_sql`        | Çalıştırmadan **EXPLAIN** ile kontrol                               |
| `explain_sql`         | SQLite açıklama planı                                               |
| `repair_sql`          | Sık SQL hatalarını otomatik düzelt (örn. kolon adı)                 |
| `safe_run_sql`        | **SELECT-only** + **hard LIMIT** ile güvenli çalıştır               |
| `paginate_sql`        | Sonuçları sayfalara böl                                             |
| `cache_get/cache_put` | TTL’li sorgu önbelleği                                              |
| `tabulate_result`     | JSON sonucu Markdown tabloya çevir                                  |
| `summarize_answer`    | Doğal dilde özet + kısa SQL alıntısı                                |
| `schema_intel`        | list/describe/find\_join\_path/profile fonksiyonları                |

---

## Gözlemlenebilirlik (Telemetry)

* Her çalıştırma `logs/runs.jsonl` dosyasına tek satır olarak yazılır:

  * `trace_id`, `question`, toplam gecikme
  * adım adım tool çağrıları (`tool`, süre, hata/başarı, satır sayısı, cache hit, onarım denemeleri)
* UI’da **Run History** bölümünden incelenebilir.

Örnek kayıt:

```json
{
  "ts": "2025-09-18T14:22:11+03:00",
  "trace_id": "a9f41b7e2c3d",
  "question": "Kullanıcı yaşlarının min/ortalama/maks değerleri",
  "latency_ms": 1245,
  "steps": [
    {"tool":"plan_query","ms":128,"ok":true},
    {"tool":"validate_sql","ms":22,"ok":true},
    {"tool":"safe_run_sql","ms":980,"ok":true,"rows":1,"paged":false}
  ]
}
```

---

## Değerlendirme (Eval Suite)

Çalıştırma:

```bash
python scripts/eval.py
```

* `eval/eval_questions.jsonl` içindeki soru setini çalıştırır.
* `eval/eval_report.csv` üretir.
* Özet metrikler: **Exec-Accuracy**, **Avg Latency**, **Repair-Rate**.

Örnek `eval/eval_questions.jsonl` satırları:

```json
{"id":"Q001","question":"Toplam kullanıcı sayısı?","expect":{"type":"agg_value","value":100,"tol":0}}
{"id":"Q002","question":"İlk 3 birim kullanıcı sayısı?","expect":{"type":"rows_contains","rows":[["Research Lab",12],["Design Team",11],["Product Development",9]]}}
```

---

## Güvenlik ve Politikalar

* **SELECT-only**: DDL/DML engellenir.
* **Hard LIMIT**: Aşırı sonuçlar üstten limitlenir.
* **Tablo/Kolon Politikaları**: `allow_tables/deny_tables/deny_columns` ile kapsam kısıtlanabilir.
* **PII Maskesi (opsiyonel)**: Özetleyicide TCKN/e-posta/telefon için basit maskeleme uygulanabilir.
* **OpenRouter free modeller**: Bazı modellerde sağlayıcı loglama yapabilir; gizlilik gereksiniminize göre model/policy seçin.

---

## Örnek Sorular

* “Kullanıcı yaşlarının min/ortalama/maks değerleri nedir?”
* “En çok kullanıcısı olan ilk 5 birim hangileri?”
* “LLM kullanılan oturumların toplam oturumlara oranı nedir?”
* “Bu sağlayıcıyı kullanan oturumlarda ortalama mesaj sayısı en yüksek ilk 5 sağlayıcı hangileri?”
* “En az iki farklı LLM deneyen kaç kullanıcı var?”

> Daha kapsamlı liste için `eval/eval_questions.jsonl`.

---

## Sorun Giderme

**`RuntimeError: no usable providers configured`**

* `OPENROUTER_API_KEY` set mi? `config.yaml → llm.openrouter.api_key` dolu mu?
* `pip install -U langchain-openai openai` yüklü mü?

**`401 No auth credentials found`**

* Anahtar boş/yanlış. Terminalde `echo $OPENROUTER_API_KEY` ile doğrulayın.

**`404 No endpoints found that support tool use`**

* Seçilen model tool-calling desteklemiyor. `openrouter/sonoma-sky-alpha` kullanın.

**Yavaş sorgular**

* Büyük tablolar için sayfalama (`paginate_sql`).
* SQLite PRAGMA ayarları `tools/db.py` içinde.

---

## Geliştirme Notları

* **Çalışma modu**: `runtime.mode=guarded` → planla/doğrula/onar/çalıştır/sayfala/özetle.
* **Fallback**: OpenRouter başarısız olursa opsiyonel HF serverless devreye girer.
* **Test DB**: `app.db`; kendi DB’niz için `config.yaml → db.path` güncelleyin.

---

## Lisans

MIT License.
