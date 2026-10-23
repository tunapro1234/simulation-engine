# simengine

**English summary.** simengine is a Python library for population simulation with LLMs and for the evidence needed to
trust it. It builds personas from tables or histories, runs simulated surveys, A/B panels, segment distributions and
per-person predictions with Claude (or an offline `FakeLLM`), and ships the baselines that usually win: persistence,
marginals, TF-IDF, logistic and GBDT. It also has agent-based models for adoption, opinion, markets and cascades, an
evidence pipeline (baseline ladder, paired bootstrap CIs, audits, "worth N humans" via PPI++, sealed predictions,
cards and dashboards), a pre-registered research loop, and benchmark tasks. A second porting round added a persona
method zoo (10 methods) with a Big Five fidelity check, schema-constrained answers with re-prompting, a one-shot
economic game battery scored by distance to published human numbers, a who-to-punish agent society, theory-persona
mixture calibration, extra evidence sections (variance ratio, completion, coefficient recovery) and harness-variant
sweeps. A persona injection zoo (`simengine.persona.inject`: 52 methods in 12 families plus 2 controls, each scored against
no-persona and persona-swap controls; literature effects only, not yet measured here) compares ways of putting a
real person's data into the prompt. The measured lesson: pure persona panels
rarely beat simple baselines. The LLM helps when there is no data, when it is stacked with a learner, and when it
serves as a prior in small tests. Every number below has a source in [docs/RESULTS.md](docs/RESULTS.md). This is
prior work: declare it if you use it at a hackathon (open source, AGPL-3.0-or-later; cite the dated `v0.1.0` release).

---

## Nedir

simengine, "bir kitle bu soruya / bu metne / bu fiyata nasıl tepki verir?" sorusunu LLM ile simüle edip sonucun ne
kadar güvenilir olduğunu aynı çağrıda ölçen bir kütüphane. TALP hazırlık araştırmasında (Ekim 2026) yazılan
deney kodlarının (`uretim/<dal>/`) temizlenip tek pakette toplanmış hali.

İki şeyi birlikte yapar:

1. **Simülasyon:** tablodan ya da satın alma geçmişinden persona üretir; anket, A/B paneli, segment dağılımı, kişi
   başı tahmin, tartışma turu ve ajan tabanlı modeller çalıştırır.
2. **Kanıt:** her LLM sonucunu basit tabanların (persistence, marjinal, TF-IDF, lojistik, GBDT) yanına koyar; eşli
   bootstrap güven aralığı, karıştırma kontrolü, sızıntı ve ezber denetimi, "kaç insana bedel" (PPI++) ve bir rozet
   verir. Tahminler doğru cevap açılmadan önce mühürlenebilir.

## Kurulum

```bash
pip install -e .[all]       # anthropic + lightgbm + statsmodels
pip install -e .            # yalnız çekirdek (numpy, pandas, scikit-learn); FakeLLM ile çevrimdışı çalışır
```

Kurmadan (bağımlılıklar ortamda hazırsa), depo kökünden: `PYTHONPATH=. python -m simengine demo`. Aşağıdaki her
`simengine ...` komutu `python -m simengine ...` olarak da çalışır.

API anahtarı ortam değişkeninden okunur, depoya yazılmaz. Arka uçlar: `se.llm.Claude` (anthropic SDK),
`se.llm.ClaudeCode` (`claude -p`), `se.llm.FakeLLM` (çevrimdışı, deterministik).

## Hızlı başlangıç

```python
import simengine as se, pandas as pd
df = pd.read_csv("data/musteriler.csv")                              # kişi başı bir satır; data/ git dışı
panel = se.persona.from_table(df, id_col="uid", segment="income_tier", forbidden=["hedef"])
llm = se.FakeLLM(se.sim.schema_responder())   # gerçek: se.llm.Claude(model="claude-haiku-5-5", budget_usd=2)
res = se.sim.survey(panel, "Online alışverişten keyif alırım.", llm=llm)
se.seal(se.sim.survey_frame(res), "anket tahminleri, gerçek açılmadan önce")   # tek mühür -> .onkayit/zincir.jsonl
frame = se.sim.survey_frame(res, {"q1": dict(zip(df.uid, df.hedef))}).dropna(axis=1, how="all")
frame["pred_persistence"] = frame.group.map(dict(zip(df.uid, df.gecen_yil)))
rep = se.evaluate(frame, title="keyif", ours="score", task="continuous", unlabeled_baselines=["persistence"],
                  prereg=".onkayit")
print(se.evidence.summary(rep)["headline_en"]); se.card(rep, "runs/kart.md"); se.dashboard([rep], "runs/pano.html")
```

Çalışan sürümü: `python examples/quickstart.py` (çevrimdışı, ~4 sn). Uçtan uca demo: `simengine demo`.

`se.seal` ile `se.evidence.seal` aynı işlevdir (tek mühür); `evaluate(prereg=...)` ve `--prereg` hem bunun
`zincir.jsonl`'ini hem `se.research.seal`'in `chain.jsonl`'ini okur.

Kör provadan (D2, 22 Eki) sonra eklenenler, her biri bir iki satır (ayrıntı modül belgelerinde):

- **Haiku düşünme bütçesi:** küçük `max_tokens` + açık düşünme boş metin döndürebilir; `thinking="off"` ver.
  1024'ün altında `ThinkingBudgetWarning`, `max_tokens`'ta kesilen boş yanıtta `EmptyResponseError` (önbelleğe yazılmaz).
- **Ortak harcama logu:** `SIMENGINE_SPEND_LOG` / `SIMENGINE_AGENT` ortam değişkenleri ya da `Claude(spend_log=, agent=)`;
  her gerçek çağrı için bir TSV satırı (zaman, ajan, model, giriş, çıkış, usd, etiket).
- **Yumuşak ret:** `se.looks_like_refusal`; `survey` reti önbelleğe yazmaz, bir kez farklı çerçeveyle yeniden sorar,
  `refused` / `unparsed` / `refusal_rate` alanlarını ayrı verir. Kendi çağrında `complete(..., cache_if=...)`.
- **Anket:** `kind="numeric"` (0-100 kaydırıcı, `min`/`max`), `free_text=False`, kişi başı uyarıcı (`stimulus_col=`
  ya da `context={pid: metin}`); `from_table` boş/tekrarlı id'de sayısıyla durur, `id_col=None` satır numarası verir.
- **Grup aktarımı:** `se.GroupTransfer` (grup düzeyi ridge + kişi sapması, isteğe bağlı LLM grup önseli) ve
  `se.oof_predict(model, df, features=, target=, group=)` (grup dışarıda çapraz uydurma).
- **Kanıt:** `baselines_oof=` (etiketle eğitilmiş, çapraz uydurmalı tabanlar; rozet en güçlü tabana karşı),
  `ceiling_col=` (model/oracle tavanı), `aggregate_by=` / `se.evaluate_aggregate` (grup düzeyi), büyük tabloda
  çarpan 2.000 satırlık alt örneklemle, gerçek yokken mühürlü `awaiting truth` kartı; sayısal ezber için
  `se.correlation_audit` ve `memorization_audit(tolerance=, relative=True)`.
- **Talep tahmini (D4):** `se.methods.forecast`: tabanlar (`persistence_forecast`, `seasonal_naive`, `moving_mean`,
  `ses`), `make_features`, `se.DirectForecaster` (ufuk başına model, LightGBM Tweedie ya da sklearn), kayan köken
  `se.backtest(..., origins=[...])` + `se.summarize_backtest`, L1 ölçütleri için `se.point_for_metric`, adlı/kör/
  karıştırılmış LLM önseli `se.llm_prior_feature`. Kanıt tarafı: `evaluate(task="forecast", primary="wmape")`
  (yanlılık yanında), `aggregate="sum"` (hiyerarşi toplamları), `ci="basic"`, `n_configs=`; CLI'de `--task forecast
  --primary wmape --aggregate sum --ci basic --n-configs N`.

Kör prova D3'ten (yeni film soğuk başlangıcı, 22 Eki) sonra eklenenler:

- **Soğuk başlangıç:** `se.item_holdout_features` (katta üye istatistiği + SVD zevk vektörü, Procrustes hizası),
  `se.MFBaseline` (ALS ile yanlı matris ayrışımı tabanı), `oof_predict(..., folds=)` (hazır katlar).
- **Kanıt:** `offset_col=` (ofsetli grup artığı), `noise_ceiling="variance_components"` (örneklem gürültüsü tavanı, tüm gruplar; eski adı `"split_half"`),
  `blind_col=` ve ezber uyarısında rozet notu, `compare=[("P1", "L1"), ...]` (kayıtlı düşme koşulları) ve
  `se.paired_diff`; `correlation_audit` `max_calls`'ı çağrıdan önce denetler.
- **Mühür:** `se.seal(..., store_copy=False)` ya da `archive_dir=`: lisanslı veride tahmin kopyası repoya düşmez;
  `.onkayit` ve `.prereg` `.gitignore`'da.
- **Anket / etiket:** `survey` bir koldaki bütün uyarıcı gruplarını tek `llm.map` ile, `parallel=` iş parçacığıyla
  gönderir; `refused` / `unparsed` sayıdır, kimlikler `refused_ids` / `unparsed_ids`; `survey(tag=)`,
  `Claude(tag_prefix=)` ya da `SIMENGINE_SPEND_TAG_PREFIX` harcama logunda etikete önek koyar.

## Komut satırı

```bash
simengine demo                                                   # sentetik veri, FakeLLM -> runs/demo/
simengine evaluate data/frame.csv --ours stack --raw llm --task binary --primary auc \
    --unlabeled-baselines persistence --prereg .onkayit            # -> runs/frame-evidence/
simengine evaluate data/test_frame.csv --ours stack --task binary --primary auc --prereg .onkayit   # gerçek yok: awaiting truth
simengine evaluate data/dev_pairs.csv --ours ours --primary spearman --aggregate-by movie_id \
    --offset-col member_mean --noise-ceiling variance_components --memorization runs/audit.json \
    --blind-col ours_blind --compare P1:L1 --compare "P2::c: P2 ≈ 0"              # grup artığı + ezber + düşme koşulları
simengine seal runs/test_pred.csv "test tahmini" --registry runs/.onkayit --no-copy   # yalnız zincir kaydı
simengine card runs/demo/report.json --dashboard runs/pano.html --lang en
simengine bench list -v                                          # görevler, kabul edilen seçenekler, dosya düzeni
simengine bench run gss upworthy:split=holdout --data-root <veri-kökü>
simengine loop examples/loop.json --max-turns 10
```

- `--primary` verilmezse görev varsayılanı kullanılır: binary/multiclass `accuracy` (en sık sınıf > %60 ise
  `macro_f1`), distribution `one_minus_tvd`, continuous `pearson` (gerçek değer 0-1 aralığındaysa `one_minus_mae`),
  ab `pair_sign`. Ön kayıtta yazdığın metriği her zaman `--primary` ile ver; `evaluate` verilmediğinde uyarır.
- Çıktılar varsayılan olarak `runs/` altına yazılır (git dışı). `--prereg` dizininde ne `zincir.jsonl` ne
  `chain.jsonl` varsa `evaluate` hata verir; `se.seal(...)` (= `se.evidence.seal`) varsayılan olarak `.onkayit/`
  dizinine yazar.
- `evaluate` ek seçenekleri: `--baselines-oof T1,T2`, `--ceiling-col pred_oracle --ceiling-label oracle`,
  `--aggregate-by group`, `--multiplier-max-rows N` (0 = tüm tablo), `--no-multiplier`, `--memorization audit.json`
  (`--ours-no-llm` ezber notunu kapatır), `--offset-col`, `--noise-ceiling variance_components` (eski adı `split_half`; `--noise-ceiling-repeats` split-half denetimi),
  `--blind-col`, `--compare A:B` (tekrarlanabilir; `A` tek başına 0'a karşı, `A:B:etiket`). `truth` sütunu yoksa ya da
  tümü boşsa `awaiting truth` kartı yazılır (çıkış kodu 0).
- `seal`: `--registry` (varsayılan `.onkayit`), `--no-copy` (yalnız zincir kaydı), `--archive-dir` (kopya başka
  yerde, zincirde yalnız dosya adı), `--git`. Aynı dizini `evaluate --prereg`'e ver.
- Rozetler: `high confidence`, `inconclusive`, `below baseline` (bizim − en güçlü taban GA'sı tümüyle sıfırın
  altında; araştırma kuralında bu da "kararsız" sayılır), `awaiting truth`.

## Persona enjeksiyonu

Gerçek bir kişinin verisi (öznitelikler, diğer cevapları, mülakat, davranış logu, komşuları, grubu) prompta nasıl
girmeli? `simengine.persona.inject` literatürden 12 ailede 52 yöntemi (+2 kontrol) aynı hücrelerde karşılaştırır:
her kol `no_persona` ve `swap` (başkasının verisi) kontrolüne karşı, kişi kümeli eşli bootstrap GA'sıyla.

```python
import simengine.persona as sp
arms = ["plain_fields", ("own_answers_rag", {"questions": bank, "k": 3}), ("similar_respondents", {"pool": people})]
plan = sp.compare_injectors(llm, people, bank, "answers", arms, dry_run=True)   # çağrı sayısı + üst sınır USD
res = sp.compare_injectors(llm, people, bank, "answers", arms, max_usd=2.0); res.table
```

Çevrimdışı örnek: `python examples/persona_inject_compare.py` (FakeLLM, ~2 sn). Hangi veride hangi yöntem, beklenen
etki (kaynaklı), maliyet ve risk tablosu: [docs/modules/persona-inject.md](docs/modules/persona-inject.md).
Hiçbiri simengine'de gerçek LLM ile ölçülmedi; tablodaki sayılar makalelerin kendi sayıları.

## Modül haritası

| modül | ne yapar | belge |
|---|---|---|
| `simengine.llm` | Claude / `claude -p` / FakeLLM arka uçları, disk önbelleği, maliyet defteri, bütçe, 401'de durma | [llm](docs/modules/llm.md) |
| `simengine.persona` | persona şeması, tablodan / geçmişten persona, LLM ile üretim, enum doğrulama, sızıntı koruması; persona yöntem hayvanat bahçesi (10 yöntem, ajan bankası), Big Five kurucu + Mini-IPIP sadakat kontrolü (`validate_bigfive`); persona enjeksiyonu (`persona.inject`, 12 aile, 52 yöntem + 2 kontrol, kontrollü kıyas) | [persona](docs/modules/persona.md), [persona-inject](docs/modules/persona-inject.md) |
| `simengine.sim` | anket, SSR, A/B paneli + güven kontrolü, tartışma turu, segment dağılımı, kişi başı tahmin (L0/L1/L2); şemalı cevap + hata geri bildirimiyle yeniden sorma (`ask_constrained`, tahmin/kurtarma yok) | [sim](docs/modules/sim.md) |
| `simengine.methods` | tabanlar, sepet yöntemleri, A/B (TF-IDF, LLM, stack, erken test Bayes), stack, büzülme, MNL, sonraki eylem; teori-persona karışım ağırlıkları + tanımlanabilirlik uyarısı (`fit_mixture_weights`) | [methods](docs/modules/methods.md) |
| `simengine.abm` | benimseme ABM + ABC, fikir kayması, pazar ABM, elastisite, retweet eğrileri (Hawkes, SEISMIC), ajan toplumu; ekonomik oyun toplumları (kimi cezalandırmalı oylaması, tekrarlı PD, güven); ızgara ve mekânsal modeller: kafes çekirdeği (Moran's I, Duncan D), Schelling + Zhang + LLM taşınanlar, Sugarscape, hareketlilik (GEO-BLEU/DTW, Markov, EPR), yerleşim (CELF açgözlü kapsama), mekânsal SEIR + üretken ajan salgını | [abm](docs/modules/abm.md) |
| `simengine.evidence` | `evaluate`, taban merdiveni, eşli CI, denetimler, PPI++ çarpanı, mühür, kart, pano; ek bölümler: varyans oranı, tamamlama satırı (sessiz kayıp), yayımlanmış bulgularla fark, stilize olgu puanı, katsayı geri kazanımı | [evidence](docs/modules/evidence.md) |
| `simengine.research` | ön kayıt + kilit, arama uzayı, öneriler (rastgele, ızgara, açgözlü, halving, LLM), dev/seçim/test döngüsü; ön kayıtta beyan edilen hedefler (`declare_targets`), tek değişiklikli düzenek taraması (`harness_variants`, `harness_sweep`: ifade, max_tokens, sıcaklık); arama ve yöntem modülleri: haydutlar (UCB1, Thompson, successive rejects, LUCB), Bayes optimizasyonu (GP-EI/UCB, TPE), Hyperband + ASHA, AIDE ağaç araması + MCTS, istem evrimi (OPRO, APE, Promptbreeder), otomatik sosyal bilim + deney sonucu tahmini | [research](docs/modules/research.md) |
| `simengine.games` | 15 tek atışlık ekonomik oyun (10 kalibrasyon, 5 ayrılmış), maliyet kapılı koşucu, alıntılı insan değerlerine uzaklık (ajan bootstrap CI, W1/KS), tutarlılık denetimi, kimi cezalandırmalı 9 ölçütlü karne | [games](docs/modules/games.md) |
| `simengine.bench` | 27 kıyas görevi: 24 veri görevi (bank marketing, GSS, Open e-commerce, OpinionQA, SimBench, Twin-2K-500, Upworthy; yeni: Instacart, M5, MovieLens, RetailRocket ×2, choices13k, CPC18, SocSci210, SubPOP / Global OpinionQA, Swissmetro, Hillstrom / Criteo / X5 uplift, WebShop ×2, YJMob insan hareketliliği) + 3 oyun görevi (`econ_games_b01`, `ertan_wtp` veri istemez; `beebench_b01_prior` yerel b01 tablosunu okur) | [bench](docs/modules/bench.md) |
| `simengine.cli` | `evaluate`, `seal`, `card`, `bench`, `loop`, `demo` | bu dosya |

Kısayollar: `se.Claude`, `se.FakeLLM`, `se.CostLedger`, `se.EmptyResponseError`, `se.ThinkingBudgetWarning`,
`se.looks_like_refusal`, `se.from_table`, `se.survey`, `se.survey_frame`, `se.ab_test`, `se.segment_distribution`,
`se.predict_people`, `se.GroupTransfer`, `se.oof_predict`, `se.item_holdout_features`, `se.MFBaseline`,
`se.compose_tag`, `se.paired_diff`, `se.evaluate`, `se.evaluate_aggregate`, `se.card`,
`se.dashboard`, `se.worth_humans`, `se.memorization_audit`, `se.correlation_audit`, `se.seal`, `se.chain`, `se.Loop`,
`se.Space`, `se.Prereg`, `se.play_games`, `se.distance_to_human`, `se.persistence_forecast`, `se.seasonal_naive`,
`se.moving_mean`, `se.ses`, `se.make_features`, `se.DirectForecaster`, `se.backtest`, `se.summarize_backtest`,
`se.point_for_metric`, `se.llm_prior_feature`. Alt paketler ilk erişimde yüklenir.

Örnekler (`examples/`): `quickstart.py`, `ab_test.py`, `segment_distribution.py`, `research_loop.py` + `loop.json`,
`abm_adoption.py`, `persona_inject_compare.py` (persona enjeksiyon kolları, kontroller ve GA; çevrimdışı, ~2 sn),
`econ_games.py` (persona yöntemleri → oyunlar → insana uzaklık → kıyas → kimi cezalandırmalı karnesi; çevrimdışı,
~3 sn) ve bir günlük yarışma planı [`hackathon_day.md`](examples/hackathon_day.md). Yöntem
notları: [docs/METHODS.md](docs/METHODS.md). Alan haritası ve doğrulama: [docs/landscape.md](docs/landscape.md)
(rakip iddiaları, açık simülatörler, kıyaslar; her sayı "makale" ya da "satıcı beyanı" etiketli),
[docs/validation.md](docs/validation.md) (jürilerin tanıdığı doğrulama ölçütleri → simengine işlevi),
[docs/failure-modes.md](docs/failure-modes.md) (bilinen hata türleri, sayıları ve onları yakalayan denetim).

## Hackathon recipes

Eight end-to-end scripts in `examples/recipes/`, one per common brief, each 210-230 lines with numbered steps.
They share one contract: offline by default (FakeLLM + synthetic data; needs numpy and pandas), `--data <csv>` for
the day's table, `--out <dir>` for the card, dashboard, seals, cache and ledger, and real Claude calls only with
`--live --max-usd <cap>` (the cap is checked against the ledger in `<out>/.cache/` before every call; the key is
never printed). Every recipe fits its LLM-free baselines on training data only, seals the predictions before the
truth is joined, and ends with an evidence card. Offline, FakeLLM knows nothing, so the badges say "inconclusive";
read the offline run as a check of the plumbing, not of the LLM. The day-of settings are in step 0; each recipe has
a guide in [docs/recipes/](docs/recipes/), and [`hackathon_day.md`](examples/hackathon_day.md) says at which step
to reach for which one.

| script | the brief it answers | guide |
|---|---|---|
| [`ad_ab.py`](examples/recipes/ad_ab.py) | rank N ad creatives for an audience, lift of each over the control ad with CIs; past campaigns give the baselines and calibration | [ad_ab](docs/recipes/ad_ab.md) |
| [`adoption.py`](examples/recipes/adoption.py) | new product adoption N periods out: curve baselines and analogy Bass / ABM backtested, LLM Bass prior from the market text, marketing what-ifs | [adoption](docs/recipes/adoption.md) |
| [`cold_start.py`](examples/recipes/cold_start.py) | rating or early demand of a brand-new item: LLM prior from its text blended with similar past items, against the same blend without the LLM | [cold_start](docs/recipes/cold_start.md) |
| [`pricing.py`](examples/recipes/pricing.py) | price sensitivity and WTP from a choice-based conjoint: logit part-worths, WTP with person-bootstrap CIs and a revenue curve, humans and LLM side by side | [pricing](docs/recipes/pricing.md) |
| [`purchase_pred.py`](examples/recipes/purchase_pred.py) | will this person buy in this category next window: persona + history, L0/L1/L2 ladder, calibration, out-of-fold stack (the talp73 setup) | [purchase_pred](docs/recipes/purchase_pred.md) |
| [`segment_survey.py`](examples/recipes/segment_survey.py) | survey answer shares per segment and for the population: verbalised distributions (3 framings), train-survey prior update, PPI++ population estimate | [segment_survey](docs/recipes/segment_survey.md) |
| [`site_visitors.py`](examples/recipes/site_visitors.py) | landing page / checkout funnel: personas walk each variant's pages, P(continue) per step, drop-off and the variant lift against past traffic | [site_visitors](docs/recipes/site_visitors.md) |
| [`spread.py`](examples/recipes/spread.py) | a message spreading on a network: who shares, how far, which seeding strategy reaches most; independent-cascade fit plus LLM P(share) per segment | [spread](docs/recipes/spread.md) |

## Ölçülen sonuçlar

Hepsi araştırma kodunda ölçüldü; kaynak dal ve dosya sağda. Son üç satır önceki iki depodan (replicant,
edsl-test) gelir ve Claude dışı modellerde ölçüldü; ayrıntı ve uyarılar RESULTS §8'de. Tam liste ve güven aralıkları
[docs/RESULTS.md](docs/RESULTS.md)'de. "(yeniden)" işaretli satırlar bu kütüphanede gerçek veriyle tekrar üretildi.

| görev | sonuç | kaynak |
|---|---|---|
| Upworthy A/B, 300 test, yön doğruluğu (şans 0.50) | TF-IDF + Opus stack **0.817** [0.770, 0.860]; TF-IDF 0.773; Opus tek başına 0.765; uzun başlık 0.533 | deney `RAPOR.md` 1.1 |
| aynı, Haiku 8 personalı panel | 0.522 [0.468, 0.575]; personasız okur 0.528 | sahne `RAPOR.md` |
| erken test, kol başı 100 gösterim, 20 testle kalibrasyon (`simulate_saving_calibrated`) | LLM önseli **2.27x** (1.67-3.11) trafik tasarrufu; 400'de 1.34x; 800+ kazanç yok; 2 katlı `simulate_saving` aynı veride 2.47x | deney `RAPOR.md` 2.2-2.3, `ornek_tasarrufu_llm*.json` |
| CES 2020 segment dağılımı, Haiku | doğrudan dağılım W1 0.077, açığın **0.818**'i kapandı; persona paneli 0.55 | makro `RAPOR.md` C1, K2-5 |
| 5 tartışma turu | ΔW1 −0.005 [−0.012, +0.002], ölçülebilir etki yok | makro `RAPOR.md` |
| talp73 kişi başı satın alma, 1a | her zaman "hayır" 0.764; L1 Haiku persona **0.694**; geçmiş LightGBM 0.807 | talp73 `kart/KART.md` |
| talp73 1b(c) | geçmiş LightGBM 0.672; anket modeli 0.588; L1 Haiku 0.564; SSR 0.547-0.573 | talp73 `kart/KART.md`, `kod/llm/SONUC-v1.7d.md` |
| OE aylık sipariş günü (Pearson) | persona + geçen yıl çapası 0.860; persistence 0.848; çapasız persona 0.618 | sahne `RAPOR.md` |
| Twin-2K-500 kişi düzeyi korelasyon | adil LLM'siz stack **0.342**; + LLM +0.006 [+0.004, +0.007]; tekrar-test tavanı 0.565 | persona `RAPOR.md` |
| "kaç insana bedel" (PPI++, n = 100) | persistence **4.36x**; LLM persona davranış tahmini 1.06x; LLM ikiz 1.08x | kanit `RAPOR.md` §1a |
| OWID internet benimseme, test MAE | ABM + ABC + analoji **8.48**; Bass kontrolü 9.73; kazancın %81-85'i ABC + analojiden | makro `RAPOR.md` M1, `abm/m1/NOTLAR.md` |
| retweet eğrisi, 24 sa RMSE | şekil tabanı 0.391 = Hawkes 0.391; LLM ajan toplumu 0.771 = tek Haiku 0.772 | makro `abm/yayilim24/SONUC.md` |
| OPeRA sonraki web eylemi | LightGBM lambdarank **31.60%** (3 tohum ortalaması); Markov-2 26.94; sıfır atış Haiku persona ajanı 13.67 (n=300 alt örneklem) | web `RAPOR.md` |
| bank marketing AUC | lojistik **0.748**; LightGBM 0.665; Haiku persona 0.684 (yeniden: tabanlar) | intent `gbdt/lig/bank_lig.json` |
| GSS dalga tahmini, ortalama z | PCA ridge **0.560**; ridge 0.512; persistence 0.491 (yeniden) | dongu `turlar/gss_dalga/test/OZET.md` |
| araştırma döngüsü, Twin kiti | LLM gözlemci − rastgele arama +0.0018 [−0.0011, +0.0046]: rastgele yetti | dongu `turlar/twin_kit/test/OZET.md` |
| Big Five sadakati (BFI-2 ile atanan, Mini-IPIP ile ölçülen), N=20 — önceki depo, Claude değil | havuzlanmış r .911, ama uyumluluk +0.74, nevrotiklik −0.69 kayıyor | replicant `papers/ertan2009/results/validation_continuous.json` |
| Ertan vinyeti, düşük katkıyı cezalandırma oyu — önceki depo, Claude değil | personasız 0.94; rastgele persona 0.80; aşırı işbirlikçi 0.46; aşırı işbirlikçide ceza sayfasını yalnız 7/50 bitirdi | replicant `papers/ertan2009/results/otree_*` |
| b01 oyunları, ayrılmış oyunlarda insana MAE — önceki depo verisi, simengine hesabı | en iyi persona (gps) 0.113; düzgün rastgele 0.201; çıplak model 0.341: çıplak model rastgeleden kötü | edsl-test `benchmark/beebench/b01/results.csv`, RESULTS §8 |

## Dürüst sınırlar

- **Saf persona panelleri genelde tabanları geçemedi.** A/B'de Haiku paneli şans düzeyinde; kişi başı satın almada L1
  persona "her zaman hayır"ın altında; çapasız persona persistence'ın çok altında; tartışma turları ve ajan toplumu
  ölçülebilir bir şey eklemedi. LLM persona tahmini PPI++'ta ~1.0-1.1 insan değerinde.
- **LLM'nin değer kattığı yerler ölçüldü ve dar:** hiç veri yokken segment dağılımı, bir öğreniciyle stack (A/B
  TF-IDF + LLM), küçük erken testlerde önsel. Bunların dışında önce basit tabanı dene.
- **Dev'deki kazanç test'te küçülebilir.** OpinionQA A'da döngünün dev kazancı test'te tutmadı. Twin'de LLM'ye
  atfedilen +0.071 adil LLM'siz stack kurulunca +0.006'ya indi. Dominick's elastisitesindeki kazancı LLM-karıştırma
  plasebosu da verdi.
- **Ezber riski.** CES, Upworthy, OpinionQA gibi kamu verilerinde modelin cevabı görmüş olması dışlanmadı. Bellek
  denetimi ve mühür bunu azaltır, kanıtlamaz.
- **Bu paket sayıları yeniden ölçmedi.** Tabloda "(yeniden)" yazmayan her sayı araştırma kodundan aktarıldı. Paketin
  kendi testleri `FakeLLM` ile çalışır; gerçek API çağrısı içermez. Bazı varyantlar taşınmadı: hiyerarşik Bayes (HB)
  konjoint, NPE benimseme varyantı, retweet GBM'i. Veri setleri pakette yok; `bench` veri kökü verilmezse
  `missing_data` döner.
- **Oyun ve persona sonuçları Claude'da ölçülmedi.** Persona yöntem sıralamaları, Big Five sapmaları ve oyun
  sonuçları stepfun/step-3.5-flash ve gpt-oss-120b koşularından gelir; `simengine.llm` ile yeniden koşulmadan
  Claude'a aktarılamaz. Bütün araçlar ve normlar İngilizce ve ABD temelli; Türkçe persona iddiası için
  `validate_bigfive` ve oyunlar Türkçe yeniden koşulmalı.
- Kanıt raporunun metinleri (manşet cümlesi, kart ve panodaki bazı etiketler, denetim ayrıntıları) Türkçedir;
  rapor sözlüğü `manset`, `merdiven`, `kanit_kart` anahtarlarını korur. `summary()` İngilizce düz bir görünüm ve
  `headline_en` verir. İngilizce bir etkinlik için `se.card(rep, path, lang="en")` (ya da `--lang en`) Türkçe
  rozet gerekçesini ve manşeti çıkarır, ondalıkları noktaya çevirir; merdiven ve denetim etiketleri yine Türkçe kalır.
- `research.Loop` kaynak döngü gibi varsayılan olarak "eşit ve daha sade" kuralını uygular (`simpler_ties=True`,
  yalnız `Space(complexity=...)` verilince etkili).

## Veri setleri ve lisanslar

Depoda hiçbir veri yok. Her seti kendin, kendi koşullarıyla indir; `bench` adaptörleri yalnız yerel kopyayı okur
(`--data-root` ya da `SIMENGINE_DATA`, düzenler `simengine bench list -v`'de). Ödüllü bir teslimde kullanmadan önce
koşulları oku: ticari olmayan (NC) ve yalnız akademik setler bir yarışma teslimine uymayabilir.

| veri seti | nerede kullanılıyor | lisans / koşul | dikkat |
|---|---|---|---|
| UCI Bank Marketing | `bench` bank_marketing | CC BY 4.0 | atıf |
| Upworthy Research Archive | `bench` upworthy, A/B sonuçları | CC BY 4.0 | atıf |
| Open e-commerce 1.0 | `bench` open_ecommerce | CC BY 4.0 | atıf |
| Twin-2K-500 | `bench` twin2k500 | CC BY 4.0 (veri kartına göre) | atıf |
| OPeRA | sonraki eylem sonuçları | CC BY 4.0 | atıf |
| SimBench | `bench` simbench | **CC BY-NC-SA 4.0** | **ticari olmayan**; türev işler aynı lisansla |
| Pew ATP / OpinionQA / SubPOP | `bench` opinionqa_groups | **araştırma amaçlı** (Pew koşulları) | ödüllü/ticari kullanımda koşulları kontrol et |
| GSS panelleri (NORC) | `bench` gss | ücretsiz, atıf zorunlu, **yeniden dağıtım yok** | veriyi teslimle birlikte paylaşma |
| Dominick's | ABM elastisite sonuçları | **yalnız akademik araştırma** | yarışmada kullanmadan önce organizatörle netleştir |
| SEISMIC (SNAP) | retweet eğrisi sonuçları | **lisans belirtilmemiş**, araştırma; atıf Zhao vd. 2015 | ödüllü kullanımda kaynak sahibine danış |
| GSS örneklemi (NORC) | `persona.methods` demographic (`gss_sample.csv`) | ücretsiz, atıf zorunlu, **yeniden dağıtım yok** | veriyi teslimle birlikte paylaşma |
| Twin-2K-500 persona örneklemi | `persona.methods` twin2k (`twin2k_personas_sample.json`) | CC BY 4.0 (veri kartına göre) | atıf; pakette yok |
| Anthology backstories (Moon vd. 2024, arXiv:2407.06576) | `persona.methods` anthology (`anthology_backstories.json`) | **lisans doğrulanmadı**; veri setinin kartından kontrol et | pakette yok; ödüllü kullanımdan önce koşulları oku |
| GPS madde ifadeleri (Falk vd., QJE 2018) | `persona.methods` gps | yayımlanmış anket maddeleri; atıf | GPS verisi pakette yok, yalnız atıflı madde ifadeleri |
| BFI-2 madde içeriği (Soto & John 2017) | `persona.bigfive` cümle bankası (düzey-0 cümleleri madde köklerini izler) | © yazarlar; **ticari olmayan araştırma** için ücretsiz | ticari kullanımdan önce yazarlardan izin al |
| BeeBench b01 sonuç tablosu (edsl-test) | `bench` beebench_b01_prior / econ_games_b01, `games.read_results` | aynı sahip; LICENSE yok (tüm hakları saklı) | pakette yok; yerel kopyadan okunur |
| Ertan, Page & Putterman (2009) çalışma metni | `games.wtp` hedefleri (sayfa atıflı sabitler) | **telif hakkı yazarlarda** | PDF pakette yok; yalnız atıflı sayılar |

Koşullar araştırma notlarından (`makro/arastirma/veri-ham.md`, `makro/abm/arastirma/literatur.md`, `web/PROTOKOL.md`)
ve adaptörlerin `meta["source"]` alanından alındı; bağlayıcı olan her setin kendi sayfasıdır.

## Lisans

Özgür yazılım: **GNU Affero General Public License v3.0 ya da sonrası** (`AGPL-3.0-or-later`, tam metin `LICENSE`).
Kütüphaneyi değiştirip dağıtan ya da ağ üzerinden hizmet olarak sunan herkes, değiştirdiği kaynak kodu aynı
lisansla kullanıcılarına açmak zorundadır. Telif: the simengine authors (github.com/tunapro1234). Taşınan kaynaklar
ve lisansları (replicant MIT, atfı korunur; edsl-test aynı sahip) `NOTICE` dosyasında; üçüncü taraf veri setleri
dağıtılmaz ve kendi lisanslarına tabidir. BFI-2 maddeleri ticari olmayan araştırma kullanımı içindir (`NOTICE`).

## Önceden yapılmış iş beyanı

simengine bir yarışmadan **önce** yazıldı. Bir hackathonda kullanılırsa teslim metninde açıkça belirtilmeli ve
etkinliğin önceden yazılmış kod kurallarına uyulmalı. Beyan, etkinlik öncesi durumun kontrol edilebilmesi için
tarihli bir commit ya da etikete atıf yapmalı; örnek:

> "Etkinlikten önce yazdığımız simengine kütüphanesi (v0.1.0, commit `<hash>`, `<tarih>`) üzerine kurulu."

`<hash>` ve `<tarih>` etkinlik başlamadan önceki son commit'ten alınır (`git log -1 --format='%h %ad' --date=short`;
tercihen o commit'e bir etiket konur). Kütüphane özeldir ve açık lisanslı değildir: etkinlik önceden yazılmış kodun
herkese açık olmasını şart koşuyorsa, kullanmadan önce depoyu bir lisansla açmak ya da organizatöre sormak gerekir;
"açık kütüphane" demek yanlış olur. Etkinlik sırasında yapılan iş; veri hazırlığı, yapılandırma, mühürlenen
tahminler ve kanıt kartıdır. Önceden yazılmış kodu "yarışma sırasında yapıldı" diye sunmak yok.

## Test

```bash
python -m unittest discover -s tests -t .
```

Çevrimdışı. Gerçek veriyle yayımlanan sayıları yeniden üreten testler (`tests/test_bench_data.py`) veri kökünü
`SIMENGINE_TEST_DATA`, yoksa `SIMENGINE_DATA` ortam değişkeninden okur ve veri yoksa atlanır. Araştırma paketiyle
eşdeğerlik testi (`tests/test_evidence_equivalence.py`) yalnız `SIMENGINE_KANIT_SRC` orijinal `kanit/kod` dizinini
gösterdiğinde çalışır. Ayrıntılı çıktı için `-v`.
