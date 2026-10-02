# Metadata census двух авторских кейсов

## Запрос и границы

Дата: 2026-10-02. Baseline: `f312efb8f5fa291a961c337e6efa3b67cc247be4`.
Владелец разрешил исполнение исследовательского плана; эта задача — подготовка входных
метаданных для кейсов «Булгаков / Двенадцать стульев» и «Шолохов / Тихий Дон».
Документ предлагает следующий research input и не меняет нормативный `research/ROADMAP.md`.

Прочитаны tracked конфиги/агрегаты и разрешённые локальные JSON manifests. Корпусные `.txt`,
raw/private passages и секреты не читались; художественные тексты не скачивались.
Изменение этой подзадачи — только настоящий документ. Fits, новые quality claims,
атрибуционные решения, регистрация корпуса и внешние записи не выполнялись.

**Результат.** Есть воспроизводимые списки файлов/work IDs и конкретные библиографические
приоритеты. Полная независимая панель для любого из двух кейсов ещё не установлена.
Разные hashes и content components не удостоверяют автора, издание или отсутствие
невыявленного заимствования; источник-label не равен проверенной библиографической provenance.

## 1. Три отдельных снимка

| Снимок | Источник и identity | Что установлено | Ограничение |
|---|---|---|---|
| H: исходный исторический каталог | `docs/corpus_manifest.json:4`, SHA256 `4f4e5e941c1ae953e5a07e007c0e5bf3211f8fcdee47c6f0aed4160cd46dd521` | 290 файлов, 51 author label, 156 source labels | Каталог старого корпуса; текущие тексты и bibliographic provenance не проверялись |
| A: audit parent | `data/audit_corpus/15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029/corpus_manifest.json` | 255 work IDs, digest `15d265e0878dbf1acd9224e2558598ff7266fd6fc650585d1433fbd65a717029` | Это отдельный parent benchmark universe, не H из 290 файлов |
| C: доступный corrected bundle | `research/local/v3_2_bundle_20260824/paired_audit_v3_2_bundles/ff620b05f20b81c21732014b553aa739a393c74fe344e6d9f2bd8d80996cef21/` | 252 work IDs; corrected corpus digest `1a9a0779e4e578f38664fd974c7ac4565f12fb4992cf29773a34061fddee8531` | Локальный candidate bundle; case candidate panel не сформирован |

Для C далее используется **LOBO work catalog**, `lobo_fold_manifest_v3_2.json`, а не
ограниченная RUAA selection. Его SHA256:
`85a90b76f9860a6bdb5125d1ab377ce5c7bd64cd0fcdc68cb53dd95d0933abe1`.
SHA256 `corrected_corpus/corrected_corpus_manifest_v3_2.json`:
`cd3bf5e37b6ca12295d7a86bf3bb1d19427993cae118b59db106280740e40468`.

Наблюдаемый статус `candidate.json`: `local_candidate_preparation_pending_review`,
`independent_manifest_review_required`, freeze unapproved, evaluator unregistered,
confirmatory execution hard-disabled, headline/publication not authorized. Это статус
самого локального candidate, а не новое решение governance.

## 2. Исторический каталог H: количество, источник, пробелы

«Известен источник» ниже означает только `source != local/неизвестно` в H.
Ни у одной из этих 54 строк нет полей даты/периода или конкретного издания.
Проверенная source→edition→clean-bytes цепочка здесь не установлена.

| Author label | Файлов H | Source labels H | Target / independence issue | Edition/date gaps |
|---|---:|---:|---|---|
| `bulgakov` | 5 | 0/5 | Пять разных названий; самостоятельность редакций и регистр референсов не подтверждены | 5/5 без издания и дат |
| `ilf-petrov` | 7 | 2/7 | Две книги дилогии входят в label: обе должны быть targets при проверке дилогии. Из оставшихся пяти `rasskazi` и циклы требуют composition map | 7/7; source labels только у targets |
| `sholohov` | 15 | 0/15 | Четыре файла ТД — один роман, не четыре независимых контроля. Семь ранних рассказов и четыре поздних файла — кандидаты в ref, авторские метки требуют отдельного обоснования | 15/15; дополнительная информация есть в config, но bytes mapping не подтверждён |
| `krukov` | 4 | 0/4 | Четыре разных basename; genre labels: три очерка, одна повесть. Нет verified content/edition independence для кейса | 4/4 |
| `serafimovich` | 9 | 9/9 | Source labels Викитеки; `у_нас_и_у_них` далее исключён из C по recorded authorship mismatch | 9/9; нужны точные страницы издания/редакции |
| `sevsky` | 9 | 9/9 | Source labels Викитеки; `дон_на_костылях` далее исключён из C по recorded source quality reason | 9/9; `/ДО` в двух source labels — повод проверить редакцию и нормализацию |
| `kataev` | 3 | 0/3 | Три basename; библиографическая идентификация и сопоставимость с сатирой 1920-х неизвестны | 3/3 |
| `olesha` | 2 | 0/2 | `zavist` и `ni_dnya_bez_strochki`: нельзя считать одинаковыми по жанру/периоду без разметки | 2/2 |

Источники строк H: `docs/corpus_manifest.json:110` (Булгаков), `:764` (Ильф–Петров),
`:812` (Катаев), `:884` (Крюков), `:1224` (Олеша), `:1488` (Серафимович),
`:1548` (Севский), `:1608` (Шолохов). Это наблюдаемые метаданные, не авторская экспертиза.

## 3. Corrected bundle C: независимые вычислительные identities

| Author label | Work IDs A → C | Разных work-content IDs / content components C | Source metadata прямо в C LOBO catalog | Следствие для case panel |
|---|---:|---:|---|---|
| `bulgakov` | 5 → 5 | 5 / 5 | Библиографических source/edition/date fields нет | Сохранились вычислительные identities; 0/5 source labels в H не исправлены этим фактом |
| `ilf-petrov` | 0 → 0 | 0 / 0 | Строк нет | C не содержит основной ref label первого кейса |
| `sholohov` | 0 → 0 | 0 / 0 | Строк нет | C не содержит основного ref label второго кейса |
| `krukov` | 4 → 4 | 4 / 4 | Библиографических полей нет | Четыре вычислительно разных ref candidates; библиографические gaps H остаются |
| `serafimovich` | 9 → 8 | 8 / 8 | Библиографических полей нет | Из восьми можно составлять ref shortlist после edition/date checks |
| `sevsky` | 9 → 8 | 8 / 8 | Библиографических полей нет | Восьми identities недостаточно для вывода о мощности; слова/жанры отдельно не измерялись |
| `kataev` | 3 → 3 | 3 / 3 | Библиографических полей нет | Нужна period/register и source аттестация |
| `olesha` | 2 → 2 | 2 / 2 | Библиографических полей нет | Разные identities не снимают жанровую неоднородность |

В C `exclusion_policy.exclusions` записывает:
`serafimovich/у_нас_и_у_них` → `adjudicated_authorship_mismatch`,
`sevsky/дон_на_костылях` → `adjudicated_source_quality_exclusion`,
`turgenev/записки_охотника` → `collection_umbrella_content_component`.
Первые две причины здесь переписаны как факты manifest; сам литературный спор/качество
текста повторно не расследовались.

`content_isolation_audit_v3_2.json` содержит пустые findings для exact byte duplicates,
normalized duplicates, cross-work component overlap, collection-member overlap и
train/test leakage. Это recorded aggregate audit данного C; corpus texts и выполнение
аудитора в этой задаче не проверялись. Отсутствие находок не устанавливает библиографическую
или историческую независимость.

## 4. Явная карта work IDs и групп

Во всех строках prefix — точный author label. Приведённые basename сверены H и C,
кроме двух labels, отсутствующих в C. Предложенные book/content groups ещё не зарегистрированы.

| Label | Exact basename/work set | Предлагаемая единица независимого ref / исключение |
|---|---|---|
| `bulgakov/` | `belaya_gvardiya`, `diavoliada`, `master_i_margarita`, `rokovie_yaytca`, `sobache_serdce` | По одному work на название, edition grouping обязателен. Сборник «Дьяволиада» целиком нельзя добавить как независимый work рядом с входящими повестями |
| `ilf-petrov/` | `1001_den`, `neobiknovennie_istorii`, `odnoetazhnaya_america`, `rasskazi`, `svetlaya_lichnost`, `двенадцать_стульев`, `золотой_телёнок` | Обе последние — target group дилогии; остальные пять только shortlist. `rasskazi` без contents не сертифицировать; циклы разбить на named works с collection membership |
| `sholohov/` early | `aleshkino_serdce`, `batraki`, `chuzhaya_krov`, `lazorevaya_step`, `pastuh`, `rodinka`, `zherebenok` | Семь named stories; collection «Донские рассказы» не восьмой ref. Early author labels не объявлять независимо доказанными |
| `sholohov/` late | `nauka_nenavisti`, `oni_srazhalis`, `podnyataya_celina_2`, `sudba_cheloveka` | Четыре file candidates вне ТД; поздний panel отдельный. Две книги «Поднятой целины» — одна novel group при work-level independence |
| `sholohov/` target | `tihiy_don_1`, `tihiy_don_2`, `tihiy_don_3`, `tihiy_don_4` | Одна novel group; все четыре одновременно вне fit, calibration и настройки |
| `krukov/` | `na_tihom_donu`, `oficersha`, `v_glubine`, `v_rodnih_mestah` | Четыре named work candidates; source, contents и первое издание неизвестны |
| `serafimovich/` C | `бомбы`, `железный_поток`, `как_он_умер`, `лихорадка`, `на_льдине`, `пески`, `степные_люди`, `стрелочник` | Восемь identities; запрещённый C work `у_нас_и_у_них` не возвращать по старому config |
| `sevsky/` C | `без_карболки`, `дворянин_гукасов`, `дядино_отопление`, `жорж_корсаров`, `игнатов_бугор`, `пропавшая_грамота`, `сквозные_ворота`, `степняки` | Восемь identities; `дон_на_костылях` оставить исключённым по C policy |
| `kataev/` | `o_dolgom_yashike`, `poveliteli_zheleza`, `vremya_vpered` | Три named candidates; не подменять Валентина Катаева сольным Евгением Петровым |
| `olesha/` | `ni_dnya_bez_strochki`, `zavist` | Два named candidates; первоиздание и состав записей нужны отдельно |

## 5. Config и case passports: дополнительные сведения, не их смешение с H/C

`configs/sholokhov.yaml:12`–`:31` указывает source URLs и `period` для четырёх книг ТД,
«Поднятой целины», поздней прозы и группового entries ранних рассказов. Так, четыре ТД
имеют dates 1928/1929/1933/1940; поздние labels — 1942/1957/1959/1969; early group — 1925.
Это **values config**, не проверка первой публикации или времени создания. Группа семи
рассказов одной строкой не даёт семь индивидуальных дат и источников.

URLs militera/lib.ru в config конкретнее неизвестного source label H, но нет доказанной
связи fetched edition → clean bytes H → work identity. `local/donskie` и
`local/podnyataya_celina_1` не внешние источники. «Поднятая целина, кн.1» помечена
`unknown`/спорным target, не входит в 15 H строк `sholohov`; её нельзя незаметно добавлять
как бесспорный ref. Contents поздних изданий также не удостоверяют год написания.

`configs/classics_don.yaml:5`–`:17` содержит 13 плановых entries Серафимовича,
`:20`–`:26` — семь Севского. Это не counts H (9/9) или C (8/8). Config всё ещё включает
оба исключённых C titles; один config не разрешает возвращать их в corrected panel.
`classics2.yaml` и `classics3.yaml` не содержат entries восьми рассматриваемых labels.

`docs/ilf_vs_petrov.json:10` записывает `genre_confounded=true`; `:23` описывает
единственный сольный Ильф как записные книжки, Петрова как военную публицистику/мемуар.
Агрегат не содержит проверяемого named-work/edition catalog этих сольных references.
По нему нельзя сертифицировать раннюю solo panel или вклад каждого соавтора.
`docs/ilfpetrov_timeline.json` и `docs/cases_attribution.json` содержат исторические
назначения/сходства, не новую атрибуцию и не текущую provenance chain.

`docs/sholokhov_rigor12.json:12` перечисляет семь early basenames, `:21` — четыре TD
basenames; test-registry note сохраняет reference circularity limitation. Ранние рассказы
могут быть отдельной relational panel «тот же профиль», но фамилия референсного автора
остаётся предпосылкой. `docs/case_passport.json` содержит агрегаты пяти других кейсов;
точного corpus passport для этих двух panels он не предоставляет.

## 6. Приоритетная библиография первого кейса

Тарасова (2020), DOI `10.25205/2410-7883-2020-1-117-145`, документирует перенос материала
сольных текстов в дилогию. Ниже — bibliography leads, не уже независимые refs.
[Статья и библиография](https://www.philology.nsc.ru/journals/sis/article.php?id=264),
[PDF журнала](https://www.philology.nsc.ru/journals/sis/pdf/SS2020-1/08.pdf).

| Solo label proposal | Work leads / dates по статье | Страницы статьи | Что проверить |
|---|---|---|---|
| `ilf` | «Рыболов стеклянного батальона» (1923); «Ярмарка в Нижнем» (1924); «Катя-Китти-Кет» (1924) | 122, 124, 130 | Первая публикация, подпись, edition, reused-passages group |
| `ilf` | Цикл «В Средней Азии» (1925): «Азия без покрывала», «Глиняный рай», «Перегон Москва – Азия», «Энвер-басмач» | 134–136 | Четыре named works; contents cycle; overlap с дилогией |
| `petrov` | «Гусь и украденные доски» (1924); «Юморист Физикевич» (1927); «Всеобъемлющий зайчик» (1927) | 132, 124, 140 | Подписи и точные даты; provenance reuse map |
| `petrov` | «Несантиментальное путешествие»; «Коричневый город» / «Граждане туристы» (публикации 1928) | 138 | Предположение о создании в 1927 не установленная дата; отдельный stratum |

Начать сверку изданий с Ильф, «Дом с кренделями» (Текст, 2009); Петров, «День борьбы
с мухами» (Текст, 2009), «Невероятно, но...» (Гудок, 1928), «Шевели ногами» (Огонёк,
1930). Это сборники, не по одному independent work. Подтверждённое текстовое повторение
объединять content group или исключать из ref; тематическая перекличка сама по себе не duplicate.

Для Булгакова доступен каталог прижизненных публикаций НЭБ: сборник «Дьяволиада»
(«Недра», 1925) включает также «Роковые яйца», «№ 13. Дом Эльпит-Рабкоммуна»,
«Китайскую историю», «Похождения Чичикова». НЭБ перечисляет и сборник «Рассказы»
(1926), включая «Воспаление мозгов», «Летучий голландец», «Паршивый тип» и другие
named stories. Это приоритетные period/register leads; сборник и его члены не независимые
пробы. [Каталог НЭБ](https://bulgakovs.rusneb.ru/).

Музей подтверждает сотрудничество Булгакова с газетами в 1921–1926 и наличие прижизненных
сборников. Следующий bibliography input — named фельетоны «Гудка»/«Накануне» с подписью,
номером, датой и страницей; газетный регистр выделить отдельно от романов.
[Выставка музея «Тощая книжонка»](https://bulgakovmuseum.ru/events/vystavka-toshhaya-knizhonka).
Для проверки редакций есть metadata архива РГБ, ф. 562; наличие фонда не удостоверяет
использованную локальную редакцию. [Карточка НЭБ](https://rusneb.ru/catalog/000199_000009_004709881/).
Карточки книг `009153100` и `005454800`, связанные с каталогом выставки НЭБ, при проверке
вернули HTTP 403; metadata titles подтверждены самой коллекционной страницей, contents
оцифровок не проверялись.

Совместные works вне дилогии нужны отдельной дуэтной panel; исторические пять кандидатов
из раздела 4 требуют contents и первого издания. Сольные профили не заменяют профиль дуэта.
Катаев и Олеша остаются необходимыми author-label controls после source/date/register checks;
их представленность 3/2 файлами не является измерением достаточной мощности.

## 7. Приоритетная metadata сверка второго кейса

ФЭБ описывает собрание сочинений в восьми томах (ГИХЛ, 1956–1960), «Лазоревую степь»
(Новая Москва, 1926), письма (ИМЛИ РАН, 2003), рукописи ТД и библиографические указатели.
Эти издания позволяют разделить дату произведения, дату редакции и дату source edition.
Письма, публицистика и fiction составляют разные registers.
[Описание ЭНИ](https://feb-web.ru/feb/sholokh/rub1.html?cmd=1).

[Алфавитный указатель ФЭБ](https://feb-web.ru/feb/feb/atindex/atindex9.htm?cmd=show)
перечисляет семь early titles H, другие рассказы («Нахалёнок», «Коловерть», «Семейный
человек», «Бахчевник», «Один язык»), обе книги «Поднятой целины», поздние рассказы,
четыре книги ТД и автограф. Указатель добавляет варианты заглавий; aliases не дают новые works.
«Лазоревая степь» как сборник 1926 и одноимённый рассказ требуют разных edition/collection
roles. Новые early titles — shortlist, не автоматически бесспорные референсы.

| Metadata task | Приоритетный источник | Результат для следующей panel |
|---|---|---|
| Индивидуальные даты/первые публикации семи early stories | Указатель/примечания ФЭБ; metadata сборника 1926 | Seven work rows; separate collection membership, edition, label confidence |
| Late fiction: `nauka_nenavisti`, `sudba_cheloveka`, `oni_srazhalis`, `podnyataya_celina_2` | Конкретные том/страницы и редакция, сопоставленные config URLs | Late stratum отдельно; дата source не заменяет composition date |
| ТД четыре книги/редакции/автограф | Библиография и каталог рукописей ФЭБ | Один held-out novel group, edition-alias map |
| Крюков четыре named refs | Библиографические карточки и first-edition contents ещё предстоит установить | Source/date gaps закрыть до расширения количества файлов |
| Серафимович/Севский | C exclusion policy + точные editions для восьми оставшихся works каждого | Не возвращать два excluded titles; не считать OCR/редакцию новым work |

## 8. Что ещё нельзя признавать независимым

- Оба target романа дилогии, либо любая из четырёх книг ТД, в references проверяемого кейса.
- Сборник вместе с его рассказами; общие `rasskazi`/cycles без composition map.
- Две редакции одного произведения, chapter split, duplicate basename alias как отдельные works.
- Ранний сольный lead, повторяющий target passage, до overlap adjudication; metadata дата
  «до романа» не снимает target contamination.
- Seven early Sholokhov author labels как независимое доказательство фамилии, если гипотеза
  допускает спор об авторстве самих рассказов.
- `local/неизвестно`, либо config URL без связи с проверенной редакцией/clean bytes.
- Different content components C как историко-библиографическую экспертизу независимости.

Следующий reviewable input — таблица `author_label × work_id × content_group × genre/register ×
composition/first-publication date × edition/editor × source_url × source_sha256 ×
attribution_status × target/ref role`. Для каждого поля нужны source и статус
`observed / inferred / unknown`; неизвестное не заполнять типичным значением. Установленные
источники ещё не были привязаны к локальным copyrighted bytes, и эта задача не разрешает
добавлять их в Git или проводить fits.

## 9. Воспроизведение metadata counts

Команда читает только JSON metadata, не chunk payloads. Для проверки H/C counts и identities:

```python
import json
from pathlib import Path

labels = ["bulgakov", "ilf-petrov", "sholohov", "krukov", "serafimovich",
          "sevsky", "kataev", "olesha"]
h = json.loads(Path("docs/corpus_manifest.json").read_text())
base = Path("research/local/v3_2_bundle_20260824/paired_audit_v3_2_bundles/")
p = base / "ff620b05f20b81c21732014b553aa739a393c74fe344e6d9f2bd8d80996cef21/lobo_fold_manifest_v3_2.json"
c = json.loads(p.read_text())
for author in labels:
    books = h["authors"][author]["books"]
    works = [r for r in c["works"] if r["author_id"] == author]
    known = sum(b.get("source") not in (None, "", "local/неизвестно") for b in books)
    print(author, len(books), known, len(works),
          len({r["work_content_identity"] for r in works}),
          len({r["content_component_identity"] for r in works}))
```

Наблюдения H и C пересчитаны этой логикой; имена/ключи, exclusion policy и linked metadata
проверены отдельно. Веб-проверка 2026-10-02 — primary publisher, музей, НЭБ и ФЭБ;
библиографическая карточка двух сборников недоступна (403), DOI redirect не открывался,
издательские HTML и PDF статьи доступны. Accuracy, мощность и corpus content quality
в данной задаче не измерялись. Historical JSON, configs, governance и public claims неизменны.
