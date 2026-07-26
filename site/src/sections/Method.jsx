import { Fragment } from "react";
import { Card, Stat, Timeline, WhyBlock } from "@dmitrymake/rk-ui";
import { FEATURES, LOBO_STRICT, TOMSK, HEADLINE, MODELS } from "../data.js";
import { CORPUS } from "../corpus.js";
import { CASES, BENCH, BENCH_EXT } from "../segdata.js";
import { fmtScore, fmtP, fmtRange, fmtPct } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";
import Sources from "../components/Sources.jsx";

const TA = CASES.tolstoyAn;
const C = CASES;
// значимость — все числа из генератора, не литералы
const BOW_M = MODELS.find((m) => m.id === "bow_lr");
// author-clustered CI macro-F1 ОТОЗВАН (HEADLINE.macroF1CI === null) — показываем точку.
// доверительный интервал точности считаем по выборкам КНИГ (author-clustered) — единообразно
// с macro-F1 CI и с WhyBlock «единица оценки — книга»; более узкий чанк-интервал не берём.
const ACC_CI = fmtRange(HEADLINE.accCIAuthor[0], HEADLINE.accCIAuthor[1]);
// max шкал берётся из данных, не литералом
const CH_MAX = Math.max(...BENCH.channels.map((r) => r.v));
const PROZA_MAX = Math.max(...BENCH_EXT.prozaCompare.map((r) => r.v));
const SIL_MAX = Math.max(...TA.sil.map((r) => r.v));
const SIL_MIN = Math.min(...TA.sil.map((r) => r.v));
// группы признаков ищем по имени, а не по хрупкому индексу отсортированного массива
const CH_CHAR = BENCH.channels.find((r) => r.c.startsWith("char"));
const CH_SYN = BENCH.channels.find((r) => r.c.includes("синтаксис") || r.c.startsWith("syntax"));
// Русские подписи каналов прямо в диаграмме: данные приходят с англ. слагами — переводим на месте,
// чтобы в графике не осталось технического жаргона (find выше работает по исходному r.c, не по подписи).
const CH_RU = {
  "char-ngrams (2-5)": "цепочки букв (2–5)",
  "word-ngrams (1-2)": "мешок слов (1–2)",
  "синтаксис dep+pos+syn": "синтаксис (связи и части речи)",
  "dependency (чистый идиолект)": "синтаксические связи",
  "function words": "служебные слова",
  morphology: "морфология",
  "словообразование (DSP)": "словообразование (суффиксы)",
  "АНСАМБЛЬ (равновесный)": "ансамбль (все группы поровну)",
};
const chRu = (c) => CH_RU[c] || c;
// разброс «цельного одиночки» берём из бесспорных контролей, не литералом
const SINGLE_LOW = Math.min(...C.controls.map((c) => c.v));
const SINGLE_HIGH = Math.max(...C.controls.map((c) => c.v));
// худший по узнаваемости автор публикуемого среза + сколько его книг помечено верно
const WORST = BENCH.worstRecall;
const WORST_OK = Math.round(WORST.recall * WORST.books);
const ruBooks = (n) => {
  const mod100 = Math.abs(n) % 100;
  const mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "книг";
  if (mod10 === 1) return "книга";
  if (mod10 >= 2 && mod10 <= 4) return "книги";
  return "книг";
};
// склонение «автор» по образцу ruBooks — чтобы число из данных не ломало грамматику
// (только для именительного счётного: «51 автор», «22 автора», «43 автора», «5 авторов»).
const ruAuthors = (n) => {
  const mod100 = Math.abs(n) % 100;
  const mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "авторов";
  if (mod10 === 1) return "автор";
  if (mod10 >= 2 && mod10 <= 4) return "автора";
  return "авторов";
};
// пересчёт по целым книгам на открытых данных группы из ТУСУР: строка для 50 авторов
// и диапазон масштабов (числа для читаемого примечания — из данных, не литералами).
const TOMSK_50 = TOMSK.headToHead.table.find((r) => r.k === 50);
const TOMSK_KMIN = Math.min(...TOMSK.headToHead.table.map((r) => r.k));
const TOMSK_KMAX = Math.max(...TOMSK.headToHead.table.map((r) => r.k));

// Короткие устойчивые ярлыки трёх срезов корпуса — чтобы читатель не путал 22 / 43 / 51 автора
// (числа приходят из данных, склонение — через ruAuthors; форма именительная для скобочного вида).
const SLICE_OPEN = `открытый срез (${BENCH.nAuthors} ${ruAuthors(BENCH.nAuthors)})`;
const SLICE_BOOK = `срез по книгам (${CORPUS.benchmark.authors} ${ruAuthors(CORPUS.benchmark.authors)})`;
const SLICE_ALL = `весь корпус (${CORPUS.research.authors} ${ruAuthors(CORPUS.research.authors)})`;

// Русские подписи строк графика Proza.ru прямо в диаграмме: данные приходят с англ. слагами
// (reliability^6, char-SVM…) — переводим на месте, чтобы в графике не осталось жаргона.
const PROZA_RU = {
  "char-SVM (консервативный лидер)": "цепочки букв (char-SVM) · лидер",
  "reliability^6 (test-favoured diagnostic)": "взвешивание по надёжности (настроено под тест)",
  "word-SVM (мой)": "мешок слов (word-SVM)",
  "равновесный ансамбль (наивный)": "все группы поровну (наугад)",
  "ruBERT-tiny2 (pretrained нейро)": "облегчённая нейросеть (ruBERT-tiny2)",
};
const prozaRu = (m) => PROZA_RU[m] || m;

// Единый стиль заголовка сворачиваемых блоков «детали».
const SUMMARY_STYLE = { cursor: "pointer", fontFamily: "var(--font-text)", fontSize: "var(--fs-caption)", fontWeight: "var(--fw-semibold)", letterSpacing: "var(--tracking-caption)", textTransform: "uppercase", color: "var(--text-muted)" };

// Признаки перечислены без статусных бейджей. Факультативные блоки (kind: «opt») приглушены через opacity.
const KIND_STYLE = { opt: { dim: true } };

// Русские подписи карточек признаков прямо при рендере: данные приходят с англ. слагами,
// data.js не трогаем — переводим на месте (как chRu/prozaRu в диаграммах), чтобы в карточках
// не осталось сырых кодов («dependency», «off», «spaCy morph»).
const FEAT_NAME_RU = { dependency: "синтаксические связи" };
const FEAT_NOTE_RU = {
  "off — риск утечки темы": "выключено — риск утечки темы",
  "падеж/время/вид (spaCy morph)": "падеж, время, вид (разбор spaCy)",
};
const featName = (n) => FEAT_NAME_RU[n] || n;
const featNote = (n) => FEAT_NOTE_RU[n] || n;

const PROTOCOL = [
  { marker: "01", title: "очистка", body: "Нормализация текста, удаление точных дублей по контрольной сумме, отсев почти-одинаковых отрывков одного автора — чтобы куски своих же книг не подсматривали из обучения в проверку.", color: "var(--icon-blue)", state: "done" },
  { marker: "02", title: "разделение", body: "Книга — неделимая единица. Отрывки одной книги никогда не попадают по разные стороны границы между обучением и проверкой.", color: "var(--icon-blue)", state: "done" },
  { marker: "03", title: "проверка по целым книгам", body: `Каждую книгу по очереди убираем и проверяем — ${HEADLINE.books} пересчётов, по одному на отложенную книгу. На её шаге всё обучаемое учится строго без неё: словарь, частоты слов, веса классификатора. Вердикт по книге — среднее вероятностей её отрывков, чтобы одна удачная страница не стала выводом. Всего проверено ${HEADLINE.authors} ${ruAuthors(HEADLINE.authors)} / ${HEADLINE.books} ${ruBooks(HEADLINE.books)}.`, color: "var(--gold)", state: "done" },
  { marker: "04", title: "значимость", body: "Каждую цифру проверяем отдельно: насколько она точна, устойчива ли к пересчётам и не случайно ли выше простого метода.", color: "var(--cinnabar)", state: "done" },
];

export default function Method() {
  return (
    <section className="section" id="method">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Метод</p>
          <h2>Проверяем по целым книгам — без подсматривания</h2>
          <p className="prose lead muted">
            Главное в атрибуции не принять смену темы за смену руки. Автор
            берёт новый сюжет, других героев, другой словарь — и слабый метод решает, что
            сменился человек. Страхуемся так: проверяемую книгу целиком убираем из корпуса
            и возвращаем только на шаге предсказания. Она не участвует ни в словаре, ни в
            частотах слов, ни в обучении — текст не может подсказать сам себя. Поэтому
            итоговая цифра показывает, как метод переносится на новую книгу, а не насколько
            он её запомнил.
          </p>
          <p className="prose muted">
            Заголовочная проверка — по одной отложенной книге за раз; её доля верных
            попаданий — {fmtScore(LOBO_STRICT.styloFullLobo, 3)}.
            Деление корпуса на 5 частей — быстрый ориентир для перебора
            ({fmtScore(LOBO_STRICT.proxyTop1, 3)}). Числа близкие, но считаются по-разному
            и на разных наборах авторов. Поэтому между собой их не сравнивают.
          </p>
          {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
            <p className="mono muted" style={{ fontSize: 12 }}>
              Оговорка: при обучении длинная книга сейчас весит больше короткой. Пересчёт «одна книга —
              один голос» ещё впереди — заголовочная цифра может немного сдвинуться.
            </p>
          )}
        </div>

        {/* Протокол без подсматривания */}
        <div className="split reveal" style={{ alignItems: "start" }}>
          <div className="prose">
            {/* Заголовок раздела — вручную свёрстанный <h3>, а не <StageHeader>:
                у того title рендерится как <h1> и ломает иерархию заголовков. */}
            <header style={{ marginBottom: 18 }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", marginBottom: "var(--space-2)" }}>
                <span style={{ width: 10, height: 10, background: "var(--gold)", flex: "0 0 auto" }} />
                <span style={{ fontFamily: "var(--font-text)", fontSize: "var(--fs-caption)", fontWeight: "var(--fw-semibold)", letterSpacing: "var(--tracking-caption)", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  Протокол · без подсматривания
                </span>
              </div>
              <h3 style={{ fontFamily: "var(--font-display)", fontSize: "var(--fs-display-2)", lineHeight: "var(--lh-display-2)", fontWeight: "var(--fw-bold)", letterSpacing: "var(--tracking-tight)", margin: 0, color: "var(--text)" }}>
                Модель не видит проверяемую книгу
              </h3>
              <p style={{ fontFamily: "var(--font-text)", fontSize: "var(--fs-body)", lineHeight: "var(--lh-body)", color: "var(--text-muted)", margin: "var(--space-2) 0 0", maxWidth: "60ch" }}>
                На каждом шаге словарь, частоты слов и классификатор видят только остальные книги. Отложенная книга появляется ровно один раз — на шаге предсказания.
              </p>
            </header>
            <details style={{ marginTop: 4 }}>
              <summary style={SUMMARY_STYLE}>Как проверяли значимость</summary>
              <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 10px", maxWidth: "62ch" }}>
                Каждую цифру считали отдельной проверкой и сравнивали с простыми опорными
                методами — настоящей Burrows Delta (отклонения по частым словам), мешком слов
                и точкой отсчёта «всегда самый частый автор».
              </p>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Точность по авторам (macro-F1 — средняя по всем авторам поровну, чтобы авторы с большим числом книг не перетягивали среднее) — единая оценка {fmtScore(HEADLINE.macroF1)}. Разброс по авторам не приводим: пересборка по авторам меняет набор учитываемых авторов, поэтому такой интервал для macro-F1 недействителен и отозван.
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Доверительный интервал точности по книгам [{ACC_CI}] — из повторных пересчётов на случайных выборках книг.
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Тест МакНемара (сравнивает два метода на одних и тех же книгах) против опорных методов даёт p {fmtP(BOW_M.p)}: stylo надёжно обходит и Burrows Delta, и мешок слов — отрыв не случайный.
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Калибровка (ECE — насколько заявленная уверенность расходится с реальной долей попаданий) {fmtScore(HEADLINE.ece)}: плохая (хорошая около 0.02–0.05). Модель склонна переоценивать себя, поэтому доли вероятностей в разборах читаем как порядок версий, а не как точную уверенность.
              </span>
            </details>
          </div>
          <Card padding={20} parade>
            <Timeline items={PROTOCOL} />
          </Card>
        </div>

        {/* почему книга — единица оценки */}
        <div className="reveal module">
          <WhyBlock title="Почему по книгам, а не по отрывкам">
            Отрывки внутри одной книги похожи между собой: общая тема, лексика, герои.
            Считать их независимыми — значит занижать погрешность: точность кажется
            надёжнее, чем она есть. Единственная честная единица оценки — книга. Поэтому и
            погрешность, и голосование, и все проверки на неслучайность работают на уровне книг.
          </WhyBlock>
        </div>

        {/* ХУК: держит ли метод одного автора сквозь жанры */}
        <div className="reveal module">
          <h3>Расщепит ли метод одного автора?</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 16 }}>
            Тест: взять <strong style={{ color: "var(--text)" }}>бесспорно одного</strong> автора,
            писавшего в совсем разных жанрах, и посмотреть, распадётся ли он на разные «руки».
            Удобный подопытный — <strong style={{ color: "var(--text)" }}>А. Н. Толстой</strong>: {TA.genres}.
            Меряем на чистом признаке — синтаксических связях (кто с кем в предложении связан):
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              <div className="mono muted" style={{ fontSize: 11, marginBottom: 8 }}>
                разброс между книгами одного автора (чем ниже — тем цельнее автор):
              </div>
              {TA.sil.slice().sort((a, b) => a.v - b.v).map((r) => (
                <div key={r.a} style={{ display: "grid", gridTemplateColumns: "14ch 1fr 5ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                  <span style={{ fontSize: 12.5, color: r.hi ? "var(--text)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400 }}>{r.a}</span>
                  <MeterBar value={r.v} max={SIL_MAX} accent={r.hi ? "var(--icon-blue)" : "var(--text-muted)"} />
                  <span className="mono" style={{ fontSize: 11, color: r.hi ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gap: 10, alignContent: "start" }}>
              <p className="verdict" style={{ margin: 0 }}>
                А. Н. Толстой ({fmtScore(TA.sil[0].v, 3)}) держится в узком диапазоне бесспорных
                одиночек ({fmtRange(SIL_MIN, SIL_MAX)}) — вровень с Тургеневым. Научная фантастика,
                историческая проза, эмигрантский роман <em>не расщепляют</em> его на разные руки.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Больше того: <strong style={{ color: "var(--text)" }}>{TA.nSelf} из {TA.nBooks}</strong>{" "}
                его книг тяготеют к нему самому, и <strong style={{ color: "var(--text)" }}>ни одна не путается со
                Львом Толстым</strong> — полным тёзкой-однофамильцем.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Метод ловит <em>руку</em>, а не тему. Без этого вердикты по «Тихому Дону» и «12 стульям»
                ничего бы не стоили.
              </p>
            </div>
          </div>
        </div>

        {/* РАСПЛАТА: шкала цельности — одиночки низко, коллектив высоко */}
        <div className="reveal module">
          <h3>Та же мера ловит коллектив</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 16 }}>
            Внутренний разброс работает и как <strong style={{ color: "var(--text)" }}>детектор нескольких рук</strong>:
            у цельного одиночного автора он низкий (как у бесспорных одиночек — {fmtRange(SINGLE_LOW, SINGLE_HIGH)}),
            а если текст или корпус склеены из разных рук — высокий. Вот все проверенные случаи на одной шкале:
          </p>
          {(() => {
            const rows = [
              ...C.controls.map((c) => ({ a: c.a + " (1 автор)", v: c.v, kind: "single" })),
              { a: "А. Н. Толстой (4 жанра)", v: TA.sil[0].v, kind: "single" },
              { a: "«Чевенгур» Платонова", v: C.platonov.chevengurSil, kind: "work" },
              { a: "«Чапаев» Фурманова", v: C.chapaev.internalSil, kind: "work" },
              { a: "«Железный поток» Серафимовича", v: C.serafimovich.internalSil, kind: "work" },
              { a: "Козьма Прутков (коллектив 4-х)", v: C.prutkov.sil, kind: "collective" },
            ].sort((x, y) => x.v - y.v);
            const max = Math.max(...rows.map((r) => r.v));
            return (
              <div className="split" style={{ alignItems: "center" }}>
                <div>
                  {rows.map((r) => (
                    <div key={r.a} style={{ display: "grid", gridTemplateColumns: "20ch 1fr 5ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                      <span style={{ fontSize: 12, color: r.kind === "collective" ? "var(--cinnabar)" : r.kind === "work" ? "var(--icon-blue)" : "var(--text-muted)", fontWeight: r.kind === "single" ? 400 : 700 }}>{r.a}</span>
                      <MeterBar value={r.v} max={max} accent={r.kind === "collective" ? "var(--cinnabar)" : r.kind === "work" ? "var(--icon-blue)" : "var(--text-muted)"} />
                      <span className="mono" style={{ fontSize: 10.5, color: r.kind === "collective" ? "var(--cinnabar)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                    </div>
                  ))}
                </div>
                <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
                  <p className="verdict" style={{ margin: 0 }}>
                    <strong style={{ color: "var(--cinnabar)" }}>Козьма Прутков</strong> ({fmtScore(C.prutkov.sil)}) —
                    на верхнем конце шкалы: коллективная маска (А. К. Толстой и три брата Жемчужниковы) в разы
                    разнороднее любого одиночки. Метод «видит» коллектив.{" "}
                    <span className="muted">Оговорка: четыре текста — разных форм (афоризмы, пьеса, сатира)
                    и без разбивки по соавторам, так что часть раскола — от формы; вывод предварительный.</span>
                  </p>
                  <p className="note" style={{ margin: 0 }}>
                    <strong style={{ color: "var(--icon-blue)" }}>«Чапаев» Фурманова</strong> ({fmtScore(C.chapaev.internalSil, 3)}) —
                    наоборот, среди самых цельных. Документальная основа (материалы Фрунзе, редактура) не оставила
                    отдельного слоя: книга написана единой рукой и уверенно относится к самому Фурманову.
                    Версия «автор против документов» не подтверждается.
                  </p>
                </div>
              </div>
            );
          })()}

          {/* автор vs редактор — и граница по Шолохову, короткими абзацами */}
          <p className="verdict" style={{ marginBottom: 8 }}>
            <strong style={{ color: "var(--text)" }}>«Автор или редактор».</strong>{" "}
            Тот же вопрос, что неразрешим у «Тихого Дона», на других книгах <em>решается</em>.
          </p>
          <p className="note" style={{ margin: "0 0 8px", maxWidth: "80ch" }}>
            «Железный поток» Серафимовича долго подозревали в тяжёлой правке Горького. По стилю книга — рука
            самого Серафимовича, правка не глубже косметической (вероятность «это Серафимович»{" "}
            {fmtScore(C.serafimovich.pVsGorky, 3)}; к своему почерку {fmtScore(C.serafimovich.dSelf, 1)} ближе,
            чем к горьковскому {fmtScore(C.serafimovich.dGorky)}).
          </p>
          <p className="note" style={{ margin: "0 0 8px", maxWidth: "80ch" }}>
            «Чевенгур» Платонова ходил в разных редакциях — и остаётся предельно однородным
            ({fmtScore(C.platonov.chevengurSil, 3)}). Сильный личный почерк редактура <em>не стирает</em>.
          </p>
          <p className="note" style={{ margin: 0, maxWidth: "80ch" }}>
            Разница с Шолоховым одна. У Серафимовича и Платонова есть <strong style={{ color: "var(--text)" }}>бесспорный
            собственный профиль</strong>, с которым можно сверить спорную книгу. У Шолохова такого якоря нет:
            вся его бесспорная донская проза — из того же периода, что обсуждается в споре. Поэтому «автор или
            редактор» там не закрывается — не из-за метода, а из-за отсутствия независимой точки отсчёта.
          </p>
        </div>

        {/* точность инструмента — бенчмарк */}
        <div className="reveal module">
          <h3>Насколько точен инструмент</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 16 }}>
            Открытый срез — <strong style={{ color: "var(--text)" }}>{BENCH.nBooks} {ruBooks(BENCH.nBooks)}</strong> по{" "}
            <strong style={{ color: "var(--text)" }}>{BENCH.nAuthors} авторам-классикам</strong>, умершим больше 70 лет назад:
            их тексты может докачать и перепроверить кто угодно. Проверяем по целым книгам простой линейной моделью
            (без нейросетей). Вклад каждой группы признаков по отдельности:
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              {BENCH.channels.map((r) => (
                <div key={r.c} style={{ display: "grid", gridTemplateColumns: "21ch 1fr 5ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                  <span style={{ fontSize: 11.5, color: r.hi ? "var(--text)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400 }}>{chRu(r.c)}</span>
                  <MeterBar value={r.v} max={CH_MAX} accent={r.hi ? "var(--icon-blue)" : "var(--gold)"} />
                  <span className="mono" style={{ fontSize: 10.5, color: r.hi ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gap: 10, alignContent: "start" }}>
              <div className="grid cols-2">
                <Stat label="точность по авторам (главная метрика)" value={fmtScore(BENCH.topMacroF1)} accent="var(--icon-blue)" />
                <Stat label="верных попаданий" value={fmtScore(BENCH.topTop1)} accent="var(--text)" />
              </div>
              <p className="note" style={{ margin: 0 }}>
                Ансамбль «все группы поровну» — это простое усреднение групп признаков. Его веса{" "}
                <strong style={{ color: "var(--text)" }}>не зависят от проверяемого текста</strong>, поэтому текст не помогает угадать сам себя.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Под <em>одним</em> классификатором{" "}
                <strong style={{ color: "var(--text)" }}>цепочки букв ({fmtScore(CH_CHAR.v, 3)}) и синтаксис ({fmtScore(CH_SYN.v, 3)}) идут вровень</strong>.
                Независимость от темы здесь — <em>размен</em>, а не выигрыш в точности.
              </p>
            </div>
          </div>
          <p className="muted" style={{ fontSize: 12.5, marginTop: 12, maxWidth: "80ch" }}>
            Это открытый срез из {BENCH.nAuthors} {ruAuthors(BENCH.nAuthors)}; заголовочные{" "}
            {fmtScore(LOBO_STRICT.styloFullLobo, 3)} считаются на другом, большем срезе. Числа с разных
            срезов между собой не сравнивают.
          </p>
          <details style={{ marginTop: 6 }}>
            <summary style={SUMMARY_STYLE}>Три среза корпуса</summary>
            <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 8px", maxWidth: "80ch" }}>
              На <strong style={{ color: "var(--text)" }}>{SLICE_OPEN}</strong> верных попаданий больше ({fmtScore(BENCH.topTop1)}) и точность по авторам выше ({fmtScore(BENCH.topMacroF1)}): меньше авторов — их труднее спутать.
              Заголовочные числа считаются на <strong style={{ color: "var(--text)" }}>{SLICE_BOOK}</strong>: {fmtScore(LOBO_STRICT.styloFullLobo, 3)} верных попаданий, точность по авторам в диапазоне [{MF1_CI}].
              И то, и другое зависит от состава среза, поэтому числа с разных срезов между собой не сравнивают.
            </p>
            <p className="muted" style={{ fontSize: 12.5, margin: "0 0 8px", maxWidth: "80ch" }}>
              Слабее всех узнаётся {WORST.name} — коллективная маска: своим именем помечена лишь половина его книг ({WORST_OK} из {WORST.books}).
              Маску из нескольких почерков трудно свести к одному профилю.
              Доверительный интервал точности по авторам на открытом срезе — {fmtRange(BENCH.ci[0], BENCH.ci[1])}: верхняя точка оптимистична, честная нижняя граница — {fmtScore(BENCH.ci[0])}.
            </p>
            <p className="muted" style={{ fontSize: 12.5, margin: "0 0 8px", maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>{SLICE_OPEN}</strong> — классики, умершие больше 70 лет назад ({BENCH.nBooks} {ruBooks(BENCH.nBooks)}):
              тексты в репозитории не хранятся, их докачивает скрипт по адресам из манифеста, поэтому результат может перепроверить кто угодно.
              У двоих (Гумилёв, Пильняк) срок охраны в России продлён после реабилитации — это помечено в манифесте, их тексты тоже не распространяются.
            </p>
            <p className="muted" style={{ fontSize: 12.5, margin: "0 0 8px", maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>{SLICE_BOOK}</strong> — полная проверка по книгам; в него входят и авторы под защитой авторских прав, поэтому он доступен только локально.
              Именно в этом срезе ({CORPUS.benchmark.authors} {ruAuthors(CORPUS.benchmark.authors)}) есть и полные нули — Зощенко, Олеша, Катаев: их метод ни разу не отметил верно.
            </p>
            <p className="muted" style={{ fontSize: 12.5, margin: 0, maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>{SLICE_ALL}</strong> — всё исследование целиком. Для сравнения с <em>чужими</em> работами берут стандартные наборы данных.
            </p>
          </details>
        </div>

        {/* каталог признаков — справочник */}
        <div className="reveal module">
          <h3>Из чего собран профиль автора</h3>
          <p className="prose muted" style={{ marginBottom: 22, maxWidth: "78ch" }}>
            Часть признаков ближе к поверхности текста: цепочки букв, частые слова, повторяющиеся
            обороты. Другие описывают структуру: служебные слова, синтаксические связи, пунктуацию.
            Каждый блок проверяется отдельно — правдоподобная идея признака не считается результатом,
            пока не показала вклад в общей оценке.
          </p>
          <div className="grid cols-3">
            {FEATURES.map((f) => {
              const k = KIND_STYLE[f.kind] || {};
              return (
                <Card key={f.id} padding={18} style={{ opacity: k.dim ? 0.66 : 1 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ fontFamily: "var(--font-display)", fontSize: "1.05rem", color: "var(--text)" }}>{featName(f.name)}</span>
                  </div>
                  <p className="muted mono" style={{ margin: 0, fontSize: 12.5 }}>{featNote(f.note)}</p>
                </Card>
              );
            })}
          </div>
          <details style={{ marginTop: 14 }}>
            <summary style={SUMMARY_STYLE}>Перевод сокращений</summary>
            <p className="muted" style={{ fontSize: 12.5, marginTop: 10, maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>n-граммы</strong> — цепочки из нескольких
              подряд идущих букв или слов; <strong style={{ color: "var(--text)" }}>MFW-300</strong> — 300 самых частых
              слов; <strong style={{ color: "var(--text)" }}>POS</strong> — часть речи; <strong style={{ color: "var(--text)" }}>TTR</strong> —
              доля неповторяющихся слов; <strong style={{ color: "var(--text)" }}>Hapax</strong> — слова, встреченные ровно
              один раз; <strong style={{ color: "var(--text)" }}>Yule</strong> — мера богатства словаря;{" "}
              <strong style={{ color: "var(--text)" }}>topic-bleaching</strong> — стирание темы (оставляем скелет из частей
              речи, чтобы измерять почерк, а не о чём текст); <strong style={{ color: "var(--text)" }}>синтаксические связи</strong> —
              кто с кем в предложении связан и как глубоко ветвится дерево разбора; <strong style={{ color: "var(--text)" }}>морфология</strong> —
              грамматические пометы слов (падеж, время, вид), взятые разбором spaCy (программой грамматического разбора).
            </p>
          </details>
        </div>

        {/* технические сверки — вынесены под кейсы, свёрнуты */}
        <div className="reveal module">
          <h3>Технические сверки</h3>
          <p className="prose muted" style={{ fontSize: 13, borderLeft: "2px solid var(--gold)", paddingLeft: 14, maxWidth: "74ch", marginBottom: 14 }}>
            Три внешние проверки: англоязычный набор CCAT50, русский Proza.ru и сверка протокола с группой из
            ТУСУР, плюс числа проверки по целым книгам.
          </p>

          <details style={{ marginBottom: 10 }}>
            <summary style={SUMMARY_STYLE}>Чужие наборы данных (CCAT50, Proza.ru) и сравнение с нейросетями</summary>
            <div className="split" style={{ alignItems: "start", marginTop: 12 }}>
              <div>
                <p className="prose muted" style={{ fontSize: 13, marginBottom: 10 }}>
                  <strong style={{ color: "var(--text)" }}>Стандартный CCAT50</strong> — общепринятый англоязычный
                  набор (Reuters, 50 авторов). Наш ансамбль без подсматривания <strong style={{ color: "var(--text)" }}>{fmtScore(BENCH_EXT.ccat50Ensemble, 3)}</strong>{" "}
                  держится вровень с эталоном Valla на буквенных n-граммах ({fmtScore(BENCH_EXT.ccat50Valla.ngramA, 3)})
                  и выше нейросетевого варианта BERT ({fmtScore(BENCH_EXT.ccat50Valla.bertA, 3)}) — это равенство при одном фиксированном делении, не уверенное
                  превосходство. Лучший известный результат на CCAT50 ({fmtScore(BENCH_EXT.ccat50Valla.record, 3)} — нейросеть по синтаксису) получен другим способом деления данных. Он несопоставим, и как достижение мы его не заявляем.
                </p>
                <div className="mono muted" style={{ fontSize: 11, marginBottom: 6 }}>
                  Внешний РУССКИЙ набор данных (Proza.ru, 50 авторов), одно деление, верных попаданий:
                </div>
                {BENCH_EXT.prozaCompare.map((r) => (
                  <div key={r.m} style={{ display: "grid", gridTemplateColumns: "20ch 1fr 5ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                    <span style={{ fontSize: 11.5, color: r.hi ? "var(--text)" : r.neuro ? "var(--cinnabar)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400, fontStyle: r.old ? "italic" : "normal" }}>{prozaRu(r.m)}</span>
                    <MeterBar value={r.v} max={PROZA_MAX} accent={r.hi ? "var(--icon-blue)" : r.neuro ? "var(--cinnabar)" : r.old ? "var(--border-strong)" : "var(--gold)"} />
                    <span className="mono" style={{ fontSize: 10.5, color: r.hi ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                  </div>
                ))}
              </div>
              <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
                <p className="verdict" style={{ margin: 0 }}>
                  <strong style={{ color: "var(--text)" }}>На этих срезах классические признаки сильнее готовых нейросетевых векторов.</strong> Один
                  классификатор по цепочкам букв (<strong style={{ color: "var(--text)" }}>{fmtScore(BENCH_EXT.prozaLeader, 3)}</strong>) обходит готовую{" "}
                  <strong style={{ color: "var(--cinnabar)" }}>облегчённую нейросеть ({fmtScore(BENCH_EXT.prozaNeuro, 3)})</strong>: такие модели
                  улавливают тему, а не стиль. <span className="muted">Оговорка: это самый слабый нейросетевой опорный метод; дообученные и
                  профильные модели для атрибуции авторства здесь не сравнивались, поэтому «классика бьёт нейросети вообще» мы не утверждаем.</span>
                </p>
                <p className="note" style={{ margin: 0 }}>
                  <strong style={{ color: "var(--text)" }}>Слить группы наугад — вредит.</strong> Простое
                  равновесное усреднение ({fmtScore(BENCH_EXT.prozaEqualEnsemble, 3)}) проседает <em>ниже</em> лучшей одиночной группы: слабые
                  группы тянут вниз сильную. Лидер на этом срезе — один классификатор по цепочкам букв ({fmtScore(BENCH_EXT.prozaLeader, 3)}).
                </p>
                <details style={{ margin: 0 }}>
                  <summary style={SUMMARY_STYLE}>Почему ансамбль {fmtScore(BENCH_EXT.prozaEnsemble, 3)} здесь не считаем за победу</summary>
                  <p className="note" style={{ margin: "8px 0 0" }}>
                    Взвешивание по надёжности (веса групп пропорциональны их точности на{" "}
                    <em>отложенной части обучения</em> — так, что проверяемый текст в этом не участвует) поднимает ансамбль до{" "}
                    <strong style={{ color: "var(--text)" }}>{fmtScore(BENCH_EXT.prozaEnsemble, 3)}</strong>. Но его настройка выбрана
                    по лучшему результату из небольшого перебора на этом же тесте, поэтому перевес +{fmtScore(BENCH_EXT.prozaEnsemble - BENCH_EXT.prozaLeader, 3)} над лидером ({fmtScore(BENCH_EXT.prozaLeader, 3)})
                    настроен под тест и лежит в пределах шума.
                  </p>
                </details>
              </div>
            </div>
          </details>

          <details style={{ marginBottom: 10 }}>
            <summary style={SUMMARY_STYLE}>Числа проверки по целым книгам и сверка с протоколом группы из ТУСУР</summary>
            <div className="split" style={{ alignItems: "start", marginTop: 12 }}>
              <div>
                <div className="grid cols-3">
                  <Stat label="проверка по целым книгам · главная цифра" value={fmtScore(LOBO_STRICT.styloFullLobo, 3)} accent="var(--gold)" />
                  <Stat label="деление на 5 частей · ориентир" value={fmtScore(LOBO_STRICT.proxyTop1, 3)} accent="var(--text)" />
                  <Stat label={`нижняя граница · голый посимвольный признак (${LOBO_STRICT.trueLoboBooks} ${ruBooks(LOBO_STRICT.trueLoboBooks)})`} value={fmtScore(LOBO_STRICT.trueLoboTop1, 3)} accent="var(--text-muted)" />
                </div>
                <p className="muted" style={{ fontSize: 12.5, marginTop: 10, maxWidth: "54ch" }}>
                  Заголовочная цифра — {fmtScore(LOBO_STRICT.styloFullLobo, 3)} ({HEADLINE.authors} {ruAuthors(HEADLINE.authors)} / {HEADLINE.books} {ruBooks(HEADLINE.books)}).
                  Деление на 5 частей ({fmtScore(LOBO_STRICT.proxyTop1, 3)}) сходится с ней в пределах шума. Голый посимвольный
                  косинус без обучаемого словаря даёт строгую нижнюю границу на одном признаке — {fmtScore(LOBO_STRICT.trueLoboTop1, 3)}.
                </p>
                {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
                  <p className="mono muted" style={{ fontSize: 12, marginTop: 8, maxWidth: "54ch" }}>
                    Оговорка: при обучении длинная книга сейчас весит больше короткой. Пересчёт «одна книга —
                    один голос» ещё впереди — заголовочная цифра может немного сдвинуться.
                  </p>
                )}
              </div>
              <div>
                <div className="mono muted" style={{ fontSize: 11, margin: "0 0 10px" }}>
                  Протокол ТУСУР · {TOMSK_50.k} {ruAuthors(TOMSK_50.k)} · их заявленное против пересчёта по книгам:
                </div>
                <div className="note" style={{ fontSize: 13 }}>
                  <p style={{ margin: 0 }}>
                    Опубликовано {fmtPct(TOMSK.theirAcc, 1)} на {TOMSK_50.k} авторах. В их открытом
                    демо-коде отрывки одной книги попадают и в обучение, и в проверку — деления по
                    книге нет, поэтому текст помогает угадать сам себя. На тех же данных и признаках,
                    но с делением по книгам, точность на {TOMSK_50.k} авторах — около {fmtPct(TOMSK_50.grouped)}{" "}
                    (против {fmtPct(TOMSK_50.rand)} без деления). Такой разрыв между двумя способами
                    деления держится на всех масштабах — от {TOMSK_KMIN} до {TOMSK_KMAX} авторов. Это
                    не пересчёт их закрытого полного корпуса, а указание на то, где в открытом коде
                    текст подсматривает сам себя.
                  </p>
                  <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "5ch 1fr 1fr 4.5ch", gap: "4px 10px", fontSize: 12, alignItems: "center" }}>
                    <span className="mono muted">авт.</span>
                    <span className="mono muted">их протокол</span>
                    <span className="mono muted">по книге</span>
                    <span className="mono muted" style={{ textAlign: "right" }}>утечка</span>
                    {TOMSK.headToHead.table.map((r) => (
                      <Fragment key={r.k}>
                        <span className="mono" style={{ color: "var(--text)" }}>{r.k}</span>
                        <span className="mono muted">{fmtScore(r.rand, 3)}</span>
                        <span className="mono" style={{ color: "var(--text)" }}>{fmtScore(r.grouped, 3)}</span>
                        <span className="mono" style={{ color: "var(--gold)", textAlign: "right" }}>+{r.prem}</span>
                      </Fragment>
                    ))}
                  </div>
                </div>
                <Sources
                  label="Источник"
                  items={[
                    { cite: `${TOMSK.ref.cite} · ${TOMSK.ref.group}`, url: TOMSK.ref.url },
                    { cite: TOMSK.ref.baseCite, url: TOMSK.ref.baseUrl },
                    { cite: `Код + демо-корпус — ${TOMSK.data.repo}`, url: TOMSK.data.repoUrl },
                    { cite: TOMSK.headToHead.prCite, url: TOMSK.headToHead.prUrl },
                  ]}
                  note={TOMSK.data.note}
                />
              </div>
            </div>
          </details>

          <details>
            <summary style={SUMMARY_STYLE}>Как перепроверить у себя</summary>
            <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 8px", maxWidth: "72ch" }}>
              Каждый прогон запускается одной командой и пишет результат в отдельный файл — числа берутся именно из этих выходов.
              Полный путь от корпуса до вердикта разобран в разделе «Можно повторить у себя».
            </p>
            <div style={{ display: "grid", gap: 8, maxWidth: "72ch" }}>
              {[
                { what: "Публикуемый бенчмарк классиков", cmd: "scripts/run_benchmark.py --pd-only", out: "docs/validation_pd.json" },
                { what: "Русский набор Proza.ru", cmd: "scripts/run_proza_ru.py", out: null },
                { what: "Проверка по целым книгам", cmd: "final.py / lobo.py", out: "final_comparison.csv" },
              ].map((r) => (
                <div key={r.cmd} style={{ display: "grid", gridTemplateColumns: "20ch 1fr", gap: 10, alignItems: "baseline", borderBottom: "1px solid color-mix(in srgb, var(--line) 40%, transparent)", paddingBottom: 7 }}>
                  <span style={{ fontSize: 12.5, color: "var(--text)" }}>{r.what}</span>
                  <span className="mono muted" style={{ fontSize: 11 }}>
                    {r.cmd}{r.out ? <> → {r.out}</> : null}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </div>
      </div>
    </section>
  );
}
