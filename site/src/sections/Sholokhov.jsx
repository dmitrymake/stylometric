import ArticleContents from "../components/ArticleContents.jsx";
import { EditorialPanel, EditorialMetric, EditorialFigure, AnomalyGlyph, EditorialBar } from "@dmitrymake/rk-ui";
import { SHOLOKHOV, RIGOR, CONSISTENCY, MULTIHANDS } from "../segdata.js";
import { DISPUTED } from "../data.js";
import { TD_CANDIDATES } from "../candidates.js";
import { fmtScore, fmtPct, fmtP, fmtZ } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";
import Sources from "../components/Sources.jsx";

const THEM = SHOLOKHOV.thematic;
const MS = SHOLOKHOV.manuscript;
const PC = DISPUTED.podnyataya;

// часть ярлыков в данных — сырые имена папок; приводим к человекочитаемым.
const DISPLAY = {
  serafimovich: "А. Серафимович", sevsky: "В. Севский", kumov: "Р. Кумов",
  krukov: "Ф. Крюков", kuprin: "А. Куприн",
  "Михаил Шолохов": "М. Шолохов", "Фёдор Крюков": "Ф. Крюков",
  "Константин Каргин": "К. Каргин", "Михаил Булгаков": "М. Булгаков",
  "Исаак Бабель": "И. Бабель", "Борис Акунин": "Б. Акунин",
  "Андрей Платонов": "А. Платонов", "Антон Чехов": "А. Чехов",
  "Иван Бунин": "И. Бунин", "Максим Горький": "М. Горький",
};
const nm = (s) => DISPLAY[s] || s;

// Русское склонение существительного при числе: [форма для 1, для 2–4, для многих].
const plural = (n, one, few, many) => {
  const d = Math.abs(n) % 100, d1 = d % 10;
  if (d > 10 && d < 20) return many;
  if (d1 === 1) return one;
  if (d1 >= 2 && d1 <= 4) return few;
  return many;
};


// Один и тот же предел у всех проверок на цельность — один короткий указатель на «Пределы»
// вердикта вместо повторения оговорки в каждом под-тесте.
function SimilarHandLimit() {
  return (
    <p className="muted" style={{ fontSize: 16, marginTop: 16, textAlign: "center" }}>
      Для похожих донских авторов и малой доли примеси чувствительность ограничена.
      Пороги конкретных проверок сведены в условиях интерпретации.
    </p>
  );
}

const COLOR_MAP = {
  "М. Шолохов": "var(--icon-blue)",
  "Ф. Крюков": "var(--cinnabar)",
  "А. Серафимович": "var(--gold)",
};
const accentOf = (k) => COLOR_MAP[nm(k)] || "var(--text-muted)";


// Короткий ответ теста: один крючок-вывод сразу под вопросом, чтобы читатель
// получал итог до разбора улик и не тонул в повторных развёрнутых вердиктах.
function TestSummary({ children }) {
  return <p className="test-summary">{children}</p>;
}

// Пояснения к некоторым карточкам в исходных данных содержат служебные пометки и жаргон.
// Для читателя без подготовки заменяем их обычным русским; смысл и направление вывода сохранены.
const WHY_CLEAN = {
  "Николай Гумилёв":
    "Поэт Серебряного века. В отдельном контрольном сравнении профиль «Тихого Дона» далёк от представленных произведений Гумилёва.",
  "Андрей Платонов":
    "В дополнительном сравнении «Поднятая целина» дальше от профиля Платонова и ближе к профилю Шолохова. Тем же методом проверена собственная проза Платонова. Это результат для данных эталонов.",
};

// Кандидаты с уцелевшей, но слишком тонкой прозой: формально в корпусе, но профиль по ней ненадёжен.
const THIN_CORPUS = new Set(["Виктор Севский (Краснушкин)", "Роман Кумов"]);

function CandidateCard({ c }) {
  const thin = THIN_CORPUS.has(c.name);
  const why = WHY_CLEAN[c.name] || c.why;
  return (
    <EditorialPanel>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
        <strong style={{ color: "var(--text)", fontSize: 17 }}>{c.name}</strong>
        <span className="mono muted" style={{ fontSize: 16 }}>† {c.death}</span>
      </div>
      <div style={{ display: "flex", gap: 7, margin: "9px 0 10px", flexWrap: "wrap" }}>
        <span className="chip" style={{ opacity: 0.85 }}>
          {c.inCorpus ? (thin ? "в корпусе · профиль слабый" : "в корпусе · проверяем") : "вне теста"}
        </span>
      </div>
      <p className="muted" style={{ fontSize: 16, lineHeight: 1.5, margin: 0 }}>{why}</p>
    </EditorialPanel>
  );
}

function ThematicRow({ rank, name, score, max }) {
  const hi = nm(name) === "Ф. Крюков" || nm(name) === "М. Шолохов";
  return (
    <div className="thematic-row">
      <span className="mono muted" style={{ fontSize: 16 }}>{rank}</span>
      <span style={{ fontSize: 16, color: hi ? "var(--text)" : "var(--text-muted)", fontWeight: hi ? 600 : 400 }}>
        {nm(name)}
      </span>
      <MeterBar value={score} max={max} accent={hi ? accentOf(name) : "var(--border-strong)"} />
      <span className="mono" style={{ fontSize: 16, color: hi ? "var(--text)" : "var(--text-muted)" }}>{fmtScore(score, 3)}</span>
    </div>
  );
}

export default function Sholokhov() {
  const inCorpus = TD_CANDIDATES.filter((c) => c.inCorpus);
  const offCorpus = TD_CANDIDATES.filter((c) => !c.inCorpus);
  const powMax = Math.max(...RIGOR.power.map((x) => x.frac)) * 1.08;
  // доля подмеса-эквивалента для военной прозы — из калибровки (не литерал).
  const warPct = MULTIHANDS.hiddenPositive.calib.find((c) => c.g === "война")?.pct;
  // Числа для текста — из данных, не литералами: близость ТД к раннему Шолохову,
  // диапазоны долей и «согласия» по книгам, верхняя опора шкалы разнокнижности.
  const tdSelfDist = RIGOR.tdCandDist.find((r) => r.self)?.d;
  const fullMin = Math.min(...RIGOR.attrib.map((r) => r.full));
  const fullMax = Math.max(...RIGOR.attrib.map((r) => r.full));
  const agreeMin = Math.min(...RIGOR.attrib.map((r) => r.agree));
  const agreeMax = Math.max(...RIGOR.attrib.map((r) => r.agree));
  const homTop = Math.max(RIGOR.homFloor, RIGOR.homSholohov, RIGOR.homCeil, ...RIGOR.homCtrls.map((c) => c.auc));
  // диапазон страниц на автора в рукописном тесте — по факту min/max всех строк (не по одной строке).
  const msPagesMin = Math.min(...MS.rows.map((r) => r.n));
  const msPagesMax = Math.max(...MS.rows.map((r) => r.n));

  return (
    <section className="section" id="sholokhov">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Донская проза</p>
          <h1>«Тихий Дон»: с какими текстами он сходен</h1>
          <p className="prose lead muted">
            «Тихий Дон» сравнивается с ранней и поздней прозой под именем Шолохова
            и произведениями других донских авторов. Проверяем, как выбор эталонов
            меняет сходство и какие различия между частями романа замечает метод.
          </p>
          <p className="prose muted">
            Авторские метки эталонов приняты как предпосылка. Общая донская тема
            может сближать тексты. Этот корпус не входит в проверку метода
            на 248 произведениях.
          </p>
        </div>
        <ArticleContents chapter="sholokhov" items={[
            ["sholokhov-section-1", "Кто, кроме Шолохова"],
            ["sholokhov-section-2", "Тест №1 · Сравнение при равном объёме текста"],
            ["sholokhov-section-3", "Тест №2 · Что показывает сходство словарей"],
            ["sholokhov-section-4", "Тест №3 · Поиск неоднородности между произведениями"],
            ["sholokhov-section-5", "Тест №4 · Насколько различаются книги одного автора"],
            ["sholokhov-section-6", "Тест №5 · Как уменьшить влияние тематической лексики"],
            ["sholokhov-section-7", "Рукопись · глубина авторской правки"],
            ["sholokhov-section-8", "Вердикт"]
          ]} />


        {/* 1. Поле кандидатов */}

        <div className="reveal module">
          <h2 id="sholokhov-section-1">Кто, кроме Шолохова</h2>
          <p className="prose muted" style={{ maxWidth: "64ch", marginBottom: 22 }}>
            Список фиксирует кандидатов, включённых в исследование. Для сравнения нужны
            собственные произведения каждого автора. Объём и состав сохранившейся прозы
            определяют, насколько содержателен его профиль.
          </p>
          <p className="eyebrow" style={{ marginBottom: 12 }}>Проверяемые · есть корпус</p>
          <div className="grid cols-2">
            {inCorpus.map((c) => <CandidateCard key={c.name} c={c} />)}
          </div>
          <p className="eyebrow" style={{ margin: "26px 0 12px" }}>Вне анализа стиля · нет прозы / не писатель</p>
          <div className="grid cols-2">
            {offCorpus.map((c) => <CandidateCard key={c.name} c={c} />)}
          </div>
          <p className="note">
            В использованной подборке у Кумова ~1,6&nbsp;тыс. слов, у Севского ~19&nbsp;тыс. —
            этого мало для надёжного профиля. Их кандидатура не отводится, но остаётся{" "}
            <strong style={{ color: "var(--text)" }}>с ограниченными данными для сравнения</strong>.
          </p>

          {/* Первый взгляд: ТД против ВСЕХ кандидатов — разминка перед пятью тестами */}
          <div className="reveal" style={{ marginTop: 34 }}>
            <p className="eyebrow" style={{ marginBottom: 6 }}>Первый взгляд</p>
            <h3 style={{ marginBottom: 6 }}>«Тихий Дон» против всех кандидатов сразу</h3>
            <p className="prose muted" style={{ maxWidth: "70ch", marginBottom: 16 }}>
              Сначала сравниваем усреднённые профили по синтаксису. Проверяемый текст
              отложен при построении профилей; меньшая величина означает большую близость.
              Сходство относится к конкретным опорным произведениям каждого кандидата:
            </p>
            <div className="split" style={{ alignItems: "center" }}>
              <div>
                {RIGOR.tdCandDist.map((r) => (
                  <div key={r.a} className="data-row" style={{ display: "grid", gridTemplateColumns: "16ch 1fr 4ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                    <span style={{ fontSize: 16, color: r.self ? "var(--text)" : "var(--text-muted)", fontWeight: r.self ? 700 : 400 }}>{r.a}</span>
                    <MeterBar value={r.d} max={Math.max(...RIGOR.tdCandDist.map((x) => x.d))} accent={r.self ? "var(--icon-blue)" : "var(--text-muted)"} />
                    <span className="mono" style={{ fontSize: 16, color: r.self ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.d)}</span>
                  </div>
                ))}
              </div>
              <p className="callout" style={{ margin: 0 }}>
                «Тихий Дон» ближе всего к <strong style={{ color: "var(--text)" }}>ранним рассказам под именем Шолохова</strong> ({fmtScore(tdSelfDist)}) —
                и так во всех 4 томах. Парное сравнение в одном жанре подтверждает: против <em>каждого</em> из{" "}
                {RIGOR.tdCandGm.length} кандидатов ТД уходит к Шолохову{" "}
                ({RIGOR.tdCandGm.map((c) => `${c.a} ${c.p}`).join(", ")} — все&nbsp;&gt;&nbsp;0.5).{" "}
                <strong style={{ color: "var(--text)" }}>Ни один</strong> кандидат этой подборки не даёт более близкого профиля.
              </p>
            </div>
            <p className="muted" style={{ fontSize: 16, marginTop: 12 }}>
              Сравнение относится к перечисленным профилям. Кандидата без сопоставимой
              прозы ({RIGOR.tdCandUntestable}) такой набор не представляет.
            </p>
          </div>
        </div>

        {/* 3. Атрибуция Тихого Дона — две модели */}
        <div className="reveal module">
          <h2 id="sholokhov-section-2">Тест №1 · Сравнение при равном объёме текста</h2>
          <p className="prose muted" style={{ maxWidth: "68ch", marginBottom: 22 }}>
            Усреднённый профиль Шолохова строим <strong style={{ color: "var(--text)" }}>без единой страницы «Тихого
            Дона»</strong> (ранние рассказы и поздняя проза с корпусной меткой «Шолохов») и{" "}
            <strong style={{ color: "var(--text)" }}>уравниваем объём текста у всех авторов</strong> (чтобы
            обилие текстов Шолохова не давало перекоса). Каждую книгу прогоняем двумя
            моделями: <strong style={{ color: "var(--text)" }}>полной</strong> (со словами) и{" "}
            <strong style={{ color: "var(--text)" }}>с урезанной долей слов</strong>. Доли — к Шолохову.
          </p>
          <TestSummary>
            При равном объёме текста «Тихий Дон» уходит к Шолохову (медиана {fmtScore(RIGOR.dsTdFullMed, 3)}). Но
            стоит выровнять ещё и жанр — счёт по словам почти ничейный, так что сам по себе он ничего не решает.
          </TestSummary>
          <EditorialFigure label="Таблица 1" caption="Доли назначений к профилю Шолохова при равном объёме эталонного текста. Полная модель и вариант с меньшим вкладом слов сопоставлены для каждой книги. Согласие — отдельная мера между LR и Delta.">
            <p className="table-scroll-hint">Таблица прокручивается по горизонтали →</p>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Сравнение книг романа — таблица прокручивается по горизонтали">
              <table><thead><tr><th scope="col">Произведение</th><th scope="col">Полная модель</th><th scope="col">Меньше слов</th><th scope="col">Согласие</th></tr></thead><tbody>
                {RIGOR.attrib.map((r) => <tr key={r.book}><th scope="row">{r.book}</th><td>{fmtPct(r.full, 0)}<MeterBar value={r.full} max={1} /></td><td>{fmtPct(r.topic, 0)}<MeterBar value={r.topic} max={1} /></td><td>{r.agree}</td></tr>)}
              </tbody></table>
            </div>
          </EditorialFigure>

          <p className="verdict">
            Когда объём текста у всех авторов выровнен, полная модель отдаёт «Тихий Дон» Шолохову ({fmtScore(fullMin, 2)}–{fmtScore(fullMax, 3)}) —
            устойчиво, в том числе когда все авторы урезаны до {RIGOR.dsNmin} отрывков (= объём Крюкова): медиана полной модели{" "}
            {RIGOR.dsTdFullMed} [{RIGOR.dsTdFullLo}–{RIGOR.dsTdFullHi}]. Это показывает результат при выровненном
            объёме. Вклад темы и жанра проверяется отдельно.
          </p>
          <p className="note">
            <strong style={{ color: "var(--editorial-accent)" }}>Самый строгий тест ослабляет вывод:</strong>{" "}
            если уравнять не только объём, но и <em>жанр</em> — собрать профиль Шолохова <strong style={{ color: "var(--text)" }}>только
            из ранних донских рассказов</strong> ({RIGOR.earlyPoolN}, тот же тип текста, что ТД) и взять у Крюкова столько же —
            полная модель на «Тихом Доне» даёт почти <strong style={{ color: "var(--editorial-accent)" }}>ничью</strong>:
            Шолохов {RIGOR.gmlrTdShFull} vs Крюков {RIGOR.gmlrTdKrFull}. Ответ меняется вместе с составом
            опорных произведений. По этому сравнению нельзя отдельно установить причину изменения.
            С поправкой на тему ТД всё ещё к Шолохову ({RIGOR.gmlrTdShTopic}) — но и тематические признаки несут жанр.
            Словарный сигнал «ТД = Шолохов, не Крюков» при выровненном жанре — <strong style={{ color: "var(--text)" }}>неубедителен</strong>.
          </p>
          <details style={{ margin: "6px 0" }}>
            <summary style={{ cursor: "pointer", color: "var(--icon-blue)", fontSize: 16, fontWeight: 600 }}>
              Почему столбец «согласие» — не мера надёжности
            </summary>
            <p className="muted" style={{ marginTop: 8, marginBottom: 0, fontSize: 16 }}>
              «Согласие» — насколько по каждому отрывку сходятся две разные модели: одна взвешивает признаки, другая
              мерит близость ({fmtScore(agreeMin, 2)}–{fmtScore(agreeMax, 2)}). При 5 кандидатах случайное совпадение уже ≈{fmtScore(1 / 5)},
              а вторая модель по отдельным отрывкам ведёт себя как шум. Поэтому вывод по книгам опирается на общий
              ответ первой модели и усреднённый профиль автора, а не на этот флаг.
            </p>
          </details>
        </div>

        {/* Калибровка: «Поднятая целина» — заведомо Шолохов (негативный контроль) */}
        <div className="reveal module">
          <h3 style={{ marginBottom: 6 }}>«Поднятая целина»: дополнительное сравнение</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 14 }}>
            Через тот же набор моделей проходит «Поднятая целина»: {PC.fragments}{" "}
            {plural(PC.fragments, "фрагмент", "фрагмента", "фрагментов")}. Это ещё одно наблюдение
            о том, как меняется ответ при выборе эталонов и признаков.
          </p>
          <div style={{ display: "grid", gap: 12, maxWidth: "54ch" }}>
            {PC.candidates.map((c, i) => (
              <EditorialBar
                key={c.name}
                value={c.full}
                valueText={fmtScore(c.full, 3)}
                label={<span style={{ color: i === 0 ? "var(--gold)" : "var(--text-muted)", fontWeight: i === 0 ? 700 : 400 }}>{c.name}</span>}
                accent={i === 0 ? "var(--gold)" : "var(--text-muted)"}
              />
            ))}
          </div>
          <p className="callout">
            Набор моделей со словами ожидаемо относит текст к Шолохову ({fmtPct(PC.candidates[0].full, 0)}, отрыв&nbsp;+{fmtScore(PC.margin)}).
            Но это и мера осторожности: при строгом уравнивании жанра отдельный тест ошибочно относит «Поднятую целину»
            к Крюкову ({RIGOR.gmlrPcKrFull}) — на донском материале счёт по словам ненадёжен, и тот же предел касается «Тихого Дона».
          </p>
        </div>

        {/* 4. Почему все указывают на Крюкова */}
        <div className="reveal module">
          <h2 id="sholokhov-section-3">Тест №2 · Что показывает сходство словарей</h2>
          <p className="prose muted" style={{ maxWidth: "66ch", marginBottom: 20 }}>
            Близость по словам показывает совпадение словарей. В этом сравнении
            ближайший к «Тихому Дону» — <strong style={{ color: "var(--text)" }}>сам Шолохов</strong> ({fmtScore(THEM.tihiyDon[0][1])}),
            а Крюков — близкий второй ({fmtScore(THEM.tihiyDon[1][1])}). Общий донской материал
            может сближать словари: у «Донских рассказов» с корпусной меткой
            «Шолохов» ближайший после самого автора — снова Крюков ({fmtScore(THEM.donskie[1][1])}), но следом
            почти вплотную идут и недонские авторы ({fmtScore(THEM.donskie[2][1])}).
          </p>
          <TestSummary>
            Ранние рассказы с меткой «Шолохов» тоже близки к Крюкову по словарю.
            Такое сходство согласуется с общей темой, но само по себе не отделяет её от авторской манеры.
          </TestSummary>
          <div className="grid cols-2" style={{ gap: 22 }}>
            <EditorialPanel>
              <p className="eyebrow" style={{ marginBottom: 14 }}>«Тихий Дон» — ближайшие по словам</p>
              <div style={{ display: "grid", gap: 9 }}>
                {THEM.tihiyDon.map(([n, s], i) =>
                  <ThematicRow key={n} rank={i + 1} name={n} score={s} max={THEM.tihiyDon[0][1]} />)}
              </div>
            </EditorialPanel>
            <EditorialPanel>
              <p className="eyebrow" style={{ marginBottom: 14 }}>«Донские рассказы» Шолохова — ближайшие</p>
              <div style={{ display: "grid", gap: 9 }}>
                {THEM.donskie.map(([n, s], i) =>
                  <ThematicRow key={n} rank={i + 1} name={n} score={s} max={THEM.donskie[0][1]} />)}
              </div>
              <p className="muted" style={{ fontSize: 16, marginTop: 12 }}>
                Ранние рассказы с меткой «Шолохов» тоже близки к Крюкову по словарю.
              </p>
            </EditorialPanel>
          </div>

          <div className="split" style={{ marginTop: 28, alignItems: "center" }}>
            <div className="prose">
              <p className="callout gold" style={{ marginTop: 0 }}>
                Тематическая лексика — одно из возможных объяснений близости: у Крюкова
                и Шолохова общий донской материал. Поэтому результат по словам
                сопоставляется с результатами по построению фразы.
              </p>
              <p>
                Это видно по тому, что «Донские рассказы» с корпусной меткой «Шолохов» тоже стоят к
                Крюкову вплотную. При равном объёме эталонов (Тест&nbsp;№1)
                модель по словам относит и «Тихий Дон», и «Поднятую целину»{" "}
                <strong style={{ color: "var(--text)" }}>Шолохову</strong>, а не Крюкову.
                Смена ответа показывает, почему важно указывать состав и объём эталонов.
              </p>
            </div>
            <div style={{ display: "grid", placeItems: "center", gap: 12 }}>
              <AnomalyGlyph kind="relation_mismatch" size={46} />
              <span className="muted mono" style={{ fontSize: 16, textAlign: "center", maxWidth: "22ch" }}>
                сходство словарей<br />требует проверки по другим признакам
              </span>
            </div>
          </div>
        </div>

        {/* Тест №3: много рук — LEAK-FREE */}
        <div className="reveal module">
          <h2 id="sholokhov-section-4">Тест №3 · Поиск неоднородности между произведениями</h2>
          <p className="prose muted" style={{ maxWidth: "70ch", marginBottom: 8 }}>
            Проверяем различия между произведениями под одним авторским именем.
            Важно учесть и влияние большого объёма эталона: широкий профиль может
            приближать к себе разные тексты. Здесь пространство признаков строится
            по другим авторам, без Шолохова, а проверяемая книга откладывается.
          </p>
          <TestSummary>
            Разброс внутри подборки Шолохова попадает в диапазон контрольных одиночек
            (z&nbsp;{fmtZ(MULTIHANDS.avMultiHand.zPseudo)}, это ~5 обычных разбросов ниже настоящих смесей). Предел — ниже.
          </TestSummary>

          <div className="split" style={{ alignItems: "start", marginTop: 18 }}>
            <div>
              <p className="eyebrow" style={{ marginBottom: 6 }}>Каждая книга → к какому автору ближе (по одной отложенной книге за раз)</p>
              <p className="muted" style={{ fontSize: 16, margin: "0 0 12px" }}>
                Эталон «Шолохов» здесь <em>включает</em> его ранние донские рассказы — круг сравнения замкнут на самого
                автора. Тест в вердикте исключает проверяемые работы из обучения (там первый том «Тихого Дона» уже
                спорный), но зависимость от меток оставшихся опорных текстов сохраняется.
              </p>
              <div style={{ display: "grid", gap: 7 }}>
                {RIGOR.perBook.map((r) => {
                  const td = r.book.startsWith("Тихий Дон");
                  const td1 = r.book === "Тихий Дон кн.1";
                  return (
                    <div key={r.book} style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", alignItems: "center", gap: 10, padding: "3px 0" }}>
                      <span style={{ fontSize: 16, color: td ? "var(--text)" : "var(--text-muted)", fontWeight: td ? 600 : 400 }}>
                        {r.book}{td1 && <span className="muted" style={{ fontWeight: 400 }}> · спорный при строгом тесте</span>}
                      </span>
                      <span style={{ fontSize: 16, color: r.stays ? "var(--icon-blue)" : "var(--cinnabar)", fontWeight: r.stays ? 500 : 700 }}>
                        {r.stays ? "→ Шолохов" : `→ ${r.nearest}`}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="muted" style={{ fontSize: 16, marginTop: 12 }}>
                <strong style={{ color: "var(--text)" }}>{RIGOR.b2Stay}/{RIGOR.b2N}</strong> книг ближе к «Шолохову без
                этой книги», включая все 4 тома «Тихого Дона» и обе книги «Поднятой целины». Два иных ответа получены на коротких
                поздних рассказах (14 и 30 отрывков, профиль на них шумный): «Судьба человека»→Крюков (но Крюков †1920,
                рассказ 1957 — назначение требует отдельного исторического обоснования), «Наука ненависти»→Булгаков. Причину этих назначений
                сами значения сходства не устанавливают.
              </p>
            </div>
            <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
              <EditorialMetric label="книг ближе к Шолохову" value={`${RIGOR.b2Stay}/${RIGOR.b2N}`} accent="var(--success)" />
              <EditorialMetric label="разброс книг: место среди одиночек" value={`${RIGOR.dispRank} / ${RIGOR.dispPanelN}`} accent="var(--icon-blue)" hint={`${RIGOR.dispSholohov} против ${RIGOR.dispControl}±${RIGOR.dispControlStd} у одиночек — в нижней четверти по разбросу: метка не раздута, ведёт себя как обычный автор`} />
            </div>
          </div>

          {/* Решающий тест «много рук»: supervised pairwise authorship-verification (author-disjoint) */}
          <div className="module" style={{ marginTop: 26 }}>
            <p className="eyebrow" style={{ marginBottom: 4 }}>Попарное сравнение с одноавторскими текстами и смесями</p>
            <p className="muted" style={{ fontSize: 16, margin: "0 0 14px" }}>
              авторы в обучении и проверке не пересекаются, отрывки равного объёма; настроено на {MULTIHANDS.avMultiHand.nPos} псевдонимных смесях (разные авторы под одним именем) и {MULTIHANDS.avMultiHand.nNeg} одиночках
            </p>
            <p className="prose" style={{ margin: 0, fontSize: 17 }}>
              Для каждой пары книг считаем, насколько они «разные авторские профили». У настоящего коллектива (смесь под псевдонимом)
              оценка ≈ <strong style={{ color: "var(--text)" }}>{MULTIHANDS.avMultiHand.posMean}</strong>, у одиночек ≈{" "}
              <strong style={{ color: "var(--text)" }}>{MULTIHANDS.avMultiHand.negMean}</strong>. Корпус Шолохова даёт{" "}
              <strong style={{ color: "var(--editorial-positive)" }}>{MULTIHANDS.avMultiHand.score}</strong> — то есть он{" "}
              <strong style={{ color: "var(--editorial-positive)" }}>близок к контрольным одиночкам</strong> по этой оценке:
              z = <strong style={{ color: "var(--text)" }}>{MULTIHANDS.avMultiHand.zPseudo}</strong> (насколько велико отклонение
              против обычного разброса — здесь ~5 таких разбросов ниже смесей), различимость{" "}
              {MULTIHANDS.avMultiHand.auc} [{MULTIHANDS.avMultiHand.aucCi[0]}–{MULTIHANDS.avMultiHand.aucCi[1]}] (1.0 — идеально, 0.5 — наугад),
              проверка на случайность p {fmtP(MULTIHANDS.avMultiHand.permP)}. Вывод относится к смесям,
              представленным в этой проверке.
            </p>
            <div className="grid cols-3" style={{ marginTop: 14 }}>
              <EditorialMetric label="отрыв от псевдонимной смеси (z)" value={fmtZ(MULTIHANDS.avMultiHand.zPseudo)} accent="var(--success)" hint="ниже контрольных смесей по этой оценке" />
              <EditorialMetric label="различимость: смеси и одиночки" value={fmtScore(MULTIHANDS.avMultiHand.auc)} accent="var(--icon-blue)" hint={`p ${fmtP(MULTIHANDS.avMultiHand.permP)} (проверка на случайность)`} />
              <EditorialMetric label="оценка «много рук» у Шолохова" value={fmtScore(MULTIHANDS.avMultiHand.score)} accent="var(--success)" hint={`≈ одиночки ${MULTIHANDS.avMultiHand.negMean}, далеко от смеси ${MULTIHANDS.avMultiHand.posMean}`} />
            </div>
            <p className="muted" style={{ fontSize: 16, marginTop: 12, marginBottom: 0 }}>
              Этот тест различает смеси <em>разных</em> авторов. Смесь <em>похожих</em> донских авторов здесь не отделяется от одиночек,
              поэтому выявление похожего соавтора с малой долей текста здесь не подтверждено (пороги — в условиях интерпретации).
            </p>
          </div>

          <div className="grid cols-2" style={{ marginTop: 22, gap: 16 }}>
            <EditorialPanel>
              <p className="eyebrow" style={{ marginBottom: 4 }}>Что тест вообще способен заметить</p>
              <p className="muted" style={{ fontSize: 16, margin: "0 0 12px" }}>подмешиваем Крюкова → доля «крюковских» отрывков</p>
              <div style={{ display: "grid", gap: 6 }}>
                {RIGOR.power.map((x) => (
                  <div key={x.k} className="data-row" style={{ display: "grid", gridTemplateColumns: "4ch 1fr 4ch", alignItems: "center", gap: 8 }}>
                    <span className="mono muted" style={{ fontSize: 16 }}>{x.k}%</span>
                    <MeterBar value={x.frac} max={powMax} accent={x.k >= RIGOR.powerDetectK ? "var(--cinnabar)" : "var(--border-strong)"} />
                    <span className="mono" style={{ fontSize: 16 }}>{x.frac}</span>
                  </div>
                ))}
              </div>
              <p className="muted" style={{ fontSize: 16, marginTop: 10 }}>
                В этой серии тест обнаруживает примесь начиная с <strong style={{ color: "var(--editorial-accent)" }}>~{RIGOR.powerDetectK}%</strong>{" "}
                примеси похожего по стилю автора. Для меньших долей надёжное обнаружение в этой серии не показано.
              </p>
            </EditorialPanel>
            <p className="callout gold" style={{ margin: 0 }}>
              И у «поправки на тему» есть предел. Строгая проверка — военная проза одного писателя против сельской прозы
              другого — показывает: даже структурные признаки различают{" "}
              <strong style={{ color: "var(--editorial-accent)" }}>жанр</strong>{" "}
              (<strong style={{ color: "var(--text)" }}>{RIGOR.crossGenreAuc}</strong>, где 0.5 — наугад, 1.0 — безошибочно).
              Значит, они несут ещё и жанр с эпохой, а не только личный почерк — поэтому говорим «стилистически похоже»,
              а не «доказано авторство».
            </p>
          </div>

          <p className="note">
            <strong style={{ color: "var(--text)" }}>Предел ещё жёстче:</strong> две половины <em>одной</em> книги уже
            различаются с оценкой&nbsp;{RIGOR.sepFloor} (нижняя граница ≠ 0.5) — признаки так чувствительны к местному
            содержанию, что чисто проверить авторство, обучая модель их разделять, нельзя. При этом вывод «ТД ближе к
            Шолохову» устойчив: держится в <strong style={{ color: "var(--text)" }}>{RIGOR.embRobustConfigs}/{RIGOR.embRobustN}</strong>{" "}
            вариантах настройки пространства стиля (число осей, нормировка, способ мерить расстояние).
          </p>

          <p className="verdict">
            Итог: «много рук» <strong style={{ color: "var(--text)" }}>не подтверждается</strong> — корпус не разнороднее
            одиночных авторов (по разбросу место {RIGOR.dispRank}/{RIGOR.dispPanelN}), метка не раздута. Похожего соавтора
            с малой долей текста тесты не поймали бы — чувствительность указана в условиях интерпретации.
          </p>
        </div>

        {/* 5d. Гомогенность: разные люди писали разные работы? */}
        <div className="reveal module">
          <h2 id="sholokhov-section-5">Тест №4 · Насколько различаются книги одного автора</h2>
          <p className="prose muted" style={{ maxWidth: "70ch", marginBottom: 18 }}>
            Сравниваем попарную различимость книг с диапазоном у контрольных авторов.
            У одного писателя произведения тоже меняются вместе с темой, периодом и жанром.
            Для версии о нескольких авторах важно, отличается ли исследуемая подборка
            от такого обычного разброса.
          </p>
          <TestSummary>
            Книги Шолохова разнятся между собой как у одного «широкого» автора — Бунин и вовсе
            разнообразнее. Версия «разные люди» не подтверждается, но и не исключается для похожих донских рук.
          </TestSummary>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              <div className="mono muted" style={{ fontSize: 16, display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span>← похожи</span><span>отделимы →</span>
              </div>
              {[
                { a: "ориентир: две половины одной книги", v: RIGOR.homFloor, hi: false, anchor: true },
                ...RIGOR.homCtrls.map((c) => ({ a: c.a + " (1 автор)", v: c.auc, hi: false })),
                { a: "ШОЛОХОВ (все его книги)", v: RIGOR.homSholohov, hi: true },
                { a: "ориентир: разные авторы", v: RIGOR.homCeil, hi: false, anchor: true },
              ].sort((x, y) => x.v - y.v).map((r) => (
                <div key={r.a} className="data-row" style={{ display: "grid", gridTemplateColumns: "20ch 1fr 4ch", alignItems: "center", gap: 8, padding: "2px 0" }}>
                  <span style={{ fontSize: 16, color: r.hi ? "var(--text)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400, fontStyle: r.anchor ? "italic" : "normal" }}>{r.a}</span>
                  <MeterBar value={r.v - 0.5} max={homTop - 0.5} accent={r.hi ? "var(--icon-blue)" : r.anchor ? "var(--border-strong)" : "var(--gold)"} />
                  <span className="mono" style={{ fontSize: 16, color: r.hi ? "var(--text)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
              <p className="callout" style={{ margin: 0 }}>
                Книги Шолохова отделимы друг от друга (различимость&nbsp;{fmtScore(RIGOR.homSholohov, 3)}) — но{" "}
                <strong style={{ color: "var(--text)" }}>Бунин</strong>, бесспорно один автор, ещё{" "}
                <strong style={{ color: "var(--text)" }}>отделимее</strong> ({fmtScore(RIGOR.homCtrls[0].auc, 3)}). Высокая
                различимость книг встречается и у одного автора; признаки также чувствительны к жанру:
                даже две половины одной книги дают {fmtScore(RIGOR.homFloor, 3)}.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Внутри «Тихого Дона» 4 тома похожи друг на друга (различимость&nbsp;<strong style={{ color: "var(--icon-blue)" }}>{fmtScore(RIGOR.homTdInternal, 3)}</strong>){" "}
                <strong style={{ color: "var(--text)" }}>больше</strong>, чем на остальные его работы ({fmtScore(RIGOR.homSholohov, 3)}) —
                описывает внутреннее сходство томов романа. {RIGOR.homNStay}/{RIGOR.homNWorks} работ тяготеют к самому Шолохову.
              </p>
            </div>
          </div>
          <p className="verdict">
            Вывод по «разным людям»: <strong style={{ color: "var(--text)" }}>не подтверждается</strong> —
            разнокнижность Шолохова в диапазоне одиночных авторов, Бунин и вовсе разнообразнее.
          </p>

          <p className="note">
            <strong style={{ color: "var(--text)" }}>Проверено ещё одним способом</strong> — чёткостью деления книг на
            кластеры: Шолохов ({CONSISTENCY.sholokhovSil}) на высоком краю одиночек (место {CONSISTENCY.sholokhovRank}/{CONSISTENCY.nPanel},
            бесспорный Лесков ({CONSISTENCY.scale.find((x) => x.a === "Лесков").v}) разнороднее), но коллектив Прутков
            ({CONSISTENCY.prutkov}) — в <strong style={{ color: "var(--editorial-accent)" }}>×{CONSISTENCY.prutkovRatio}</strong> выше.
            Книги образуют две слабо разделённые группы, связанные с донским и советским
            <em>материалом</em> (совпадение разбиений&nbsp;{RIGOR.cxAriDonskoy}).
            Эта связь не устанавливает число авторов.
          </p>

          {/* Может ли тест поймать подделку вообще: контроли + скрытый позитив */}
          <div className="reveal" style={{ marginTop: 34 }}>
            <h3 style={{ marginBottom: 6 }}>Проверка на искусственных смесях</h3>
            <p className="prose muted" style={{ maxWidth: "76ch", marginBottom: 16 }}>
              Чувствительность метода проверяем на искусственных смесях известных авторов.
              Склейку из трёх <strong style={{ color: "var(--text)" }}>разных</strong> авторов метод{" "}
              <strong style={{ color: "var(--text)" }}>ловит</strong>: единому автору не приписался ни один кусок
              ({MULTIHANDS.fakeDifferentCaught} → «свой»), различимость {fmtScore(MULTIHANDS.fakeDifferent, 3)}. Но склейка{" "}
              <strong style={{ color: "var(--text)" }}>похожих</strong> донских авторов (Крюков+Серафимович+Севский, {MULTIHANDS.fakeSimilar})
              <strong style={{ color: "var(--editorial-accent)" }}>не отделяется от одиночек в этой проверке</strong> — она даже ниже Шолохова
              ({MULTIHANDS.sholokhovSep}). Это показывает ограничение поиска соавторов со сходной манерой.
            </p>
            <div className="split" style={{ alignItems: "start" }}>
              <div>
                <p className="eyebrow" style={{ marginBottom: 4 }}>Чувствительность к добавлению Крюкова</p>
                <p className="muted" style={{ fontSize: 16, margin: "0 0 10px" }}>
                  подмешиваем реального Крюкова к Шолохову → порог обнаружения ~{MULTIHANDS.hiddenPositive.flagThreshold}%; куда попадают реальные работы:
                </p>
                {MULTIHANDS.hiddenPositive.calib.map((c) => {
                  const over = c.pct >= MULTIHANDS.hiddenPositive.flagThreshold;
                  const near = c.pct >= 25;
                  return (
                    <div key={c.g} className="data-row" style={{ display: "grid", gridTemplateColumns: "16ch 1fr 6ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                      <span style={{ fontSize: 16, color: near ? "var(--gold)" : "var(--text-muted)", fontWeight: near ? 700 : 400 }}>{c.g}</span>
                      <span style={{ height: 9, borderRadius: 4, background: "var(--surface-sunken)", overflow: "hidden", position: "relative" }}>
                        <span style={{ display: "block", height: "100%", width: `${c.pct}%`, background: over ? "var(--cinnabar)" : near ? "var(--gold)" : "var(--text-muted)" }} />
                        <span style={{ position: "absolute", left: `${MULTIHANDS.hiddenPositive.flagThreshold}%`, top: -2, bottom: -2, width: 1, background: "var(--cinnabar)", opacity: 0.6 }} />
                      </span>
                      <span className="mono" style={{ fontSize: 16, color: over ? "var(--cinnabar)" : "var(--text-muted)" }}>~{c.pct}%</span>
                    </div>
                  );
                })}
                <div className="mono muted" style={{ fontSize: 16, marginTop: 6 }}>
                  ┊ красная черта — порог {MULTIHANDS.hiddenPositive.flagThreshold}%. «Война» (~{warPct}%) сидит ровно под ним.
                </div>
              </div>
              <p className="callout" style={{ margin: 0 }}>
                Ни одна работа не переходит порог, а «не-свои» фрагменты <strong style={{ color: "var(--text)" }}>рассыпаны</strong>,
                а не собраны в одну руку (война: Крюков ≈ Бунин ≈ Достоевский; «Тихий Дон» ведёт <em>Горький</em>, не донской).
                Карта не выделяет один устойчивый внешний профиль. Но «война» (≈{warPct}%) — у самого порога, поэтому <em>частичный</em> вклад
                стилистически <strong style={{ color: "var(--text)" }}>похожего</strong> донского соавтора в самые расходящиеся
                работы метод исключить не может. <span className="mono muted" style={{ fontSize: 16 }}>(доля дрожит ±{MULTIHANDS.hiddenPositive.runNoise})</span>
              </p>
            </div>
          </div>
          <SimilarHandLimit />
        </div>

        {/* 5e. Поиск чистого от темы признака → dependency */}
        <div className="reveal module">
          <h2 id="sholokhov-section-6">Тест №5 · Как уменьшить влияние тематической лексики</h2>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 16 }}>
            Сравниваем, насколько каждая группа признаков различает авторов и насколько
            чувствительна к жанру. Для DSP — профиля словообразовательных суффиксов —
            первоначальная проверка жанра была ограничена <em>одним автором</em>.
            В расширенном сравнении есть{" "}
            <strong style={{ color: "var(--text)" }}>проза из общественного достояния</strong>: военная ({RIGOR.faWarAuthors}{" "}
            авторов — Толстой, Гаршин, Фурманов…) и сельская ({RIGOR.faRuralAuthors} — Тургенев, Короленко,
            Бунин…). На этих данных проверяем, сохраняет ли признак различение авторов при меньшем
            различении военной и сельской прозы у <strong style={{ color: "var(--text)" }}>других</strong> авторов.
          </p>
          <TestSummary>
            На группе признаков с меньшим влиянием тематической лексики в этой проверке — синтаксических связях — «Тихий Дон» склоняется
            к Шолохову отчётливее, чем на любом другом. Но по целым книгам запас всё ещё дотягивается до ничьей.
          </TestSummary>
          <div className="split" style={{ alignItems: "start" }}>
            <div>
              <div className="mono muted" style={{ fontSize: 16, marginBottom: 8 }}>
                по горизонтали: ◼ различает АВТОРА (выше — лучше) · ◻ путает с жанром поперёк чужих авторов (ниже — лучше)
              </div>
              {RIGOR.fa2.map((r) => (
                <div key={r.feat} className="data-row" style={{ display: "grid", gridTemplateColumns: "15ch 1fr 4ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                  <span style={{ fontSize: 16, color: r.idi > 0.45 ? "var(--text)" : "var(--text-muted)", fontWeight: r.idi > 0.45 ? 700 : 400 }}>{r.feat}</span>
                  <span style={{ position: "relative", height: 13, background: "var(--surface-sunken)", borderRadius: 3 }}>
                    <span style={{ position: "absolute", left: 0, top: 1, height: 5, width: `${r.author * 100}%`, background: r.idi > 0.45 ? "var(--icon-blue)" : "var(--text-muted)", borderRadius: 2 }} title={`автор ${r.author}`} />
                    <span style={{ position: "absolute", left: 0, bottom: 1, height: 5, width: `${r.genreXA * 100}%`, background: "var(--cinnabar)", opacity: 0.55, borderRadius: 2 }} title={`жанр ${r.genreXA}`} />
                  </span>
                  <span className="mono" style={{ fontSize: 16, color: r.idi > 0.45 ? "var(--icon-blue)" : "var(--text-muted)" }}>+{fmtScore(r.idi)}</span>
                </div>
              ))}
              <p className="callout">
                Победитель — <strong style={{ color: "var(--text)" }}>синтаксические связи</strong>:
                различает авторов с оценкой&nbsp;<strong style={{ color: "var(--icon-blue)" }}>{fmtScore(RIGOR.fa2[0].author)}</strong>, но войну
                от деревни у чужих авторов почти не видит (<strong style={{ color: "var(--text)" }}>{fmtScore(RIGOR.fa2[0].genreXA)}</strong> —
                ниже случайного угадывания). В этой проверке связи слов лучше разделяют авторов,
                чем две выбранные тематические группы. А{" "}
                <strong style={{ color: "var(--text)" }}>DSP — в самом низу</strong>: жанр он ловит ({fmtScore(RIGOR.fa2.at(-1).genreXA)}) почти
                так же, как автора ({fmtScore(RIGOR.fa2.at(-1).author)}). Его результат на расширенном наборе также чувствителен к жанру.
              </p>
            </div>
            <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
              <div className="mono muted" style={{ fontSize: 16 }}>
                «Тихий Дон» по синтаксису, частям речи и связям слов; ось построена на отложенных текстах:
              </div>
              {[
                { a: "эталон: ранние рассказы Шолохова", v: RIGOR.caEnsShRef, kind: "sh" },
                { a: "«Поднятая целина» (контроль)", v: RIGOR.caEnsPc, kind: "ctrl" },
                { a: "«Тихий Дон» (спорный)", v: RIGOR.caEnsTd, kind: "td" },
                { a: "эталон: проза Крюкова", v: RIGOR.caEnsKrRef, kind: "kr" },
              ].map((r) => (
                <div key={r.a} className="data-row" style={{ display: "grid", gridTemplateColumns: "1fr 4ch", alignItems: "center", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 16, color: r.kind === "td" ? "var(--text)" : "var(--text-muted)", fontWeight: r.kind === "td" ? 700 : 400, marginBottom: 3 }}>{r.a}</div>
                    <span style={{ display: "block", height: 8, borderRadius: 4, background: "var(--surface-sunken)", position: "relative", overflow: "hidden" }}>
                      <span style={{ position: "absolute", left: `${RIGOR.caEnsMid * 100}%`, top: 0, bottom: 0, width: 1, background: "var(--cinnabar)" }} title="середина оси" />
                      <span style={{ display: "block", height: "100%", width: `${r.v * 100}%`, background: r.kind === "td" ? "var(--icon-blue)" : r.kind === "kr" ? "var(--cinnabar)" : r.kind === "sh" ? "var(--gold)" : "var(--border-strong)" }} />
                    </span>
                  </div>
                  <span className="mono" style={{ fontSize: 16, color: r.kind === "td" ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v)}</span>
                </div>
              ))}
              <p className="callout" style={{ margin: 0 }}>
                Ось <strong style={{ color: "var(--text)" }}>широкая</strong> (Шолохов&nbsp;{RIGOR.caEnsShRef} ↔
                Крюков&nbsp;{RIGOR.caEnsKrRef}), и «Тихий Дон» ({RIGOR.caEnsTd}) сидит <strong style={{ color: "var(--icon-blue)" }}>заметно
                на стороне Шолохова</strong>, контроль «Целина» проходит ({RIGOR.caEnsPc}). Это{" "}
                <em>сильнее</em>, чем давал DSP у середины. Покнижно по синтаксису: кн.1 уверенно Шолохов
                ({RIGOR.caEnsTdBooks[0].p}), кн.4 — ничья ({RIGOR.caEnsTdBooks[3].p}). В покнижной проверке с исключением проверяемых работ из обучения
                (Тест&nbsp;№3) слабым выходит, наоборот, первый том — какой том «шатается», зависит от набора признаков, и это ожидаемо.
              </p>
            </div>
          </div>
          <p className="verdict">
            В этом сравнении по синтаксису (жанр&nbsp;{fmtScore(RIGOR.fa2[0].genreXA)}), на
            расширенном корпусе — «Тихий Дон» <strong style={{ color: "var(--text)" }}>склоняется к Шолохову
            отчётливее</strong>, чем на любом другом признаке ({RIGOR.caEnsTd} на оси 0.10–0.94; усреднённый профиль:{" "}
            {fmtPct(RIGOR.caEnsCentTdFracPos, 0)} пересчётов к Шолохову). Это поддерживает сходство с выбранным профилем Шолохова и ещё
            сильнее давит «Крюкова». Но по целым книгам, а их всего {RIGOR.bcTdNbooks}, разброс правдоподобных значений всё ещё{" "}
            <strong style={{ color: "var(--editorial-accent)" }}>включает 0</strong> (от {RIGOR.caEnsCentTdCiLo} до {RIGOR.caEnsCentTdCiHi}),
            кн.4 — ничья, а на одних только синтаксических связях контроль «Целины» на грани. «Склоняется» — да; «доказано» — нет.
          </p>
        </div>

        {/* 5f. Рукопись: глубина авторской правки (палеография через VertexAI) */}
        <div className="reveal module">
          <h2 id="sholokhov-section-7">Рукопись · глубина авторской правки</h2>
          <p className="prose muted" style={{ maxWidth: "76ch", marginBottom: 12 }}>
            Отдельный скептический довод — не про стиль, а про <strong style={{ color: "var(--text)" }}>почерк</strong>:
            будто бы черновики «Тихого Дона» слишком чистые, как переписанные с чужого готового текста. Проверяем на
            самих листах. По случайным страницам чернового автографа ТД (отдел рукописей ИМЛИ) и черновиков трёх заведомо
            сочинявших классиков оцениваем глубину правки 1–5 единой шкалой (от правки одного слова до сплошной
            переработки — когда страница переписана поверх стёртого). Оценка — моделью Gemini 3.1 Pro, читающей
            изображения, через VertexAI; это <em>грубая оценка по почерку</em>, а не анализ стиля.
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              <div className="mono muted" style={{ fontSize: 16, marginBottom: 8 }}>
                средняя глубина правки (1 — почти чисто · 5 — сплошь переписано), случайные страницы:
              </div>
              {MS.rows.map((r) => (
                <div key={r.name} className="data-row" style={{ display: "grid", gridTemplateColumns: "22ch 1fr 4ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                  <span style={{ fontSize: 16, color: r.isTarget ? "var(--text)" : "var(--text-muted)", fontWeight: r.isTarget ? 700 : 400 }}>
                    {r.name} <span className="mono muted" style={{ fontWeight: 400 }}>n={r.n}</span>
                  </span>
                  <MeterBar value={r.mean} max={5} accent={r.isTarget ? "var(--icon-blue)" : "var(--gold)"} />
                  <span className="mono" style={{ fontSize: 16, color: r.isTarget ? "var(--icon-blue)" : "var(--text-muted)" }}>{r.mean}</span>
                </div>
              ))}
              <p className="mono muted" style={{ fontSize: 16, marginTop: 8 }}>
                доля страниц со «структурной» переработкой: {MS.rows.map((r) => `${r.name.split(" ")[0]} ${fmtPct(r.structFrac, 0)}`).join(" · ")}
              </p>
            </div>
            <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
              <p className="verdict" style={{ margin: 0 }}>
                На просмотренных страницах модель <strong style={{ color: "var(--text)" }}>отмечает правку</strong>.
                В этой модельной разметке правка у Шолохова — <strong style={{ color: "var(--text)" }}>самая лёгкая</strong> из четырёх (средняя
                {" "}{MS.test.shMean} против {MS.test.ctrlMean} у контролей), и <strong style={{ color: "var(--text)" }}>ни на одной</strong>
                {" "}из {MS.test.shN} страниц она не доходит до сплошной переработки (максимум — уровень фразы), тогда
                как у всех трёх классиков такие листы есть.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Но копирование это <strong style={{ color: "var(--text)" }}>не доказывает</strong>: разница средних статистически
                незначима (тест Манна–Уитни — проверка, различаются ли две группы; p&nbsp;{fmtP(MS.test.p)}, размер эффекта средний d&nbsp;=&nbsp;{MS.test.cohenD}), распределения
                перекрываются, а «ошибки переписчика» в модельной разметке не отмечены. Более лёгкая правка
                совместима и с обдумыванием «в уме», и с тем, что сохранившийся автограф — уже не первый черновик.
              </p>
            </div>
          </div>
          <p className="muted" style={{ fontSize: 16, marginTop: 12, maxWidth: "82ch" }}>
            Оговорки: оценка 1–5 грубая и субъективная (модель зрения, не текстолог); по {msPagesMin}–{msPagesMax} страниц на
            автора, по одному-двум произведениям; наборы контролей смещены (у Достоевского взяты страницы с набросками → доля «схем»
            завышена). Это наблюдение про <em>потолок</em> правки, а не вывод об авторстве.
          </p>
        </div>

        {/* 6. Вердикт */}
        <div className="reveal module">
          <h2 id="sholokhov-section-8">Вердикт</h2>
          <div className="verdict-layout">
              <p className="callout" style={{ marginTop: 0 }}>
                В этих сравнениях «Тихий Дон» <strong style={{ color: "var(--gold-ink)" }}>чаще ближе к профилю Шолохова</strong>.
                Ответ зависит от опорных произведений и признаков. Данные позволяют сравнить
                выбранные профили, но авторские метки ранних рассказов остаются предпосылкой.
              </p>
              <p className="verdict">
                <strong style={{ color: "var(--text)" }}>Проверка с одновременным исключением целевых произведений.</strong>{" "}
                Проверка по целым книгам, где все спорные тома и донские контроли разом вынуты из обучения (опора — лишь
                оставшиеся работы с корпусной меткой «Шолохов»), относит <strong style={{ color: "var(--editorial-positive)" }}>{RIGOR.loboTd.tdAttrib} тома → Шолохову</strong>{" "}
                при отсутствии ложных срабатываний на проверенных донских контролях. Это исключает утечку самих проверяемых
                произведений, но не замкнутость эталона по меткам опорных текстов. Против «много рук» — попарная проверка авторства
                против {MULTIHANDS.avMultiHand.nPos} смесей под чужими именами: корпус Шолохова близок к контрольным одиночкам по этой оценке
                (z&nbsp;=&nbsp;{MULTIHANDS.avMultiHand.zPseudo}, ~5 обычных разбросов, p&nbsp;{fmtP(MULTIHANDS.avMultiHand.permP)}). А доля
                «чужих» отрывков по томам падает к финалу — назначения к основному профилю учащаются к концу романа:
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, margin: "6px 0 4px" }}>
                {RIGOR.loboTd.gradient.map((g) => (
                  <div key={g.book} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 16, color: "var(--text-muted)", marginBottom: 3 }}>{g.book}</div>
                    <MeterBar value={g.ff} accent={g.ff < 0.1 ? "var(--success)" : g.ff < 0.3 ? "var(--gold)" : "var(--cinnabar)"} />
                    <div className="mono" style={{ fontSize: 16, marginTop: 3, color: "var(--text)" }}>{fmtPct(g.ff, 0)}</div>
                  </div>
                ))}
              </div>
              <p className="muted" style={{ fontSize: 16, marginTop: 0 }}>
                «Чужая» доля первого тома ({RIGOR.loboTd.gradient[0].ff}) значимо выше фона (p&nbsp;{fmtP(RIGOR.loboTd.td1PermP)},
                блоками соседних отрывков p&nbsp;{fmtP(RIGOR.tdLoboBlockP)}); донские контроли дают ноль ложных срабатываний
                ({RIGOR.loboTd.donFpr}). Спад к четвёртому тому ({RIGOR.loboTd.gradient[0].ff}&nbsp;→&nbsp;{RIGOR.loboTd.gradient[3].ff}) —
                описание картины, не отдельный тест. В этой серии сравнений наблюдается согласованная
                картина нескольких способов сопоставления без утечки проверяемых произведений, но с оговорённой ниже зависимостью от меток эталона.
              </p>
              <p>
                <strong style={{ color: "var(--text)" }}>Где проходит граница.</strong> По целым книгам, а их всего{" "}
                {RIGOR.bcTdNbooks}, разброс правдоподобных значений перевеса (от {RIGOR.bcTdCiLo} до {RIGOR.bcTdCiHi}){" "}
                <strong style={{ color: "var(--editorial-accent)" }}>включает 0</strong> — формально неотличимо от ничьей
                ({fmtPct(RIGOR.bcTdFracPos, 0)} пересчётов к Шолохову). Самая строгая проверка при выровненном жанре и вовсе
                даёт почти ничью ({RIGOR.gmlrTdShFull} vs {RIGOR.gmlrTdKrFull}), а «Поднятую целину» относит
                к Крюкову ({RIGOR.gmlrPcKrFull}). Эти результаты зависят от авторских меток опорных произведений.
                Внутренние сравнения также не разделяют автора и редактора; ниже перечислены
                условия, ограничивающие исторический вывод.
              </p>
              <details style={{ margin: "8px 0 4px" }}>
                <summary style={{ cursor: "pointer", color: "var(--icon-blue)", fontSize: 16, fontWeight: 600 }}>
                  Условия интерпретации и исследовательский контекст
                </summary>
              <p className="muted" style={{ marginTop: 12 }}>
                Чего утверждать <em>нельзя</em> — открытые пределы:
              </p>
              <ul className="muted" style={{ lineHeight: 1.6, paddingLeft: "1.1em" }}>
                <li><strong style={{ color: "var(--text)" }}>Автор и редактор:</strong> «Шолохов
                  писал сам» и «единый редактор переработал чужой материал» внутренними тестами на цельность{" "}
                  <em>неразличимы</em>. В отдельном сравнении профиль{" "}
                  <strong style={{ color: "var(--text)" }}>Серафимовича</strong> не оказывается{" "}
                  <strong style={{ color: "var(--text)" }}>ближайшим</strong> — ТД ближе к ранним рассказам Шолохова
                  ({RIGOR.serafEdShDon}), чем к Серафимовичу ({RIGOR.serafEdSeraf}) или Крюкову ({RIGOR.serafEdKrukov}).
                  Это сравнение профилей не измеряет объём или характер редакторского участия.</li>
                <li><strong style={{ color: "var(--text)" }}>Влияние жанра:</strong> у
                  признаков по словам военная и сельская проза различается у разных авторов с оценкой&nbsp;{RIGOR.crossGenreAuc}.
                  У блока <strong style={{ color: "var(--icon-blue)" }}>синтаксических связей</strong> в тесте&nbsp;№5
                  AUC различения авторов составляет {fmtScore(RIGOR.fa2[0].author)}, а жанров — {fmtScore(RIGOR.fa2[0].genreXA)}.
                  Ориентир случайного ранжирования для AUC — 0,5; одна оценка на выбранной подборке
                  не устанавливает отсутствия жанрового влияния. DSP на расширенном наборе — <em>среди худших</em>:
                  ограничение лексических признаков снижает здесь качество.</li>
                <li><strong style={{ color: "var(--editorial-accent)" }}>Замкнутый круг с эталоном (важно):</strong> ТД ближе
                  всего к <em>ранним донским рассказам</em> Шолохова (1924–26) — но именно этот период входит в спорную зону.
                  Дополнительный вариант: обучаем на <em>поздней</em> прозе (война+ПЦ-2, 1942–69) против Крюкова и проецируем.
                  Результат <em>смешанный</em>: поздний Шолохов узнаёт ТД ({RIGOR.circTd}) <em>примерно как
                  собственные ранние рассказы</em> ({RIGOR.circEarly}) — то есть ТД не <em>менее</em> шолоховский, чем
                  ранняя проза, но сам сигнал слаб (мешает разрыв в жанре). «ТД = Шолохов» нельзя доказать, не
                  приняв метки ранних рассказов. Сопоставимая донская опора вне обсуждаемого
                  периода в этой подборке не представлена.</li>
                <li><strong style={{ color: "var(--text)" }}>Ограниченная чувствительность:</strong> похожего по
                  стилю соавтора, давшего меньшую часть текста, тесты не различают; порог у каждой проверки свой —{" "}
                  <span className="mono" style={{ color: "var(--text)" }}>целые книги ~{RIGOR.loboTd.minAdmix}% · кривая примеси ~{RIGOR.powerDetectK}% · скрытая рука ~{MULTIHANDS.hiddenPositive.flagThreshold}%</span>.</li>
                <li>Для Кумова и Севского в подборке мало текста; эти профили ограничивают
                  сравнение. Вопрос заимствования исходного материала требует текстологической проверки.</li>
                <li><strong style={{ color: "var(--text)" }}>Закрытый список кандидатов:</strong> атрибуция по целым
                  книгам ({RIGOR.tdLoboAttributed} тома → Шолохову) держится внутри короткого списка «Шолохов, Крюков
                  или Серафимович». Если убрать короткий список и открыть выбор на весь корпус авторов, поздние тома всё равно
                  уверенно уходят к Шолохову (ТД-4: {fmtScore(RIGOR.openSetTd.td4Share, 3)}), а ранние — нет
                  (ТД-1: {fmtScore(RIGOR.openSetTd.td1Share, 3)}, больше всего — к {RIGOR.openSetTd.td1TopName} с долей{" "}
                  {fmtScore(RIGOR.openSetTd.td1TopShare, 3)}). Контрольный текст {RIGOR.platonovInject.name} относится
                  к собственному профилю ({fmtScore(RIGOR.platonovInject.selfShare, 1)}), к Шолохову — {fmtScore(RIGOR.platonovInject.toSholokhovShare, 1)}.
                  Профиль Платонова присутствует среди кандидатов: этот контроль проверяет различение представленных
                  классов, но не способность отказаться от ответа, когда настоящий автор отсутствует в списке.
                  Отдельная проверка «а тот ли это автор вообще» (модель учится узнавать именно почерк Шолохова и
                  отвергать чужих) на «Тихом Доне» даёт {RIGOR.verifTd.tdAttributed} — но в такой постановке она
                  чувствительна к составу данных: зрелые произведения под именем Шолохова тоже её не проходят, поэтому вывод «ТД не Шолохов»
                  из неё не следует.</li>
              </ul>
              <p className="muted">
                <strong style={{ color: "var(--text)" }}>Соотнесение с предшественниками.</strong> Компьютерный анализ
                Хьетсо и коллег (1984) пришёл к тому же направлению: Шолохов, не Крюков. К его методике известны
                претензии — узкий набор признаков (длины предложений, частотные распределения), отсутствие жанрового
                контроля и зависимость от авторских меток эталона. Исторические расчёты этой главы исключают
                проверяемые произведения из обучения, варьируют признаки и список кандидатов, сравнивают отдельные
                жанровые и донские контроли. Они характеризуют эти конкретные панели; независимость от темы,
                редакции и происхождения эталонов из них не следует. Сходное направление результатов можно
                сопоставить, сохраняя различия в корпусах и методиках.
              </p>
              </details>
              <p className="verdict">
                Итог: эти сравнения <strong style={{ color: "var(--gold-ink)" }}>совместимы</strong> с авторством Шолохова
                при принятых метках опорных произведений. Но превратить это в «доказано» анализ стиля
                не может — и здесь такой вывод не делается.
              </p>
            <aside className="verdict-aside">
              <AnomalyGlyph kind="relation_mismatch" size={52} />
              <span className="muted mono" style={{ fontSize: 16, textAlign: "center", maxWidth: "26ch" }}>
                «один автор» и «один редактор»<br />изнутри неразличимы
              </span>
            </aside>
          </div>
        </div>

        <Sources
          label="Внешние исследования"
          items={[
            { cite: "Н. П. Великанова, Б. В. Орехов (2019). «Цифровая текстология: атрибуция текста на примере романа М. А. Шолохова “Тихий Дон”»", url: "https://publications.hse.ru/pubs/share/direct/314793949.pdf#page=4", format: "PDF, статья со страницы 4 файла" },
            { cite: "К. А. Маслинский (2022). «Уточненная цифровая текстология: еще раз к вопросу об авторстве романа “Тихий Дон”»", url: "https://ruslitras.ru/index.php?dispatch=products.print_publication&format=pdf&product_id=95733&version_id=93851", format: "PDF" },
          ]}
          note="Эти работы описывают задачу и методы сравнения. Графики Stylo используют отдельные агрегаты перечисленных ниже протоколов."
        />
        <Sources
          label="Рукописи и научные издания"
          items={[
            { cite: "ИМЛИ РАН: научное издание «Тихого Дона», история рукописей, факсимиле и транскрипция", url: "https://imli.ru/index.php/izdaniya/izdatelstvo/249-tikhij-don" },
            { cite: "ФЭБ: описание электронного научного издания «Шолохов», включая собрание факсимиле", url: "https://feb-web.ru/feb/sholokh/rub1.html?cmd=1" },
          ]}
        />
        <Sources label="Данные графиков" artifact="sholokhov"
          note="Файл содержит показатели главы и общих контрольных сравнений, названия файлов расчётов и их контрольные суммы. Разметка правки рукописей выполнена моделью Gemini 3.1 Pro через VertexAI; это автоматическая оценка изображений." />
      </div>
    </section>
  );
}
