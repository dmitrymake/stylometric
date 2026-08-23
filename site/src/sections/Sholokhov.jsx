import { Card, Stat, AnomalyGlyph, ConfidenceBar } from "@dmitrymake/rk-ui";
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

// Способы проверки = пять нумерованных тестов ниже (Тест №1…№5). Слово-числительное
// в лиде берём из длины списка, чтобы обещание в лиде не расходилось с нумерацией.
const METHODS = [
  "узнаём автора с поправкой на тему",                    // №1
  "сравниваем словари",                                   // №2
  "напрямую ищем «много рук»",                             // №3
  "смотрим, насколько ровен его стиль от книги к книге",   // №4
  "отделяем личный почерк от темы",                       // №5
];
const NUM_INSTR = { 3: "тремя", 4: "четырьмя", 5: "пятью", 6: "шестью", 7: "семью" };

// Один и тот же предел у всех проверок на цельность — один короткий указатель на «Пределы»
// вердикта вместо повторения оговорки в каждом под-тесте.
function SimilarHandLimit() {
  return (
    <p className="muted" style={{ fontSize: 12, marginTop: 16, textAlign: "center" }}>
      У всех проверок этого раздела предел один и тот же — похожего донского соавтора с малой долей текста
      анализ стиля не различает; пороги по каждому тесту сведены в «Пределы» вердикта.
    </p>
  );
}

const COLOR_MAP = {
  "М. Шолохов": "var(--icon-blue)",
  "Ф. Крюков": "var(--cinnabar)",
  "А. Серафимович": "var(--gold)",
};
const accentOf = (k) => COLOR_MAP[nm(k)] || "var(--text-muted)";

const PLAUS_CHIP = { "высокая": "hot", "средняя": "gold", "низкая": "", "маргинальная": "" };

// Короткий ответ теста: один крючок-вывод сразу под вопросом, чтобы читатель
// получал итог до разбора улик и не тонул в повторных развёрнутых вердиктах.
function TestSummary({ children }) {
  return (
    <p style={{
      display: "flex", gap: 12, alignItems: "baseline",
      margin: "0 0 22px", padding: "10px 14px",
      borderLeft: "3px solid var(--icon-blue)",
      background: "var(--surface-sunken)", borderRadius: "0 6px 6px 0",
    }}>
      <strong style={{ color: "var(--icon-blue)", whiteSpace: "nowrap", fontSize: 11.5, letterSpacing: "0.05em", textTransform: "uppercase" }}>
        Короткий ответ
      </strong>
      <span style={{ fontSize: 14.5, lineHeight: 1.55, color: "var(--text)" }}>{children}</span>
    </p>
  );
}

// Пояснения к некоторым карточкам в исходных данных содержат служебные пометки и жаргон.
// Для читателя без подготовки заменяем их обычным русским; смысл и направление вывода сохранены.
const WHY_CLEAN = {
  "Николай Гумилёв":
    "Поэт Серебряного века; версия о его авторстве — на дальней обочине спора. Проверили его отдельно, как заведомо чужого автора: «Тихий Дон» лежит далеко от Гумилёва — версия не поддержана.",
  "Андрей Платонов":
    "Назывался как возможный автор «Поднятой целины». Проверили отдельно: «Поднятая целина» дальше всего от Платонова и ближе к Шолохову. При этом тот же метод уверенно узнаёт подлинного Платонова в его собственной прозе — значит это настоящее «нет», а не слепота теста.",
};

// Кандидаты с уцелевшей, но слишком тонкой прозой: формально в корпусе, но профиль по ней ненадёжен.
const THIN_CORPUS = new Set(["Виктор Севский (Краснушкин)", "Роман Кумов"]);

function CandidateCard({ c }) {
  const thin = THIN_CORPUS.has(c.name);
  const why = WHY_CLEAN[c.name] || c.why;
  return (
    <Card padding={18}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
        <strong style={{ color: "var(--text)", fontSize: 15.5 }}>{c.name}</strong>
        <span className="mono muted" style={{ fontSize: 12 }}>† {c.death}</span>
      </div>
      <div style={{ display: "flex", gap: 7, margin: "9px 0 10px", flexWrap: "wrap" }}>
        <span className={"chip " + PLAUS_CHIP[c.plaus]}>{c.plaus}</span>
        <span className="chip" style={{ opacity: 0.85 }}>
          {c.inCorpus ? (thin ? "в корпусе · профиль слабый" : "в корпусе · проверяем") : "вне теста"}
        </span>
      </div>
      <p className="muted" style={{ fontSize: 13, lineHeight: 1.5, margin: 0 }}>{why}</p>
    </Card>
  );
}

function ThematicRow({ rank, name, score, max }) {
  const hi = nm(name) === "Ф. Крюков" || nm(name) === "М. Шолохов";
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.4ch 13ch 1fr 5ch", alignItems: "center", gap: 10 }}>
      <span className="mono muted" style={{ fontSize: 12 }}>{rank}</span>
      <span style={{ fontSize: 13.5, color: hi ? "var(--text)" : "var(--text-muted)", fontWeight: hi ? 600 : 400 }}>
        {nm(name)}
      </span>
      <MeterBar value={score} max={max} accent={hi ? accentOf(name) : "var(--border-strong)"} />
      <span className="mono" style={{ fontSize: 12, color: hi ? "var(--text)" : "var(--text-muted)" }}>{fmtScore(score, 3)}</span>
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
          <p className="eyebrow">Кейс · «Тихий Дон»</p>
          <h2>«Шолохов вообще не писатель»?</h2>
          <p className="prose lead muted">
            Известный спор об авторстве в русской литературе. Двадцать два года, четыре
            класса образования — и роман-эпопея. Молва с первых лет твердила: настоящий
            автор — донской предшественник, а то и несколько рук сразу. Подозрение старое и
            упрямое. Берём его всерьёз. Проверяем{" "}
            <strong style={{ color: "var(--text)" }}>{NUM_INSTR[METHODS.length]} независимыми способами</strong>, на{" "}
            <em>всех</em> уцелевших его книгах (весь «Тихий Дон», обе «Поднятой целины», поздняя
            проза и ранние рассказы). Без готового вывода — только по фактам.
          </p>
          <ol className="prose muted" style={{ maxWidth: "62ch", margin: "4px 0 0", paddingLeft: "1.4em", lineHeight: 1.6, fontSize: 14 }}>
            {METHODS.map((m, i) => <li key={i} style={{ marginBottom: 2 }}>{m}</li>)}
          </ol>
        </div>

        {/* 1. Поле кандидатов */}
        <div className="reveal module">
          <h3>Кто, кроме Шолохова</h3>
          <p className="prose muted" style={{ maxWidth: "64ch", marginBottom: 22 }}>
            За век накопилось около десятка версий — от академических до маргинальных.
            Собрали всех, кого называли, и оценили, насколько это правдоподобно. Шесть
            кандидатов с уцелевшей прозой можно проверить по стилю. У остальных либо нет
            текстов, либо они не были писателями.
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
            Оговорка о пределах теста: у Кумова уцелело ~1,6&nbsp;тыс. слов, у Севского ~19&nbsp;тыс. —
            этого мало для надёжного профиля. Их кандидатура не отводится, но остаётся{" "}
            <strong style={{ color: "var(--text)" }}>непроверяемой по стилю</strong>.
          </p>

          {/* Первый взгляд: ТД против ВСЕХ кандидатов — разминка перед пятью тестами */}
          <div className="reveal" style={{ marginTop: 34 }}>
            <p className="eyebrow" style={{ marginBottom: 6 }}>Первый взгляд</p>
            <h4 style={{ marginBottom: 6 }}>«Тихий Дон» против всех кандидатов сразу</h4>
            <p className="prose muted" style={{ maxWidth: "70ch", marginBottom: 16 }}>
              Один прямой замер на чистом признаке (по синтаксису): к чьему усреднённому профилю
              ближе «Тихий Дон»? Считаем честно: <strong style={{ color: "var(--text)" }}>модель не подглядывает</strong> в
              проверяемый текст — он не участвует в обучении. Меньше — ближе:
            </p>
            <div className="split" style={{ alignItems: "center" }}>
              <div>
                {RIGOR.tdCandDist.map((r) => (
                  <div key={r.a} style={{ display: "grid", gridTemplateColumns: "16ch 1fr 4ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                    <span style={{ fontSize: 12.5, color: r.self ? "var(--text)" : "var(--text-muted)", fontWeight: r.self ? 700 : 400 }}>{r.a}</span>
                    <MeterBar value={r.d} max={Math.max(...RIGOR.tdCandDist.map((x) => x.d))} accent={r.self ? "var(--icon-blue)" : "var(--text-muted)"} />
                    <span className="mono" style={{ fontSize: 11, color: r.self ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.d)}</span>
                  </div>
                ))}
              </div>
              <p className="callout" style={{ margin: 0 }}>
                «Тихий Дон» ближе всего к <strong style={{ color: "var(--text)" }}>раннему Шолохову</strong> ({fmtScore(tdSelfDist)}) —
                и так во всех 4 томах. Парное сравнение в одном жанре подтверждает: против <em>каждого</em> из{" "}
                {RIGOR.tdCandGm.length} кандидатов ТД уходит к Шолохову{" "}
                ({RIGOR.tdCandGm.map((c) => `${c.a} ${c.p}`).join(", ")} — все&nbsp;&gt;&nbsp;0.5).{" "}
                <strong style={{ color: "var(--text)" }}>Ни один</strong> выдвигавшийся автор не ближе к ТД, чем сам Шолохов.
              </p>
            </div>
            <p className="muted" style={{ fontSize: 12.5, marginTop: 12 }}>
              Сильный <em>коллективный</em> довод против «писал кто-то из них» — включая маргинальные версии (Гумилёв).
              Но спор он не закрывает: эталоном служит сам Шолохов, а кандидата без сохранившихся текстов
              ({RIGOR.tdCandUntestable}) так не проверишь.
            </p>
          </div>
        </div>

        {/* 3. Атрибуция Тихого Дона — две модели */}
        <div className="reveal module">
          <h3>Тест №1 · Кому уходит «Тихий Дон»</h3>
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
          <div style={{ display: "grid", gap: 9, marginTop: 8 }}>
            <div className="mono muted" style={{ display: "grid", gridTemplateColumns: "16ch 1fr 1fr 8ch", gap: 10, fontSize: 11 }}>
              <span></span><span>полная модель</span><span>меньше слов</span><span>согласие</span>
            </div>
            {RIGOR.attrib.map((r) => (
              <div key={r.book} style={{ display: "grid", gridTemplateColumns: "16ch 1fr 1fr 8ch", gap: 10, alignItems: "center" }}>
                <span style={{ fontSize: 13, color: "var(--text)" }}>{r.book}</span>
                {[r.full, r.topic].map((v, i) => (
                  <span key={i} style={{ position: "relative", height: 16, borderRadius: 4, background: "var(--surface-sunken)", overflow: "hidden" }}>
                    <span style={{ display: "block", height: "100%", width: `${v * 100}%`, background: "var(--icon-blue)", opacity: 0.55 + 0.45 * v }} />
                    <span className="mono" style={{ position: "absolute", right: 5, top: 1, fontSize: 10.5, color: "var(--text)" }}>{fmtPct(v, 0)}</span>
                  </span>
                ))}
                <span className="mono" style={{ fontSize: 11, color: r.reliable ? "var(--success)" : "var(--gold)" }} title="согласие LR и Delta">
                  {r.agree}
                </span>
              </div>
            ))}
          </div>

          <p className="verdict">
            Когда объём текста у всех авторов выровнен, полная модель отдаёт «Тихий Дон» Шолохову ({fmtScore(fullMin, 2)}–{fmtScore(fullMax, 3)}) —
            устойчиво, в том числе когда все авторы урезаны до {RIGOR.dsNmin} отрывков (= объём Крюкова): медиана полной модели{" "}
            {RIGOR.dsTdFullMed} [{RIGOR.dsTdFullLo}–{RIGOR.dsTdFullHi}]. То есть это <strong style={{ color: "var(--text)" }}>не
            побочный эффект частоты слов</strong>. «≈40% уходит Крюкову» — след перекоса по объёму в корпусе.
          </p>
          <p className="note">
            <strong style={{ color: "var(--cinnabar)" }}>Самый строгий тест ослабляет вывод:</strong>{" "}
            если уравнять не только объём, но и <em>жанр</em> — собрать профиль Шолохова <strong style={{ color: "var(--text)" }}>только
            из ранних донских рассказов</strong> ({RIGOR.earlyPoolN}, тот же тип текста, что ТД) и взять у Крюкова столько же —
            полная модель на «Тихом Доне» даёт почти <strong style={{ color: "var(--cinnabar)" }}>ничью</strong>:
            Шолохов {RIGOR.gmlrTdShFull} vs Крюков {RIGOR.gmlrTdKrFull}. Значит высокие доли держались не на совпадении
            почерка. Держались на том, что у Шолохова в обучении <em>есть донские тексты</em>, каких нет у других авторов.
            С поправкой на тему ТД всё ещё к Шолохову ({RIGOR.gmlrTdShTopic}) — но и тематические признаки несут жанр.
            Словарный сигнал «ТД = Шолохов, не Крюков» при выровненном жанре — <strong style={{ color: "var(--text)" }}>неубедителен</strong>.
          </p>
          <details style={{ margin: "6px 0" }}>
            <summary style={{ cursor: "pointer", color: "var(--icon-blue)", fontSize: 13, fontWeight: 600 }}>
              Почему столбец «согласие» — не мера надёжности
            </summary>
            <p className="muted" style={{ marginTop: 8, marginBottom: 0, fontSize: 13 }}>
              «Согласие» — насколько по каждому отрывку сходятся две разные модели: одна взвешивает признаки, другая
              мерит близость ({fmtScore(agreeMin, 2)}–{fmtScore(agreeMax, 2)}). При 5 кандидатах случайное совпадение уже ≈{fmtScore(1 / 5)},
              а вторая модель по отдельным отрывкам ведёт себя как шум. Поэтому вывод по книгам опирается на общий
              ответ первой модели и усреднённый профиль автора, а не на этот флаг.
            </p>
          </details>
        </div>

        {/* Калибровка: «Поднятая целина» — заведомо Шолохов (негативный контроль) */}
        <div className="reveal module">
          <h4 style={{ marginBottom: 6 }}>«Поднятая целина» — ответ известен заранее</h4>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 14 }}>
            Прежде чем доверять методу на спорном «Тихом Доне», его стоит проверить на тексте с известным ответом.
            «Поднятая целина» — бесспорно Шолохов; её {PC.fragments}{" "}
            {plural(PC.fragments, "фрагмент", "фрагмента", "фрагментов")} прогоняем через тот же набор моделей.
          </p>
          <div style={{ display: "grid", gap: 12, maxWidth: "54ch" }}>
            {PC.candidates.map((c, i) => (
              <ConfidenceBar
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
          <h3>Тест №2 · Откуда «крюковский след»</h3>
          <p className="prose muted" style={{ maxWidth: "66ch", marginBottom: 20 }}>
            Близость по словам (насколько совпадают словари; тема при этом не вычищена). На <em>чистом</em> корпусе
            ближайший к «Тихому Дону» — <strong style={{ color: "var(--text)" }}>сам Шолохов</strong> ({fmtScore(THEM.tihiyDon[0][1])}),
            а Крюков — близкий второй ({fmtScore(THEM.tihiyDon[1][1])}). Крюков «подозрительно близко» не потому, что писал роман, а
            потому что оба пишут <em>один и тот же</em> донской мир: у «Донских рассказов» с корпусной меткой
            «Шолохов» ближайший после самого автора — снова Крюков ({fmtScore(THEM.donskie[1][1])}), но следом
            почти вплотную идут и недонские авторы ({fmtScore(THEM.donskie[2][1])}).
          </p>
          <TestSummary>
            «Крюковский след» — это общая донская тема, а не общая рука: к Крюкову вплотную жмутся
            даже ранние рассказы с корпусной меткой «Шолохов».
          </TestSummary>
          <div className="grid cols-2" style={{ gap: 22 }}>
            <Card padding={22}>
              <p className="eyebrow" style={{ marginBottom: 14 }}>«Тихий Дон» — ближайшие по словам</p>
              <div style={{ display: "grid", gap: 9 }}>
                {THEM.tihiyDon.map(([n, s], i) =>
                  <ThematicRow key={n} rank={i + 1} name={n} score={s} max={THEM.tihiyDon[0][1]} />)}
              </div>
            </Card>
            <Card padding={22}>
              <p className="eyebrow" style={{ marginBottom: 14 }}>«Донские рассказы» Шолохова — ближайшие</p>
              <div style={{ display: "grid", gap: 9 }}>
                {THEM.donskie.map(([n, s], i) =>
                  <ThematicRow key={n} rank={i + 1} name={n} score={s} max={THEM.donskie[0][1]} />)}
              </div>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 12 }}>
                Ранний корпус с меткой «Шолохов» тоже почти упирается в Крюкова — донская тема роднит всех.
              </p>
            </Card>
          </div>

          <div className="split" style={{ marginTop: 28, alignItems: "center" }}>
            <div className="prose">
              <p className="callout gold" style={{ marginTop: 0 }}>
                Откуда тогда вековой «крюковский след»? Из <strong style={{ color: "var(--text)" }}>темы</strong>:
                донские военно-казачьи слова у Крюкова и Шолохова общие, и простая модель по словам на
                <strong style={{ color: "var(--text)" }}> неполном и перекошенном</strong> корпусе легко
                принимает совпадение материала за совпадение руки.
              </p>
              <p>
                Это видно по тому, что «Донские рассказы» с корпусной меткой «Шолохов» тоже стоят к
                Крюкову вплотную. Но как только корпус починен, а объёмы у авторов уравнены (Тест&nbsp;№1), даже
                модель по словам отдаёт и «Тихий Дон», и «Поднятую целину»{" "}
                <strong style={{ color: "var(--text)" }}>Шолохову</strong>, а не Крюкову. «Крюков» —
                метка <strong style={{ color: "var(--gold)" }}>донского материала</strong>, а не почерка.
              </p>
            </div>
            <div style={{ display: "grid", placeItems: "center", gap: 12 }}>
              <AnomalyGlyph kind="relation_mismatch" size={46} />
              <span className="muted mono" style={{ fontSize: 12, textAlign: "center", maxWidth: "22ch" }}>
                общий донской мир ≠<br />общая рука
              </span>
            </div>
          </div>
        </div>

        {/* Тест №3: много рук — LEAK-FREE */}
        <div className="reveal module">
          <h3>Тест №3 · Один автор или много рук?</h3>
          <p className="prose muted" style={{ maxWidth: "70ch", marginBottom: 8 }}>
            Две острые гипотезы. Первая: за Шолоховым стоял <em>не один</em> автор. Вторая, про метод: не
            превращается ли «Шолохов» в{" "}
            <strong style={{ color: "var(--text)" }}>метку, которая тянет к себе любой текст</strong> — раздутую
            обилием книг так, что вбирает и чужое. Чтобы это исключить, пространство стиля строим на{" "}
            <strong style={{ color: "var(--text)" }}>нейтральных авторах без Шолохова</strong> — и снова честно,
            модель не подглядывает в проверяемую книгу.
          </p>
          <TestSummary>
            «Много рук» не подтверждается: разброс внутри корпуса Шолохова — как у одного автора
            (z&nbsp;{fmtZ(MULTIHANDS.avMultiHand.zPseudo)}, это ~5 обычных разбросов ниже настоящих смесей). Предел — ниже.
          </TestSummary>

          <div className="split" style={{ alignItems: "start", marginTop: 18 }}>
            <div>
              <p className="eyebrow" style={{ marginBottom: 6 }}>Каждая книга → к какому автору ближе (по одной отложенной книге за раз)</p>
              <p className="muted" style={{ fontSize: 12, margin: "0 0 12px" }}>
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
                      <span style={{ fontSize: 13, color: td ? "var(--text)" : "var(--text-muted)", fontWeight: td ? 600 : 400 }}>
                        {r.book}{td1 && <span className="muted" style={{ fontWeight: 400 }}> · спорный при строгом тесте</span>}
                      </span>
                      <span style={{ fontSize: 12.5, color: r.stays ? "var(--icon-blue)" : "var(--cinnabar)", fontWeight: r.stays ? 500 : 700 }}>
                        {r.stays ? "→ Шолохов" : `→ ${r.nearest}`}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 12 }}>
                <strong style={{ color: "var(--text)" }}>{RIGOR.b2Stay}/{RIGOR.b2N}</strong> книг ближе к «Шолохову без
                этой книги», все 4 тома «Тихого Дона» и обе «Поднятых целины» — уверенно его. Два промаха — совсем короткие
                поздние рассказы (14 и 30 отрывков, профиль на них шумный): «Судьба человека»→Крюков (но Крюков †1920,
                рассказ 1957 — по времени невозможно), «Наука ненависти»→Булгаков. Это шум малых выборок, не вторая рука.
              </p>
            </div>
            <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
              <Stat label="книг ближе к Шолохову" value={`${RIGOR.b2Stay}/${RIGOR.b2N}`} accent="var(--success)" parade />
              <Stat label="разброс книг: место среди одиночек" value={`${RIGOR.dispRank} / ${RIGOR.dispPanelN}`} accent="var(--icon-blue)" hint={`${RIGOR.dispSholohov} против ${RIGOR.dispControl}±${RIGOR.dispControlStd} у одиночек — в нижней четверти по разбросу: метка не раздута, ведёт себя как обычный автор`} />
            </div>
          </div>

          {/* Решающий тест «много рук»: supervised pairwise authorship-verification (author-disjoint) */}
          <div className="module" style={{ marginTop: 26 }}>
            <p className="eyebrow" style={{ marginBottom: 4 }}>Решающий тест · попарная проверка авторства</p>
            <p className="muted" style={{ fontSize: 12.5, margin: "0 0 14px" }}>
              авторы в обучении и проверке не пересекаются, отрывки равного объёма; настроено на {MULTIHANDS.avMultiHand.nPos} псевдонимных смесях (разные авторы под одним именем) и {MULTIHANDS.avMultiHand.nNeg} одиночках
            </p>
            <p className="prose" style={{ margin: 0, fontSize: 14 }}>
              Для каждой пары книг считаем, насколько они «разные руки». У настоящего коллектива (смесь под псевдонимом)
              оценка ≈ <strong style={{ color: "var(--text)" }}>{MULTIHANDS.avMultiHand.posMean}</strong>, у одиночек ≈{" "}
              <strong style={{ color: "var(--text)" }}>{MULTIHANDS.avMultiHand.negMean}</strong>. Корпус Шолохова даёт{" "}
              <strong style={{ color: "var(--success)" }}>{MULTIHANDS.avMultiHand.score}</strong> — то есть он{" "}
              <strong style={{ color: "var(--success)" }}>неотличим от одного автора</strong>, а не от коллектива:
              z = <strong style={{ color: "var(--text)" }}>{MULTIHANDS.avMultiHand.zPseudo}</strong> (насколько велико отклонение
              против обычного разброса — здесь ~5 таких разбросов ниже смесей), уверенность различения{" "}
              {MULTIHANDS.avMultiHand.auc} [{MULTIHANDS.avMultiHand.aucCi[0]}–{MULTIHANDS.avMultiHand.aucCi[1]}] (1.0 — идеально, 0.5 — наугад),
              проверка на случайность p {fmtP(MULTIHANDS.avMultiHand.permP)}. Это главный довод против версии о нескольких авторах.
            </p>
            <div className="grid cols-3" style={{ marginTop: 14 }}>
              <Stat label="отрыв от псевдонимной смеси (z)" value={fmtZ(MULTIHANDS.avMultiHand.zPseudo)} accent="var(--success)" hint="~5 обычных разбросов — сильно ниже смесей" />
              <Stat label="уверенность: много рук или один" value={fmtScore(MULTIHANDS.avMultiHand.auc)} accent="var(--icon-blue)" hint={`p ${fmtP(MULTIHANDS.avMultiHand.permP)} (проверка на случайность)`} />
              <Stat label="оценка «много рук» у Шолохова" value={fmtScore(MULTIHANDS.avMultiHand.score)} accent="var(--success)" hint={`≈ одиночки ${MULTIHANDS.avMultiHand.negMean}, далеко от смеси ${MULTIHANDS.avMultiHand.posMean}`} />
            </div>
            <p className="muted" style={{ fontSize: 12, marginTop: 12, marginBottom: 0 }}>
              Предел: этот тест бьёт «много <em>разных</em> рук». Смесь <em>похожих</em> донских рук от одиночки неотличима,
              поэтому скрыть похожего соавтора с малой долей текста метод не может (пороги — в «Пределах» вердикта).
            </p>
          </div>

          <div className="grid cols-2" style={{ marginTop: 22, gap: 16 }}>
            <Card padding={22}>
              <p className="eyebrow" style={{ marginBottom: 4 }}>Что тест вообще способен заметить</p>
              <p className="muted" style={{ fontSize: 12.5, margin: "0 0 12px" }}>подмешиваем Крюкова → доля «крюковских» отрывков</p>
              <div style={{ display: "grid", gap: 6 }}>
                {RIGOR.power.map((x) => (
                  <div key={x.k} style={{ display: "grid", gridTemplateColumns: "4ch 1fr 4ch", alignItems: "center", gap: 8 }}>
                    <span className="mono muted" style={{ fontSize: 11 }}>{x.k}%</span>
                    <MeterBar value={x.frac} max={powMax} accent={x.k >= RIGOR.powerDetectK ? "var(--cinnabar)" : "var(--border-strong)"} />
                    <span className="mono" style={{ fontSize: 11 }}>{x.frac}</span>
                  </div>
                ))}
              </div>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
                Чужую руку тест уверенно ловит лишь от <strong style={{ color: "var(--cinnabar)" }}>~{RIGOR.powerDetectK}%</strong>{" "}
                примеси похожего по стилю автора. Меньшую долю различить не способен — честный предел.
              </p>
            </Card>
            <p className="callout gold" style={{ margin: 0 }}>
              И у «поправки на тему» есть предел. Строгая проверка — военная проза одного писателя против сельской прозы
              другого — показывает: даже признаки, которые считаются нечувствительными к теме, уверенно различают{" "}
              <strong style={{ color: "var(--cinnabar)" }}>жанр</strong>{" "}
              (<strong style={{ color: "var(--text)" }}>{RIGOR.crossGenreAuc}</strong>, где 0.5 — наугад, 1.0 — безошибочно).
              Значит, они несут ещё и жанр с эпохой, а не только личный почерк — поэтому говорим «стилистически похоже»,
              а не «доказано авторство».
            </p>
          </div>

          <p className="note">
            <strong style={{ color: "var(--text)" }}>Предел ещё жёстче:</strong> две половины <em>одной</em> книги уже
            различаются с уверенностью&nbsp;{RIGOR.sepFloor} (нижняя граница ≠ 0.5) — признаки так чувствительны к местному
            содержанию, что чисто проверить авторство, обучая модель их разделять, нельзя. При этом вывод «ТД ближе к
            Шолохову» устойчив: держится в <strong style={{ color: "var(--text)" }}>{RIGOR.embRobustConfigs}/{RIGOR.embRobustN}</strong>{" "}
            вариантах настройки пространства стиля (число осей, нормировка, способ мерить расстояние).
          </p>

          <p className="verdict">
            Итог: «много рук» <strong style={{ color: "var(--text)" }}>не подтверждается</strong> — корпус не разнороднее
            одиночных авторов (по разбросу место {RIGOR.dispRank}/{RIGOR.dispPanelN}), метка не раздута. Похожего соавтора
            с малой долей текста тесты не поймали бы — общий предел см. в «Пределах» вердикта.
          </p>
        </div>

        {/* 5d. Гомогенность: разные люди писали разные работы? */}
        <div className="reveal module">
          <h3>Тест №4 · «Разные люди писали разные его работы»?</h3>
          <p className="prose muted" style={{ maxWidth: "70ch", marginBottom: 18 }}>
            Сильная версия гипотезы: почти всё, что издавал Шолохов, написано <em>разными</em> людьми.
            Если так — его книги должны быть <strong style={{ color: "var(--text)" }}>отделимы друг от друга
            сильнее</strong>, чем книги настоящего одиночного автора. Меряем, насколько попарно отличимы по стилю все его
            книги (снова честно — модель не подглядывает в проверяемую книгу). Ставим их в ряд с одиночными авторами
            и опорными точками.
          </p>
          <TestSummary>
            Книги Шолохова разнятся между собой как у одного «широкого» автора — Бунин и вовсе
            разнообразнее. Версия «разные люди» не подтверждается, но и не исключается для похожих донских рук.
          </TestSummary>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              <div className="mono muted" style={{ fontSize: 11, display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span>← похожи (одна рука)</span><span>отделимы →</span>
              </div>
              {[
                { a: "ориентир: две половины одной книги", v: RIGOR.homFloor, hi: false, anchor: true },
                ...RIGOR.homCtrls.map((c) => ({ a: c.a + " (1 автор)", v: c.auc, hi: false })),
                { a: "ШОЛОХОВ (все его книги)", v: RIGOR.homSholohov, hi: true },
                { a: "ориентир: разные авторы", v: RIGOR.homCeil, hi: false, anchor: true },
              ].sort((x, y) => x.v - y.v).map((r) => (
                <div key={r.a} style={{ display: "grid", gridTemplateColumns: "20ch 1fr 4ch", alignItems: "center", gap: 8, padding: "2px 0" }}>
                  <span style={{ fontSize: 12.5, color: r.hi ? "var(--text)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400, fontStyle: r.anchor ? "italic" : "normal" }}>{r.a}</span>
                  <MeterBar value={r.v - 0.5} max={homTop - 0.5} accent={r.hi ? "var(--icon-blue)" : r.anchor ? "var(--border-strong)" : "var(--gold)"} />
                  <span className="mono" style={{ fontSize: 11, color: r.hi ? "var(--text)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
              <p className="callout" style={{ margin: 0 }}>
                Книги Шолохова отделимы друг от друга (уверенность различения&nbsp;{fmtScore(RIGOR.homSholohov, 3)}) — но{" "}
                <strong style={{ color: "var(--text)" }}>Бунин</strong>, бесспорно один автор, ещё{" "}
                <strong style={{ color: "var(--text)" }}>отделимее</strong> ({fmtScore(RIGOR.homCtrls[0].auc, 3)}). Высокая
                «разнокнижность» — не подпись разных рук, а свойство автора с долгой, разнообразной
                карьерой (плюс признаки чувствительны к жанру: даже две половины одной книги дают {fmtScore(RIGOR.homFloor, 3)}).
              </p>
              <p className="note" style={{ margin: 0 }}>
                Внутри «Тихого Дона» 4 тома похожи друг на друга (уверенность различения&nbsp;<strong style={{ color: "var(--icon-blue)" }}>{fmtScore(RIGOR.homTdInternal, 3)}</strong>){" "}
                <strong style={{ color: "var(--text)" }}>больше</strong>, чем на остальные его работы ({fmtScore(RIGOR.homSholohov, 3)}) —
                согласуется с одной рукой на весь роман. {RIGOR.homNStay}/{RIGOR.homNWorks} работ тяготеют к самому Шолохову.
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
            ({CONSISTENCY.prutkov}) — в <strong style={{ color: "var(--cinnabar)" }}>×{CONSISTENCY.prutkovRatio}</strong> выше.
            Его книги собираются в два рыхлых сгустка — и это донской и советский <em>материал</em>, а не разные руки
            (совпадение разбиений с материалом&nbsp;{RIGOR.cxAriDonskoy}). Тот же вывод, что по разбросу и отделимости.
          </p>

          {/* Может ли тест поймать подделку вообще: контроли + скрытый позитив */}
          <div className="reveal" style={{ marginTop: 34 }}>
            <h4 style={{ marginBottom: 6 }}>А поймал бы тест подделку?</h4>
            <p className="prose muted" style={{ maxWidth: "76ch", marginBottom: 16 }}>
              Чтобы вывод «→ Шолохов» что-то значил, тест обязан <em>уметь</em> ловить чужую руку. Проверяем на заведомых
              подделках. Склейку из трёх <strong style={{ color: "var(--text)" }}>разных</strong> авторов метод{" "}
              <strong style={{ color: "var(--text)" }}>ловит</strong>: единому автору не приписался ни один кусок
              ({MULTIHANDS.fakeDifferentCaught} → «свой»), различимость {fmtScore(MULTIHANDS.fakeDifferent, 3)}. Но склейка{" "}
              <strong style={{ color: "var(--text)" }}>похожих</strong> донских авторов (Крюков+Серафимович+Севский, {MULTIHANDS.fakeSimilar})
              от одиночки <strong style={{ color: "var(--cinnabar)" }}>неотличима</strong> — она даже ниже Шолохова
              ({MULTIHANDS.sholokhovSep}). А это и есть реалистичный сценарий «литературных негров».
            </p>
            <div className="split" style={{ alignItems: "start" }}>
              <div>
                <p className="eyebrow" style={{ marginBottom: 4 }}>Где пряталась бы вторая рука?</p>
                <p className="muted" style={{ fontSize: 12, margin: "0 0 10px" }}>
                  подмешиваем реального Крюкова к Шолохову → порог обнаружения ~{MULTIHANDS.hiddenPositive.flagThreshold}%; куда попадают реальные работы:
                </p>
                {MULTIHANDS.hiddenPositive.calib.map((c) => {
                  const over = c.pct >= MULTIHANDS.hiddenPositive.flagThreshold;
                  const near = c.pct >= 25;
                  return (
                    <div key={c.g} style={{ display: "grid", gridTemplateColumns: "16ch 1fr 6ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                      <span style={{ fontSize: 12, color: near ? "var(--gold)" : "var(--text-muted)", fontWeight: near ? 700 : 400 }}>{c.g}</span>
                      <span style={{ height: 9, borderRadius: 4, background: "var(--surface-sunken)", overflow: "hidden", position: "relative" }}>
                        <span style={{ display: "block", height: "100%", width: `${c.pct}%`, background: over ? "var(--cinnabar)" : near ? "var(--gold)" : "var(--text-muted)" }} />
                        <span style={{ position: "absolute", left: `${MULTIHANDS.hiddenPositive.flagThreshold}%`, top: -2, bottom: -2, width: 1, background: "var(--cinnabar)", opacity: 0.6 }} />
                      </span>
                      <span className="mono" style={{ fontSize: 10, color: over ? "var(--cinnabar)" : "var(--text-muted)" }}>~{c.pct}%</span>
                    </div>
                  );
                })}
                <div className="mono muted" style={{ fontSize: 10, marginTop: 6 }}>
                  ┊ красная черта — порог {MULTIHANDS.hiddenPositive.flagThreshold}%. «Война» (~{warPct}%) сидит ровно под ним.
                </div>
              </div>
              <p className="callout" style={{ margin: 0 }}>
                Ни одна работа не переходит порог, а «не-свои» фрагменты <strong style={{ color: "var(--text)" }}>рассыпаны</strong>,
                а не собраны в одну руку (война: Крюков ≈ Бунин ≈ Достоевский; «Тихий Дон» ведёт <em>Горький</em>, не донской).
                Сосредоточенной «второй руки» нет. Но «война» (≈{warPct}%) — у самого порога, поэтому <em>частичный</em> вклад
                стилистически <strong style={{ color: "var(--text)" }}>похожего</strong> донского соавтора в самые расходящиеся
                работы метод исключить не может. <span className="mono muted" style={{ fontSize: 10 }}>(доля дрожит ±{MULTIHANDS.hiddenPositive.runNoise})</span>
              </p>
            </div>
          </div>
          <SimilarHandLimit />
        </div>

        {/* 5e. Поиск чистого от темы признака → dependency */}
        <div className="reveal module">
          <h3>Тест №5 · Признак без темы</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 16 }}>
            Признаки, нейтральные к теме, упирались в потолок: они несли жанр. Можно ли
            найти такой, что ловит <strong style={{ color: "var(--text)" }}>привычку автора</strong>, а не{" "}
            <em>о чём</em> он пишет? Первый кандидат (DSP — профиль словообразовательных суффиксов) проверки не
            выдерживает: его «чистота» — побочный эффект, ведь жанр там мерили <em>внутри одного автора</em>. Для строгой проверки в корпусе есть{" "}
            <strong style={{ color: "var(--text)" }}>проза из общественного достояния</strong>: военная ({RIGOR.faWarAuthors}{" "}
            авторов — Толстой, Гаршин, Фурманов…) и сельская ({RIGOR.faRuralAuthors} — Тургенев, Короленко,
            Бунин…). Так «жанр» отделяется от «автора»:
            признак чист, если различает автора, но <em>не</em> отличает войну от деревни у <strong style={{ color: "var(--text)" }}>чужих</strong> авторов.
          </p>
          <TestSummary>
            На самом устойчивом к теме признаке — синтаксических связях — «Тихий Дон» склоняется
            к Шолохову отчётливее, чем на любом другом. Но по целым книгам запас всё ещё дотягивается до ничьей.
          </TestSummary>
          <div className="split" style={{ alignItems: "start" }}>
            <div>
              <div className="mono muted" style={{ fontSize: 11, marginBottom: 8 }}>
                по горизонтали: ◼ различает АВТОРА (выше — лучше) · ◻ путает с жанром поперёк чужих авторов (ниже — лучше)
              </div>
              {RIGOR.fa2.map((r) => (
                <div key={r.feat} style={{ display: "grid", gridTemplateColumns: "15ch 1fr 4ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                  <span style={{ fontSize: 11.5, color: r.idi > 0.45 ? "var(--text)" : "var(--text-muted)", fontWeight: r.idi > 0.45 ? 700 : 400 }}>{r.feat}</span>
                  <span style={{ position: "relative", height: 13, background: "var(--surface-sunken)", borderRadius: 3 }}>
                    <span style={{ position: "absolute", left: 0, top: 1, height: 5, width: `${r.author * 100}%`, background: r.idi > 0.45 ? "var(--icon-blue)" : "var(--text-muted)", borderRadius: 2 }} title={`автор ${r.author}`} />
                    <span style={{ position: "absolute", left: 0, bottom: 1, height: 5, width: `${r.genreXA * 100}%`, background: "var(--cinnabar)", opacity: 0.55, borderRadius: 2 }} title={`жанр ${r.genreXA}`} />
                  </span>
                  <span className="mono" style={{ fontSize: 10.5, color: r.idi > 0.45 ? "var(--icon-blue)" : "var(--text-muted)" }}>+{fmtScore(r.idi)}</span>
                </div>
              ))}
              <p className="callout">
                Победитель — <strong style={{ color: "var(--text)" }}>синтаксические связи</strong>:
                различает автора с уверенностью&nbsp;<strong style={{ color: "var(--icon-blue)" }}>{fmtScore(RIGOR.fa2[0].author)}</strong>, но войну
                от деревни у чужих авторов почти не видит (<strong style={{ color: "var(--text)" }}>{fmtScore(RIGOR.fa2[0].genreXA)}</strong> —
                ниже случайного угадывания). Глубокая грамматическая привычка не зависит от темы. А{" "}
                <strong style={{ color: "var(--text)" }}>DSP — в самом низу</strong>: жанр он ловит ({fmtScore(RIGOR.fa2.at(-1).genreXA)}) почти
                так же, как автора ({fmtScore(RIGOR.fa2.at(-1).author)}) — на расширенном наборе он не чище прочих, это сигнал жанра, а не личного почерка речи.
              </p>
            </div>
            <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
              <div className="mono muted" style={{ fontSize: 11 }}>
                «Тихий Дон» на <strong>чистом</strong> признаке (набор: синтаксис + части речи + связи), ось на отложенных текстах:
              </div>
              {[
                { a: "эталон: ранние рассказы Шолохова", v: RIGOR.caEnsShRef, kind: "sh" },
                { a: "«Поднятая целина» (контроль)", v: RIGOR.caEnsPc, kind: "ctrl" },
                { a: "«Тихий Дон» (спорный)", v: RIGOR.caEnsTd, kind: "td" },
                { a: "эталон: проза Крюкова", v: RIGOR.caEnsKrRef, kind: "kr" },
              ].map((r) => (
                <div key={r.a} style={{ display: "grid", gridTemplateColumns: "1fr 4ch", alignItems: "center", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 12, color: r.kind === "td" ? "var(--text)" : "var(--text-muted)", fontWeight: r.kind === "td" ? 700 : 400, marginBottom: 3 }}>{r.a}</div>
                    <span style={{ display: "block", height: 8, borderRadius: 4, background: "var(--surface-sunken)", position: "relative", overflow: "hidden" }}>
                      <span style={{ position: "absolute", left: `${RIGOR.caEnsMid * 100}%`, top: 0, bottom: 0, width: 1, background: "var(--cinnabar)" }} title="середина оси" />
                      <span style={{ display: "block", height: "100%", width: `${r.v * 100}%`, background: r.kind === "td" ? "var(--icon-blue)" : r.kind === "kr" ? "var(--cinnabar)" : r.kind === "sh" ? "var(--gold)" : "var(--border-strong)" }} />
                    </span>
                  </div>
                  <span className="mono" style={{ fontSize: 11, color: r.kind === "td" ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v)}</span>
                </div>
              ))}
              <p className="callout" style={{ margin: 0 }}>
                Ось <strong style={{ color: "var(--text)" }}>широкая</strong> (Шолохов&nbsp;{RIGOR.caEnsShRef} ↔
                Крюков&nbsp;{RIGOR.caEnsKrRef}), и «Тихий Дон» ({RIGOR.caEnsTd}) сидит <strong style={{ color: "var(--icon-blue)" }}>заметно
                на стороне Шолохова</strong>, контроль «Целина» проходит ({RIGOR.caEnsPc}). Это{" "}
                <em>сильнее</em>, чем давал DSP у середины. Покнижно на чистом синтаксисе: кн.1 уверенно Шолохов
                ({RIGOR.caEnsTdBooks[0].p}), кн.4 — ничья ({RIGOR.caEnsTdBooks[3].p}). В покнижной проверке с исключением проверяемых работ из обучения
                (Тест&nbsp;№3) слабым выходит, наоборот, первый том — какой том «шатается», зависит от набора признаков, и это ожидаемо.
              </p>
            </div>
          </div>
          <p className="verdict">
            Вывод по чистому признаку: на синтаксисе — самом устойчивом к теме сигнале (жанр&nbsp;{fmtScore(RIGOR.fa2[0].genreXA)}), на
            расширенном корпусе — «Тихий Дон» <strong style={{ color: "var(--text)" }}>склоняется к Шолохову
            отчётливее</strong>, чем на любом другом признаке ({RIGOR.caEnsTd} на оси 0.10–0.94; усреднённый профиль:{" "}
            {fmtPct(RIGOR.caEnsCentTdFracPos, 0)} пересчётов к Шолохову). Это укрепляет «за Шолохова» и ещё
            сильнее давит «Крюкова». Но по целым книгам, а их всего {RIGOR.bcTdNbooks}, разброс правдоподобных значений всё ещё{" "}
            <strong style={{ color: "var(--cinnabar)" }}>включает 0</strong> (от {RIGOR.caEnsCentTdCiLo} до {RIGOR.caEnsCentTdCiHi}),
            кн.4 — ничья, а на одних только синтаксических связях контроль «Целины» на грани. «Склоняется» — да; «доказано» — нет.
          </p>
        </div>

        {/* 5f. Рукопись: глубина авторской правки (палеография через VertexAI) */}
        <div className="reveal module">
          <h3>Рукопись · глубина авторской правки</h3>
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
              <div className="mono muted" style={{ fontSize: 11, marginBottom: 8 }}>
                средняя глубина правки (1 — почти чисто · 5 — сплошь переписано), случайные страницы:
              </div>
              {MS.rows.map((r) => (
                <div key={r.name} style={{ display: "grid", gridTemplateColumns: "22ch 1fr 4ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                  <span style={{ fontSize: 12, color: r.isTarget ? "var(--text)" : "var(--text-muted)", fontWeight: r.isTarget ? 700 : 400 }}>
                    {r.name} <span className="mono muted" style={{ fontWeight: 400 }}>n={r.n}</span>
                  </span>
                  <MeterBar value={r.mean} max={5} accent={r.isTarget ? "var(--icon-blue)" : "var(--gold)"} />
                  <span className="mono" style={{ fontSize: 11, color: r.isTarget ? "var(--icon-blue)" : "var(--text-muted)" }}>{r.mean}</span>
                </div>
              ))}
              <p className="mono muted" style={{ fontSize: 10.5, marginTop: 8 }}>
                доля страниц со «структурной» переработкой: {MS.rows.map((r) => `${r.name.split(" ")[0]} ${fmtPct(r.structFrac, 0)}`).join(" · ")}
              </p>
            </div>
            <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
              <p className="verdict" style={{ margin: 0 }}>
                Предпосылка «правок нет» <strong style={{ color: "var(--text)" }}>неверна</strong>: черновик ТД правлен активно.
                Но правка у Шолохова — <strong style={{ color: "var(--text)" }}>самая лёгкая</strong> из четырёх (средняя
                {" "}{MS.test.shMean} против {MS.test.ctrlMean} у контролей), и <strong style={{ color: "var(--text)" }}>ни на одной</strong>
                {" "}из {MS.test.shN} страниц она не доходит до сплошной переработки (максимум — уровень фразы), тогда
                как у всех трёх классиков такие листы есть.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Но копирование это <strong style={{ color: "var(--text)" }}>не доказывает</strong>: разница средних статистически
                незначима (тест Манна–Уитни — проверка, различаются ли две группы; p&nbsp;{fmtP(MS.test.p)}, размер эффекта средний d&nbsp;=&nbsp;{MS.test.cohenD}), распределения
                перекрываются, а «ошибок переписчика» — основного признака копирования — не нашлось ни у кого. Более лёгкая правка
                совместима и с обдумыванием «в уме», и с тем, что сохранившийся автограф — уже не первый черновик.
              </p>
            </div>
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 12, maxWidth: "82ch" }}>
            Оговорки: оценка 1–5 грубая и субъективная (модель зрения, не текстолог); по {msPagesMin}–{msPagesMax} страниц на
            автора, по одному-двум произведениям; наборы контролей смещены (у Достоевского взяты страницы с набросками → доля «схем»
            завышена). Это наблюдение про <em>потолок</em> правки, а не приговор об авторстве.
          </p>
        </div>

        {/* 6. Вердикт */}
        <div className="reveal module">
          <h3>Вердикт</h3>
          <div className="split" style={{ alignItems: "start" }}>
            <div className="prose">
              <p className="callout" style={{ marginTop: 0 }}>
                <strong style={{ color: "var(--text)" }}>Если коротко.</strong> Данные{" "}
                <strong style={{ color: "var(--gold)" }}>склоняются к Шолохову</strong> и не поддерживают ни «писал Крюков»,
                ни «много литературных негров». Но <strong style={{ color: "var(--text)" }}>доказать</strong> авторство
                анализ стиля не может: независимых крупных книг по сути всего две, а эталоном служит сам Шолохов.
              </p>
              <p className="verdict">
                <strong style={{ color: "var(--text)" }}>За Шолохова — ключевой тест без утечки проверяемых произведений.</strong>{" "}
                Проверка по целым книгам, где все спорные тома и донские контроли разом вынуты из обучения (опора — лишь
                оставшиеся работы с корпусной меткой «Шолохов»), относит <strong style={{ color: "var(--success)" }}>{RIGOR.loboTd.tdAttrib} тома → Шолохову</strong>{" "}
                при нулевой доле ложных срабатываний на донских контролях. Это исключает утечку самих проверяемых
                произведений, но не замкнутость эталона по меткам опорных текстов. Против «много рук» — попарная проверка авторства
                против {MULTIHANDS.avMultiHand.nPos} смесей под чужими именами: корпус Шолохова неотличим от одного автора
                (z&nbsp;=&nbsp;{MULTIHANDS.avMultiHand.zPseudo}, ~5 обычных разбросов, p&nbsp;{fmtP(MULTIHANDS.avMultiHand.permP)}). А доля
                «чужих» отрывков по томам падает к финалу — преобладающая рука крепнет к концу романа:
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, margin: "6px 0 4px" }}>
                {RIGOR.loboTd.gradient.map((g) => (
                  <div key={g.book} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 3 }}>{g.book}</div>
                    <MeterBar value={g.ff} accent={g.ff < 0.1 ? "var(--success)" : g.ff < 0.3 ? "var(--gold)" : "var(--cinnabar)"} />
                    <div className="mono" style={{ fontSize: 11, marginTop: 3, color: "var(--text)" }}>{fmtPct(g.ff, 0)}</div>
                  </div>
                ))}
              </div>
              <p className="muted" style={{ fontSize: 12.5, marginTop: 0 }}>
                «Чужая» доля первого тома ({RIGOR.loboTd.gradient[0].ff}) значимо выше фона (p&nbsp;{fmtP(RIGOR.loboTd.td1PermP)},
                блоками соседних отрывков p&nbsp;{fmtP(RIGOR.tdLoboBlockP)}); донские контроли дают ноль ложных срабатываний
                ({RIGOR.loboTd.donFpr}). Спад к четвёртому тому ({RIGOR.loboTd.gradient[0].ff}&nbsp;→&nbsp;{RIGOR.loboTd.gradient[3].ff}) —
                описание картины, не отдельный тест. Поэтому «склоняется к Шолохову» — не подбрасывание монеты, а согласованная
                картина нескольких независимых тестов без утечки проверяемых произведений, но с оговорённой ниже зависимостью от меток эталона.
              </p>
              <p>
                <strong style={{ color: "var(--text)" }}>Где проходит граница.</strong> По целым книгам, а их всего{" "}
                {RIGOR.bcTdNbooks}, разброс правдоподобных значений перевеса (от {RIGOR.bcTdCiLo} до {RIGOR.bcTdCiHi}){" "}
                <strong style={{ color: "var(--cinnabar)" }}>включает 0</strong> — формально неотличимо от ничьей
                ({fmtPct(RIGOR.bcTdFracPos, 0)} пересчётов к Шолохову). Самая строгая проверка при выровненном жанре и вовсе
                даёт почти ничью ({RIGOR.gmlrTdShFull} vs {RIGOR.gmlrTdKrFull}), а бесспорную «Поднятую целину» ошибочно относит
                к Крюкову ({RIGOR.gmlrPcKrFull}). Доказать авторство нельзя: эталон = сам Шолохов (частично замкнутый круг),
                а автора и редактора изнутри не разделить. Авторство <strong style={{ color: "var(--text)" }}>не доказано</strong> —
                куда именно упираются границы, ниже.
              </p>
              <details style={{ margin: "8px 0 4px" }}>
                <summary style={{ cursor: "pointer", color: "var(--icon-blue)", fontSize: 13.5, fontWeight: 600 }}>
                  Открытые пределы и прежние работы (подробно)
                </summary>
              <p className="muted" style={{ marginTop: 12 }}>
                Чего утверждать <em>нельзя</em> — открытые пределы:
              </p>
              <ul className="muted" style={{ lineHeight: 1.6, paddingLeft: "1.1em" }}>
                <li><strong style={{ color: "var(--text)" }}>Два объяснения не разделить (зазор сужен, но не снят):</strong> «Шолохов
                  писал сам» и «единый редактор переработал чужой материал» внутренними тестами на цельность
                  <em>неразличимы</em>. Но <em>названный</em> кандидат-правщик проверен: рука{" "}
                  <strong style={{ color: "var(--text)" }}>Серафимовича</strong> (покровитель, продвигавший ТД) в «Тихом
                  Доне» <strong style={{ color: "var(--text)" }}>не обнаруживается</strong> — ТД ближе к раннему Шолохову
                  ({RIGOR.serafEdShDon}), чем к Серафимовичу ({RIGOR.serafEdSeraf}) или Крюкову ({RIGOR.serafEdKrukov}).
                  Остаются лишь литературные редакторы без сохранившихся текстов — их рукой проверить нельзя.</li>
                <li><strong style={{ color: "var(--text)" }}>Примесь жанра (ослаблена, не устранена):</strong> у
                  признаков по словам военная vs сельская проза делится поперёк авторов с уверенностью&nbsp;{RIGOR.crossGenreAuc}.
                  На расширенном корпусе устойчивый к теме признак — это{" "}
                  <strong style={{ color: "var(--icon-blue)" }}>синтаксические связи</strong> (Тест&nbsp;№5: автор&nbsp;{fmtScore(RIGOR.fa2[0].author)},
                  жанр&nbsp;{fmtScore(RIGOR.fa2[0].genreXA)}). На нём ТД склоняется к Шолохову. Но и он не идеален ({fmtScore(RIGOR.fa2[0].genreXA)}&nbsp;≠&nbsp;0), а{" "}
                  DSP на расширенном наборе — <em>среди худших</em>: независимость от темы даётся не даром.</li>
                <li><strong style={{ color: "var(--cinnabar)" }}>Замкнутый круг с эталоном (важно):</strong> ТД ближе
                  всего к <em>ранним донским рассказам</em> Шолохова (1924–26) — но именно этот период входит в спорную зону.
                  Решающий тест: обучаем на <em>бесспорно поздней</em> прозе (война+ПЦ-2, 1942–69) против Крюкова и проецируем.
                  Результат <em>смешанный</em>: поздний Шолохов узнаёт ТД ({RIGOR.circTd}) <em>примерно как
                  собственные ранние рассказы</em> ({RIGOR.circEarly}) — то есть ТД не <em>менее</em> шолоховский, чем
                  ранняя проза, но сам сигнал слаб (мешает разрыв в жанре). «ТД = Шолохов» нельзя доказать, не
                  предположив, что и ранние рассказы его — а бесспорной донской опоры вне обсуждаемого периода в природе нет.</li>
                <li><strong style={{ color: "var(--text)" }}>Ограниченная чувствительность:</strong> похожего по
                  стилю соавтора, давшего меньшую часть текста, тесты не различают; порог у каждой проверки свой —{" "}
                  <span className="mono" style={{ color: "var(--text)" }}>целые книги ~{RIGOR.loboTd.minAdmix}% · кривая примеси ~{RIGOR.powerDetectK}% · скрытая рука ~{MULTIHANDS.hiddenPositive.flagThreshold}%</span>.</li>
                <li>Кумов и Севский непроверяемы (мало текста); заимствование сырого донского материала
                  тесты по стилю не закрывают.</li>
                <li><strong style={{ color: "var(--text)" }}>Закрытый список кандидатов:</strong> атрибуция по целым
                  книгам ({RIGOR.tdLoboAttributed} тома → Шолохову) держится внутри короткого списка «Шолохов, Крюков
                  или Серафимович». Если убрать короткий список и открыть выбор на весь корпус авторов, поздние тома всё равно
                  уверенно уходят к Шолохову (ТД-4: {fmtScore(RIGOR.openSetTd.td4Share, 3)}), а ранние — нет
                  (ТД-1: {fmtScore(RIGOR.openSetTd.td1Share, 3)}, больше всего — к {RIGOR.openSetTd.td1TopName} с долей{" "}
                  {fmtScore(RIGOR.openSetTd.td1TopShare, 3)}: эпический роман тянет к романисту-эпику). Что список не
                  «засасывает» чужого — проверено: подброшенный {RIGOR.platonovInject.name} уходит к самому себе
                  ({fmtScore(RIGOR.platonovInject.selfShare, 1)}), к Шолохову — {fmtScore(RIGOR.platonovInject.toSholokhovShare, 1)}.
                  Отдельная проверка «а тот ли это автор вообще» (модель учится узнавать именно почерк Шолохова и
                  отвергать чужих) на «Тихом Доне» даёт {RIGOR.verifTd.tdAttributed} — но в такой постановке она
                  перекошена: бесспорные зрелые вещи самого Шолохова тоже её не проходят, поэтому вывод «ТД не Шолохов»
                  из неё не следует.</li>
              </ul>
              <p className="muted">
                <strong style={{ color: "var(--text)" }}>Соотнесение с предшественниками.</strong> Компьютерный анализ
                Хьетсо и коллег (1984) пришёл к тому же направлению: Шолохов, не Крюков. К его методике известны
                претензии — узкий набор признаков (длины предложений, частотные распределения), отсутствие жанрового
                контроля и проверки замкнутого круга эталона. Здесь эти претензии проверены отдельными тестами: проверка по
                целым книгам без утечки проверяемых произведений, но с сохраняющейся зависимостью от меток эталона; нулевой фон ложных срабатываний на донских контролях,
                контроль жанра и темы, открытый список кандидатов. Совпадение направления у двух независимых методик —
                самостоятельный аргумент.
              </p>
              </details>
              <p className="verdict">
                Итог: данные <strong style={{ color: "var(--gold)" }}>совместимы</strong> с авторством Шолохова и
                складываются в согласованную картину «писал он сам». Но превратить это в «доказано» анализ стиля
                не может — и здесь такой вывод не делается.
              </p>
            </div>
            <div style={{ display: "grid", placeItems: "center", gap: 14 }}>
              <AnomalyGlyph kind="relation_mismatch" size={52} />
              <span className="muted mono" style={{ fontSize: 12.5, textAlign: "center", maxWidth: "26ch" }}>
                «один автор» и «один редактор»<br />изнутри неразличимы
              </span>
            </div>
          </div>
        </div>

        <Sources
          items={[
            { cite: "Черновой автограф «Тихого Дона» — отдел рукописей ИМЛИ РАН (по материалам ФЭБ)", url: "http://feb-web.ru/feb/sholokh/" },
            { cite: "Проза кандидатов (Крюков, Серафимович и др.), военная и сельская проза — открытые публикации az.lib.ru; используется локально для расчётов и не распространяется", url: "http://az.lib.ru/" },
          ]}
          note="Палеографическая оценка правки рукописи — мультимодальной моделью Gemini 3.1 Pro через VertexAI."
        />
      </div>
    </section>
  );
}
