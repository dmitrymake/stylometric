import { Card, Stat, Badge } from "@dmitrymake/rk-ui";
import MeterBar from "../components/MeterBar.jsx";
import { fmtScore, fmtP, fmtRange } from "../format.js";
import { LIMITS } from "../segdata.js";

// Честный протокол: метрический урок (единица голоса) + карта режимов
// (где метод делит руки, где честно отказывает). Все числа — из LIMITS, формат — format.js.

const T = LIMITS.threshold; // порог надёжной атрибуции (0.80)

const SEP_ACCENT = { sovremennik: "var(--success)", kolokol: "var(--gold)", petersburg: "var(--icon-blue)" };
const LIM_ACCENT = { nekrasov: "var(--cinnabar)", pair: "var(--gold)", chekhonte: "var(--cosmos)" };

// русское склонение существительного после числа: форма следует за данными, а не хардкодится под падеж
const plural = (n, one, few, many) => {
  const t = Math.abs(n) % 100, o = t % 10;
  if (t >= 11 && t <= 14) return many;
  if (o === 1) return one;
  if (o >= 2 && o <= 4) return few;
  return many;
};

// оговорки карточек приходят из данных и читаются простым языком: научные обороты
// заменяются бытовыми словами, а число объёма остаётся из данных — переформатируется
// только запись «Nk слов» → «N тыс. слов» (сам N берётся из строки, не хардкодится).
const plainCaveat = (text = "") =>
  text
    .replace(
      "позитив-контроль выполнимости и тай-брейк к филологии по одноавторским передовым",
      "проверка, что руки в принципе делятся, и дополнительный довод к текстологии по передовым одного автора",
    )
    .replace(
      "не разрешение атрибуции соавторских текстов",
      "а не установление авторства текстов, написанных вдвоём",
    )
    .replace("часть прогона — калибровка", "часть расчёта — калибровочная")
    .replace(/(\d+)\s*k(?=\s*слов)/gi, "$1 тыс.")
    .replace("спорного Н.Н.", "спорного фельетона под подписью «Н.Н.»");

// что подтверждает разделение в каждом «работает»-кейсе
const SEP_NOTE = {
  sovremennik: "Критики двух школ одного журнала и одной эпохи делятся идеально: каждый отложенный на проверку текст уходит к своей школе.",
  kolokol: "Герцен и Огарёв десятилетиями правили тексты друг друга, профили почти слиты — и руки всё равно делятся.",
  petersburg: "При каждой пересборке выборки фельетоны Ф.Д. снова ближе к публицистике Достоевского. Результат на самой грани, с оговоркой на сдвиг эпохи.",
};
// почему именно метод отказывает в каждом «предел»-кейсе
const LIM_NOTE = {
  nekrasov: "По служебным словам, не зависящим от темы, руки соавторов неразличимы; деление появляется только на содержании — значит несёт тему, а не почерк.",
  pair: "Пару учитель↔ученик внутри одной школы метод держит ровно у порога. Проверка на случайность перевес не подтверждает: руки сошлись слишком близко.",
  chekhonte: "Образцы авторов — из «Осколков», спорная подборка — из «Будильника»: журнал и рубрика смешиваются с почерком, а оригиналы 1885 года не оцифрованы.",
};

// одна строка метрики внутри карточки
function Row({ label, value, sub, accent, good, bad }) {
  const color = bad ? "var(--cinnabar)" : good ? "var(--success)" : accent || "var(--text)";
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "baseline", borderTop: "1px solid color-mix(in srgb, var(--line) 40%, transparent)", paddingTop: 8 }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{label}</div>
        {sub && <div style={{ fontSize: 11.5, color: "var(--text-muted)", opacity: 0.78, marginTop: 2 }}>{sub}</div>}
      </div>
      <div className="mono" style={{ fontSize: 14, fontWeight: 600, whiteSpace: "nowrap", color }}>{value}</div>
    </div>
  );
}

// заголовочная доля с порогом 0.80: строго выше — зелёная «выше», ровно на пороге —
// нейтральная «у порога», ниже — красная «ниже». Равенство порогу не выдаётся за успех.
function MacroHead({ value, accent }) {
  const at = Math.abs(value - T) < 1e-9;
  const above = value > T && !at;
  const color = above ? "var(--success)" : at ? "var(--text)" : "var(--cinnabar)";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>доля верных опознаний (один текст — один голос)</span>
        <span className="mono" style={{ fontSize: 15, fontWeight: 700, color }}>
          {at ? `ровно ${fmtScore(value)} · у порога` : `${fmtScore(value)} · ${above ? "выше" : "ниже"} ${fmtScore(T)}`}
        </span>
      </div>
      <div style={{ position: "relative" }}>
        <MeterBar value={value} max={1} accent={accent} />
        <span title={`порог ${fmtScore(T)}`} style={{ position: "absolute", left: `${T * 100}%`, top: -2, bottom: -2, width: 2, background: "var(--cinnabar)" }} />
      </div>
      <div className="mono" style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
        0 — наугад · {fmtScore(T)} — порог надёжной атрибуции · 1 — всегда верно
      </div>
    </div>
  );
}

export default function Limits() {
  const cal = LIMITS.calibration;
  const nCases = LIMITS.separates.length + LIMITS.limitsCases.length;
  const nModes = LIMITS.limitsCases.length;
  return (
    <section className="section" id="limits">
      <div className="wrap flow">
        {/* ───────────────────────── шапка ───────────────────────── */}
        <div className="section-head reveal">
          <p className="eyebrow">Урок счёта и карта режимов</p>
          <h2>Где метод работает, а где нет</h2>
          <p className="prose lead muted">
            Хороший инструмент честно говорит о своих пределах. Один и тот же корпус — а вывод
            переворачивается, стоит иначе сосчитать голоса. Граница между «можно утверждать»
            и «может быть интересно» — и есть смысл честного протокола.
          </p>
          <div className="grid cols-2 reveal" style={{ maxWidth: 520 }}>
            <Stat label={`${plural(nCases, "кейс", "кейса", "кейсов")} на карте`} value={nCases} accent="var(--icon-blue)" parade />
            <Stat label={`${plural(nModes, "режим", "режима", "режимов")} отказа`} value={nModes} accent="var(--cinnabar)" hint="автор ≡ тема · сросшиеся руки · панель у случайности" />
          </div>
        </div>

        {/* ──────────────── метрический урок: единица голоса ──────────────── */}
        <div className="module reveal">
          <h3>Как считать голоса</h3>
          <p className="prose muted">
            У каждой проверки есть выбор: что считать одним голосом. Можно дать голос
            каждому отрывку — тогда толстая книга, нарезанная на сотню кусков, голосует
            сотню раз. А куски одного текста похожи и тянут в одну сторону, раздувая
            уверенность. Можно иначе: один голос — целому тексту, и тогда
            каждая книга весит одинаково. Так считать честно. Разница не косметическая:
            на одном и том же корпусе она переносит вывод через порог.
          </p>

          <div style={{ display: "grid", gap: 18, marginTop: "var(--beat-group)" }}>
            {LIMITS.metric.map((m) => {
              const accent = SEP_ACCENT[m.id] || LIM_ACCENT[m.id] || "var(--icon-blue)";
              const workAbove = m.work > T;
              return (
                <div key={m.id} style={{ display: "grid", gap: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text)" }}>{m.label}</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: 10, alignItems: "center" }}>
                    <span style={{ fontSize: 12.5, color: "var(--text-muted)", whiteSpace: "nowrap" }}>по кускам</span>
                    <MeterBar value={m.chunk} max={1} accent="color-mix(in srgb, var(--text-muted) 55%, transparent)" />
                    <span className="mono" style={{ fontSize: 13, color: "var(--cinnabar)", whiteSpace: "nowrap" }}>{fmtScore(m.chunk)}</span>
                    <span style={{ fontSize: 12.5, color: "var(--text-muted)", whiteSpace: "nowrap" }}>по текстам</span>
                    <div style={{ position: "relative" }}>
                      <MeterBar value={m.work} max={1} accent={accent} />
                      <span style={{ position: "absolute", left: `${T * 100}%`, top: -2, bottom: -2, width: 2, background: "var(--cinnabar)" }} />
                    </div>
                    <span className="mono" style={{ fontSize: 13, fontWeight: 700, color: workAbove ? "var(--success)" : "var(--text)", whiteSpace: "nowrap" }}>{fmtScore(m.work)}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="callout reveal" style={{ marginTop: "var(--beat-group)" }}>
            Современник по кускам даёт {fmtScore(LIMITS.metric[0].chunk)} — ниже порога,
            «неубедительно». По текстам — {fmtScore(LIMITS.metric[0].work)}, уверенное
            разделение. Корпус один; вывод решает единица голоса. Голос отдаём
            целому тексту.
          </p>
        </div>

        {/* ──────────────── калибровочная линейка ──────────────── */}
        <div className="module reveal">
          <h3>Как выглядит уверенное разделение</h3>
          <p className="prose muted">
            Числам нужна мера. Две пары известных авторов,
            прогнанные тем же протоколом, задают шкалу. Пары отличаются эпохой и
            регистром — типом речи: проза, критика, публицистика.
            Косинус показывает, насколько похожи усреднённые профили: 1 — совпадают,
            0 — совсем разные. У русской прозы косинусы высоки. Поэтому судить
            надо по доле верных опознаний, а косинус читать по этой линейке.
          </p>
          <div className="split" style={{ alignItems: "start", marginTop: "var(--beat-group)" }}>
            <Card padding={24}>
              <div style={{ display: "grid", gap: 20 }}>
                {[
                  { tag: "разные эпоха и регистр", accent: "var(--gold)", ...cal.easy },
                  { tag: "тот же регистр и эпоха", accent: "var(--icon-blue)", ...cal.medium },
                ].map((r, i) => (
                  <div key={i} style={{ display: "grid", gap: 6 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: r.accent }}>{r.tag}</span>
                      <span className="mono" style={{ fontSize: 13, color: "var(--success)" }}>доля {fmtScore(r.macro)}</span>
                    </div>
                    <MeterBar value={r.macro} max={1} accent={r.accent} />
                    <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)" }}>косинус профилей {fmtScore(r.cos, 3)}</div>
                  </div>
                ))}
              </div>
            </Card>
            <p className="prose muted" style={{ margin: 0 }}>
              Обе опорные пары уверенно проходят порог {fmtScore(T)}: доля {fmtScore(cal.easy.macro)}
              {" "}у авторов разных эпох и {fmtScore(cal.medium.macro)} даже у авторов одного
              регистра. Это образец уверенного разделения: известные разные руки в одном
              регистре дают долю примерно {fmtRange(cal.medium.macro, cal.easy.macro)}.
              С этим сравниваем кейсы карты. Чем ближе доля к этой полосе, тем крепче деление.
              Где она сползает к порогу {fmtScore(T)} и ниже — метод отказывает.
            </p>
          </div>
        </div>

        {/* ──────────────── карта: метод делит руки ──────────────── */}
        <div className="reveal">
          <h3>Где метод делит руки</h3>
          <p className="prose muted">
            На честном счёте «один текст — один голос» метод уверенно отделяет одну руку
            от другой даже там, где профили кажутся почти слитыми. У каждой карточки —
            вопрос, закрытый круг кандидатов и доля выше порога {fmtScore(T)}.
          </p>
        </div>
        <div className="grid cols-2 reveal" style={{ marginTop: "var(--beat-group)" }}>
          {LIMITS.separates.map((c) => {
            const accent = SEP_ACCENT[c.id] || "var(--text-muted)";
            return (
              <Card key={c.id} padding={24}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="case-kicker" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ width: 22, height: 2, background: accent }} />
                    <Badge className="case-badge" tone="success">делит руки · выше порога</Badge>
                  </div>
                  <h4 style={{ margin: 0, color: "var(--text)", fontSize: "1.12rem" }}>{c.title}</h4>
                  <p className="muted" style={{ margin: 0, fontSize: 14.5 }}>{c.question}</p>
                  <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.5 }}>
                    круг кандидатов: {c.candidates}
                  </div>
                  <div style={{ display: "grid", gap: 8, marginTop: 2 }}>
                    <MacroHead value={c.macro} accent={accent} />
                    {c.perm != null && <Row label="проверка на случайность" value={`p = ${fmtP(c.perm)}`} sub="перевес устойчив против случайной расстановки ярлыков" good />}
                    {c.boot != null && <Row label="доля побед при пересборках" value={fmtScore(c.boot)} sub="при каждой случайной пересборке выборки — снова к Достоевскому" accent={accent} />}
                    {c.cos != null && <Row label="косинус профилей" value={fmtScore(c.cos, 3)} sub="руки близки, и метод всё равно их делит" accent="var(--text)" />}
                    {c.dist != null && <Row label="отрыв от соседей" value={fmtScore(c.dist)} sub="мал — оговорка на близость к центру панели" accent="var(--text)" />}
                  </div>
                  <p className="note" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5 }}>{SEP_NOTE[c.id]}</p>
                  {c.caveat && (
                    <p className="note" style={{ margin: 0, fontSize: 12, lineHeight: 1.5, color: "var(--text-muted)", fontStyle: "italic" }}>
                      Оговорка: {plainCaveat(c.caveat)}
                    </p>
                  )}
                </div>
              </Card>
            );
          })}
        </div>

        {/* ──────────────── карта: метод честно отказывает ──────────────── */}
        <div className="reveal">
          <h3>Где метод честно отказывает</h3>
          <p className="prose muted">
            Отказ случается в {nModes} {plural(nModes, "режиме", "режимах", "режимах")} — и метод честно
            их показывает: когда видимое
            деление несёт тему, а не почерк; когда руки сошлись слишком близко; и когда
            сам набор образцов у случайности, а источник недоступен.
          </p>
        </div>
        <div className="grid cols-2 reveal" style={{ marginTop: "var(--beat-group)" }}>
          {LIMITS.limitsCases.map((c) => {
            const accent = LIM_ACCENT[c.id] || "var(--text-muted)";
            return (
              <Card key={c.id} padding={24}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="case-kicker" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ width: 22, height: 2, background: accent }} />
                    <Badge className="case-badge" tone="warning">предел · {c.reason}</Badge>
                  </div>
                  <h4 style={{ margin: 0, color: "var(--text)", fontSize: "1.12rem" }}>{c.title}</h4>
                  <p className="muted" style={{ margin: 0, fontSize: 14.5 }}>{c.question}</p>
                  <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.5 }}>
                    круг кандидатов: {c.candidates}
                  </div>
                  <div style={{ display: "grid", gap: 8, marginTop: 2 }}>
                    {/* автор ≡ тема: две дорожки — почерк не делит, содержание делит */}
                    {c.fwMacro != null && <Row label="по личному почерку (служебные слова)" value={fmtScore(c.fwMacro)} sub="ниже порога — руки соавторов неразличимы" bad />}
                    {c.fwPerm != null && <Row label="проверка на случайность (почерк)" value={`p = ${fmtP(c.fwPerm)}`} sub="не подтверждает — как при случайном разбиении" accent="var(--text)" />}
                    {c.char3Macro != null && <Row label="по содержанию (символьные триграммы)" value={fmtScore(c.char3Macro)} sub="делит уверенно — но это тема, не почерк" accent={accent} />}
                    {c.kappa != null && <Row label="согласие двух признаков (κ)" value={fmtScore(c.kappa, 2)} sub="чуть выше случая: почерк и тема размечают по-разному" accent="var(--text)" />}
                    {/* учитель↔ученик: ровно у порога, но незначимо */}
                    {c.macro != null && <MacroHead value={c.macro} accent={accent} />}
                    {c.perm != null && <Row label="проверка на случайность" value={`p = ${fmtP(c.perm)}`} sub="перевес у порога не подтверждается" bad />}
                    {/* панель у случайности */}
                    {c.recall != null && <Row label="верных опознаний хозяина" value={c.recall} sub="ровно половина отрезков — панель почти не различает авторов" bad />}
                    {c.cos != null && <Row label="косинус профилей" value={fmtScore(c.cos, 3)} sub="профили почти слиты" accent="var(--text)" />}
                    {c.dist != null && <Row label="отрыв от соседей" value={fmtScore(c.dist)} sub="близко к нулю — текст почти не отличить от соседних" accent="var(--text)" />}
                  </div>
                  <p className="note" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5 }}>{LIM_NOTE[c.id]}</p>
                </div>
              </Card>
            );
          })}
        </div>

        {/* ──────────────────── финальный вердикт ──────────────────── */}
        <p className="verdict reveal">
          Единица голоса — текст или кусок — решает, увидим мы разделение или
          «неубедительно». Честный счёт даёт один голос целому тексту. На честном
          счёте метод делит даже сросшиеся руки Герцена и Огарёва. Но отказывает
          там, где тема притворяется почерком, где руки сошлись внутри одной школы,
          где сам набор образцов у случайности. Ценность инструмента — в границе
          между «можно утверждать» и «может быть интересно».
        </p>
        <p className="note reveal">
          Кейсы-пределы остаются открытыми вопросами для архивов и текстологов. Деление по
          содержанию или перевес у порога — повод присмотреться, а не основание для вывода
          об авторстве.
        </p>
        <p className="note reveal">
          P-значения на карточках даны без поправки на число одновременных проверок. Для кейсов,
          где проверок несколько, поправка (метод Холма) посчитана отдельно — <span className="mono">docs/holm_correction.json</span>:
          часть отдельных проверок после поправки теряет формальную значимость, вердикты кейсов не меняются.
        </p>
      </div>
    </section>
  );
}
