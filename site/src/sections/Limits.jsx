import { Card, Badge } from "@dmitrymake/rk-ui";
import MeterBar from "../components/MeterBar.jsx";
import { fmtScore, fmtP } from "../format.js";
import { LIMITS } from "../segdata.js";

// Карта режимов контрольных панелей. Гейт выполнимости зарегистрирован заранее и требует двух условий
// сразу: средняя по классам полнота по работам >= T и перестановочная проверка на уровне работ p <= P_GATE.
// Числа — из LIMITS, формат — format.js.

const T = LIMITS.threshold; // рабочий порог этой карты (0.80) — первое условие гейта
const P_GATE = 0.05; // второе условие гейта: порог проверки, а не измеренная величина
const sc = (x) => fmtScore(x, 3); // измеренные доли — три знака, порог печатается как есть
const raw = (x) => String(x); // зафиксированные калибровочные значения — без округления

const ACCENT = {
  sovremennik: "var(--success)", petersburg: "var(--icon-blue)", nekrasov: "var(--cinnabar)",
  pair: "var(--gold)", kolokol: "var(--gold)", chekhonte: "var(--cosmos)",
};

// какое условие гейта нарушено — короткая подпись бейджа, без внутренних маркеров прогона
const GATE_MISS = {
  nekrasov: "вклад автора и темы не разделён",
  pair: "разделение не подтверждено",
  kolokol: "разделение не подтверждено",
  chekhonte: "полнота ниже рабочего порога",
};

// читательские примечания карточек: по id, компактно, без пересказа внутренних оговорок
const NOTE = {
  sovremennik: "Гейт пройден для конкретных критиков этой панели; перенос на «школу как класс» не показан. Боткин представлен одной работой, часть разметки основана на гонорарных ведомостях.",
  petersburg: "Панель проходит гейт, но спорный фельетон под подписью «Н.Н.» остаётся без атрибуции: его куски делятся 1:1 между публицистикой Достоевского и Панаевым.",
  nekrasov: "Группы признаков дают разные результаты, но этот дизайн не разделяет вклад автора и темы. Значение по символьным триграммам не читается как доказательство.",
  pair: "Панель не подтверждает разделение пары учитель ↔ ученик внутри одной школы.",
  kolokol: "Панель не подтверждает разделение. Невыполненное условие по перестановке не означает доказанного равенства авторов.",
  chekhonte: "Архивный заказ Курепина относится к заметке 24 мая, а не ко всей подборке из пяти текстов: сильный документальный кандидат в пользу Чехова, но не доказательство авторства всей подборки.",
};

// оба условия гейта читаются вместе: маленькое p само по себе гейт не проходит
const gateNote = (macro, p) => {
  if (macro == null || p == null) return null;
  const okMacro = macro >= T, okP = p <= P_GATE;
  if (okMacro && okP) return "оба условия гейта выполнены";
  if (okP) return "условие по перестановке выполнено, но средняя полнота ниже рабочего порога; гейт не пройден";
  if (okMacro) return "средняя полнота не ниже рабочего порога, но условие по перестановке не выполнено; гейт не пройден";
  return "ни одно из двух условий гейта не выполнено";
};
const gatePass = (macro, p) => macro != null && p != null && macro >= T && p <= P_GATE;
const permColor = (p) => (p <= P_GATE ? "var(--success)" : "var(--cinnabar)"); // цвет — только про условие p
const pLabel = (p) => `p ${fmtP(p).startsWith("<") ? "" : "= "}${fmtP(p)}`; // без «= < 0.001»

// одна строка метрики внутри карточки
function Row({ label, value, sub, color = "var(--text)" }) {
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

// первое условие гейта: выше порога — зелёная, ровно на пороге — нейтральная, ниже — красная
function MacroHead({ value, accent }) {
  const at = Math.abs(value - T) < 1e-9;
  const above = value > T;
  const color = above ? "var(--success)" : at ? "var(--text)" : "var(--cinnabar)";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>средняя по классам доля правильно опознанных работ</span>
        <span className="mono" style={{ fontSize: 15, fontWeight: 700, color }}>{sc(value)} · {at ? "ровно рабочий порог" : above ? "выше рабочего порога" : "ниже рабочего порога"}</span>
      </div>
      <div style={{ position: "relative" }}>
        <MeterBar value={value} max={1} accent={accent} />
        <span title={`рабочий порог ${fmtScore(T)}`} style={{ position: "absolute", left: `${T * 100}%`, top: -2, bottom: -2, width: 2, background: "var(--cinnabar)" }} />
      </div>
      <div className="mono" style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>0 — ни одной правильно опознанной работы · {fmtScore(T)} — рабочий порог этой карты · 1 — все работы опознаны правильно</div>
    </div>
  );
}

export default function Limits() {
  const cal = LIMITS.calibration;
  const sovr = LIMITS.metric.find((m) => m.id === "sovremennik");
  return (
    <section className="section" id="limits">
      <div className="wrap flow">
        {/* ───────────────────────── шапка ───────────────────────── */}
        <div className="section-head reveal">
          <p className="eyebrow">Границы метода</p>
          <h2>Что показывают контрольные панели</h2>
          <p className="prose lead muted">
            У каждой контрольной панели заранее записан гейт выполнимости из двух условий: средняя по классам
            доля правильно опознанных работ не ниже {fmtScore(T)} и проверка на случайность (перестановка ярлыков)
            на уровне работ с p ≤ {P_GATE}. Обязательны оба. Гейт говорит о панели, а не об отдельном спорном тексте внутри неё.
          </p>
        </div>

        {/* ──────────────── метрический урок: единица голоса ──────────────── */}
        <div className="module reveal">
          <h3>Как считать голоса</h3>
          <p className="prose muted">
            У каждой проверки есть выбор: что считать единицей оценки. Если голос даётся каждому куску,
            толстая книга, нарезанная на сотню кусков, голосует сотню раз, а куски одного текста похожи и тянут
            в одну сторону. Протокол этой карты сначала строит профиль каждой работы, затем даёт работам равный вес
            в усреднённом профиле автора и один голос на проверяемый текст. Метрика гейта — средняя по классам доля
            правильно опознанных работ, где каждый класс получает одинаковый вес. Значения по кускам остаются
            диагностикой: в них длинные работы получают больше голосов.
          </p>

          <div style={{ display: "grid", gap: 18, marginTop: "var(--beat-group)" }}>
            {LIMITS.metric.map((m) => (
              <div key={m.id} style={{ display: "grid", gap: 8 }}>
                <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text)" }}>{m.label}</span>
                <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: 10, alignItems: "center" }}>
                  <span style={{ fontSize: 12.5, color: "var(--text-muted)", whiteSpace: "nowrap" }}>по работам · метрика гейта</span>
                  <div style={{ position: "relative" }}>
                    <MeterBar value={m.work} max={1} accent={ACCENT[m.id] || "var(--icon-blue)"} />
                    <span style={{ position: "absolute", left: `${T * 100}%`, top: -2, bottom: -2, width: 2, background: "var(--cinnabar)" }} />
                  </div>
                  <span className="mono" style={{ fontSize: 13, fontWeight: 700, color: m.work > T ? "var(--success)" : "var(--text)", whiteSpace: "nowrap" }}>{sc(m.work)}</span>
                  <span style={{ fontSize: 12.5, color: "var(--text-muted)", whiteSpace: "nowrap" }}>по кускам · диагностика</span>
                  <MeterBar value={m.chunk} max={1} accent="color-mix(in srgb, var(--text-muted) 55%, transparent)" />
                  <span className="mono" style={{ fontSize: 13, color: "var(--text-muted)", whiteSpace: "nowrap" }}>{sc(m.chunk)}</span>
                </div>
              </div>
            ))}
          </div>

          {sovr && (
            <p className="callout reveal" style={{ marginTop: "var(--beat-group)" }}>
              У «Современника» метрика гейта по работам — {sc(sovr.work)}, диагностика по кускам — {sc(sovr.chunk)}. Корпус один: расходятся единицы счёта. К условиям гейта диагностическое значение не применяется.
            </p>
          )}
        </div>

        {/* ──────────────── опорные примеры протокола ──────────────── */}
        <div className="module reveal">
          <h3>Опорные примеры протокола</h3>
          <div className="split" style={{ alignItems: "start", marginTop: "var(--beat-group)" }}>
            <Card padding={24}>
              <div style={{ display: "grid", gap: 20 }}>
                {[{ tag: "разные эпоха и регистр", accent: "var(--gold)", ...cal.easy }, { tag: "тот же регистр и эпоха", accent: "var(--icon-blue)", ...cal.medium }].map((r) => (
                  <div key={r.tag} style={{ display: "grid", gap: 6 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: r.accent }}>{r.tag}</span>
                      <span className="mono" style={{ fontSize: 13, color: "var(--success)" }}>доля {raw(r.macro)}</span>
                    </div>
                    <MeterBar value={r.macro} max={1} accent={r.accent} />
                    <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)" }}>косинус профилей {raw(r.cos)}</div>
                  </div>
                ))}
              </div>
            </Card>
            <p className="prose muted" style={{ margin: 0 }}>
              Две пары известных авторов, прогнанные тем же протоколом, — опорные примеры этого протокола, а не
              универсальная шкала и не источник рабочего порога. Косинус показывает, насколько совпадает направление
              усреднённых профилей: 1 — одинаковое направление, меньшее значение — менее похожие профили. Это
              вспомогательная диагностика: гейт читается по средней по классам полноте и перестановочному p.
            </p>
          </div>
        </div>

        {/* ──────────────── карта: панель проходит гейт ──────────────── */}
        <div className="reveal">
          <h3>Где контрольная панель проходит гейт</h3>
          <p className="prose muted">
            Обе панели выполняют оба условия сразу. Это значит, что панель различает заданные классы
            на этой закрытой панели, — и ничего не говорит о спорном тексте внутри неё.
          </p>
        </div>
        <div className="grid cols-2 reveal" style={{ marginTop: "var(--beat-group)" }}>
          {LIMITS.separates.map((c) => {
            const accent = ACCENT[c.id] || "var(--text-muted)";
            const pass = gatePass(c.macro, c.perm);
            return (
              <Card key={c.id} padding={24}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="case-kicker" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ width: 22, height: 2, background: accent }} />
                    <Badge className="case-badge" tone={pass ? "success" : "warning"}>{pass ? "панель проходит проверку" : "гейт не пройден"}</Badge>
                  </div>
                  <h4 style={{ margin: 0, color: "var(--text)", fontSize: "1.12rem" }}>{c.title}</h4>
                  <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.5 }}>круг кандидатов: {c.candidates}</div>
                  <div style={{ display: "grid", gap: 8, marginTop: 2 }}>
                    <MacroHead value={c.macro} accent={accent} />
                    {c.perm != null && <Row label="проверка на случайность (перестановка ярлыков)" value={pLabel(c.perm)} sub={gateNote(c.macro, c.perm)} color={permColor(c.perm)} />}
                    {c.cos != null && <Row label="косинус профилей" value={fmtScore(c.cos, 3)} sub="ближе к 1 — ближе направление усреднённых профилей" />}
                  </div>
                  <p className="note" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5 }}>{NOTE[c.id]}</p>
                </div>
              </Card>
            );
          })}
        </div>

        {/* ──────────────── карта: панель не проходит гейт ──────────────── */}
        <div className="reveal">
          <h3>Где контрольная панель не проходит гейт</h3>
          <p className="prose muted">
            Здесь нарушено хотя бы одно из двух условий. На карточках показано, какое именно и с какими
            значениями. Непройденный гейт — это состояние проверки, а не вывод об авторстве.
          </p>
        </div>
        <div className="grid cols-2 reveal" style={{ marginTop: "var(--beat-group)" }}>
          {LIMITS.limitsCases.map((c) => {
            const accent = ACCENT[c.id] || "var(--text-muted)";
            return (
              <Card key={c.id} padding={24}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="case-kicker" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ width: 22, height: 2, background: accent }} />
                    <Badge className="case-badge" tone="warning">гейт не пройден · {GATE_MISS[c.id]}</Badge>
                  </div>
                  <h4 style={{ margin: 0, color: "var(--text)", fontSize: "1.12rem" }}>{c.title}</h4>
                  <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.5 }}>круг кандидатов: {c.candidates}</div>
                  <div style={{ display: "grid", gap: 8, marginTop: 2 }}>
                    {/* две группы признаков — две дорожки, вклад автора и темы не разделён */}
                    {c.fwMacro != null && <Row label="по служебным словам" value={sc(c.fwMacro)} sub="ниже рабочего порога" color="var(--cinnabar)" />}
                    {c.fwPerm != null && <Row label="проверка на случайность (служебные слова)" value={pLabel(c.fwPerm)} sub={gateNote(c.fwMacro, c.fwPerm)} color={permColor(c.fwPerm)} />}
                    {c.char3Macro != null && <Row label="по символьным триграммам" value={sc(c.char3Macro)} sub="другая группа признаков даёт другой результат" color={accent} />}
                    {c.kappa != null && <Row label="согласие двух групп признаков (κ)" value={sc(c.kappa)} sub="группы признаков размечают тексты по-разному" />}
                    {c.macro != null && <MacroHead value={c.macro} accent={accent} />}
                    {c.perm != null && <Row label="проверка на случайность (перестановка ярлыков)" value={pLabel(c.perm)} sub={gateNote(c.macro, c.perm)} color={permColor(c.perm)} />}
                    {c.cos != null && <Row label="косинус профилей" value={fmtScore(c.cos, 3)} sub="ближе к 1 — ближе направление усреднённых профилей" />}
                  </div>
                  <p className="note" style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5 }}>{NOTE[c.id]}</p>
                </div>
              </Card>
            );
          })}
        </div>

        <p className="verdict reveal">
          <strong style={{ color: "var(--text)" }}>Одна работа — один голос.</strong>{" "}
          Метрика гейта считается по работам, значения по кускам остаются диагностикой. Гейт требует
          двух условий сразу и относится к панели, а не к спорному тексту внутри неё.
        </p>
        <p className="note reveal">Панели без пройденного гейта остаются открытыми вопросами для архивов и текстологов.</p>
      </div>
    </section>
  );
}
