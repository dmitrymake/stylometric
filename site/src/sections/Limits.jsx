import { fmtScore, fmtP } from "../format.js";
import { LIMITS } from "../segdata.js";
import Sources from "../components/Sources.jsx";

const T = LIMITS.threshold;
const P_THRESHOLD = 0.05; // Registered decision threshold, not a measured value.
const sc = (value) => fmtScore(value, 3);
const pLabel = (value) => `p ${fmtP(value).startsWith("<") ? "" : "= "}${fmtP(value)}`;

function ComparisonTable({ caption, headings, rows }) {
  return <table className="limits-table">
    <caption>{caption}</caption>
    <thead><tr><th scope="col">Сравнение</th>{headings.map((heading) => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
    <tbody>{rows.map((row) => <tr key={row.id}>
      <th scope="row">{row.label}{row.note && <span className="limits-row-note">{row.note}</span>}</th>
      {row.values.map((value, index) => <td key={headings[index]} data-label={headings[index]}>{value}</td>)}
    </tr>)}</tbody>
  </table>;
}

function CaseReading({ title, candidates, children }) {
  return <div className="limits-reading">
    <h4>{title}</h4>
    <p className="limits-candidates">{candidates}</p>
    <p>{children}</p>
  </div>;
}

export default function Limits() {
  const calibration = LIMITS.calibration;
  const sovremennik = LIMITS.metric.find((row) => row.id === "sovremennik");
  const cases = Object.fromEntries([...LIMITS.separates, ...LIMITS.limitsCases].map((row) => [row.id, row]));
  const comparisonRows = LIMITS.limitsCases.flatMap((row) => row.fwMacro != null ? [
    { id: `${row.id}-fw`, label: row.title, note: "Служебные слова", values: [sc(row.fwMacro), fmtP(row.fwPerm), "—"] },
    { id: `${row.id}-char3`, label: row.title, note: "Символьные триграммы", values: [sc(row.char3Macro), fmtP(row.char3Perm), "—"] },
  ] : [{ id: row.id, label: row.title, values: [sc(row.macro), fmtP(row.perm), sc(row.cos)] }]);

  return <section className="section" id="limits"><div className="wrap flow">
    <div className="section-head">
      <p className="eyebrow">Границы метода</p>
      <h2>Что показывают контрольные сравнения</h2>
      <p className="prose lead">Прежде чем обсуждать спорный текст, нужно проверить, различает ли метод известных авторов. Эти литературные подборки показывают, как результат зависит от состава кандидатов, признаков и единицы оценки.</p>
    </div>

    <div className="module limits-unit">
      <h3>Произведение и фрагмент дают разные оценки</h3>
      <p className="prose">Протокол сначала строит профиль каждого произведения, затем усредняет их с равным весом в профиле автора. При проверке одно произведение даёт один ответ. Средняя полнота — это доля правильно опознанных произведений, усреднённая по авторским классам с равным весом каждого класса.</p>
      <p className="prose">При подсчёте по фрагментам длинные книги дают больше наблюдений, а соседние фрагменты связаны между собой. Поэтому значения по фрагментам служат отдельной диагностикой и не заменяют оценку по произведениям.</p>
      <ComparisonTable caption="Один корпус, две единицы оценки" headings={["По произведениям", "По фрагментам"]} rows={LIMITS.metric.map((row) => ({ id: row.id, label: row.label, values: [sc(row.work), sc(row.chunk)] }))} />
      <p className="prose">У «Современника» значения {sc(sovremennik.work)} и {sc(sovremennik.chunk)} описывают одну подборку. Различие возникает из-за единицы счёта; к условиям дальнейшей интерпретации относится только оценка по произведениям.</p>
    </div>

    <div className="module limits-calibration">
      <h3>Два опорных сравнения</h3>
      <p className="prose">Тот же протокол применён к двум парам известных авторов. Они показывают результат на конкретном материале; универсальную шкалу сходства и рабочий порог по этим двум примерам не устанавливают.</p>
      <ComparisonTable caption="Различение известных авторов тем же методом" headings={["Средняя полнота", "Косинус профилей"]} rows={[
        { id: "easy", label: calibration.easy.label.split(" — ")[0], note: "Разные эпоха и регистр", values: [String(calibration.easy.macro), String(calibration.easy.cos)] },
        { id: "medium", label: calibration.medium.label.split(" — ")[0], note: "Один регистр и эпоха", values: [String(calibration.medium.macro), String(calibration.medium.cos)] },
      ]} />
      <p className="prose">Косинус описывает направление усреднённых профилей: значение 1 означает одинаковое направление, меньшее значение — менее похожие профили. Это вспомогательная характеристика, которая сама по себе не определяет, различимы ли авторы.</p>
    </div>

    <div className="module limits-conditions">
      <h3>Условия интерпретации</h3>
      <p className="prose">Для этих контрольных подборок заранее приняты два совместных условия: средняя полнота по произведениям не ниже {fmtScore(T)} и перестановочная проверка на уровне произведений с p ≤ {P_THRESHOLD}. Перестановка меняет авторские метки и проверяет, насколько результат отличается от случайного распределения. Малого p недостаточно без требуемой полноты.</p>
      <p className="prose">Полнота лежит от 0 до 1: от отсутствия верно опознанных произведений до правильных ответов для всех произведений. Выполнение обоих условий относится к известным классам в данной подборке. Авторство спорного текста требует отдельного сравнения.</p>
    </div>

    <div className="module limits-separates">
      <h3>Где известные классы различимы</h3>
      <ComparisonTable caption="Обе подборки выполняют два принятых условия" headings={["Средняя полнота", "p", "Косинус"]} rows={LIMITS.separates.map((row) => ({ id: row.id, label: row.title, values: [sc(row.macro), fmtP(row.perm), sc(row.cos)] }))} />
      <CaseReading title={cases.sovremennik.title} candidates={cases.sovremennik.candidates}>
        Результат относится к перечисленным критикам; перенос на любую «школу как класс» не показан. Боткин представлен одной работой, часть авторской разметки основана на гонорарных ведомостях.
      </CaseReading>
      <CaseReading title={cases.petersburg.title} candidates={cases.petersburg.candidates}>
        Известные авторы различимы, но спорный фельетон под подписью «Н.Н.» остаётся без атрибуции: его фрагменты делятся 1:1 между публицистикой Достоевского и Панаевым.
      </CaseReading>
    </div>

    <div className="module limits-unresolved">
      <h3>Где различение остаётся недостаточным</h3>
      <p className="prose">В этих сравнениях не выполнено хотя бы одно из двух условий либо авторское различие смешано с тематическим. Это ограничение конкретной проверки, а не доказательство равенства авторов или вывод об авторстве спорного текста.</p>
      <ComparisonTable caption="Результаты зависят от пары авторов и группы признаков" headings={["Средняя полнота", "p", "Косинус"]} rows={comparisonRows} />
      <p className="limits-table-note">Знак «—» означает, что отдельное значение в сводке не приведено. Показатели округлены для чтения; условия интерпретации применяются к исходным значениям.</p>
      <CaseReading title={cases.nekrasov.title} candidates={cases.nekrasov.candidates}>
        Служебные слова не достигают ни одного из двух принятых условий. Символьные триграммы дают другую картину, но автор и тема в этом дизайне не разделены. Косинус профилей в сводке — {sc(cases.nekrasov.cos)}; согласие двух групп признаков низкое: κ = {sc(cases.nekrasov.kappa)}. Высокая полнота на триграммах не доказывает авторство.
      </CaseReading>
      <CaseReading title={cases.pair.title} candidates={cases.pair.candidates}>
        Полнота ниже {fmtScore(T)}, а {pLabel(cases.pair.perm)} превышает {P_THRESHOLD}. Эта проверка не подтверждает разделение учителя и ученика внутри одной школы.
      </CaseReading>
      <CaseReading title={cases.kolokol.title} candidates={cases.kolokol.candidates}>
        Полнота и перестановочная проверка не достигают принятых условий. Отсутствие подтверждённого разделения не означает, что авторские манеры одинаковы.
      </CaseReading>
      <CaseReading title={cases.chekhonte.title} candidates={cases.chekhonte.candidates}>
        Перестановочное условие выполнено, но полнота ниже {fmtScore(T)}. Архивный заказ Курепина относится к заметке 24 мая, а не ко всей подборке из пяти текстов. Это документальный довод в пользу Чехова для одной заметки; авторство всей подборки он не устанавливает.
      </CaseReading>
    </div>
    <Sources label="Данные контрольных сравнений" artifact="controls" />
  </div></section>;
}
