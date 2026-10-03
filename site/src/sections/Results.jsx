import { EditorialFigure, EditorialNote } from "@dmitrymake/rk-ui";
import { MEASUREMENT } from "../data.js";
import { fmtCount, fmtInt, fmtPct } from "../format.js";

const pct = (value) => fmtPct(value, 2).replace(".", ",");
const CONDITIONS = {
  A0: "Обучение на фрагментах",
  A4: "Пакет с балансировкой произведений",
};
const ARMS = { current: "Полный набор", topic_strict: "Сокращённый набор" };
const PAIRS = [
  { left: ["A0", "current"], right: ["A4", "current"], label: "Пакет обучения", note: "Полный набор признаков: A0 → A4" },
  { left: ["A0", "current"], right: ["A0", "topic_strict"], label: "Сокращение признаков при A0", note: "Полный → сокращённый набор" },
  { left: ["A4", "current"], right: ["A4", "topic_strict"], label: "Сокращение признаков при A4", note: "Полный → сокращённый набор" },
];
const sameArm = (a, b) => a[0] === b[0] && a[1] === b[1];

export default function Results() {
  const analysis = MEASUREMENT.pairedAnalysis;
  const comparisons = PAIRS.map((pair) => ({ ...pair, ...analysis.comparisons.find((row) => sameArm(row.left, pair.left) && sameArm(row.right, pair.right)) }));
  const weight = comparisons[0];
  const strictA0 = comparisons[1];
  const strictA4 = comparisons[2];
  const arm = (cell, name) => analysis.arms.find((row) => row.cell === cell && row.arm === name);
  const gainAuthors = weight.per_author.filter((row) => row.net_correct > 0).sort((a, b) => b.net_correct - a.net_correct);
  const maximumChange = Math.max(...comparisons.flatMap((row) => [row.corrected, row.lost]));
  return <section className="section" id="results"><div className="wrap flow">
    <div className="section-head">
      <p className="eyebrow">03 / Результаты проверки</p>
      <h2>Одна точность не описывает все ответы</h2>
      <p className="prose lead">Каждый из четырёх вариантов проверен на одних и тех же {fmtInt(MEASUREMENT.works)} произведениях. Качество оценено для {MEASUREMENT.testedAuthors} авторских меток; всего выполнено {fmtInt(MEASUREMENT.fits)} обучений. Меняются пакет обучения и набор признаков.</p>
    </div>
    <EditorialFigure label="Таблица 1" caption={<>Узнавание автора целого отложенного произведения. Точность учитывает каждую книгу, средняя полнота (recall) — каждого из {MEASUREMENT.testedAuthors} авторов с одинаковым весом. Поэтому показатели могут меняться в разные стороны.</>} source={<a href={`${import.meta.env.BASE_URL}${analysis.publicArtifact}`} download>Исходные результаты по авторам и парные сравнения</a>}>
      <div className="result-table-region">
        <table className="result-table"><thead><tr><th scope="col">Обучение и признаки</th><th scope="col">Верно / всего</th><th scope="col">Точность</th><th scope="col">Средняя полнота<br />по авторам</th></tr></thead><tbody>
          {analysis.arms.map((row) => <tr key={`${row.cell}-${row.arm}`}><th scope="row"><span className="result-model">{row.cell} · {CONDITIONS[row.cell]}</span><small>{ARMS[row.arm]}</small></th><td data-label="Верно / всего">{row.correct} / {row.total}</td><td data-label="Точность">{pct(row.accuracy)}</td><td data-label="Средняя полнота">{pct(row.macro_author_recall)}</td></tr>)}
        </tbody></table>
      </div>
    </EditorialFigure>
    <div className="result-explanation">
      <h3>Что даёт пакет с балансировкой</h3>
      <p>На полном наборе признаков пакет A4 исправляет {fmtCount(weight.corrected, "ошибку", "ошибки", "ошибок")} варианта A0 и теряет {fmtCount(weight.lost, "верный ответ", "верных ответа", "верных ответов")}. Улучшение затрагивает {fmtCount(weight.authors_net_improved, "автора", "авторов", "авторов")}. На метку «{analysis.authorNames[gainAuthors[0].author] ?? gainAuthors[0].author}» приходятся {gainAuthors[0].net_correct} из {weight.net_correct} дополнительных верных ответов.</p>
      <p>A4 объединяет равный суммарный вес произведений внутри автора, построение словаря и IDF по произведениям и нормирование частот по полной длине текста. Это сравнение оценивает пакет целиком; отдельный эффект весов из него не следует.</p>
    </div>
    <EditorialFigure label="Рисунок 1" caption="Как меняются ответы при переходе от левого варианта к правому. Исправление означает: первый вариант ошибся, второй ответил верно. Потеря означает обратный переход. Числа относятся к одним и тем же отложенным произведениям.">
      <div className="paired-chart">
        <div className="paired-chart-heading"><span>Сравнение</span><span>Исправлено</span><span>Потеряно</span></div>
        {comparisons.map((row) => <div className="paired-chart-row" key={row.label}>
          <div><strong>{row.label}</strong><span>{row.note}</span></div>
          <div className="paired-bar"><span className="paired-bar-track" aria-hidden="true"><i style={{ width: `${row.corrected / maximumChange * 100}%` }} /></span><b>{row.corrected}</b><span className="paired-mobile-label">исправлено</span></div>
          <div className="paired-bar paired-bar--lost"><span className="paired-bar-track" aria-hidden="true"><i style={{ width: `${row.lost / maximumChange * 100}%` }} /></span><b>{row.lost}</b><span className="paired-mobile-label">потеряно</span></div>
        </div>)}
      </div>
    </EditorialFigure>
    <div className="result-explanation">
      <h3>Что скрывает общий процент</h3>
      <p>При обучении на фрагментах сокращение признаков добавляет {strictA0.net_correct} попадания: {arm("A0", "current").correct} → {arm("A0", "topic_strict").correct}. Но средняя полнота по авторам снижается с {pct(arm("A0", "current").macro_author_recall)} до {pct(arm("A0", "topic_strict").macro_author_recall)}. Выигрыш по книгам распределён между авторами неравномерно.</p>
      <p>При A4 точность обоих наборов равна {pct(arm("A4", "current").accuracy)}, но {fmtCount(strictA4.changed_top1, "ответ", "ответа", "ответов")} различаются: сокращённый набор исправляет {strictA4.corrected} ошибку и теряет {strictA4.lost} верный ответ. Совпадение общей точности не означает совпадения решений.</p>
    </div>
    <EditorialNote title="Область результата"><p>Это описательное сравнение вариантов на данном корпусе. Под автором здесь понимается метка корпуса: она может обозначать в том числе коллективный псевдоним. Его модель содержит {MEASUREMENT.candidateClasses} классов кандидатов; у авторов с единственным произведением нет другой книги для независимой проверки. Статистическая значимость различий и доверительные интервалы здесь не оцениваются.</p><p>В корпусе нет авторских классов Ильфа–Петрова и Шолохова. Результат проверки метода не устанавливает авторство их романов: главы о литературных случаях используют собственные эталоны и протоколы.</p></EditorialNote>
    <details><summary>Состав проверки и воспроизводимые источники</summary><p className="prose">Сокращённый набор использует фиксированные служебные слова и сокращённый синтаксис. В каждом варианте произведение целиком исключается из обучения. Корпус проходит проверку пересечений содержания; библиографические связи изданий требуют отдельной сверки. Macro-F1 в этом анализе не рассчитывается.</p><p><a href={`${import.meta.env.BASE_URL}${MEASUREMENT.publicArtifact}`} download>Скачать агрегат четырёх вариантов</a></p><p className="source-path">{MEASUREMENT.source}</p><p className="source-path">{analysis.source}</p></details>
  </div></section>;
}
