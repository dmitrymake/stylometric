import { Card, Stat } from "@dmitrymake/rk-ui";
import { MEASUREMENT } from "../data.js";
import { fmtCount, fmtInt, fmtPct } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";
import HistoricalResults from "./HistoricalResults.jsx";

const pct = (value) => fmtPct(value, 2).replace(".", ",");
const CONDITIONS = {
  A0: { title: "Обучение на фрагментах", note: "Авторские классы сбалансированы, но внутри автора длинные произведения сильнее влияют на словарь, частоты и обучение." },
  A4: { title: "Пакет с балансировкой произведений", note: "Внутри каждого автора книги имеют равный суммарный вес; выравниваются авторские классы. Меняются также словарь, IDF и нормировка частот." },
};
const ARMS = {
  current: "Исходный набор признаков",
  topic_strict: "Фиксированные служебные слова и сокращённый синтаксис",
};

export default function Results() {
  const balanced = MEASUREMENT.cells.find((cell) => cell.cell === "A4");
  const fragment = MEASUREMENT.cells.find((cell) => cell.cell === "A0");
  const changedBalanced = balanced.transitions.current_only_correct + balanced.transitions.topic_strict_only_correct
    + balanced.transitions.both_wrong_changed_prediction;
  return (
    <section className="section" id="results">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Завершённый замер</p>
          <h2>Сохраняется ли качество при изменении признаков</h2>
          <p className="prose lead muted">
            Сравнили настройки обучения и признаки на одних и тех же {fmtInt(MEASUREMENT.works)} отложенных
            произведениях. Качество проверено для {MEASUREMENT.testedAuthors} авторов. Каждый ответ
            оценивался по целому произведению; всего выполнено {fmtInt(MEASUREMENT.fits)} обучений.
          </p>
        </div>
        <div className="grid cols-2 reveal">
          <Stat label="с балансировкой · исходные признаки" value={pct(balanced.accuracy.current.value)}
            accent="var(--gold)" hint={`${balanced.accuracy.current.correct} из ${balanced.accuracy.current.total} произведений опознаны верно`} parade />
          <Stat label="с балансировкой · сокращённый набор" value={pct(balanced.accuracy.topic_strict.value)}
            accent="var(--icon-blue)" hint={`${balanced.accuracy.topic_strict.correct} из ${balanced.accuracy.topic_strict.total} произведений опознаны верно`} />
        </div>
        <div className="grid cols-2 reveal">
          {MEASUREMENT.cells.map((cell) => (
            <Card key={cell.cell} padding={24}>
              <h3>{CONDITIONS[cell.cell].title}</h3>
              <p className="note">{CONDITIONS[cell.cell].note}</p>
              {Object.entries(ARMS).map(([arm, label]) => (
                <div key={arm} style={{ display: "grid", gap: 8, marginTop: 20 }}>
                  <span>{label}</span>
                  <MeterBar value={cell.accuracy[arm].value} max={1}
                    accent={arm === "current" ? "var(--gold)" : "var(--icon-blue)"} />
                  <span className="mono">{cell.accuracy[arm].correct}/{cell.accuracy[arm].total} · {pct(cell.accuracy[arm].value)}</span>
                </div>
              ))}
            </Card>
          ))}
        </div>
        <div className="split reveal">
          <div className="prose">
            <h3>Равная точность, разные ответы</h3>
            <p>
              В пакете с балансировкой число попаданий осталось прежним, но изменились ответы
              для {fmtCount(changedBalanced, "произведения", "произведений", "произведений")}: сокращённый набор
              исправил {fmtCount(balanced.transitions.topic_strict_only_correct, "ошибку", "ошибки", "ошибок")}
              {" "}и потерял {fmtCount(balanced.transitions.current_only_correct, "попадание", "попадания", "попаданий")}.
              Поэтому одинаковая общая точность не означает полного совпадения решений.
            </p>
          </div>
          <div className="prose">
            <h3>Что изменилось без балансировки книг</h3>
            <p>
              При обучении на фрагментах сокращённый набор исправил
              {" "}{fmtCount(fragment.transitions.topic_strict_only_correct, "ошибку", "ошибки", "ошибок")}
              {" "}и потерял {fmtCount(fragment.transitions.current_only_correct, "попадание", "попадания", "попаданий")}.
              Итоговая разница — {fmtCount(fragment.delta.numerator, "произведение", "произведения", "произведений")} из {fragment.delta.denominator}.
              Это наблюдение на данном наборе, без оценки статистической значимости разницы.
            </p>
          </div>
        </div>
        <p className="callout reveal">
          Эти проценты описывают узнавание известных авторов по отложенным произведениям.
          Версии о Булгакове и «Тихом Доне» этим прогоном не перепроверялись: состав эталонов
          и поиск локального участия требуют отдельных сравнений.
        </p>
        <details className="reveal">
          <summary>Состав проверки и источник чисел</summary>
          <p className="prose muted">
            В модели {MEASUREMENT.candidateClasses} класса кандидатов, качество проверено для
            {" "}{MEASUREMENT.testedAuthors} авторов. У авторов с единственным произведением нет
            другой книги для независимой проверки. Метрика — доля правильных ответов по работам;
            macro-F1 и доверительные интервалы в этом расчёте не оценивались.
          </p>
          <p className="prose muted">
            Пакеты различаются сразу несколькими настройками. Этот прогон не выделяет
            отдельный эффект каждого механизма балансировки.
          </p>
          <a href={`${import.meta.env.BASE_URL}${MEASUREMENT.publicArtifact}`} download="topic-validity-aggregate.json">
            Скачать исходный агрегат
          </a>
          <p className="note mono" style={{ overflowWrap: "anywhere" }}>{MEASUREMENT.source}</p>
        </details>
        <details className="reveal">
          <summary>История: результаты исходного корпуса</summary>
          <HistoricalResults />
        </details>
      </div>
    </section>
  );
}
