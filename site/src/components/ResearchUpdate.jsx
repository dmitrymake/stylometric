import { HEADLINE } from "../data.js";
import { fmtPct, fmtScore } from "../format.js";

export default function ResearchUpdate() {
  return (
    <aside
      id="research-update"
      className="research-update"
      role="note"
      aria-label="Статус оценки модели"
    >
      <p className="research-update-kicker">Исследование продолжается</p>
      <p className="research-update-lead">
        Прежние <strong>{fmtPct(HEADLINE.accuracy, 1)}</strong> и macro-F1{" "}
        {fmtScore(HEADLINE.macroF1, 3)} относятся к первому эксперименту, а не к
        итоговой оценке. Macro-F1 усредняет качество по авторам, давая каждому
        одинаковый вес.
      </p>
      <p>
        Аудит показал, что совпадающие тексты попадали и в обучение, и в проверку.
        Такое пересечение может завысить оценку модели. При подготовке нового корпуса
        обучение и проверку разделили по содержанию; новая итоговая оценка ещё не
        опубликована.
      </p>
      <details>
        <summary>Что изменилось в проверке</summary>
        <p>
          Раньше из обучения исключалась только проверяемая книга. Теперь вместе с
          ней исключаются тексты с тем же содержанием, в том числе рассказ в сборнике
          под другим названием.
        </p>
      </details>
    </aside>
  );
}
