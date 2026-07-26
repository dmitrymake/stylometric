import { HEADLINE, HEADLINE_PUBLICATION } from "../data.js";
import { fmtScore } from "../format.js";

export default function HistoricalHeadlineNotice({ compact = false }) {
  return (
    <aside
      className="note"
      role="note"
      aria-label="Статус исторического LOBO-результата"
      style={{
        borderLeft: "4px solid var(--danger)",
        padding: compact ? "12px 16px" : "16px 20px",
        margin: compact ? "12px 0" : "18px 0 24px",
        maxWidth: "88ch",
      }}
    >
      <strong style={{ color: "var(--danger)" }}>
        Исторический LOBO headline отозван.
      </strong>{" "}
      В границе train/test найдено перекрытие содержания между разными произведениями.
      Поэтому accuracy {fmtScore(HEADLINE.accuracy, 4)}, macro-F1{" "}
      {fmtScore(HEADLINE.macroF1, 4)}, интервалы и p ниже — только сохранённая
      историческая арифметика: это не leakage-free оценка точности и не действующее
      свидетельство значимости. Нужны новая версия корпуса и полный пересчёт.{" "}
      <span className="mono" style={{ fontSize: "0.88em" }}>
        {HEADLINE_PUBLICATION.corpusStatus} · {HEADLINE_PUBLICATION.claimStatus}
      </span>
    </aside>
  );
}
