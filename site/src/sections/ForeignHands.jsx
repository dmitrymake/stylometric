import { Stat } from "@dmitrymake/rk-ui";
import { SEGMENT } from "../data.js";
import { fmtScore } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";

const S = SEGMENT;

export default function ForeignHands() {
  // грубая биномиальная верхняя 95%-граница по правилу трёх: три делить на число книг.
  const falseUpperPct = Math.ceil((3 / S.fpr.totalBooks) * 100);

  return (
    <section className="section" id="foreign">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Проверка · сегментный анализ</p>
          <h2>Сегментный анализ</h2>
          <p className="prose lead muted">
            Книга режется на последовательные фрагменты. Для каждого фрагмента логистическая регрессия оценивает
            вероятности кандидатов; после сглаживания выбирается метка с наибольшей вероятностью. Метод ищет
            непрерывную последовательность с меткой, отличной от основного автора: он не доказывает соавторство
            и не обязан находить точную границу склейки. Контролей два — синтетические склейки и цельные одноавторские книги.
          </p>
        </div>

        <div className="split reveal" style={{ alignItems: "start" }}>
          {/* Контроль на синтетических склейках + кривая подмеса */}
          <div>
            <h3 style={{ marginTop: 0 }}>Синтетическая склейка</h3>
            <p className="prose muted" style={{ marginBottom: 16 }}>
              Две книги разных авторов сшиваются встык: чужой сегмент должен быть обнаружен.
              Вторая панель — близкие по манере авторы.
            </p>
            <div className="grid cols-2" style={{ marginBottom: 22 }}>
              <Stat label="разные авторы" value={`${S.recallDissimilar.detected}/${S.recallDissimilar.total}`} accent="var(--success)" parade hint="малая панель" />
              <Stat label="похожие авторы" value={`${S.recallSimilar.detected}/${S.recallSimilar.total}`} accent="var(--gold)" hint="малая панель" />
            </div>
            <div className="mono muted" style={{ fontSize: 11, marginBottom: 8 }}>одна серия подмеса: слева доля добавленного текста, справа доля чужих фрагментов</div>
            {S.admixture.map((a) => (
              <div key={a.pct} style={{ display: "grid", gridTemplateColumns: "5ch 1fr 14ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>{a.pct}%</span>
                <MeterBar value={a.foreign} accent={a.detected ? "var(--icon-blue)" : "var(--border-strong)"} />
                <span className="mono" style={{ fontSize: 10.5, color: a.detected ? "var(--icon-blue)" : "var(--text-muted)" }}>
                  {fmtScore(a.foreign)}{a.detected ? " — сегмент" : ""}
                </span>
              </div>
            ))}
            <p className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
              В этой серии сегмент появился при <strong style={{ color: "var(--text)" }}>{S.minDetectedAdmixPct}%</strong> подмеса;
              меньшие ненулевые доли не проверялись.
            </p>
          </div>

          {/* Контроль на цельных книгах + предел по похожим авторам */}
          <div style={{ display: "grid", gap: 18, alignContent: "start" }}>
            <h3 style={{ marginTop: 0 }}>Цельная книга</h3>
            <p className="verdict" style={{ margin: 0 }}>
              В {S.fpr.totalBooks} одноавторских книгах ложных сегментных срабатываний не наблюдалось:
              {" "}{S.fpr.falseBooks}/{S.fpr.totalBooks}. Грубая биномиальная верхняя 95%-граница для этой
              выборки — около&nbsp;{falseUpperPct}%.
            </p>
            <p className="note" style={{ margin: 0 }}>
              <strong style={{ color: "var(--cinnabar)" }}>Предел.</strong> Похожих авторов проверили только
              на <strong style={{ color: "var(--text)" }}>{S.recallSimilar.total}</strong> синтетических склейках,
              где добавленная доля была не меньше <strong style={{ color: "var(--text)" }}>{S.similarDetectionFloorPct}%</strong>.
              Меньшие доли не проверялись. Малый или распределённый вклад стилистически похожего автора исключить нельзя.
            </p>
          </div>
        </div>

        <p className="muted reveal" style={{ fontSize: 13.5, maxWidth: "64ch" }}>
          В конкретных кейсах детектор не обнаружил непрерывного сегмента, но такой результат не исключает малый,
          распределённый или стилистически близкий вклад. Это сохранённая диагностика первого корпуса; на новом
          содержательно разделённом корпусе она ещё не повторена.
        </p>
      </div>
    </section>
  );
}
