import { Card, CodeBlock } from "@dmitrymake/rk-ui";
import { REPRO, BENCH } from "../segdata.js";
import { LOBO_STRICT, HEADLINE } from "../data.js";
import { CORPUS } from "../corpus.js";
import { fmtScore, fmtPct } from "../format.js";

const GUARANTEES = [
  {
    title: "Проверяемо",
    accent: "var(--success)",
    body: `Настройки лежат в одном файле, версии библиотек закреплены. Все ${REPRO.gatesBitExact} контрольных расчётов литературных кейсов повторяются бит-в-бит. Самый долгий из них — «${REPRO.longestGateName}», но и он проверяется автоматически.`,
  },
  {
    title: "Экономно",
    accent: "var(--icon-blue)",
    body: "Языковой разбор текста делается один раз и ложится на диск, признаки отрывков считаются заранее. Тяжёлую часть машина не повторяет — каждый следующий прогон только разгоняется.",
  },
  {
    title: "Без подсказок",
    accent: "var(--gold)",
    body: "Перед обучением программа ищет совпадающие и вложенные произведения. Проверяемая книга уходит вместе со всей группой текстов того же содержания.",
  },
];

export default function Repro() {
  return (
    <section className="section" id="repro">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Можно повторить у себя</p>
          <h2>Как повторить исследование</h2>
          <p className="prose lead muted">
            Код, настройки, список источников и промежуточные результаты лежат в
            репозитории. Открытую часть корпуса можно собрать заново и пройти тот же
            путь: проверить тексты, выделить стилевые признаки, обучить модель и
            получить отчёт. Сохранённый первый расчёт тоже воспроизводится — его
            точка на открытой выборке была {fmtScore(BENCH.topTop1, 3)}, около{" "}
            {fmtPct(BENCH.topTop1)} верных ответов.
          </p>
          <p className="prose muted">
            Открытая выборка меньше: {BENCH.nAuthors} автора и {BENCH.nBooks} книг.
            В полном первом эксперименте было {CORPUS.benchmark.authors} автора, и
            доля верных ответов составила {fmtScore(LOBO_STRICT.styloFullLobo, 3)}.
            Эти числа нельзя сравнивать напрямую: наборы авторов разные. После
            пересборки корпуса оба расчёта будут выполнены заново.
          </p>
          {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
            <p className="note" style={{ fontSize: 13 }}>
              В первом эксперименте длинные книги сильнее влияли на авторский
              профиль. В новом расчёте действует простое правило: одна книга — один голос.
            </p>
          )}
          <p className="prose muted">
            В полный срез входят ещё и книги под защитой авторских прав. Они живут только на локальной
            машине, в общий корпус не отдаются. Можете добавить их самостоятельно.
          </p>
        </div>

        {/* Полный пайплайн */}
        <div className="module reveal" style={{ display: "grid", gap: 16 }}>
          <CodeBlock language="bash" title="собрать открытую часть и запустить расчёт">
            {`./run.sh fetch-classics   # загрузить классику из открытых источников
./run.sh all              # проверить корпус → обучить → оценить → собрать отчёт`}
          </CodeBlock>

          <p className="prose muted" style={{ margin: 0, fontSize: 13.5 }}>
            Перед обучением скрипт проверяет, разделены ли совпадающие произведения.
            Если корпус ещё не готов, расчёт останавливается — модель не успевает
            увидеть данные и не публикует новую цифру.
          </p>

          <CodeBlock language="bash" title="по шагам">
            {`./run.sh sweep      # проверить, какие группы признаков действительно помогают
./run.sh evaluate   # спрятать книги по очереди и оценить ответы
./run.sh predict    # определяем автора неизвестного текста по профилям авторов`}
          </CodeBlock>
        </div>

        {/* Гарантии */}
        <div className="grid cols-3 module reveal">
          {GUARANTEES.map((g) => (
            <Card key={g.title} padding={22} style={{ borderTop: `3px solid ${g.accent}` }}>
              <h3 style={{ margin: "0 0 8px", fontSize: "1.2rem", color: g.accent }}>{g.title}</h3>
              <p className="prose muted" style={{ margin: 0, fontSize: 14.5 }}>
                {g.body}
              </p>
            </Card>
          ))}
        </div>

        <p className="verdict reveal">
          Главное правило повторного эксперимента: модель
          <strong style={{ color: "var(--gold)" }}> не видит ни проверяемую книгу,
          ни другой текст с тем же содержанием</strong>. Если программа находит такое
          совпадение, она останавливается до обучения.
        </p>
      </div>
    </section>
  );
}
