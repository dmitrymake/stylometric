import { Card, Stat } from "@dmitrymake/rk-ui";
import { HEADLINE, MODELS, AUTHOR_RECALL } from "../data.js";
import { BENCH_EXT } from "../segdata.js";
import { fmtPct, fmtScore } from "../format.js";

const BOW = MODELS.find((m) => m.id === "bow_lr");
const PROZA_NEURO = BENCH_EXT.prozaNeuro;                                  // ruBERT-tiny2 без дообучения
const PROZA_LEADER = BENCH_EXT.prozaLeader;                                // лучший классический метод на той же прозе

// склонения без литералов о результатах — только грамматика
const ruBooks = (n) => {
  const t = Math.abs(n) % 100, o = t % 10;
  if (t >= 11 && t <= 14) return "книг";
  if (o === 1) return "книга";
  if (o >= 2 && o <= 4) return "книги";
  return "книг";
};
const listNames = (arr) => arr.map((a) => a.name).join(" и ");

// Примеры «мало книг» берутся из данных, не из литералов: авторы с нулевой узнаваемостью
// при минимальном числе книг vs авторы с почти тем же числом книг, но верные во всех случаях.
const zeroRecall = AUTHOR_RECALL.filter((a) => a.recall === 0).sort((a, b) => a.books - b.books);
const minZeroBooks = zeroRecall.length ? zeroRecall[0].books : 0;
const zeroLowMin = zeroRecall.filter((a) => a.books === minZeroBooks);
const perfectSmall = AUTHOR_RECALL
  .filter((a) => a.recall === 1 && a.books <= minZeroBooks + 1)
  .sort((a, b) => a.books - b.books)
  .slice(0, 2);
// «столько же / почти столько же» — сравнение чисел из данных, без литерала о равенстве числа книг
const sameOrAlmost =
  perfectSmall.length && perfectSmall[0].books === minZeroBooks ? "таком же" : "почти таком же";
// Проверяем на данных, а не на глаз: все нераспознанные авторы лежат в нижнем краю по числу книг.
const failuresAllFewBooks = zeroRecall.length > 0 && zeroRecall.every((a) => a.books <= minZeroBooks + 1);
const failClusterLine = failuresAllFewBooks
  ? " Все нулевые результаты относятся к авторам с наименьшим числом книг."
  : "";

const LEVERS = [
  {
    num: "01",
    accent: "var(--gold)",
    title: "Число книг на автора",
    body: `${listNames(zeroLowMin)}: по ${minZeroBooks} ${ruBooks(minZeroBooks)} на автора и 0 верных ответов из ${minZeroBooks} в историческом эксперименте.${failClusterLine} ${listNames(perfectSmall)} при ${sameOrAlmost} числе книг — ${perfectSmall[0].books} ${ruBooks(perfectSmall[0].books)} на автора — во всех случаях определены верно. Дополнительные книги помогут точнее оценить разброс.`,
  },
  {
    num: "02",
    accent: "var(--icon-blue)",
    title: "Чувствительность к теме",
    body: "В следующем корпусе темы и жанры нужно распределить между авторами равномернее. Затем результат следует сравнить с исходным, чтобы проверить его чувствительность к тематической лексике.",
  },
  {
    num: "03",
    accent: "var(--success)",
    title: "Базовый нейросетевой вариант",
    body: `ruBERT-tiny2 без дообучения — один базовый нейросетевой вариант, не настроенный на определение автора. На внешней русской прозе он уступает классическому методу (${fmtScore(PROZA_NEURO, 2)} против ${fmtScore(PROZA_LEADER, 2)}); другие нейросетевые варианты в сравнении не участвовали.`,
  },
];

export default function Conclusion() {
  return (
    <section className="section" id="conclusion">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Вывод</p>
          <h2>Надёжность начинается с корпуса</h2>
          <p className="prose lead muted">
            Корпус задаёт границы вывода: важны число книг на автора, распределение
            тем и жанров и схема проверки. Сравнение групп признаков помогает оценить
            чувствительность результата к теме, но не доказывает, что тема полностью
            отделена от авторской манеры.
          </p>
        </div>

        {/* признаки окупаются */}
        <div className="split reveal module" style={{ alignItems: "start" }}>
          <div className="prose">
            <p className="verdict">
              В первом эксперименте сочетание синтаксиса, служебных слов и пунктуации
              дало более высокую долю верных ответов, чем модель по частотам слов:{" "}
              <strong style={{ color: "var(--gold)" }}> {fmtPct(HEADLINE.accuracy, 1)} против {fmtPct(BOW.acc, 1)}</strong>,
              соответственно. Эти значения относятся только к исходному корпусу;
              их устойчивость должен проверить новый расчёт.
            </p>
            <p>
              Сравнивать кандидатов одного времени, школы или круга тем труднее.
              Отдельного замера по каждой такой группе пока нет, поэтому данные не
              поддерживают уверенных выводов о донской, одесской или деревенской
              школах.
            </p>
          </div>
          <div className="grid cols-2" style={{ alignContent: "start" }}>
            <Stat label="точность · первый замер" value={fmtScore(HEADLINE.accuracy, 3)} accent="var(--gold)" parade />
            <Stat label="macro-F1 · первый замер" value={fmtScore(HEADLINE.macroF1, 3)} accent="var(--icon-blue)" hint="Каждый автор получает одинаковый вес." />
          </div>
        </div>

        {/* три рычага */}
        <div className="reveal module">
          <h3>Что нужно проверить дальше</h3>
          <div className="grid cols-3">
            {LEVERS.map((l) => (
              <Card key={l.title} padding={24}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span className="mono" style={{ fontSize: 13, color: l.accent, fontWeight: 700, letterSpacing: "0.04em" }}>{l.num}</span>
                    <span style={{ width: 22, height: 2, background: l.accent }} />
                  </div>
                  <h3 style={{ margin: 0, color: l.accent }}>{l.title}</h3>
                  <p className="muted" style={{ margin: 0 }}>{l.body}</p>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* протокол как метод */}
        <div className="reveal module">
          <p className="prose muted">
            Результату нужны оценка разброса, проверка на случайность и ясная схема
            проверки. Без них одна метрика не показывает, насколько надёжен вывод.
          </p>
          <p className="verdict">
            Хорошая проверка не обязана отвечать на каждый вопрос. Её задача —
            показать, где данных достаточно для вывода, а где ответ пока лучше
            отложить.
          </p>
        </div>
      </div>
    </section>
  );
}
