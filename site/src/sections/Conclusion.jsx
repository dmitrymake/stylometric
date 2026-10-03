import { Card, Stat } from "@dmitrymake/rk-ui";
import { HEADLINE, MODELS, AUTHOR_RECALL, MEASUREMENT } from "../data.js";
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
          <p className="eyebrow">Следующий шаг</p>
          <h2>От сравнения профилей к вопросу об авторстве</h2>
          <p className="prose lead muted">
            Ранжирование показывает, к каким произведениям текст ближе по выбранным
            признакам. Поиск участков помогает увидеть изменения внутри книги.
            Для исторического вывода оба ответа нужно соотнести с составом эталонов,
            изданиями и тем, какие различия способен заметить метод.
          </p>
        </div>
        <div className="split reveal module">
          <div className="prose">
            <h3>«Двенадцать стульев»</h3>
            <p>
              Сравнение с Булгаковым требует совместной прозы Ильфа и Петрова вне
              проверяемой дилогии. Поиск отдельных рук соавторов — другая задача:
              для неё нужны сопоставимые сольные произведения, а различие записных
              книжек и военных очерков может отражать жанр.
            </p>
          </div>
          <div className="prose">
            <h3>«Тихий Дон»</h3>
            <p>
              Здесь важен выбор между ранними рассказами и поздней прозой в эталоне,
              а также общая донская тема у сравниваемых авторов. Сходство с текстами
              под именем Шолохова устанавливает отношение между произведениями;
              достоверность авторских меток остаётся отдельным основанием.
            </p>
          </div>
        </div>
        <p className="prose reveal">
          Завершённое сравнение показывает, как меняются ответы на известных авторах
          при другом обучении и ограничении признаков. Пакет с балансировкой даёт
          {" "}{MEASUREMENT.cells.find((cell) => cell.cell === "A4").accuracy.current.correct} верных ответов
          из {MEASUREMENT.works} в обоих вариантах признаков. Это основание для дальнейших
          проверок метода. Для новых выводов о романах нужны отдельные сравнения с проверенным
          составом эталонных произведений.
        </p>
        <details className="reveal module">
          <summary>Дополнительные наблюдения первого эксперимента</summary>
          <p className="prose">
            Сочетание признаков дало {fmtPct(HEADLINE.accuracy, 1)} верных ответов,
            модель по частотам слов — {fmtPct(BOW.acc, 1)}.
            Подробное сравнение показано выше в результатах первого эксперимента.
          </p>
          <div className="grid cols-2">
            <Stat label="точность · первый замер" value={fmtScore(HEADLINE.accuracy, 3)} accent="var(--gold)" parade />
            <Stat label="macro-F1 · первый замер" value={fmtScore(HEADLINE.macroF1, 3)} accent="var(--icon-blue)" hint="Каждый автор получает одинаковый вес." />
          </div>
          <div className="grid cols-3" style={{ marginTop: 24 }}>
            {LEVERS.map((lever) => (
              <Card key={lever.title} padding={24}>
                <h3 style={{ color: lever.accent }}>{lever.title}</h3>
                <p className="muted">{lever.body}</p>
              </Card>
            ))}
          </div>
        </details>
      </div>
    </section>
  );
}
