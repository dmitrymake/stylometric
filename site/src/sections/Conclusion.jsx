import { Card, Stat } from "@dmitrymake/rk-ui";
import { HEADLINE, MODELS, AUTHOR_RECALL, CHANNELS } from "../data.js";
import { BENCH_EXT } from "../segdata.js";
import { fmtPct, fmtScore, fmtInt } from "../format.js";

const BOW = MODELS.find((m) => m.id === "bow_lr");
const PROZA_NEURO = BENCH_EXT.prozaNeuro;                                  // ruBERT-tiny2 без дообучения
const PROZA_LEADER = BENCH_EXT.prozaLeader;                                // лучший классический метод на той же прозе
const DSP_TOP1 = CHANNELS.byId("DSP (suffixes)").top1;                     // слабейший канал первого полного среза (43 автора / 251 книга)

// склонения без литералов о результатах — только грамматика
const ruBooks = (n) => {
  const t = Math.abs(n) % 100, o = t % 10;
  if (t >= 11 && t <= 14) return "книг";
  if (o === 1) return "книга";
  if (o >= 2 && o <= 4) return "книги";
  return "книг";
};
const ruAuthors = (n) => {
  const t = Math.abs(n) % 100, o = t % 10;
  if (t >= 11 && t <= 14) return "авторов";
  if (o === 1) return "автор";
  if (o >= 2 && o <= 4) return "автора";
  return "авторов";
};
const listNames = (arr) => arr.map((a) => a.name).join(" и ");

// Примеры «мало книг» берутся из данных, не из литералов: авторы с нулевой узнаваемостью
// при минимальном числе книг vs авторы с тем же скромным объёмом, но верные во всех случаях.
const zeroRecall = AUTHOR_RECALL.filter((a) => a.recall === 0).sort((a, b) => a.books - b.books);
const minZeroBooks = zeroRecall.length ? zeroRecall[0].books : 0;
const zeroLowMin = zeroRecall.filter((a) => a.books === minZeroBooks);
const perfectSmall = AUTHOR_RECALL
  .filter((a) => a.recall === 1 && a.books <= minZeroBooks + 1)
  .sort((a, b) => a.books - b.books)
  .slice(0, 2);
// «столько же / почти столько же» — сравнение чисел из данных, без литерала о равенстве объёмов
const sameOrAlmost =
  perfectSmall.length && perfectSmall[0].books === minZeroBooks ? "ровно столько же" : "почти столько же";
// Проверяем на данных, а не на глаз: все нераспознанные авторы лежат в нижнем краю по числу книг.
const failuresAllFewBooks = zeroRecall.length > 0 && zeroRecall.every((a) => a.books <= minZeroBooks + 1);
const failClusterLine = failuresAllFewBooks
  ? " И все, кого он путает во всех случаях, — из нижнего края по числу книг."
  : "";

const LEVERS = [
  {
    num: "01",
    accent: "var(--gold)",
    title: "Больше текстов на автора",
    body: `${listNames(zeroLowMin)} — по ${minZeroBooks} ${ruBooks(minZeroBooks)} на каждого, и метод не узнаёт их ни разу: 0 попаданий из ${minZeroBooks}.${failClusterLine} Но объём — не единственный фактор: ${listNames(perfectSmall)} — ${sameOrAlmost}, ${perfectSmall[0].books} ${ruBooks(perfectSmall[0].books)}, а узнаются во всех случаях. Больше книг на автора — самый простой рычаг сузить этот разброс.`,
  },
  {
    num: "02",
    accent: "var(--icon-blue)",
    title: "Отделить тему от почерка",
    body: "Тема пока примешивается к почерку: когда пишут про один и тот же край и уклад, метод цепляется за слова темы. Выровнять жанры и приглушить самые частые тематические слова — и на первый план выйдет почерк, а не материал.",
  },
  {
    num: "03",
    accent: "var(--success)",
    title: "Нейросеть под стиль",
    body: `Большие языковые модели могли бы поднять потолок — но лишь если учить их читать почерк, а не тему. На внешней русской прозе нейросеть ruBERT-tiny2 уступает простой классике (${fmtScore(PROZA_NEURO, 2)} против ${fmtScore(PROZA_LEADER, 2)}). Её взяли как есть, без подстройки под авторов, — и она ловит, о чём текст, а не как он написан. Это не приговор нейросетям: модели, обученные под автора, здесь не участвуют — речь об одном слабом опорном методе.`,
  },
];

export default function Conclusion() {
  return (
    <section className="section" id="conclusion">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Вывод</p>
          <h2>Почерк виден, когда данных хватает</h2>
          <p className="prose lead muted">
            Когда книг мало и авторы далеки друг от друга, автора выдаёт сама тема —
            по словам видно, кто писал. Стоит собрать больше имён, пишущих об одном,
            и лексика начинает подводить. Держит почерк другое: как построена фраза,
            какие служебные слова, где стоят знаки.
          </p>
        </div>

        {/* признаки окупаются */}
        <div className="split reveal module" style={{ alignItems: "start" }}>
          <div className="prose">
            <p className="verdict">
              Первый эксперимент подсказал, что сочетание синтаксиса, служебных слов
              и пунктуации различает авторов лучше, чем один словарь:{" "}
              <strong style={{ color: "var(--gold)" }}> {fmtPct(HEADLINE.accuracy, 1)} против {fmtPct(BOW.acc, 1)}</strong>,
              соответственно. Насколько велик этот выигрыш после очистки корпуса,
              должен показать новый расчёт.
            </p>
            <p>
              Особенно интересны группы, где кандидаты принадлежат одной
              школы. Донская школа, одесситы, деревенщики стоят тесно: общий край, общее
              время, общий круг сюжетов. На таких соседях голая лексика легко путает автора
              с его же школой — а синтаксис и расстановка знаков ещё различают руку.
              Отдельного замера по каждой школе пока нет: это вопрос для следующего опыта.
            </p>
          </div>
          <div className="grid cols-2" style={{ alignContent: "start" }}>
            <Stat label="верные книги · первый опыт" value={fmtScore(HEADLINE.accuracy, 3)} accent="var(--gold)" parade />
            <Stat label="macro-F1 · первый опыт" value={fmtScore(HEADLINE.macroF1, 3)} accent="var(--icon-blue)" hint="Каждый автор получает одинаковый вес." />
          </div>
        </div>

        {/* три рычага */}
        <div className="reveal module">
          <h3>Что усилит следующие проверки</h3>
          <p className="prose muted" style={{ marginBottom: 22 }}>
            Дело не в эффектных новых признаках. Самый слабый канал — разбор по
            хвостам-суффиксам слов (DSP): в том же первом эксперименте ({fmtInt(HEADLINE.authors)}{" "}
            {ruAuthors(HEADLINE.authors)}, {fmtInt(HEADLINE.books)} {ruBooks(HEADLINE.books)})
            верных ответов лишь {fmtPct(DSP_TOP1)}. Настоящие рычаги проще: больше текстов
            на автора, ровнее подобранные жанры и отдельная работа с темой.
          </p>
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
            Правило одно для любого исхода: и удачное опознание, и провал, и тупик идут
            с границами погрешности и проверкой на случайность. Цифра без протокола —
            украшение, а не результат.
          </p>
          <p className="verdict">
            Сильная стилометрия не обязана каждый раз выдавать громкий ответ. Её работа —
            отделить устойчивый след от правдоподобной догадки. Видно, где вывод держат
            данные, где версия рушится, а где вопрос честнее передать архиву и текстологам.
          </p>
        </div>
      </div>
    </section>
  );
}
