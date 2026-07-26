import { Card, Stat, ConfidenceBar } from "@dmitrymake/rk-ui";
import { AUTHOR_RECALL, HEADLINE } from "../data.js";
import { CORPUS } from "../corpus.js";
import { BENCH } from "../segdata.js";
import { fmtScore, fmtWordsM } from "../format.js";
import HistoricalHeadlineNotice from "../components/HistoricalHeadlineNotice.jsx";

// рус. плюрализация: f=[ед., 2-4, мн.]
const plu = (n, f) => { const a = n % 10, b = n % 100; return f[(a === 1 && b !== 11) ? 0 : (a >= 2 && a <= 4 && (b < 12 || b > 14)) ? 1 : 2]; };

const SEVERITY = {
  error: "var(--danger)",
  warn: "var(--warning)",
  info: "var(--icon-blue)",
};
// Метки находок по-русски (без сырых статусов «error/warn/info» в читаемом тексте).
const SEVERITY_LABEL = {
  error: "утечка",
  warn: "внимание",
  info: "к сведению",
};

// Находки валидатора — простыми словами, без жаргона (провенанс: docs/corpus_validation.json).
// Единственный множитель тянем из данных (перекос по объёму между авторами). Точка дрейфа:
// имена и объёмы самого много-/немногословного автора (Достоевский/Волошин) и процент
// совпадения почти-дублей в site-data пока не выведены — см. needed_keys; при пересборке
// корпуса крайние авторы могут смениться, поэтому имена подставлять из данных, а не литералом.
const FINDINGS = [
  {
    severity: "warn",
    title: "Почти-дубли внутри одного автора",
    text:
      "У Пруткова разные сборники афоризмов почти дословно повторяют друг друга, у Гаршина " +
      "встречаются пересекающиеся куски. Такой повтор — скрытая подсказка: знакомый по " +
      "обучению отрывок всплывает на проверке и завышает точность. Валидатор помечает эти " +
      "места. Мы помним о них, когда судим, насколько цельный почерк у псевдонима-коллектива " +
      "Прутков.",
  },
  {
    severity: "warn",
    title: "Cross-work content leakage",
    text:
      "После исторического прогона обнаружилось, что разные work-id содержат вложенные " +
      "произведения и точное совпадение чанка. Они пересекали train/test-границу, поэтому " +
      "headline и производные интервалы/p отозваны до новой версии корпуса.",
  },
  {
    severity: "warn",
    title: `Перекос по объёму ${CORPUS.research.imbalanceRatio}×`,
    text:
      "У самого многословного автора, Достоевского, текста в сотни раз больше, чем у самого " +
      "немногословного, Волошина. В исторической арифметике такие авторы раздували accuracy, " +
      "поэтому рядом сохраняется описательная macro-F1 point. Её author-clustered интервал " +
      "недействителен и отозван; accuracy-интервал тоже не является текущим claim.",
  },
  {
    severity: "info",
    title: "Авторы с одной книгой — вне зачёта",
    text:
      "Гончаров, Григорович, Решетников, Волошин — по одной книге на каждого. Отложить эту " +
      "книгу для проверки нельзя: без неё у автора не остаётся образца, по которому его " +
      "узнавать. Поэтому в проверке по целым книгам они не участвуют.",
  },
  {
    severity: "info",
    title: "Дуэт и дневники — отдельно",
    text:
      "Ильф и Петров писали вдвоём, а дневники Николая II — не проза. Их вынесли из основного " +
      "зачёта и разбирают отдельно.",
  },
];

function recallAccent(r) {
  if (r >= 0.99) return "var(--success)";
  if (r === 0) return "var(--danger)";
  return "var(--warning)";
}

// Отсортированы от худших к лучшим — разрыв сразу виден сверху.
const RANKED = [...AUTHOR_RECALL].sort((a, b) => a.recall - b.recall);
// «X из Y книг» — сколько книг автора модель узнаёт верно (числа из данных, без литералов в прозе).
const recBooks = (id) => {
  const a = AUTHOR_RECALL.find((x) => x.id === id);
  if (!a) return { correct: 0, books: 0 };
  return { correct: Math.round(a.recall * a.books), books: a.books };
};
const recPhrase = (id) => { const { correct, books } = recBooks(id); return `${correct} из ${books} книг`; };
// Худший по узнаваемости автор исторического PD-only среза — динамически из данных.
const worstPd = BENCH.worstRecall; // { name, recall, books } из docs/validation_pd.json
const worstPdCorrect = Math.round(worstPd.recall * worstPd.books);

export default function Corpus() {
  return (
    <section className="section" id="corpus">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Корпус</p>
          <h2>{CORPUS.research.authors} {plu(CORPUS.research.authors, ["автор","автора","авторов"])}, {CORPUS.research.books} {plu(CORPUS.research.books, ["книга","книги","книг"])} — и брак в самих данных</h2>
          <p className="prose lead muted">
            Прежде чем спрашивать «чей это стиль», нужно проверить сами данные. До
            обучения валидатор — программа-ревизор — проходит по корпусу и ищет
            дубли, тексты-подсказки (те, что выдают ответ заранее) и перекос между
            авторами. Без такой проверки пара лишних процентов точности говорила бы
            не о стиле, а о дефекте данных, и вердикт был бы ненадёжным.
          </p>
        </div>
        <HistoricalHeadlineNotice compact />

        {/* Сводка */}
        <div className="grid cols-4 reveal module">
          <Stat label="Авторов" value={CORPUS.research.authors} accent="var(--icon-blue)" hint={`${CORPUS.benchmark.authors} в бенчмарке`} />
          <Stat label="Книг" value={CORPUS.research.books} accent="var(--gold)" />
          <Stat label="Слов" value={fmtWordsM(CORPUS.research.words)} accent="var(--cosmos)" />
          <Stat label="Перекос между авторами" value={CORPUS.research.imbalanceRatio + "×"} accent="var(--warning)" hint="у многословных авторов текста намного больше" />
        </div>

        {/* Находки валидатора */}
        <div className="grid cols-2 reveal module">
          {FINDINGS.map((f) => {
            const color = SEVERITY[f.severity];
            return (
              <Card
                key={f.title}
                padding={22}
                style={{ borderLeft: `3px solid ${color}` }}
              >
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 8 }}>
                  <span
                    className="chip"
                    style={{ color, borderColor: `color-mix(in srgb, ${color} 50%, transparent)` }}
                  >
                    {SEVERITY_LABEL[f.severity] || f.severity}
                  </span>
                  <h3 style={{ margin: 0, fontSize: "1.1rem" }}>{f.title}</h3>
                </div>
                <p className="prose muted" style={{ margin: 0, fontSize: 14.5 }}>
                  {f.text}
                </p>
              </Card>
            );
          })}
        </div>

        {/* Per-author recall */}
        <div className="reveal module">
          <h3>Где у модели падает точность</h3>
          <p className="prose muted" style={{ marginBottom: 8 }}>
            В историческом, ныне отозванном расчёте accuracy{" "}
            <span className="mono" style={{ color: "var(--success)" }}>{fmtScore(HEADLINE.accuracy)}</span>, а
            описательная macro-F1 point{" "}
            <span className="mono" style={{ color: "var(--warning)" }}>{fmtScore(HEADLINE.styloMacroF1)}</span>.
            Разность показывает влияние объёма внутри старой арифметики, но не является
            значимым или текущим сравнением. Провалы оседали на авторах, у которых текста мало.
            Олеша ({recPhrase("olesha")}) и Катаев ({recPhrase("kataev")}) не узнаются ни разу: одесская школа
            теряется среди соседей — Олеша уходит к Волчеку, Катаев — к А. Н. Толстому. Зощенко — {recPhrase("zoshenko")}.
            Козьма Прутков — {recPhrase("prutkov")} в полном бенчмарке из {CORPUS.benchmark.authors} авторов: это коллективный псевдоним, по определению не одна рука.
            А у Достоевского, Гоголя, Бунина и Чехова почерк ровный — их модель узнаёт всегда, без единого промаха.
            (Шолохов как спорный автор из этого бенчмарка исключён — его узнаваемость здесь не измеряется.)
          </p>
          {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
            <p className="mono muted" style={{ fontSize: 12, marginBottom: 8 }}>
              И при обучении длинная книга сейчас весит больше короткой. Пересчёт «одна книга — один голос»
              возможен после content-safe миграции; эти цифры уже отозваны.
            </p>
          )}
          <p className="prose muted" style={{ marginBottom: 8 }}>
            Каждая строка — историческая доля книг автора, которую модель пометила верно:
            модель учим на всех книгах автора, кроме одной, и проверяем на отложенной.
            Сверху вниз — от тех, кого модель узнаёт хуже всего, к тем, кого узнаёт всегда.
          </p>
          <p className="prose muted" style={{ marginBottom: 22 }}>
            Эти исторические агрегаты описывали весь старый корпус, а не пригодность метода к
            одному тексту. Для отдельного спорного текста важна узкая панель: близкие авторы той
            же школы, эпохи и типа письма — и то, куда именно утекает текст. Но даже эта
            диагностическая интерпретация требует content-safe корпуса. Один и тот же автор в
            старых панелях узнавался по-разному — например, {worstPd.name}.{" "}
            В историческом PD-only срезе классиков — тех, кто умер больше 70 лет назад ({CORPUS.pd.authors}{" "}
            {plu(CORPUS.pd.authors, ["автор", "автора", "авторов"])}; Олеша,
            Катаев и Зощенко в него не входят), — хуже всех давался именно он: старая модель
            пометила верно {worstPdCorrect} из {worstPd.books} книг. В полном историческом
            бенчмарке из {CORPUS.benchmark.authors} авторов тот же recall был ещё ниже —{" "}
            {recPhrase(worstPd.id)}. Это сохранённая арифметика ineligible snapshot, а не
            действующая оценка качества или основа нового авторского вывода.
          </p>

          <Card padding={20}>
            <div className="grid cols-2" style={{ gap: "12px 32px" }}>
              {RANKED.map((a) => (
                <ConfidenceBar
                  key={a.id}
                  value={a.recall}
                  valueText={fmtScore(a.recall)}
                  label={
                    <span style={{ color: a.recall >= 0.99 ? "var(--text-muted)" : "var(--text)" }}>
                      {a.name}{" "}
                      <span className="mono muted" style={{ fontSize: 11.5 }}>· {a.books} кн.</span>
                    </span>
                  }
                  accent={recallAccent(a.recall)}
                />
              ))}
            </div>
          </Card>
        </div>

        {/* Заметка про докачку классиков */}
        <p className="muted reveal" style={{ fontSize: 13.5, maxWidth: "62ch" }}>
          Часть профилей дополнена классикой из общественного достояния (свободные тексты):
          Гоголь, Чехов, Достоевский, Толстой. Корпус собран без скидок
          на чистоту — каждый текст проходит через тот же валидатор.
        </p>

        <p className="verdict reveal">
          <strong style={{ color: "var(--text)" }}>Сначала нужен content-safe корпус.</strong>{" "}
          Валидатор должен проверять не только дубли и целостность book-id, но и вложенные
          произведения между разными work-id. Старой группировки по книгам оказалось
          недостаточно: до миграции корпуса здесь нет действующего claim ни об accuracy, ни о
          macro-F1.
        </p>
      </div>
    </section>
  );
}
