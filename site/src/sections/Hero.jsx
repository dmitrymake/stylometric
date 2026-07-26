import { Stat } from "@dmitrymake/rk-ui";
import RingStat from "../components/RingStat.jsx";
import { fmtScore, fmtRange, fmtP, fmtPct, fmtWordsM } from "../format.js";
import { HEADLINE, MODELS } from "../data.js";
import { CORPUS } from "../corpus.js";
import HistoricalHeadlineNotice from "../components/HistoricalHeadlineNotice.jsx";

const MF1 = HEADLINE.macroF1;                       // историческая описательная точка
// author-clustered 95% CI macro-F1 ОТОЗВАН (HEADLINE.macroF1CI === null): ресэмпл авторов меняет
// набор классов macro-усреднения → это не CI фиксированной функции. Показываем только точку.
const BOW = MODELS.find((m) => m.id === "bow_lr");

// Русское склонение слова «книга» после числа (ruBooks в data.js не экспортируется — локальная копия).
const ruBooks = (n) => {
  const mod100 = Math.abs(n) % 100, mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "книг";
  if (mod10 === 1) return "книга";
  if (mod10 >= 2 && mod10 <= 4) return "книги";
  return "книг";
};

export default function Hero() {
  return (
    <header className="hero wrap" id="top">
      <p className="eyebrow reveal in">Исследование · различаем авторов по стилю</p>

      <div className="split" style={{ alignItems: "center" }}>
        <div>
          <h1 className="reveal in">
            Кто написал<br />спорный текст
          </h1>
          <p className="sub reveal in">
            Стиль у каждого свой, как почерк. Важно не о чём человек пишет, а как:
            где ставит запятую, какими служебными словами сшивает речь, как строит
            фразу. Проект читает этот почерк в русской прозе. Спорный текст
            сравниваем с большим собранием книг и отделяем тему от стиля. Виден не
            просто ближайший автор — видно, насколько твёрдо держится вывод.
            Правильный протокол убирает из обучения не только проверяемую книгу,
            но и всё совпадающее с ней содержание. Исторический прогон ниже
            выполнил первое правило, но нарушил второе, поэтому его результат отозван.
          </p>

          <HistoricalHeadlineNotice />
          <div className="hero-stats">
            <Stat label="историческая точка macro-F1" value={fmtScore(MF1)} accent="var(--success)" parade hint="Описательная арифметика ineligible corpus snapshot; macro-F1 CI дополнительно отозван, inferential use запрещён." />
            <Stat label="историческая accuracy" value={fmtScore(HEADLINE.accuracy)} accent="var(--text-muted)" hint={`исторический интервал ${fmtRange(HEADLINE.accCIAuthor[0], HEADLINE.accCIAuthor[1])}, медиана ${fmtScore(HEADLINE.accBootstrapMedian)}; не текущая оценка точности`} />
            <Stat label="авторов / книг · весь корпус" value={`${CORPUS.research.authors} / ${CORPUS.research.books}`} accent="var(--icon-blue)" hint={`${CORPUS.lobo.tested_authors} / ${CORPUS.lobo.books} в историческом LOBO-срезе; cross-work leakage обнаружен позднее`} />
            <Stat label="слов · полный корпус" value={fmtWordsM(HEADLINE.words)} accent="var(--cosmos)" />
          </div>
          {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
            <p className="mono muted" style={{ marginTop: 12, fontSize: 12, maxWidth: "52ch" }}>
              Пока при обучении длинная книга весит больше короткой. Пересчёт «одна книга — один голос»
              возможен только после миграции корпуса; нынешнее число отозвано, а не просто ожидает небольшой поправки.
            </p>
          )}
        </div>

        <div className="reveal in" style={{ display: "grid", placeItems: "center", gap: 18 }}>
          <RingStat
            frac={MF1}
            wide
            big={fmtScore(MF1)}
            caption={<>историческая точка macro-F1<br />corpus snapshot непригоден · интервал отозван</>}
            accent="var(--text)"
          />
          <p className="muted" style={{ margin: 0, fontSize: "1.02em", textAlign: "center", maxWidth: "40ch" }}>
            Проверяем, нужен ли стиль. Сравниваем полную модель с простым «мешком слов» — он смотрит только на то, какие слова встречаются, но не на то, как они расставлены.
            В историческом расчёте на {HEADLINE.authors} авторах ({HEADLINE.books} {ruBooks(HEADLINE.books)}) арифметика дала {fmtPct(HEADLINE.accuracy, 1)} против {fmtPct(BOW.acc, 1)}.
            McNemar p {fmtP(BOW.p)} сохранён только как историческая диагностика и не подтверждает текущий claim о преимуществе.
            Этот результат можно проверять заново лишь после content-safe миграции корпуса.
          </p>
        </div>
      </div>
    </header>
  );
}
