import { Stat } from "@dmitrymake/rk-ui";
import RingStat from "../components/RingStat.jsx";
import { fmtScore, fmtRange, fmtP, fmtPct, fmtWordsM } from "../format.js";
import { HEADLINE, MODELS } from "../data.js";
import { CORPUS } from "../corpus.js";

const MF1 = HEADLINE.macroF1;                       // точечная оценка на пуле (оптимистичный край)
const MF1_LO = HEADLINE.macroF1CI[0], MF1_HI = HEADLINE.macroF1CI[1];   // author-clustered 95% CI
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
            Правило честное: проверяемую книгу целиком убираем из обучения. Так
            текст не подсказывает ответ.
          </p>

          <div className="hero-stats">
            <Stat label="точность по авторам (macro-F1) · 95% разброс" value={fmtRange(MF1_LO, MF1_HI)} accent="var(--success)" parade hint={`единая оценка на всех авторах сразу ${fmtScore(MF1)} — оптимистичный край: на отдельных наборах авторов бывает и ниже`} />
            <Stat label="общая точность (accuracy)" value={fmtScore(HEADLINE.accuracy)} accent="var(--text-muted)" hint={`разброс по авторам ${fmtRange(HEADLINE.accCIAuthor[0], HEADLINE.accCIAuthor[1])}, середина ${fmtScore(HEADLINE.accBootstrapMedian)}`} />
            <Stat label="авторов / книг · весь корпус" value={`${CORPUS.research.authors} / ${CORPUS.research.books}`} accent="var(--icon-blue)" hint={`${CORPUS.lobo.tested_authors} проверены без подсказок (${CORPUS.lobo.books} ${ruBooks(CORPUS.lobo.books)} в строгой проверке)`} />
            <Stat label="слов · полный корпус" value={fmtWordsM(HEADLINE.words)} accent="var(--cosmos)" />
          </div>
        </div>

        <div className="reveal in" style={{ display: "grid", placeItems: "center", gap: 18 }}>
          <RingStat
            frac={(MF1_LO + MF1_HI) / 2}
            wide
            big={fmtRange(MF1_LO, MF1_HI)}
            caption={<>точность по авторам · 95% разброс<br />(единая оценка {fmtScore(MF1)} — оптимистичный край)</>}
            accent="var(--text)"
          />
          <p className="muted" style={{ margin: 0, fontSize: "1.02em", textAlign: "center", maxWidth: "40ch" }}>
            Проверяем, нужен ли стиль. Сравниваем полную модель с простым «мешком слов» — он смотрит только на то, какие слова встречаются, но не на то, как они расставлены.
            В строгой проверке на {HEADLINE.authors} авторах ({HEADLINE.books} {ruBooks(HEADLINE.books)}) полная модель впереди: {fmtPct(HEADLINE.accuracy, 1)} против {fmtPct(BOW.acc, 1)}.
            Случайным совпадением такой разрыв не объяснить (тест МакНемара, p {fmtP(BOW.p)}).
            Значит, строй фразы, знаки и служебные слова добавляют узнаваемость поверх выбора слов.
          </p>
        </div>
      </div>
    </header>
  );
}
