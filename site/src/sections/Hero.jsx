import { Stat } from "@dmitrymake/rk-ui";
import RingStat from "../components/RingStat.jsx";
import { fmtPct, fmtWordsM } from "../format.js";
import { HEADLINE, MODELS } from "../data.js";
import { CORPUS } from "../corpus.js";
import ResearchUpdate from "../components/ResearchUpdate.jsx";

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
      <p className="eyebrow reveal in">Научпоп · как текст выдаёт автора</p>

      <div className="split" style={{ alignItems: "center" }}>
        <div>
          <h1 className="reveal in">
            Можно ли узнать<br />автора по стилю?
          </h1>
          <p className="sub reveal in">
            У каждого писателя есть языковые привычки: любимые союзы, длина фразы,
            ритм пунктуации, способ соединять слова. Мы собрали цифровые «почерки»
            русской прозы и проверяем на них литературные загадки — от «Тихого Дона»
            до второй редакции «Тараса Бульбы». Машина ищет не красивую цитату, а
            тысячи мелких повторяющихся решений, которые трудно контролировать
            сознательно.
          </p>

          <div className="hero-stats">
            <Stat label="авторов в корпусе" value={CORPUS.research.authors} accent="var(--icon-blue)" />
            <Stat label="книг" value={CORPUS.research.books} accent="var(--gold)" />
            <Stat label="слов" value={fmtWordsM(HEADLINE.words)} accent="var(--cosmos)" />
            <Stat label="литературных расследования" value="4" accent="var(--success)" />
          </div>
        </div>

        <div className="reveal in" style={{ display: "grid", placeItems: "center", gap: 18 }}>
          <RingStat
            frac={HEADLINE.accuracy}
            wide
            big={fmtPct(HEADLINE.accuracy, 1)}
            caption={<>книг распознано<br />в первом эксперименте</>}
            accent="var(--text)"
          />
          <p className="muted" style={{ margin: 0, fontSize: "1.02em", textAlign: "center", maxWidth: "40ch" }}>
            В первом прогоне модель сравнила {HEADLINE.books} {ruBooks(HEADLINE.books)} у{" "}
            {HEADLINE.authors} авторов. Полный профиль дал {fmtPct(HEADLINE.accuracy, 1)}
            {" "}верных ответов, простой «мешок слов» — {fmtPct(BOW.acc, 1)}. Это
            подсказало, что порядок слов, синтаксис и пунктуация несут дополнительный
            сигнал — и поставило следующий, более строгий эксперимент.
          </p>
          <ResearchUpdate />
        </div>
      </div>
    </header>
  );
}
