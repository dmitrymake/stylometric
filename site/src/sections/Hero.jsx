import { Stat } from "@dmitrymake/rk-ui";
import { fmtWordsM } from "../format.js";
import { HEADLINE } from "../data.js";
import { CORPUS } from "../corpus.js";
import ResearchUpdate from "../components/ResearchUpdate.jsx";

export default function Hero() {
  return (
    <header className="hero wrap" id="top">
      <p className="eyebrow reveal in">Стилометрия · сравнение авторской манеры</p>

      <div className="split" style={{ alignItems: "center" }}>
        <div>
          <h1 className="reveal in">
            Можно ли узнать<br />автора по стилю?
          </h1>
          <p className="sub reveal in">
            Стилометрия сравнивает повторяющиеся особенности языка: как автор строит
            фразы, расставляет знаки препинания и использует служебные слова —
            например, союзы и предлоги. В проекте эти признаки применяются к четырём
            вопросам об авторстве русских текстов.
          </p>

          <div className="hero-stats">
            <Stat label="авторов в корпусе" value={CORPUS.research.authors} accent="var(--icon-blue)" />
            <Stat label="книг" value={CORPUS.research.books} accent="var(--gold)" />
            <Stat label="слов" value={fmtWordsM(HEADLINE.words)} accent="var(--cosmos)" />
            <Stat label="задачи об авторстве" value="4" accent="var(--success)" />
          </div>
        </div>

        <div className="reveal in" style={{ display: "grid", placeItems: "center", gap: 18 }}>
          <ResearchUpdate />
        </div>
      </div>
    </header>
  );
}
