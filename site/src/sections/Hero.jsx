import { Stat } from "@dmitrymake/rk-ui";
import { fmtInt } from "../format.js";
import { MEASUREMENT } from "../data.js";

export default function Hero({ onChooseCase } = {}) {
  return (
    <header className="hero wrap" id="top">
      <p className="eyebrow reveal in">Стилометрия · сравнение авторской манеры</p>

      <div className="split" style={{ alignItems: "start" }}>
        <div>
          <h1 className="reveal in">
            Чей стиль<br />в тексте?
          </h1>
          <p className="sub reveal in">
            «Двенадцать стульев» сравниваем с прозой Ильфа и Петрова, Булгакова и
            их литературных соседей. «Тихий Дон» — с ранними и поздними произведениями
            под именем Шолохова и прозой донских авторов. Вопрос в том, какие сходства
            сохраняются при смене книг, признаков и состава сравнения.
          </p>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 24 }}>
            <button className="chapter-btn" type="button" onClick={() => onChooseCase?.("ilfpetrov")}>
              «12 стульев» и Булгаков
            </button>
            <button className="chapter-btn" type="button" onClick={() => onChooseCase?.("sholokhov")}>
              «Тихий Дон» и Шолохов
            </button>
          </div>
          <div className="hero-stats">
            <Stat label="произведений в новом замере" value={fmtInt(MEASUREMENT.works)} accent="var(--gold)" />
            <Stat label="авторов в проверке" value={MEASUREMENT.testedAuthors} accent="var(--icon-blue)" />
          </div>
        </div>

        <div className="prose reveal in">
          <p>
            Авторская манера складывается из повторяющихся решений: какие служебные
            слова употреблять, как строить фразу, где ставить знак препинания.
            Стилометрия переводит часть этих привычек в измерения и строит профиль текста.
          </p>
          <p>
            Профиль помогает ранжировать кандидатов и искать участки, отличающиеся от
            окружающего повествования. Сходство с другим автором приходится проверять:
            общая тема и жанр тоже сближают тексты.
          </p>
        </div>
      </div>
    </header>
  );
}
