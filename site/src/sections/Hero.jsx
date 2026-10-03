import { EditorialHeader, EditorialContents } from "@dmitrymake/rk-ui";
import { MEASUREMENT } from "../data.js";
import { fmtInt } from "../format.js";

export default function Hero() {
  return <div className="wrap" id="top">
    <EditorialHeader eyebrow="01 / Как это работает" title="Как измеряют авторскую манеру" standfirst={<p>Служебные слова, синтаксис и пунктуация помогают сравнивать тексты. Разбираем, как устроено это сравнение, что оно показывает и где заканчивается его объяснительная сила.</p>} metadata={<><span>Метод и четыре литературных случая</span><span>{fmtInt(MEASUREMENT.works)} произведений в проверке метода</span></>} />
    <div className="article-opening">
      <p>Авторская манера складывается из повторяющихся решений: как строить фразу, какие слова связывать, где ставить знак препинания. Стилометрия переводит часть этих привычек в измерения. Их сходство приходится отделять от общей темы, жанра и состава сравниваемых книг.</p>
      <EditorialContents items={[
        { href: "#framework/problem", label: "Какой вопрос решает сравнение" },
        { href: "#framework/method", label: "Как устроена проверка" },
        { href: "#framework/results", label: "Что показывают результаты" },
        { href: "#framework/conclusion", label: "Как перейти к вопросу об авторстве" },
        { href: "#framework/repro", label: "Как повторить сравнение" },
      ]} />
    </div>
  </div>;
}
