import { Card, CodeBlock } from "@dmitrymake/rk-ui";
import { REPRO, BENCH } from "../segdata.js";
import { LOBO_STRICT, HEADLINE } from "../data.js";
import { CORPUS } from "../corpus.js";
import { fmtScore, fmtPct } from "../format.js";
import HistoricalHeadlineNotice from "../components/HistoricalHeadlineNotice.jsx";

const GUARANTEES = [
  {
    title: "Повторяемо",
    accent: "var(--success)",
    body: `Один файл настроек (configs/default.yaml) и замороженные версии библиотек (requirements.lock). Исторические байты и диагностики воспроизводимы, но воспроизводимость не делает отозванный headline допустимым. Все ${REPRO.gatesBitExact} контрольных прогонов спорных кейсов повторяются бит-в-бит. Самый тяжёлый (${REPRO.longestGateName}) считается дольше остальных, но сверяется так же точно.`,
  },
  {
    title: "Быстро",
    accent: "var(--icon-blue)",
    body: "Языковой разбор текста делается один раз и ложится на диск, признаки отрывков считаются заранее. Тяжёлую часть машина не повторяет — каждый следующий прогон только разгоняется.",
  },
  {
    title: "Граница усилена",
    accent: "var(--gold)",
    body: "Старый work-id split не поймал вложенное содержание между разными произведениями. Новая версия должна делить по content-компонентам и только затем заново оценивать метод.",
  },
];

export default function Repro() {
  return (
    <section className="section" id="repro">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Можно повторить у себя</p>
          <h2>От корпуса до вердикта за один прогон</h2>
          <p className="prose lead muted">
            Хороший результат — тот, который перепроверит кто угодно. Здесь нет скрытого шага, что
            срабатывает лишь на чужой машине. Весь путь — от сборки корпуса до готового отчёта —
            запускается одной командой. На чистой машине скрипт докачивает классиков по
            списку-манифесту, заново собирает открытую часть корпуса и воспроизводит
            историческую арифметику — {fmtScore(BENCH.topTop1, 3)}, около{" "}
            {fmtPct(BENCH.topTop1)} попаданий. Из-за cross-work content leakage это
            не действующая оценка точности и не реабилитация отозванного полного LOBO.
          </p>
          <HistoricalHeadlineNotice compact />
          <p className="prose muted">
            Эта открытая часть меньше и различимее: {BENCH.nAuthors} автора, {BENCH.nBooks} книг, и
            все они — хорошо узнаваемые классики. Историческое отозванное число полного среза —{" "}
            {fmtScore(LOBO_STRICT.styloFullLobo, 3)}. Оно получено на ineligible corpus snapshot. В полный
            срез входит больше имён ({CORPUS.benchmark.authors} против {BENCH.nAuthors}) и другой
            состав; числа получены на разных наборах и не поддерживают текущий comparative claim.
          </p>
          {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
            <p className="mono muted" style={{ fontSize: 12 }}>
              И ещё: при обучении длинная книга сейчас весит больше короткой. Пересчёт «одна книга —
              один голос» не исправляет content leakage; сначала нужна новая версия корпуса.
            </p>
          )}
          <p className="prose muted">
            В полный срез входят ещё и книги под защитой авторских прав. Они живут только на локальной
            машине, в общий корпус не отдаются. Можете добавить их самостоятельно.
          </p>
        </div>

        {/* Полный пайплайн */}
        <div className="module reveal" style={{ display: "grid", gap: 16 }}>
          <CodeBlock language="bash" title="весь прогон одной командой">
            {`./run.sh all   # validate → split → warm → train → sweep → evaluate → predict → report`}
          </CodeBlock>

          <p className="prose muted" style={{ margin: 0, fontSize: 13.5 }}>
            Восемь стадий: проверка данных → нарезка книг на отрывки → прогрев кэша
            признаков → обучение → отбор признаков → оценка → предсказание → отчёт.
          </p>

          <CodeBlock language="bash" title="по шагам">
            {`./run.sh sweep      # отключаем признаки по одному — что реально работает
./run.sh evaluate   # проверка по целым книгам + простые методы-ориентиры для сравнения (классическая Дельта)
./run.sh predict    # определяем автора неизвестного текста по профилям авторов`}
          </CodeBlock>
        </div>

        {/* Гарантии */}
        <div className="grid cols-3 module reveal">
          {GUARANTEES.map((g) => (
            <Card key={g.title} padding={22} style={{ borderTop: `3px solid ${g.accent}` }}>
              <h3 style={{ margin: "0 0 8px", fontSize: "1.2rem", color: g.accent }}>{g.title}</h3>
              <p className="prose muted" style={{ margin: 0, fontSize: 14.5 }}>
                {g.body}
              </p>
            </Card>
          ))}
        </div>

        {/* Инвариант no-leakage */}
        <p className="verdict reveal">
          Главный тест держит ровно одно правило: на каждом шаге проверки модель
          <strong style={{ color: "var(--cinnabar)" }}> не видела проверяемую книгу</strong>.
          Подсмотреть ответ в проверяемом тексте завышает точность, ничего не говоря о стиле.
          Теперь content-component gate должен ронять сборку и при совпадении под разными work-id;
          исторический snapshot этого требования не выполнял.
        </p>
      </div>
    </section>
  );
}
