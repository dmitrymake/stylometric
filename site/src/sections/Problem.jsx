import { Card, Badge } from "@dmitrymake/rk-ui";
import HistoricalHeadlineNotice from "../components/HistoricalHeadlineNotice.jsx";

const TRAPS = [
  {
    num: "01",
    accent: "var(--cinnabar)",
    title: "Подсмотренный ответ",
    body: "Разрежешь одну книгу на куски, часть отдашь на обучение, часть — на проверку — и модель уже видела ответ. Общий словарь и общие частоты слов делают то же исподтишка. Она узнаёт знакомый текст, а не автора. Точность взлетает, доказательства нет. Это и есть утечка данных.",
    fix: "Для нового вывода нужен content-component holdout: вместе с проверяемой книгой из fit уходят все вложенные и совпадающие work-id. Исторический book-id LOBO этого не обеспечил, поэтому его headline отозван.",
  },
  {
    num: "02",
    accent: "var(--icon-blue)",
    title: "Тема вместо стиля",
    body: "Автор часто прикован к своей теме: донская степь, одесский двор, политический памфлет, дневник. Модель хватается за материал и выдаёт его за почерк. Сменишь тему — и «автор» пропадает.",
    fix: "Поэтому смысловые слова идут не в одиночку — рядом со служебными словами, синтаксическими связями, частями речи и пунктуацией: тем, что от темы не зависит. И отдельной проверкой — на жанр.",
  },
  {
    num: "03",
    accent: "var(--gold)",
    title: "Голая цифра",
    body: "«Точность 95%» сама по себе ничего не значит. Как мерили? Повезло ли с выборкой? И не даст ли столько же метод из трёх строк кода? Без ответов на эти три вопроса цифра — просто цифра.",
    fix: "Допустимый новый claim должен описывать content-safe проверку, неопределённость и простые baseline. Для исторического headline это условие не выполнено: его интервалы и p сохранены только для аудита.",
  },
];

export default function Problem() {
  return (
    <section className="section" id="problem">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Проблема</p>
          <h2>Когда стилометрия обманывает</h2>
          <p className="prose lead muted">
            Стилометрия читает не сюжет, а мелочи, которые автор повторяет не думая:
            любимые союзы, длину фраз, привычку к запятой. По ним она и указывает на автора.
            Но у этого метода есть три способа обмануть самого себя.
          </p>
        </div>
        <HistoricalHeadlineNotice compact />

        <div className="grid cols-3 reveal">
          {TRAPS.map((t) => (
            <Card key={t.title} padding={24}>
              <div style={{ display: "flex", flexDirection: "column", gap: 14, height: "100%" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span className="mono" style={{ fontSize: 13, color: t.accent, fontWeight: 700, letterSpacing: "0.04em" }}>{t.num}</span>
                  <span style={{ width: 22, height: 2, background: t.accent }} />
                </div>
                <h3 style={{ margin: 0, color: t.accent }}>{t.title}</h3>
                <p className="muted" style={{ margin: 0 }}>{t.body}</p>
                <p className="mono" style={{ margin: "auto 0 0", fontSize: 13, color: "var(--text)", paddingTop: 8 }}>
                  {t.fix}
                </p>
              </div>
            </Card>
          ))}
        </div>

        <div className="split reveal" style={{ alignItems: "center" }}>
          <p className="note">
            Дальше эти правила разбирают конкретные загадки. «Тихий Дон» — Шолохов, Крюков
            или несколько рук? «12 стульев» — дуэт Ильфа и Петрова или тайный Булгаков?
            «Тарас Бульба» — кто дописал вторую редакцию: сам Гоголь, писавший набело Анненков
            или готовивший издание Прокопович? Дневник Николая II — живая рука, необычная для
            дневника речь или редакторская правка? А рядом — карта пределов: где метод разделяет
            известные школы («Современник»), а где честно отвечает «не знаю» («Колокол», романы
            Некрасова и Панаевой, пара «учитель↔ученик», слабая панель «Будильника»).
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
            <Badge tone="base">«Тихий Дон»</Badge>
            <Badge tone="base">Шолохов?</Badge>
            <Badge tone="facultative">Крюков?</Badge>
            <Badge tone="base">«12 стульев»</Badge>
            <Badge tone="base">Ильф · Петров?</Badge>
            <Badge tone="facultative">Булгаков?</Badge>
            <Badge tone="base">«Тарас Бульба»</Badge>
            <Badge tone="base">Гоголь?</Badge>
            <Badge tone="facultative">Анненков?</Badge>
            <Badge tone="facultative">Прокопович?</Badge>
            <Badge tone="base">Дневник Николая II</Badge>
            <Badge tone="facultative">карта режимов</Badge>
          </div>
        </div>
      </div>
    </section>
  );
}
