import { Card, Badge } from "@dmitrymake/rk-ui";

const TRAPS = [
  {
    num: "01",
    accent: "var(--cinnabar)",
    title: "Пересечение текстов",
    body: "Если одно и то же содержание попадает в обучение и проверку, оценка может быть завышена. Тогда данные не показывают, как метод работает на новых текстах.",
    fix: "Проверка: исключать проверяемую книгу и тексты с тем же содержанием.",
  },
  {
    num: "02",
    accent: "var(--icon-blue)",
    title: "Тема и авторская манера",
    body: "Тема и жанр могут быть связаны с конкретным автором. Тогда метод различает лексику содержания, а не авторскую манеру.",
    fix: "Проверка: контролировать жанр и сравнивать группы признаков, чтобы оценить вклад темы.",
  },
  {
    num: "03",
    accent: "var(--gold)",
    title: "Метрика без контекста",
    body: "Значение «Точность 95%» мало говорит без состава выборки и схемы проверки. Без простого базового метода неясно, оправдана ли сложность модели.",
    fix: "Проверка: указывать протокол, неопределённость и сравнение с базовыми моделями.",
  },
];

export default function Problem() {
  return (
    <section className="section" id="problem">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Надёжность</p>
          <h2>Что может исказить результат</h2>
          <p className="prose lead muted">
            Стилометрический результат зависит не только от признаков текста, но и от
            корпуса и схемы проверки. Особенно важны три риска: пересечение текстов,
            связь автора с темой и метрика без контекста.
          </p>
        </div>
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
            Эти условия применяются к четырём вопросам об авторстве; кандидаты
            перечислены рядом. В контрольной панели метод различает группы критиков
            «Современника» по целым текстам, но это не общий вывод о школах. Для
            «Колокола», Некрасова с Панаевой, пары «учитель↔ученик» и «Будильника»
            данных недостаточно для уверенного вывода.
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
            <Badge tone="facultative">границы метода</Badge>
          </div>
        </div>
      </div>
    </section>
  );
}
