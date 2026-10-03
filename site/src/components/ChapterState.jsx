import { Button } from "@dmitrymake/rk-ui";

export default function ChapterState({ title, failed = false }) {
  return <section className="wrap chapter-state" aria-labelledby="chapter-state-title" aria-busy={!failed}>
    <p className="eyebrow">{failed ? "Глава недоступна" : "Открываем главу"}</p>
    <h1 id="chapter-state-title">{title}</h1>
    <p className="prose" role={failed ? "alert" : "status"}>
      {failed ? "Не удалось загрузить текст. Попробуйте открыть эту главу ещё раз." : "Загружаем текст и иллюстрации…"}
    </p>
    {failed && <Button variant="secondary" onClick={() => window.location.reload()}>
      Повторить загрузку
    </Button>}
  </section>;
}
