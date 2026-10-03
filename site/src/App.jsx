import { useState, useEffect, useRef } from "react";
import { LogoMark } from "@dmitrymake/rk-ui";
import { useReveal } from "./hooks.js";
import Hero from "./sections/Hero.jsx";
import Problem from "./sections/Problem.jsx";
import Method from "./sections/Method.jsx";
import Results from "./sections/Results.jsx";
import Corpus from "./sections/Corpus.jsx";
import Repro from "./sections/Repro.jsx";
import Limits from "./sections/Limits.jsx";
import Conclusion from "./sections/Conclusion.jsx";
import ForeignHands from "./sections/ForeignHands.jsx";
import ResearchUpdate from "./components/ResearchUpdate.jsx";

const CHAPTERS = [
  ["framework", "Как это работает"],
  ["sholokhov", "«Тихий Дон»"],
  ["ilfpetrov", "«12 стульев»"],
  ["nikolai", "Дневник Николая II"],
  ["hohol", "«Тарас Бульба»"],
];

export const CHAPTER_IDS = Object.freeze(CHAPTERS.map(([id]) => id));
const CHAPTER_LOADERS = {
  sholokhov: () => import("./sections/Sholokhov.jsx"),
  ilfpetrov: () => import("./sections/IlfPetrov.jsx"),
  nikolai: () => import("./sections/Nikolai.jsx"),
  hohol: () => import("./sections/Taras.jsx"),
};

// The browser and server-render smoke use the same actual chapter modules.
export async function loadChapterForRender(chapter) {
  if (!CHAPTER_IDS.includes(chapter)) throw new Error(`unknown chapter: ${chapter}`);
  if (chapter === "framework") return null;
  return (await CHAPTER_LOADERS[chapter]()).default;
}

export default function App({ initialChapter, chapterComponent } = {}) {
  const [chapter, setChapter] = useState(() => {
    if (initialChapter !== undefined) {
      if (!CHAPTER_IDS.includes(initialChapter)) {
        throw new Error(`unknown initial chapter: ${initialChapter}`);
      }
      return initialChapter;
    }
    const h = typeof window !== "undefined" ? window.location.hash.replace("#", "") : "";
    return CHAPTER_IDS.includes(h) ? h : "framework";
  });
  const [loadedChapters, setLoadedChapters] = useState(() =>
    chapterComponent ? { [initialChapter]: chapterComponent } : {});
  const [loadError, setLoadError] = useState(false);
  const Chapter = loadedChapters[chapter];
  const ref = useReveal(`${chapter}:${Chapter ? "ready" : "loading"}`);
  const tabRef = useRef(null);
  useEffect(() => {
    setLoadError(false);
    if (chapter === "framework" || loadedChapters[chapter]) return;
    let active = true;
    loadChapterForRender(chapter).then((component) => {
      if (active) setLoadedChapters((current) => ({ ...current, [chapter]: component }));
    }).catch(() => {
      if (active) setLoadError(true);
    });
    return () => { active = false; };
  }, [chapter, loadedChapters]);
  useEffect(() => {
    if (window.location.hash.replace("#", "") !== chapter) {
      window.history.replaceState(null, "", chapter === "framework" ? " " : `#${chapter}`);
    }
    window.scrollTo({ top: 0 });
    // лента вкладок перемонтируется (key={chapter}) и скроллится в 0 — центрируем активную
    tabRef.current?.scrollIntoView({ inline: "center", block: "nearest", behavior: "auto" });
  }, [chapter]);

  return (
    <div className="shell" ref={ref} key={chapter}>
      <a className="skip-link" href="#main">К содержанию</a>
      <nav className="topbar" aria-label="Главы исследования">
        <div className="wrap topbar-inner">
          <a className="rk-brand" href="#main" style={{ border: "none" }} onClick={() => setChapter("framework")}>
            <LogoMark className="rk-brand-mark" size={48} aria-hidden />
              <span className="rk-brand-text">
                <span className="rk-brand-word">Стилометрия</span>
                <span className="rk-brand-tagline-line">сравнение авторской манеры</span>
            </span>
          </a>
          <div className="chapters" role="tablist">
            {CHAPTERS.map(([id, label]) => (
              <button
                ref={chapter === id ? tabRef : undefined}
                key={id}
                type="button"
                role="tab"
                aria-selected={chapter === id}
                className={"chapter-btn" + (chapter === id ? " active" : "")}
                onClick={() => setChapter(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main id="main">
        {chapter !== "framework" && <aside className="wrap" aria-label="О расчётах этой главы">
          <p className="note">
            Прежний расчёт этого случая. Новое сравнение методов — в разделе «Как это работает».
          </p>
          <details>
            <summary>Что изменилось в проверке</summary>
            <ResearchUpdate />
          </details>
        </aside>}
        {chapter === "framework" && (
          <>
            <Hero onChooseCase={setChapter} />
            <Problem />
            <Method />
            <Results />
            <Conclusion />
            <details className="wrap section">
              <summary>История исследования: исходный корпус и дополнительные проверки</summary>
              <ResearchUpdate />
              <ForeignHands />
              <Corpus />
              <Repro />
              <Limits />
            </details>
          </>
        )}
        {chapter !== "framework" && (Chapter ? <Chapter /> : (
          <p className="wrap" role="status">{loadError ? "Не удалось загрузить главу. Перезагрузите страницу." : "Загрузка главы…"}</p>
        ))}
      </main>

      <footer className="foot">
        <div className="wrap">
          <span><strong style={{ color: "var(--text)" }}>Дмитрий Пуртов</strong> × Русский код</span>
          <span className="mono">
            <a href="https://github.com/dmitrymake/stylometric" target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>GitHub</a>
            {" · 2026"}
          </span>
        </div>
      </footer>
    </div>
  );
}
