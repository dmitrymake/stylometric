import { useState, useEffect, useRef } from "react";
import { LogoMark, EditorialArticle } from "@dmitrymake/rk-ui";
import Hero from "./sections/Hero.jsx";
import Problem from "./sections/Problem.jsx";
import Method from "./sections/Method.jsx";
import Results from "./sections/Results.jsx";
import Repro from "./sections/Repro.jsx";
import Limits from "./sections/Limits.jsx";
import Conclusion from "./sections/Conclusion.jsx";

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
export async function loadChapterForRender(chapter) {
  if (!CHAPTER_IDS.includes(chapter)) throw new Error(`unknown chapter: ${chapter}`);
  if (chapter === "framework") return null;
  return (await CHAPTER_LOADERS[chapter]()).default;
}
function readRoute(initialChapter) {
  if (initialChapter !== undefined) {
    if (!CHAPTER_IDS.includes(initialChapter)) throw new Error(`unknown initial chapter: ${initialChapter}`);
    return { chapter: initialChapter, anchor: "" };
  }
  const [chapter, anchor = ""] = typeof window === "undefined" ? [] : window.location.hash.slice(1).split("/");
  return { chapter: CHAPTER_IDS.includes(chapter) ? chapter : "framework", anchor };
}
export default function App({ initialChapter, chapterComponent } = {}) {
  const [route, setRoute] = useState(() => readRoute(initialChapter));
  const { chapter, anchor } = route;
  const [loadedChapters, setLoadedChapters] = useState(() => chapterComponent ? { [initialChapter]: chapterComponent } : {});
  const [loadError, setLoadError] = useState(false);
  const Chapter = loadedChapters[chapter];
  const activeChapterRef = useRef(null);
  const chapterIndex = CHAPTER_IDS.indexOf(chapter);
  const nextChapter = CHAPTERS[(chapterIndex + 1) % CHAPTERS.length];

  useEffect(() => {
    const handleHash = () => {
      if (window.location.hash === "#main") return;
      setRoute(readRoute());
    };
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);
  useEffect(() => {
    setLoadError(false);
    if (chapter === "framework" || loadedChapters[chapter]) return;
    let active = true;
    loadChapterForRender(chapter).then((component) => {
      if (active) setLoadedChapters((current) => ({ ...current, [chapter]: component }));
    }).catch(() => { if (active) setLoadError(true); });
    return () => { active = false; };
  }, [chapter, loadedChapters]);
  useEffect(() => {
    if (chapter !== "framework" && !Chapter) return;
    const centerActiveChapter = () => {
      const activeLink = activeChapterRef.current;
      if (activeLink) {
        const parent = activeLink.parentElement;
        const linkRect = activeLink.getBoundingClientRect();
        const parentRect = parent.getBoundingClientRect();
        parent.scrollLeft += linkRect.left - parentRect.left - (parent.clientWidth - activeLink.clientWidth) / 2;
      }
    };
    const frame = window.requestAnimationFrame(() => {
      centerActiveChapter();
      if (anchor) {
        const target = document.getElementById(anchor);
        for (let parent = target?.parentElement; parent; parent = parent.parentElement) {
          if (parent.tagName === "DETAILS") parent.open = true;
        }
        target?.scrollIntoView({ block: "start", behavior: "instant" });
      } else window.scrollTo({ top: 0, behavior: "instant" });
    });
    window.addEventListener("resize", centerActiveChapter);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", centerActiveChapter);
    };
  }, [chapter, anchor, Chapter]);

  return <div className="shell">
    <a className="skip-link" href={`#${chapter}/main`} onClick={(event) => {
      event.preventDefault();
      const main = document.getElementById("main");
      main?.focus({ preventScroll: true });
      main?.scrollIntoView({ block: "start", behavior: "instant" });
    }}>К содержанию</a>
    <header className="masthead">
      <div className="masthead-brand wrap">
        <a className="rk-brand" href="#framework">
          <LogoMark className="rk-brand-mark" size={48} aria-hidden />
          <span className="rk-brand-text"><span className="rk-brand-word">Стилометрия</span><span className="rk-brand-tagline-line">Русская проза · исследование авторской манеры</span></span>
        </a>
        <span className="masthead-edition">Stylo <span aria-hidden> / </span> Русский код</span>
      </div>
      <nav className="chapter-navigation" aria-label="Главы исследования">
        <div className="chapters wrap">{CHAPTERS.map(([id, label], index) => <a ref={chapter === id ? activeChapterRef : undefined} key={id} href={`#${id}`} className={`chapter-btn${chapter === id ? " active" : ""}`} aria-current={chapter === id ? "page" : undefined}><span className="chapter-number" aria-hidden>{String(index + 1).padStart(2, "0")}</span>{label}</a>)}</div>
      </nav>
    </header>
    <main id="main" tabIndex={-1}>
      <EditorialArticle className={`site-article ${chapter === "framework" ? "framework-article" : "case-article"}`} key={chapter}>
        {chapter === "framework" ? <>
          <Hero />
          <Problem />
          <Method />
          <Results />
          <Conclusion />
          <Repro />
          <details className="article-appendix wrap"><summary>Границы метода и дополнительные проверки</summary><Limits /></details>
        </> : Chapter ? <Chapter /> : <p className="wrap loading-status" role="status">{loadError ? "Не удалось загрузить главу. Перезагрузите страницу." : "Загрузка главы…"}</p>}
        <nav className="chapter-next wrap" aria-label="Продолжить чтение"><span>Следующая глава</span><a href={`#${nextChapter[0]}`}>{nextChapter[1]} <span aria-hidden>→</span></a></nav>
      </EditorialArticle>
    </main>
    <footer className="foot"><div className="wrap"><span>Дмитрий Пуртов × Русский код</span><a href="https://github.com/dmitrymake/stylometric" target="_blank" rel="noopener noreferrer">Исходный код ↗</a><span>2026</span></div></footer>
  </div>;
}
