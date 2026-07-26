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
import Taras from "./sections/Taras.jsx";
import Sholokhov from "./sections/Sholokhov.jsx";
import IlfPetrov from "./sections/IlfPetrov.jsx";
import Nikolai from "./sections/Nikolai.jsx";
import ForeignHands from "./sections/ForeignHands.jsx";

const CHAPTERS = [
  ["framework", "Фреймворк"],
  ["sholokhov", "Шолохов"],
  ["ilfpetrov", "Ильф и Петров"],
  ["nikolai", "Николай II"],
  ["hohol", "Гоголь"],
];

export const CHAPTER_IDS = Object.freeze(CHAPTERS.map(([id]) => id));

export default function App({ initialChapter } = {}) {
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
  const ref = useReveal();
  const tabRef = useRef(null);
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
      <nav className="topbar" aria-label="Главы документа">
        <div className="wrap topbar-inner">
          <a className="rk-brand" href="#main" style={{ border: "none" }} onClick={() => setChapter("framework")}>
            <LogoMark className="rk-brand-mark" size={48} aria-hidden />
            <span className="rk-brand-text">
              <span className="rk-brand-word">Стилометрия</span>
              <span className="rk-brand-tagline-line">атрибуция авторства</span>
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
        {chapter === "framework" && (
          <>
            <Hero />
            <Problem />
            <Method />
            <Results />
            <ForeignHands />
            <Corpus />
            <Repro />
            <Limits />
            <Conclusion />
          </>
        )}
        {chapter === "hohol" && <Taras />}
        {chapter === "sholokhov" && <Sholokhov />}
        {chapter === "ilfpetrov" && <IlfPetrov />}
        {chapter === "nikolai" && <Nikolai />}
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
