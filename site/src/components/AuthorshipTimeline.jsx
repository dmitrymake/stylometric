import { useId, useMemo, useState } from "react";

// A profile for each text window; selection works with keyboard and touch.

export default function AuthorshipTimeline({ timeline, host, colorMap, segments = [], caption, height = 76 }) {
  const n = timeline.length;
  const w = n ? 100 / n : 0;
  const [selected, setSelected] = useState(0);
  const inputId = useId();
  const index = Math.min(selected, Math.max(0, n - 1));
  const current = timeline[index];
  const colorOf = (name) => colorMap[name] || "var(--text-muted)";
  // Sliding the cursor does not regenerate hundreds of window elements.
  const windows = useMemo(() => timeline.map(([name, score], i) => <span key={i}
    title={`${Math.round((i / timeline.length) * 100)}% · ${name} (${score.toFixed(2)})`}
    style={{ width: `${100 / timeline.length}%`, height: "100%", background: colorMap[name] || "var(--text-muted)",
      opacity: 0.35 + 0.65 * Math.min(1, Math.max(0, score)) }} />), [timeline, colorMap]);

  return (
    <figure style={{ margin: "20px 0 8px" }}>
      <div
        role="img"
        aria-label={`Карта ближайших профилей: ${host} и кандидаты по ходу текста`}
        style={{
          position: "relative", height, borderRadius: 0,
          overflow: "hidden", border: "1px solid var(--border)",
          background: "var(--surface-sunken)", display: "flex",
        }}
      >
        {windows}
        {n > 0 && <span aria-hidden="true" style={{ position: "absolute", insetBlock: 0,
          left: `${index * w}%`, width: `${w}%`, minWidth: 3,
          boxShadow: "inset 0 0 0 2px var(--text)", pointerEvents: "none" }} />}
        {/* подсветка «чужих» сегментов снизу */}
        {segments.map(([start, end, name], k) => (
          <span
            key={`s${k}`}
            title={`Сегмент «${name}»: чанки ${start}–${end}`}
            style={{
              position: "absolute", bottom: 0, height: 6,
              left: `${(start / n) * 100}%`, width: `${((end - start + 1) / n) * 100}%`,
              background: colorOf(name), boxShadow: "none",
            }}
          />
        ))}
      </div>
      {current && <div className="timeline-selection">
        <label htmlFor={inputId}>Окно {index + 1} из {n}</label>
        <input id={inputId} type="range" min={0} max={n - 1} value={index}
          aria-valuetext={`Окно ${index + 1}: ${current[0]}, оценка ${current[1].toFixed(2)}`}
          onChange={(event) => setSelected(Number(event.target.value))} />
        <output htmlFor={inputId}>Ближайший профиль — {current[0]}; оценка модели {current[1].toFixed(2)}.</output>
      </div>}
      {/* ось */}
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", color: "var(--text-muted)", fontSize: 16, marginTop: 6 }}>
        <span>начало книги</span><span>середина</span><span>конец</span>
      </div>
      {/* легенда */}
      <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
        {Object.entries(colorMap).map(([name, color]) => (
          <span key={name} style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 16 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: color, display: "inline-block" }} />
            <span style={{ color: name === host ? "var(--text)" : "var(--text-muted)" }}>
              {name}{name === host ? " · основной" : ""}
            </span>
          </span>
        ))}
      </div>
      {caption && <figcaption className="muted" style={{ fontSize: 16, marginTop: 8, maxWidth: "60ch" }}>{caption}</figcaption>}
    </figure>
  );
}
