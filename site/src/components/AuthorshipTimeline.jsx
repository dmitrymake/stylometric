// Timeline авторства: «прокраска» книги по позиции. Каждая полоса — окно текста,
// цвет = доминирующий автор, прозрачность = уверенность. Чужие сегменты подсвечены.
// На токенах @rk/ui.

export default function AuthorshipTimeline({ timeline, host, colorMap, segments = [], caption, height = 76 }) {
  const n = timeline.length;
  const w = 100 / n; // % на полосу
  const colorOf = (name) => colorMap[name] || "var(--text-muted)";

  return (
    <figure style={{ margin: "20px 0 8px" }}>
      <div
        role="img"
        aria-label={`Карта авторства: ${host} и кандидаты по ходу текста`}
        style={{
          position: "relative", height, borderRadius: "var(--radius-sm, 8px)",
          overflow: "hidden", border: "1px solid var(--border)",
          background: "var(--surface-sunken)", display: "flex",
        }}
      >
        {timeline.map(([name, conf], i) => (
          <span
            key={i}
            title={`${Math.round((i / n) * 100)}% · ${name} (${conf.toFixed(2)})`}
            style={{
              width: `${w}%`, height: "100%",
              background: colorOf(name),
              opacity: 0.35 + 0.65 * Math.min(1, Math.max(0, conf)),
            }}
          />
        ))}
        {/* подсветка «чужих» сегментов снизу */}
        {segments.map(([start, end, name], k) => (
          <span
            key={`s${k}`}
            title={`Сегмент «${name}»: чанки ${start}–${end}`}
            style={{
              position: "absolute", bottom: 0, height: 6,
              left: `${(start / n) * 100}%`, width: `${((end - start + 1) / n) * 100}%`,
              background: colorOf(name), boxShadow: "0 0 8px " + colorOf(name),
            }}
          />
        ))}
      </div>
      {/* ось */}
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", color: "var(--text-muted)", fontSize: 11, marginTop: 6 }}>
        <span>начало книги</span><span>середина</span><span>конец</span>
      </div>
      {/* легенда */}
      <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
        {Object.entries(colorMap).map(([name, color]) => (
          <span key={name} style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 13 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: color, display: "inline-block" }} />
            <span style={{ color: name === host ? "var(--text)" : "var(--text-muted)" }}>
              {name}{name === host ? " · основной" : ""}
            </span>
          </span>
        ))}
      </div>
      {caption && <figcaption className="muted" style={{ fontSize: 13, marginTop: 8, maxWidth: "60ch" }}>{caption}</figcaption>}
    </figure>
  );
}
