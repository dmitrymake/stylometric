// Единый блок источников/провенанса для всех кейсов. Контракт: items=[{ cite, url? }].
export default function Sources({ items, note, label = "Источники" }) {
  return (
    <div className="sources">
      <p className="eyebrow">{label}</p>
      <ul className="sources-list">
        {items.map((r, i) => (
          <li key={r.url || i}>
            {r.url
              ? <a href={r.url} target="_blank" rel="noopener noreferrer">{r.cite}</a>
              : <span>{r.cite}</span>}
          </li>
        ))}
      </ul>
      {note && <p className="sources-note">{note}</p>}
    </div>
  );
}
