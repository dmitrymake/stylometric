import { CASE_DOWNLOADS } from "../data.js";

export default function Sources({ items = [], artifact, note, label = "Источники" }) {
  const download = artifact ? CASE_DOWNLOADS[artifact] : null;
  return (
    <section className="sources" aria-label={label}>
      <p className="eyebrow">{label}</p>
      <ul className="sources-list">
        {items.map((r, i) => (
          <li key={r.url || i}>
            {r.url
              ? <a href={r.url} target="_blank" rel="noopener noreferrer">{r.cite}{r.format && <span className="source-format"> · {r.format}</span>}</a>
              : <span>{r.cite}</span>}
          </li>
        ))}
        {download && <li><a href={`${import.meta.env.BASE_URL}${download.publicArtifact}`} download>
          {download.title}<span className="source-format"> · скачать JSON</span>
        </a></li>}
      </ul>
      {note && <p className="sources-note">{note}</p>}
    </section>
  );
}
