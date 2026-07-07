// Единая мини-шкала для повторного использования. max берётся из данных, не литералом.
export default function MeterBar({ value, max = 1, accent = "var(--icon-blue)", hi, height = "var(--bar-h)" }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <span className="meter" style={{ height }}>
      <span className="meter-fill" style={{ width: `${pct}%`, background: hi || accent }} />
    </span>
  );
}
