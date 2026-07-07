import { ConfidenceRing } from "@dmitrymake/rk-ui";
import { RING, ringPct } from "../format.js";

export default function RingStat({ frac, big, caption, accent = "var(--gold)", wide = false }) {
  return (
    <div style={{ position: "relative", display: "grid", placeItems: "center" }}>
      <ConfidenceRing value={ringPct(frac)} size={RING.size} stroke={RING.stroke} />
      <div className="ring-center">
        <div className={"bignum ring-num" + (wide ? " ring-num--range" : "")} style={{ color: accent }}>{big}</div>
        <div className="mono" style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 4 }}>{caption}</div>
      </div>
    </div>
  );
}
