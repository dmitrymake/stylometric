// Единая политика отображения чисел. Значения НЕ меняем по смыслу — только формат.
const EN = "–";

// доля 0..1 → процент. digits: 1 для точных метрик (89.2%), 0 для грубых долей (63%).
export const fmtPct = (frac, digits = 0) =>
  frac == null || Number.isNaN(frac) ? "—" : `${(frac * 100).toFixed(digits)}%`;

// безразмерная оценка 0..1 (recall/AUC/F1/силуэт/дистанция). По умолчанию 2 знака «0.XX».
export const fmtScore = (x, digits = 2) =>
  x == null || Number.isNaN(x) ? "—" : x.toFixed(digits);

// p-значение, научпоп: порог снизу, максимум 3 знака, без хвостовых нулей.
export const fmtP = (p) => {
  if (p == null || Number.isNaN(p)) return "—";
  if (p <= 0) return "< 0.0001";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3).replace(/\.?0+$/, "");
};

// z-оценка: фиксированные 2 знака, минус проходит естественно.
export const fmtZ = (z, digits = 2) =>
  z == null || Number.isNaN(z) ? "—" : z.toFixed(digits);

// диапазон / доверительный интервал одним правилом для обоих концов.
export const fmtRange = (lo, hi, fmt = fmtScore) => `${fmt(lo)}${EN}${fmt(hi)}`;

// объёмы. Русская запись крупных чисел: «10,4 млн» (десятичная запятая, единица словом вместо «M»).
export const fmtWordsM = (n) =>
  n == null || Number.isNaN(n) ? "—" : `${(n / 1e6).toFixed(1).replace(".", ",")} млн`;
export const fmtInt = (n) => Number(n).toLocaleString("ru-RU");

// значение дуги ConfidenceRing — всегда целый процент 0..100.
export const ringPct = (frac) => Math.round(frac * 100);

// единые габариты колец
export const RING = { size: 208, stroke: 9 };
