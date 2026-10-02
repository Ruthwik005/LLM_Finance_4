const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export const DASH = "—";

export function formatCurrency(v, { compact = false, digits = 2 } = {}) {
  if (!isNum(v)) return DASH;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: compact ? "compact" : "standard",
    minimumFractionDigits: compact ? 0 : digits,
    maximumFractionDigits: compact ? 1 : digits,
  }).format(v);
}

export function formatNumber(v, digits = 2) {
  if (!isNum(v)) return DASH;
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(v);
}

export function formatCompact(v) {
  if (!isNum(v)) return DASH;
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

export function formatPercent(v, digits = 1) {
  if (!isNum(v)) return DASH;
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
}

export function formatDate(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return DASH;
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(d);
}

export function signClass(v) {
  if (!isNum(v) || v === 0) return "";
  return v > 0 ? "pos" : "neg";
}
