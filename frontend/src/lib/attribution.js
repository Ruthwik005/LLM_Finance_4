const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export const ATTR_PARTS = [
  { key: "spread_capture_pnl", name: "Spread capture", color: "var(--accent)" },
  { key: "inventory_pnl", name: "Inventory PnL", color: "var(--blue)" },
  { key: "fees_paid", name: "Fees", color: "var(--red)", negate: true },
  { key: "rebates", name: "Rebates", color: "var(--green)" },
];

/**
 * Stacked-bar model of PnL attribution: one category row with a signed value per defined component
 * (fees are shown negative). Undefined components are omitted. Returns {row, parts, total} or null.
 */
export function attributionStack(attr) {
  if (!attr) return null;
  const parts = ATTR_PARTS.filter((p) => isNum(attr[p.key]));
  if (!parts.length) return null;
  const row = { name: "PnL" };
  parts.forEach((p) => {
    row[p.key] = p.negate ? -attr[p.key] : attr[p.key];
  });
  const total = isNum(attr.total) ? attr.total : parts.reduce((s, p) => s + row[p.key], 0);
  return { row: [row], parts, total };
}
