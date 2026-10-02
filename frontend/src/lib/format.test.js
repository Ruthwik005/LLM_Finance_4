import { describe, it, expect } from "vitest";
import { formatCurrency, formatNumber, formatCompact, formatPercent, formatDate, signClass, DASH } from "./format";

describe("format", () => {
  it("formats currency", () => {
    expect(formatCurrency(1234.5)).toBe("$1,234.50");
    expect(formatCurrency(-3)).toBe("-$3.00");
    expect(formatCurrency(null)).toBe(DASH);
    expect(formatCurrency(NaN)).toBe(DASH);
  });
  it("formats numbers, compact and percent", () => {
    expect(formatNumber(1234.567)).toBe("1,234.57");
    expect(formatNumber(undefined)).toBe(DASH);
    expect(formatCompact(1500)).toBe("1.5K");
    expect(formatPercent(0.5)).toBe("50.0%");
    expect(formatPercent(Infinity)).toBe(DASH);
  });
  it("formats dates defensively", () => {
    expect(formatDate("2026-09-25T10:00:00Z")).toMatch(/2026/);
    expect(formatDate("nope")).toBe(DASH);
    expect(formatDate(null)).toBe(DASH);
  });
  it("signClass", () => {
    expect(signClass(1)).toBe("pos");
    expect(signClass(-1)).toBe("neg");
    expect(signClass(0)).toBe("");
    expect(signClass(null)).toBe("");
  });
});
