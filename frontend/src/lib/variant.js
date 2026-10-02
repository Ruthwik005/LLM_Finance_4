/** Normalise a run variant for display. */
export function runVariant(v) {
  return v ? String(v) : "—";
}

/** True when a run id or variant looks like demo/synthetic data. */
export function looksSynthetic(runId, variant) {
  return /demo|synthetic|smoke|fake/i.test(`${runId || ""} ${variant || ""}`);
}
