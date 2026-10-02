const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** The best-checkpoint point: last is_best row (the callback only flags improvements). */
export function bestPoint(points) {
  const b = (points || []).filter((p) => p.is_best);
  return b.length ? b[b.length - 1] : null;
}

export function hasSeries(points, key) {
  return (points || []).some((p) => isNum(p[key]));
}

export function hasCurve(points) {
  return Array.isArray(points) && points.length > 0;
}

/** Last finite value of a key. */
export function lastValue(points, key) {
  const v = (points || []).map((p) => p[key]).filter(isNum);
  return v.length ? v[v.length - 1] : null;
}
