const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** Convert the /signals payload into row objects keyed by series name with t (ms). */
export function signalRows(payload) {
  if (!payload || payload.available === false || !Array.isArray(payload.timestamps)) return [];
  const names = Object.keys(payload.series || {});
  return payload.timestamps.map((ts, i) => {
    const row = { i, t: Date.parse(ts), mid: payload.midprice ? payload.midprice[i] : null };
    names.forEach((n) => {
      row[n] = payload.series[n][i];
    });
    return row;
  });
}

/** Contiguous [t0,t1] ranges where `key` < 0.5 (signal invalid) -> shading areas. */
export function invalidRanges(rows, key = "transformer_signal_valid") {
  const out = [];
  let start = null;
  let last = null;
  (rows || []).forEach((r) => {
    const bad = isNum(r[key]) && r[key] < 0.5;
    if (bad && start === null) start = r.t;
    if (!bad && start !== null) {
      out.push([start, last]);
      start = null;
    }
    last = r.t;
  });
  if (start !== null) out.push([start, last]);
  return out;
}

/** Horizons found in transformer_prob_h<H>_<cls> series, sorted ascending. */
export function probHorizons(series) {
  const hs = new Set();
  Object.keys(series || {}).forEach((k) => {
    const m = /^transformer_prob_h(\d+)_(down|flat|up)$/.exec(k);
    if (m) hs.add(Number(m[1]));
  });
  return [...hs].sort((a, b) => a - b);
}

/** Normalise the three class probabilities to sum to 1 (they should already; guards rounding after un-scaling). */
export function stackProbs(rows, h) {
  return (rows || []).map((r) => {
    const d = r[`transformer_prob_h${h}_down`];
    const f = r[`transformer_prob_h${h}_flat`];
    const u = r[`transformer_prob_h${h}_up`];
    if (![d, f, u].every(isNum)) return { t: r.t, down: null, flat: null, up: null };
    const c = (x) => Math.max(0, x);
    const s = c(d) + c(f) + c(u) || 1;
    return { t: r.t, down: c(d) / s, flat: c(f) / s, up: c(u) / s };
  });
}
