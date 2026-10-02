const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const arr = (a) => (Array.isArray(a) ? a : []);

/** Net inventory after each fill. bid = +volume, ask = -volume; resets when the episode changes. */
export function inventoryPath(fills) {
  const out = [];
  let inv = 0;
  let prev;
  arr(fills).forEach((f, idx) => {
    if (idx === 0 || f.episode !== prev) inv = 0;
    prev = f.episode;
    const vol = Number(f.volume) || 0;
    inv += f.side === "bid" ? vol : f.side === "ask" ? -vol : 0;
    out.push({ i: idx + 1, episode: f.episode, inventory: inv, pos: Math.max(inv, 0), neg: Math.min(inv, 0) });
  });
  return out;
}

export function maxAbsInventory(path) {
  const p = arr(path);
  if (!p.length) return null;
  return p.reduce((m, r) => Math.max(m, Math.abs(r.inventory)), 0);
}

export function meanAbsInventory(path) {
  const p = arr(path);
  if (!p.length) return null;
  return p.reduce((s, r) => s + Math.abs(r.inventory), 0) / p.length;
}

/** Largest peak-to-trough drop (positive magnitude) of cumulative PnL; equity starts at 0. */
export function maxDrawdown(equity) {
  const e = arr(equity);
  if (!e.length) return null;
  let peak = 0;
  let mdd = 0;
  for (const pt of e) {
    const v = pt.cumulative_pnl;
    if (!isNum(v)) continue;
    if (v > peak) peak = v;
    mdd = Math.max(mdd, peak - v);
  }
  return mdd;
}

export function winRate(episodes) {
  const e = arr(episodes).filter((x) => isNum(x.pnl));
  if (!e.length) return null;
  return e.filter((x) => x.pnl > 0).length / e.length;
}

export function episodeStats(episodes) {
  const e = arr(episodes).filter((x) => isNum(x.pnl));
  if (!e.length) return null;
  const best = e.reduce((a, b) => (b.pnl > a.pnl ? b : a));
  const worst = e.reduce((a, b) => (b.pnl < a.pnl ? b : a));
  return { best, worst, avg: e.reduce((s, x) => s + x.pnl, 0) / e.length, count: e.length };
}

/** PnL divided by mean absolute position; null when MAP is missing or zero. */
export function pnlPerMap(pnl, map) {
  if (!isNum(pnl) || !isNum(map) || map === 0) return null;
  return pnl / map;
}

export function fillStats(fills) {
  const f = arr(fills);
  if (!f.length) return null;
  const bids = f.filter((x) => x.side === "bid");
  const asks = f.filter((x) => x.side === "ask");
  const vol = (l) => l.reduce((s, x) => s + (Number(x.volume) || 0), 0);
  const totalVolume = vol(f);
  const priced = f.filter((x) => isNum(x.price));
  return {
    count: f.length,
    bidCount: bids.length,
    askCount: asks.length,
    bidVolume: vol(bids),
    askVolume: vol(asks),
    totalVolume,
    avgPrice: priced.length ? priced.reduce((s, x) => s + x.price, 0) / priced.length : null,
    avgSize: totalVolume / f.length,
  };
}

/** Downsample a numeric series to at most n points (for sparklines). */
export function downsample(values, n = 40) {
  const v = arr(values);
  if (v.length <= n) return v;
  const step = (v.length - 1) / (n - 1);
  return Array.from({ length: n }, (_, i) => v[Math.round(i * step)]);
}

/** Index of the best (max or min) numeric value in a list, or -1. */
export function bestIndex(values, higherIsBetter = true) {
  let best = -1;
  values.forEach((v, i) => {
    if (!isNum(v)) return;
    if (best === -1 || (higherIsBetter ? v > values[best] : v < values[best])) best = i;
  });
  return best;
}
