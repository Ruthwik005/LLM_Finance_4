const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const arr = (a) => (Array.isArray(a) ? a : []);

/** Turn API replay steps into chart rows (adds spread, split inventory, PnL). */
export function prepareSteps(steps) {
  return arr(steps).map((s, i) => ({
    i,
    step: s.step,
    timestamp: s.timestamp ?? null,
    mid: s.mid,
    bid_1: s.bid_1,
    bid_2: s.bid_2,
    ask_1: s.ask_1,
    ask_2: s.ask_2,
    spread: isNum(s.ask_1) && isNum(s.bid_1) ? s.ask_1 - s.bid_1 : null,
    inventory: s.inventory,
    pos: Math.max(s.inventory, 0),
    neg: Math.min(s.inventory, 0),
    mtm_pnl: s.mtm_pnl,
    canceled: s.canceled_volume,
  }));
}

/** Marker radius scaled by sqrt(volume) into [min,max] px. */
export function markerSize(volume, maxVolume, min = 3, max = 8) {
  if (!isNum(volume) || !isNum(maxVolume) || maxVolume <= 0) return min;
  return min + (max - min) * Math.sqrt(Math.min(1, Math.max(0, volume / maxVolume)));
}

/** Fills that have a step, ready for scatter: x=step, y=price. */
export function fillMarkers(fills) {
  const f = arr(fills).filter((x) => isNum(x.step) && isNum(x.price));
  const maxV = f.reduce((m, x) => Math.max(m, x.volume || 0), 0);
  return f.map((x) => ({ step: x.step, price: x.price, side: x.side, volume: x.volume, timestamp: x.timestamp ?? null, r: markerSize(x.volume, maxV) }));
}

/** Rows with step <= cursor (progressive reveal) and inside the zoom window [lo,hi]. */
export function visibleRows(rows, cursorStep, lo = -Infinity, hi = Infinity) {
  const top = Math.min(cursorStep, hi);
  return arr(rows).filter((r) => r.step >= lo && r.step <= top);
}

export function visibleFills(fills, cursorStep, lo = -Infinity, hi = Infinity) {
  const top = Math.min(cursorStep, hi);
  return arr(fills).filter((f) => f.step >= lo && f.step <= top);
}

/** Attach comparison-run fields (cmp_inventory, cmp_mtm_pnl) to base rows by forward-filling on `step`. */
export function alignComparison(rows, cmpRows) {
  const b = arr(rows);
  const c = arr(cmpRows);
  if (!c.length) return b;
  let j = -1;
  return b.map((r) => {
    while (j + 1 < c.length && c[j + 1].step <= r.step) j += 1;
    const m = j >= 0 ? c[j] : null;
    return { ...r, cmp_inventory: m ? m.inventory : null, cmp_mtm_pnl: m ? m.mtm_pnl : null };
  });
}

/** Episode KPI strip: whole-episode API summary when present, else derived from rows/fills. */
export function episodeKpis(replay) {
  if (!replay) return null;
  const s = replay.summary;
  const fills = arr(replay.fills);
  if (s) {
    return {
      fills: s.n_fills,
      volume: s.filled_volume,
      pnl: s.final_mtm_pnl,
      maxAbsInventory: s.max_abs_inventory,
      avgSpread: s.avg_quoted_spread,
      cancelVolume: s.cancel_volume,
    };
  }
  const rows = prepareSteps(replay.steps);
  const spreads = rows.map((r) => r.spread).filter(isNum);
  return {
    fills: fills.length,
    volume: fills.reduce((a, f) => a + (f.volume || 0), 0),
    pnl: rows.length ? rows[rows.length - 1].mtm_pnl : null,
    maxAbsInventory: rows.length ? rows.reduce((m, r) => Math.max(m, Math.abs(r.inventory)), 0) : null,
    avgSpread: spreads.length ? spreads.reduce((a, b) => a + b, 0) / spreads.length : null,
    cancelVolume: arr(replay.steps).reduce((a, s2) => a + (s2.canceled_volume || 0), 0),
  };
}

export function deltaPnl(base, cmp) {
  return isNum(base) && isNum(cmp) ? base - cmp : null;
}

/** Index of the step in `rows` closest to (at or before) `step`; -1 if none. */
export function indexAtOrBefore(rows, step) {
  const r = arr(rows);
  let lo = 0;
  let hi = r.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (r[mid].step <= step) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

/** Choose a default comparison run: a baseline named heuristic (or avellaneda), never the base run itself. */
export function defaultCompareRun(runs, current) {
  const list = arr(runs).filter((r) => r !== current);
  return list.find((r) => /^heuristic/i.test(r)) || list.find((r) => /avellaneda/i.test(r)) || "";
}

/** y-domain padded around mid & quote levels over rows. */
export function priceDomain(rows) {
  const vals = [];
  arr(rows).forEach((r) => {
    ["mid", "bid_1", "bid_2", "ask_1", "ask_2"].forEach((k) => isNum(r[k]) && vals.push(r[k]));
  });
  if (!vals.length) return ["auto", "auto"];
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.001 || 1;
  return [lo - pad, hi + pad];
}
