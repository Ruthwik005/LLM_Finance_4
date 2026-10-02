import { useEffect, useMemo, useRef, useState } from "react";
import ResponsiveContainer from "./AutoSize";
import { Area, CartesianGrid, ComposedChart, Line, Scatter, Tooltip, XAxis, YAxis, ReferenceLine } from "recharts";
import { Coins, Boxes, Wallet, Scale, MoveHorizontal, Ban, Play, Pause, RotateCcw, ZoomOut } from "lucide-react";
import ChartCard from "./ChartCard";
import KpiCard from "./KpiCard";
import Segmented from "./Segmented";
import { SkeletonKpis, SkeletonCard } from "./Skeleton";
import EmptyState from "./EmptyState";
import NoRun from "./NoRun";
import StatusChip from "./StatusChip";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { getReplay, getRunEpisodes } from "../api";
import { useAsync, usePrefersReducedMotion } from "../lib/hooks";
import { formatNumber, formatCurrency } from "../lib/format";
import {
  prepareSteps, fillMarkers, visibleRows, visibleFills, alignComparison, episodeKpis, deltaPnl,
  indexAtOrBefore, defaultCompareRun, priceDomain,
} from "../lib/replay";

const SPEEDS = [
  { value: "1", label: "1×" },
  { value: "4", label: "4×" },
  { value: "16", label: "16×" },
];
const TICK_MS = 60;
const NO_FILLS = [];

function Marker({ cx, cy, payload }) {
  if (cx == null || cy == null || !payload) return null;
  const r = payload.r || 5;
  const buy = payload.side === "bid";
  const pts = buy
    ? `${cx},${cy - r} ${cx - r},${cy + r * 0.8} ${cx + r},${cy + r * 0.8}`
    : `${cx},${cy + r} ${cx - r},${cy - r * 0.8} ${cx + r},${cy - r * 0.8}`;
  return <polygon points={pts} fill={buy ? "var(--green)" : "var(--red)"} stroke="var(--surface)" strokeWidth={1} opacity={0.95} />;
}

function ReadoutTooltip({ active, payload, cmpLabel }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload.find((p) => p.payload && p.payload.step !== undefined && p.payload.mid !== undefined)?.payload;
  const fill = payload.find((p) => p.payload && p.payload.side)?.payload;
  if (!row && !fill) return null;
  return (
    <div className="tooltip" role="status">
      {row && (
        <>
          <div className="tooltip-label">Step {row.step}{row.timestamp ? ` · ${row.timestamp.slice(11, 19)}` : ""}</div>
          <div className="tooltip-row"><span className="tooltip-name">Mid</span><span className="mono tooltip-val">{formatNumber(row.mid, 4)}</span></div>
          <div className="tooltip-row"><span className="dot" style={{ background: "var(--green)" }} /><span className="tooltip-name">Bid L1 / L2</span><span className="mono tooltip-val">{formatNumber(row.bid_1, 4)} / {formatNumber(row.bid_2, 4)}</span></div>
          <div className="tooltip-row"><span className="dot" style={{ background: "var(--red)" }} /><span className="tooltip-name">Ask L1 / L2</span><span className="mono tooltip-val">{formatNumber(row.ask_1, 4)} / {formatNumber(row.ask_2, 4)}</span></div>
          <div className="tooltip-row"><span className="tooltip-name">Quoted spread</span><span className="mono tooltip-val">{formatNumber(row.spread, 4)}</span></div>
          <div className="tooltip-row"><span className="tooltip-name">Inventory</span><span className="mono tooltip-val">{formatNumber(row.inventory, 4)}</span></div>
          <div className="tooltip-row"><span className="tooltip-name">MtM PnL</span><span className="mono tooltip-val">{formatNumber(row.mtm_pnl, 4)}</span></div>
          {Number.isFinite(row.cmp_mtm_pnl) && (
            <div className="tooltip-row"><span className="tooltip-name">{cmpLabel} PnL</span><span className="mono tooltip-val">{formatNumber(row.cmp_mtm_pnl, 4)}</span></div>
          )}
        </>
      )}
      {fill && (
        <div className="tooltip-fill">
          <span className={`pill pill-${fill.side}`}>{fill.side === "bid" ? "BUY" : "SELL"}</span>
          <span className="mono"> {formatNumber(fill.volume, 4)} @ {formatNumber(fill.price, 4)}</span>
        </div>
      )}
    </div>
  );
}

export default function ReplayView({ runId, runs, variants, refreshKey, synthetic }) {
  const reduced = usePrefersReducedMotion();
  const [episode, setEpisode] = useState(null);
  const [cmpRun, setCmpRun] = useState(null); // null = auto default, "" = none
  const [layers, setLayers] = useState({ mid: true, l1: true, l2: true, fills: true });
  const [cursor, setCursor] = useState(null); // step; null = fully revealed
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("4");
  const [zoom, setZoom] = useState([null, null]);

  const eps = useAsync(async () => (runId ? getRunEpisodes(runId) : []), [runId, refreshKey]);
  const episodes = useMemo(() => eps.data || [], [eps.data]);

  useEffect(() => {
    setEpisode(null);
    setCmpRun(null);
    setCursor(null);
    setPlaying(false);
    setZoom([null, null]);
  }, [runId]);

  const epIdx = episode ?? episodes[0]?.episode ?? null;
  const effCmp = cmpRun === null ? defaultCompareRun(runs, runId) : cmpRun;

  const base = useAsync(async () => (runId && epIdx !== null ? getReplay(runId, epIdx) : null), [runId, epIdx, refreshKey]);
  const cmp = useAsync(async () => (effCmp && epIdx !== null ? getReplay(effCmp, epIdx) : null), [effCmp, epIdx, refreshKey]);

  useEffect(() => {
    setCursor(null);
    setPlaying(false);
    setZoom([null, null]);
  }, [epIdx, effCmp]);

  const baseRows = useMemo(() => prepareSteps(base.data?.steps), [base.data]);
  const cmpRows = useMemo(() => prepareSteps(cmp.data?.steps), [cmp.data]);
  const rows = useMemo(() => alignComparison(baseRows, cmpRows), [baseRows, cmpRows]);
  const fills = useMemo(() => fillMarkers(base.data?.fills), [base.data]);
  const kpis = useMemo(() => episodeKpis(base.data), [base.data]);
  const cmpKpis = useMemo(() => (cmp.data ? episodeKpis(cmp.data) : null), [cmp.data]);

  const first = rows[0]?.step ?? 0;
  const last = rows[rows.length - 1]?.step ?? 0;
  const lo = zoom[0] ?? first;
  const hi = zoom[1] ?? last;
  const cur = cursor ?? last;

  // playback loop (disabled when the OS asks for reduced motion)
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  useEffect(() => {
    if (!playing || reduced) return undefined;
    const id = setInterval(() => {
      const r = rowsRef.current;
      if (!r.length) return;
      setCursor((c) => {
        const idx = indexAtOrBefore(r, c ?? r[r.length - 1].step);
        const next = Math.min(r.length - 1, idx + Number(speed));
        if (next >= r.length - 1) setPlaying(false);
        return r[next].step;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, [playing, speed, reduced]);

  const togglePlay = () => {
    if (playing) return setPlaying(false);
    if (cursor === null || cursor >= last) setCursor(first);
    setPlaying(true);
  };

  const shown = useMemo(() => visibleRows(rows, cur, lo, hi), [rows, cur, lo, hi]);
  const shownFills = useMemo(() => (layers.fills ? visibleFills(fills, cur, lo, hi) : NO_FILLS), [fills, cur, lo, hi, layers.fills]);
  const domain = useMemo(() => priceDomain(shown), [shown]);
  const curIdx = indexAtOrBefore(rows, cur);
  const curRow = curIdx >= 0 ? rows[curIdx] : null;

  if (!runId) return <NoRun />;
  if (eps.error) return <div className="card"><EmptyState hint={eps.error} /></div>;
  const busy = eps.loading || (base.loading && !base.data);
  if (!busy && !episodes.length) return <div className="card"><EmptyState hint="This run has no stored episodes." /></div>;
  const noSteps = !busy && !base.error && base.data && baseRows.length === 0;
  const cmpDelta = cmpKpis ? deltaPnl(kpis?.pnl, cmpKpis.pnl) : null;
  const toggle = (k) => setLayers((l) => ({ ...l, [k]: !l[k] }));
  const xDomain = [lo, hi];
  const xTick = (v) => String(v);
  const xAxisProps = { type: "number", dataKey: "step", domain: xDomain, allowDataOverflow: true, tick: AXIS_TICK, axisLine: AXIS_LINE, tickLine: false, tickFormatter: xTick, scale: "linear" };
  const sliderMax = Math.max(last, first + 1);

  return (
    <>
      <section className="card replay-controls fade-up" aria-label="Replay selection">
        <div className="rc-group">
          <label className="field">
            <span className="field-label">Episode</span>
            <span className="select-wrap">
              <select value={epIdx ?? ""} onChange={(e) => setEpisode(Number(e.target.value))} aria-label="Episode">
                {episodes.map((e) => <option key={e.episode} value={e.episode}>Episode {e.episode} ({formatNumber(e.pnl, 2)})</option>)}
              </select>
            </span>
          </label>
          <label className="field">
            <span className="field-label">Compare with</span>
            <span className="select-wrap">
              <select value={effCmp} onChange={(e) => setCmpRun(e.target.value)} aria-label="Compare with run">
                <option value="">None</option>
                {runs.filter((r) => r !== runId).map((r) => <option key={r} value={r}>{r}{variants?.[r] ? ` · ${variants[r]}` : ""}</option>)}
              </select>
            </span>
          </label>
        </div>
        <div className="rc-group" role="group" aria-label="Chart layers">
          {[["mid", "Mid"], ["l1", "Quotes L1"], ["l2", "Quotes L2"], ["fills", "Fills"]].map(([k, label]) => (
            <label key={k} className={`toggle ${layers[k] ? "on" : ""}`}>
              <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </section>

      {busy ? (
        <SkeletonKpis n={6} />
      ) : base.error ? (
        <div className="card"><EmptyState hint={base.error} /></div>
      ) : (
        <>
          {base.data?.downsampled && (
            <StatusChip tone="info">Showing {formatNumber(base.data.n_points, 0)} of {formatNumber(base.data.n_steps_total, 0)} steps (all fills kept). KPIs use the full episode.</StatusChip>
          )}
          <div className="kpi-grid kpi-grid-6x">
            <KpiCard delay={0} label="Fills" icon={Coins} value={kpis?.fills} format={(v) => formatNumber(Math.round(v), 0)} />
            <KpiCard delay={30} label="Filled volume" icon={Boxes} value={kpis?.volume} format={(v) => formatNumber(v, 3)} />
            <KpiCard delay={60} label="Episode PnL" icon={Wallet} value={kpis?.pnl} colorBySign format={(v) => formatCurrency(v)}
              hint={cmpKpis ? `Δ vs ${effCmp}: ${Number.isFinite(cmpDelta) ? formatCurrency(cmpDelta) : "—"}` : undefined} />
            <KpiCard delay={90} label="Max |inventory|" icon={Scale} value={kpis?.maxAbsInventory} format={(v) => formatNumber(v, 3)} />
            <KpiCard delay={120} label="Avg quoted spread" icon={MoveHorizontal} value={kpis?.avgSpread} format={(v) => formatNumber(v, 4)} />
            <KpiCard delay={150} label="Cancel volume" icon={Ban} value={kpis?.cancelVolume} format={(v) => formatNumber(v, 2)} />
          </div>
        </>
      )}

      {noSteps ? (
        <div className="card"><EmptyState hint="This run was stored without per-step data (created before the replay feature). Re-run the evaluation to enable it." /></div>
      ) : (
        <>
          <ChartCard
            title="Quotes, mid and fills"
            subtitle={`Episode ${epIdx ?? "—"} · agent bid (green) and ask (red) quote levels, L2 lighter; ▲ buy fill, ▼ sell fill`}
            loading={busy}
            empty={!!base.error || !rows.length}
            height={380}
            actions={curRow && (
              <span className="readout mono" aria-live="off">
                step {curRow.step} · mid {formatNumber(curRow.mid, 4)} · bid {formatNumber(curRow.bid_1, 4)} · ask {formatNumber(curRow.ask_1, 4)} · spr {formatNumber(curRow.spread, 4)} · inv {formatNumber(curRow.inventory, 3)}
              </span>
            )}
          >
            <div role="img" aria-label="Midprice with agent bid and ask quote levels and fill markers" style={{ width: "100%", height: 380 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={shown} syncId="replay" margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis {...xAxisProps} />
                  <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={64} domain={domain} allowDataOverflow tickFormatter={(v) => formatNumber(v, 2)} />
                  <Tooltip content={<ReadoutTooltip cmpLabel={effCmp} />} cursor={{ stroke: "var(--text-3)", strokeDasharray: "3 3" }} isAnimationActive={false} />
                  {layers.l2 && <Line type="stepAfter" dataKey="bid_2" name="Bid L2" stroke="var(--green)" strokeOpacity={0.4} strokeWidth={1.25} dot={false} isAnimationActive={false} />}
                  {layers.l2 && <Line type="stepAfter" dataKey="ask_2" name="Ask L2" stroke="var(--red)" strokeOpacity={0.4} strokeWidth={1.25} dot={false} isAnimationActive={false} />}
                  {layers.l1 && <Line type="stepAfter" dataKey="bid_1" name="Bid L1" stroke="var(--green)" strokeWidth={1.75} dot={false} isAnimationActive={false} />}
                  {layers.l1 && <Line type="stepAfter" dataKey="ask_1" name="Ask L1" stroke="var(--red)" strokeWidth={1.75} dot={false} isAnimationActive={false} />}
                  {layers.mid && <Line type="monotone" dataKey="mid" name="Mid" stroke="var(--text)" strokeWidth={1.5} dot={false} isAnimationActive={false} />}
                  {layers.fills && <Scatter data={shownFills} dataKey="price" xAxisId={0} name="Fill" shape={<Marker />} isAnimationActive={false} legendType="none" />}
                  {cur >= lo && cur <= hi && <ReferenceLine x={cur} stroke="var(--accent)" strokeOpacity={0.6} />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <ChartCard
            title="Inventory and cumulative mark-to-market PnL"
            subtitle={effCmp ? `Dashed: ${effCmp} (same episode)` : "Inventory (left axis, area) and PnL (right axis)"}
            loading={busy}
            empty={!!base.error || !rows.length}
            height={240}
          >
            <div role="img" aria-label="Inventory area and cumulative PnL line synchronized with the quote chart" style={{ width: "100%", height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={shown} syncId="replay" margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis {...xAxisProps} />
                  <YAxis yAxisId="inv" tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => formatNumber(v, 1)} />
                  <YAxis yAxisId="pnl" orientation="right" tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => formatNumber(v, 2)} />
                  <Tooltip content={<ReadoutTooltip cmpLabel={effCmp} />} cursor={{ stroke: "var(--text-3)", strokeDasharray: "3 3" }} isAnimationActive={false} />
                  <ReferenceLine yAxisId="inv" y={0} stroke="var(--border-strong)" />
                  <Area yAxisId="inv" type="stepAfter" dataKey="pos" name="Long inventory" stroke="var(--green)" fill="var(--green-soft)" strokeWidth={1.25} isAnimationActive={false} />
                  <Area yAxisId="inv" type="stepAfter" dataKey="neg" name="Short inventory" stroke="var(--red)" fill="var(--red-soft)" strokeWidth={1.25} isAnimationActive={false} />
                  <Line yAxisId="pnl" type="monotone" dataKey="mtm_pnl" name="MtM PnL" stroke="var(--accent)" strokeWidth={1.75} dot={false} isAnimationActive={false} />
                  {effCmp && <Line yAxisId="inv" type="stepAfter" dataKey="cmp_inventory" name={`${effCmp} inventory`} stroke="var(--text-2)" strokeDasharray="5 4" strokeWidth={1.25} dot={false} isAnimationActive={false} connectNulls />}
                  {effCmp && <Line yAxisId="pnl" type="monotone" dataKey="cmp_mtm_pnl" name={`${effCmp} PnL`} stroke="var(--amber)" strokeDasharray="5 4" strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls />}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          {!busy && rows.length > 0 && (
            <section className="card transport fade-up" aria-label="Playback and zoom">
              <div className="transport-row">
                <button type="button" className="btn btn-icon" onClick={togglePlay} disabled={reduced} aria-label={playing ? "Pause replay" : "Play replay"} title={reduced ? "Autoplay is disabled because your system prefers reduced motion" : undefined}>
                  {playing ? <Pause size={15} /> : <Play size={15} />} {playing ? "Pause" : "Play"}
                </button>
                <button type="button" className="btn btn-icon" onClick={() => { setPlaying(false); setCursor(null); }} aria-label="Show full episode">
                  <RotateCcw size={15} /> Full
                </button>
                <Segmented label="Playback speed" value={speed} onChange={setSpeed} options={SPEEDS} />
                <label className="slider grow">
                  <span className="field-label">Scrubber · step {cur}</span>
                  <input type="range" min={first} max={sliderMax} value={Math.min(cur, sliderMax)} onChange={(e) => { setPlaying(false); setCursor(Number(e.target.value)); }} aria-label="Replay position (step)" />
                </label>
              </div>
              <div className="transport-row">
                <label className="slider grow">
                  <span className="field-label">Zoom start · {lo}</span>
                  <input type="range" min={first} max={sliderMax} value={Math.min(lo, sliderMax)} onChange={(e) => setZoom([Math.min(Number(e.target.value), hi - 1), zoom[1]])} aria-label="Zoom window start (step)" />
                </label>
                <label className="slider grow">
                  <span className="field-label">Zoom end · {hi}</span>
                  <input type="range" min={first} max={sliderMax} value={Math.min(hi, sliderMax)} onChange={(e) => setZoom([zoom[0], Math.max(Number(e.target.value), lo + 1)])} aria-label="Zoom window end (step)" />
                </label>
                <button type="button" className="btn btn-icon" onClick={() => setZoom([null, null])} aria-label="Reset zoom"><ZoomOut size={15} /> Reset</button>
              </div>
              {reduced && <p className="card-sub">Autoplay is off because your system prefers reduced motion; use the scrubber instead.</p>}
              {synthetic && <p className="card-sub">Synthetic smoke-test data: shapes illustrate the pipeline, not market behaviour.</p>}
            </section>
          )}
          {busy && <SkeletonCard h={60} />}
        </>
      )}
    </>
  );
}
