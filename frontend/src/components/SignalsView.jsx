import { useMemo, useState } from "react";
import ResponsiveContainer from "./AutoSize";
import { Area, AreaChart, Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, Tooltip, XAxis, YAxis } from "recharts";
import { Brain, Clock3, Database, LineChart as Trend } from "lucide-react";
import StatusChip from "./StatusChip";
import EmptyState from "./EmptyState";
import ChartCard from "./ChartCard";
import ChartTooltip from "./ChartTooltip";
import Segmented from "./Segmented";
import { SkeletonCard } from "./Skeleton";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { getSignals } from "../api";
import { useAsync } from "../lib/hooks";
import { formatNumber } from "../lib/format";
import { invalidRanges, probHorizons, signalRows, stackProbs } from "../lib/signals";

const STATUS = "Computed offline · used only by the SENTIMENT / TRANSFORMER / FULL variants (not SAC_LOB_ONLY)";

const CARDS = [
  {
    icon: Brain,
    title: "FinBERT sentiment cache",
    text: "Headline-level sentiment scores (positive / neutral / negative) are computed offline with FinBERT and cached to disk for reuse.",
  },
  {
    icon: Trend,
    title: "Transformer trend probabilities",
    text: "A Transformer classifier emits up / flat / down trend probabilities from sliding windows of limit-order-book features. Outputs are cached offline and fed to the SAC_LOB_TRANSFORMER and SAC_FULL variants as extra observation features.",
  },
  {
    icon: Clock3,
    title: "As-of news alignment",
    text: "News is joined to market data with a strict as-of rule: at time t only articles published at or before t are visible. Alignment is leak-free by construction.",
  },
];

function Timeline() {
  const xs = [60, 170, 280, 390, 500, 610];
  return (
    <svg className="timeline" viewBox="0 0 680 190" role="img" aria-label="As-of alignment timeline: news published after time t is never visible at time t">
      <defs>
        <linearGradient id="tl-g" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      <rect x="40" y="52" width="285" height="86" rx="10" fill="var(--accent-soft)" />
      <rect x="325" y="52" width="315" height="86" rx="10" fill="var(--red-soft)" />
      <text x="56" y="44" className="tl-cap" fill="var(--text-2)">Visible at t (as-of ≤ t)</text>
      <text x="340" y="44" className="tl-cap" fill="var(--red)">Hidden at t (future)</text>
      <line x1="40" y1="95" x2="640" y2="95" stroke="var(--border-strong)" strokeWidth="2" />
      {xs.map((x, i) => (
        <g key={x}>
          <circle cx={x} cy="95" r="5" fill={x <= 280 ? "url(#tl-g)" : "var(--surface-3)"} stroke="var(--border-strong)" />
          <text x={x} y="120" textAnchor="middle" className="tl-tick" fill="var(--text-3)">t{i - 2 === 0 ? "" : i - 2 > 0 ? `+${i - 2}` : i - 2}</text>
        </g>
      ))}
      <path d="M280 95 V150" stroke="var(--accent)" strokeWidth="2" strokeDasharray="4 3" />
      <rect x="244" y="150" width="72" height="26" rx="13" fill="url(#tl-g)" />
      <text x="280" y="167" textAnchor="middle" className="tl-pill" fill="#fff">now = t</text>
      <g>
        <rect x="150" y="62" width="40" height="18" rx="5" fill="var(--surface)" stroke="var(--accent)" />
        <text x="170" y="75" textAnchor="middle" className="tl-tick" fill="var(--text)">news</text>
      </g>
      <g>
        <rect x="370" y="62" width="40" height="18" rx="5" fill="var(--surface)" stroke="var(--red)" strokeDasharray="3 2" />
        <text x="390" y="75" textAnchor="middle" className="tl-tick" fill="var(--red)">news</text>
        <path d="M390 82 L390 92" stroke="var(--red)" strokeWidth="1.5" />
        <path d="M383 86 L397 100 M397 86 L383 100" stroke="var(--red)" strokeWidth="1.5" strokeLinecap="round" transform="translate(0,-2)" />
      </g>
    </svg>
  );
}

const tFmt = (t) => new Date(t).toISOString().slice(11, 16);
const tAxis = { dataKey: "t", type: "number", domain: ["dataMin", "dataMax"], scale: "time", tickCount: 5, minTickGap: 48, tick: AXIS_TICK, axisLine: AXIS_LINE, tickLine: false, tickFormatter: tFmt };

function SignalCharts({ payload, synthetic }) {
  const rows = useMemo(() => signalRows(payload), [payload]);
  const bad = useMemo(() => invalidRanges(rows), [rows]);
  const horizons = useMemo(() => probHorizons(payload.series), [payload]);
  const [h, setH] = useState(null);
  const hz = h ?? horizons[0];
  const probs = useMemo(() => (hz ? stackProbs(rows, hz) : []), [rows, hz]);
  const has = (k) => k in (payload.series || {});
  const shade = bad.map(([a, b], i) => <ReferenceArea key={i} x1={a} x2={b} fill="var(--amber)" fillOpacity={0.12} stroke="none" ifOverflow="hidden" />);
  return (
    <>
      <section className="card notice fade-up" aria-label="Signal data provenance">
        <span className="notice-icon" aria-hidden="true"><Database size={18} /></span>
        <div>
          <h3 className="card-title">Feature artifacts · {payload.split} split</h3>
          <p className="card-sub">
            {formatNumber(payload.n_points, 0)} of {formatNumber(payload.n_rows_total, 0)} rows shown (evenly strided). Values are the as-of aligned features the SAC variants consume.
            {synthetic ? " Synthetic smoke-test data with stub sentiment: the numbers carry no information about markets." : ""}
          </p>
          {payload.notes?.map((n) => <p className="card-sub" key={n}>{n}</p>)}
        </div>
      </section>
      <ChartCard title="Midprice and sentiment" subtitle="Midprice (left) with EWM sentiment overlay (right); amber shading = transformer signal not yet valid" height={300}
        empty={!rows.length || !has("sentiment_ewm_30")} emptyHint="Sentiment columns are not present in this feature set.">
        <div role="img" aria-label="Midprice line with sentiment overlay" style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis {...tAxis} />
              <YAxis yAxisId="p" tick={AXIS_TICK} axisLine={false} tickLine={false} width={64} domain={["auto", "auto"]} tickFormatter={(v) => formatNumber(v, 2)} />
              <YAxis yAxisId="s" orientation="right" tick={AXIS_TICK} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => formatNumber(v, 2)} />
              <Tooltip content={<ChartTooltip labelFormatter={(l) => tFmt(l)} formatter={(v) => formatNumber(v, 4)} />} isAnimationActive={false} />
              <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />
              {shade}
              <Line yAxisId="p" type="monotone" dataKey="mid" name="Midprice" stroke="var(--text)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Line yAxisId="s" type="monotone" dataKey="sentiment_ewm_30" name="Sentiment EWM 30m" stroke="var(--accent-2)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              {has("sentiment_ewm_5") && <Line yAxisId="s" type="monotone" dataKey="sentiment_ewm_5" name="Sentiment EWM 5m" stroke="var(--blue)" dot={false} strokeWidth={1} strokeOpacity={0.7} isAnimationActive={false} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
      <div className="grid-2">
        <ChartCard title="Article counts in window" subtitle="Articles visible as-of t within 5 / 30 / 120 minutes" height={260} empty={!rows.length || !has("article_count_30")}>
          <div role="img" aria-label="Article counts over time" style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis {...tAxis} />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={40} />
                <Tooltip content={<ChartTooltip labelFormatter={(l) => tFmt(l)} formatter={(v) => formatNumber(v, 2)} />} isAnimationActive={false} />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />
                {shade}
                <Bar dataKey="article_count_120" name="120 min" fill="var(--accent-soft)" stroke="var(--accent)" isAnimationActive={false} />
                <Line type="stepAfter" dataKey="article_count_30" name="30 min" stroke="var(--accent-2)" dot={false} isAnimationActive={false} />
                <Line type="stepAfter" dataKey="article_count_5" name="5 min" stroke="var(--blue)" dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
        <ChartCard title="Transformer trend probabilities" subtitle="Stacked down / flat / up probability" height={260} empty={!rows.length || !horizons.length}
          actions={horizons.length > 1 && <Segmented label="Horizon" value={String(hz)} onChange={(v) => setH(Number(v))} options={horizons.map((x) => ({ value: String(x), label: `${x} min` }))} />}>
          <div role="img" aria-label="Stacked area chart of transformer down, flat and up probabilities" style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={probs} stackOffset="expand" margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis {...tAxis} />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={40} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip content={<ChartTooltip labelFormatter={(l) => tFmt(l)} formatter={(v) => `${formatNumber(v * 100, 1)}%`} />} isAnimationActive={false} />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />
                {shade}
                <Area type="monotone" dataKey="down" name="Down" stackId="p" stroke="var(--red)" fill="var(--red-soft)" isAnimationActive={false} />
                <Area type="monotone" dataKey="flat" name="Flat" stackId="p" stroke="var(--text-3)" fill="var(--hover)" isAnimationActive={false} />
                <Area type="monotone" dataKey="up" name="Up" stackId="p" stroke="var(--green)" fill="var(--green-soft)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>
    </>
  );
}

export default function SignalsView({ refreshKey, synthetic }) {
  const { data: payload, loading } = useAsync(async () => getSignals("test", 1500).catch(() => null), [refreshKey]);
  const live = payload && payload.available === true;
  const reason = payload && payload.available === false ? payload.reason : undefined;
  return (
    <>
      {loading && !payload && <SkeletonCard h={300} />}
      {live && <SignalCharts payload={payload} synthetic={synthetic} />}
      <section className="card notice fade-up" aria-label="Pipeline overview">
        <span className="notice-icon" aria-hidden="true"><Database size={18} /></span>
        <div>
          <h3 className="card-title">Offline NLP pipeline</h3>
          <p className="card-sub">
            Sentiment and temporal-signal features are built offline and only enter the agent variants that name them
            (SAC_LOB_SENTIMENT, SAC_LOB_TRANSFORMER, SAC_FULL). SAC_LOB_ONLY and the baselines never see them.
          </p>
        </div>
      </section>

      <div className="grid-3">
        {CARDS.map((c, i) => {
          const Icon = c.icon;
          return (
            <article className="card card-hover fade-up" key={c.title} style={{ animationDelay: `${i * 50}ms` }}>
              <span className="kpi-icon big" aria-hidden="true"><Icon size={18} /></span>
              <h3 className="card-title" style={{ marginTop: 14 }}>{c.title}</h3>
              <p className="card-sub body">{c.text}</p>
              <StatusChip tone="warn" className="status-wrap">{STATUS}</StatusChip>
            </article>
          );
        })}
      </div>

      <section className="card fade-up" aria-label="Leak-free as-of alignment">
        <header className="card-head">
          <div>
            <h3 className="card-title">Leak-free as-of alignment</h3>
            <p className="card-sub">News at t+1 is never visible at t — only information available at decision time is joined.</p>
          </div>
        </header>
        <Timeline />
      </section>

      {!live && !loading && (
        <section className="card fade-up" aria-label="Signal outputs">
          <header className="card-head">
            <div>
              <h3 className="card-title">Signal outputs</h3>
              <p className="card-sub">Real series appear here once the feature artifacts exist.</p>
            </div>
          </header>
          <EmptyState compact hint={reason || "No sentiment or trend values are displayed because the feature artifacts are unavailable."} />
        </section>
      )}
    </>
  );
}
