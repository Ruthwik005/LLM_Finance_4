import { useEffect, useMemo, useState } from "react";
import ResponsiveContainer from "./AutoSize";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import { Coins, DollarSign, Ruler, Boxes, Ban, Percent, MoveHorizontal, Crosshair, ShieldAlert } from "lucide-react";
import ChartCard from "./ChartCard";
import ChartTooltip from "./ChartTooltip";
import KpiCard from "./KpiCard";
import DataTable from "./DataTable";
import Segmented from "./Segmented";
import { SkeletonKpis, SkeletonCard } from "./Skeleton";
import EmptyState from "./EmptyState";
import NoRun from "./NoRun";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { useRunData } from "../lib/runData";
import { formatNumber, formatCurrency, formatPercent } from "../lib/format";
import { attributionStack } from "../lib/attribution";
import { fillStats } from "../lib/derive";

const PAGE = 50;

const COLUMNS = [
  { key: "episode", header: "Episode", mono: true, align: "right" },
  {
    key: "side",
    header: "Side",
    render: (r) => <span className={`pill pill-${r.side}`}>{String(r.side).toUpperCase()}</span>,
  },
  { key: "price", header: "Price", mono: true, align: "right", render: (r) => formatNumber(r.price, 4) },
  { key: "volume", header: "Volume", mono: true, align: "right", render: (r) => formatNumber(r.volume, 6) },
];

export default function BehaviorView({ runId, refreshKey }) {
  const { data, loading, error } = useRunData(runId, refreshKey, ["fills", "metrics", "attribution"]);
  const [side, setSide] = useState("all");
  const [episode, setEpisode] = useState("all");
  const [limit, setLimit] = useState(PAGE);
  const fills = useMemo(() => data?.fills || [], [data]);

  useEffect(() => {
    setSide("all");
    setEpisode("all");
    setLimit(PAGE);
  }, [runId]);

  const episodes = useMemo(() => [...new Set(fills.map((f) => f.episode))].sort((a, b) => a - b), [fills]);
  const filtered = useMemo(
    () => fills.filter((f) => (side === "all" || f.side === side) && (episode === "all" || String(f.episode) === episode)),
    [fills, side, episode],
  );
  const stats = useMemo(() => fillStats(fills), [fills]);
  const m = data?.metrics || {};
  const attr = useMemo(() => attributionStack(data?.attribution), [data]);

  if (!runId) return <NoRun />;
  if (error) return <div className="card"><EmptyState hint={error} /></div>;
  const busy = loading && !data;

  const counts = stats ? [{ name: "Bid", value: stats.bidCount, fill: "var(--green)" }, { name: "Ask", value: stats.askCount, fill: "var(--red)" }] : [];
  const vols = stats ? [{ name: "Bid", volume: stats.bidVolume, fill: "var(--green)" }, { name: "Ask", volume: stats.askVolume, fill: "var(--red)" }] : [];

  return (
    <>
      {busy ? (
        <SkeletonKpis n={4} />
      ) : (
        <div className="kpi-grid kpi-grid-4">
          <KpiCard delay={0} label="Fills" icon={Coins} value={stats?.count} format={(v) => formatNumber(Math.round(v), 0)} />
          <KpiCard delay={40} label="Total volume" icon={Boxes} value={stats?.totalVolume} format={(v) => formatNumber(v, 4)} />
          <KpiCard delay={80} label="Avg price" icon={DollarSign} value={stats?.avgPrice ?? undefined} format={(v) => formatCurrency(v)} />
          <KpiCard delay={120} label="Avg fill size" icon={Ruler} value={stats?.avgSize} format={(v) => formatNumber(v, 4)} />
        </div>
      )}

      {busy ? (
        <SkeletonKpis n={5} />
      ) : (
        <div className="kpi-grid kpi-grid-5">
          <KpiCard delay={0} label="Cancel ratio" icon={Ban} value={m.cancel_ratio} format={(v) => formatPercent(v)} hint="Canceled / placed volume" />
          <KpiCard delay={40} label="Fill rate" icon={Percent} value={m.fill_rate} format={(v) => formatPercent(v, 2)} hint="Filled / placed volume" />
          <KpiCard delay={80} label="Avg quoted spread" icon={MoveHorizontal} value={m.avg_quoted_spread} format={(v) => `${formatNumber(v, 2)} ticks`} />
          <KpiCard delay={120} label="Avg quote distance" icon={Crosshair} value={m.avg_quote_distance} format={(v) => `${formatNumber(v, 2)} ticks`} hint="From mid" />
          <KpiCard delay={160} label="Adverse selection" icon={ShieldAlert} value={m.adverse_selection_ratio} format={(v) => formatPercent(v)} hint="Fills followed by an adverse move" />
        </div>
      )}

      <ChartCard title="PnL attribution" subtitle="Spread capture vs inventory PnL vs fees and rebates (totals over all evaluation episodes)" loading={busy} empty={!attr} emptyHint="Attribution metrics are not available for this run." height={170}>
        {attr && (
          <>
            <div role="img" aria-label="Stacked bar of PnL attribution by component" style={{ width: "100%", height: 150 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={attr.row} layout="vertical" stackOffset="sign" margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid {...GRID} horizontal={false} vertical />
                  <XAxis type="number" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} tickFormatter={(v) => formatNumber(v, 2)} />
                  <YAxis type="category" dataKey="name" hide />
                  <ReferenceLine x={0} stroke="var(--border-strong)" />
                  <Tooltip cursor={{ fill: "var(--hover)" }} content={<ChartTooltip labelFormatter={() => "PnL attribution"} formatter={(v) => formatNumber(v, 4)} />} isAnimationActive={false} />
                  <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />
                  {attr.parts.map((p) => <Bar key={p.key} dataKey={p.key} name={p.name} stackId="a" fill={p.color} isAnimationActive={false} maxBarSize={44} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="card-sub">Total PnL <b className={`mono ${attr.total >= 0 ? "pos" : "neg"}`}>{formatNumber(attr.total, 4)}</b></p>
          </>
        )}
      </ChartCard>

      <div className="grid-2">
        <ChartCard title="Bid vs ask fills" subtitle="Share of fill count by side" loading={busy} empty={!stats} height={240}>
          <div className="donut-wrap" role="img" aria-label="Donut chart of bid versus ask fill counts">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={counts} dataKey="value" nameKey="name" innerRadius={62} outerRadius={92} paddingAngle={3} stroke="var(--surface)" strokeWidth={2} isAnimationActive={false}>
                  {counts.map((c) => <Cell key={c.name} fill={c.fill} />)}
                </Pie>
                <Tooltip content={<ChartTooltip labelFormatter={() => "Fills"} formatter={(v) => formatNumber(v, 0)} />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="donut-legend">
              {counts.map((c) => (
                <div key={c.name}><span className="dot" style={{ background: c.fill }} /> {c.name} <b className="mono">{formatNumber(c.value, 0)}</b></div>
              ))}
            </div>
          </div>
        </ChartCard>
        <ChartCard title="Filled volume by side" subtitle="Total base volume" loading={busy} empty={!stats} height={240}>
          <div role="img" aria-label="Bar chart of filled volume by side" style={{ width: "100%", height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={vols} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid {...GRID} />
                <XAxis dataKey="name" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} />
                <Tooltip cursor={{ fill: "var(--hover)" }} content={<ChartTooltip labelFormatter={(l) => l} formatter={(v) => formatNumber(v, 4)} />} />
                <Bar dataKey="volume" name="Volume" radius={[6, 6, 0, 0]} maxBarSize={72} isAnimationActive={false}>
                  {vols.map((c) => <Cell key={c.name} fill={c.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {busy ? (
        <SkeletonCard h={320} />
      ) : (
        <section className="card fade-up" aria-label="Fills">
          <header className="card-head">
            <div>
              <h3 className="card-title">Fills</h3>
              <p className="card-sub">{formatNumber(filtered.length, 0)} of {formatNumber(fills.length, 0)} fills</p>
            </div>
            <div className="card-actions filters">
              <Segmented
                label="Filter by side"
                value={side}
                onChange={(v) => { setSide(v); setLimit(PAGE); }}
                options={[{ value: "all", label: "All" }, { value: "bid", label: "Bid" }, { value: "ask", label: "Ask" }]}
              />
              <label className="select-wrap">
                <span className="sr-only">Filter by episode</span>
                <select value={episode} onChange={(e) => { setEpisode(e.target.value); setLimit(PAGE); }} aria-label="Filter by episode">
                  <option value="all">All episodes</option>
                  {episodes.map((e) => <option key={e} value={String(e)}>Episode {e}</option>)}
                </select>
              </label>
            </div>
          </header>
          {filtered.length === 0 ? (
            <EmptyState compact hint={fills.length ? "No fills match the current filters." : undefined} />
          ) : (
            <>
              <DataTable
                caption="Fills table"
                columns={COLUMNS}
                rows={filtered.slice(0, limit)}
                rowKey={(r, i) => i}
                maxHeight={420}
              />
              {limit < filtered.length && (
                <div className="more">
                  <button type="button" className="btn" onClick={() => setLimit((l) => l + PAGE)}>
                    Show more ({formatNumber(Math.min(PAGE, filtered.length - limit), 0)} of {formatNumber(filtered.length - limit, 0)} remaining)
                  </button>
                </div>
              )}
            </>
          )}
        </section>
      )}
    </>
  );
}
