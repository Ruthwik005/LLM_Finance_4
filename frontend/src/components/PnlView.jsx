import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import ResponsiveContainer from "./AutoSize";
import { ArrowDownRight, ArrowUpRight, Percent, TrendingDown, Gauge } from "lucide-react";
import ChartCard from "./ChartCard";
import EquityChart from "./EquityChart";
import ChartTooltip from "./ChartTooltip";
import KpiCard from "./KpiCard";
import { SkeletonKpis } from "./Skeleton";
import EmptyState from "./EmptyState";
import NoRun from "./NoRun";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { useRunData } from "../lib/runData";
import { formatCompact, formatCurrency, formatPercent } from "../lib/format";
import { episodeStats, maxDrawdown, winRate } from "../lib/derive";

export default function PnlView({ runId, refreshKey }) {
  const { data, loading, error } = useRunData(runId, refreshKey, ["episodes", "equity"]);
  if (!runId) return <NoRun />;
  if (error) return <div className="card"><EmptyState hint={error} /></div>;
  const eps = data?.episodes || [];
  const eq = data?.equity || [];
  const busy = loading && !data;
  const stats = episodeStats(eps);
  return (
    <>
      {busy ? (
        <SkeletonKpis n={5} />
      ) : (
        <div className="kpi-grid kpi-grid-5">
          <KpiCard delay={0} label="Best episode" icon={ArrowUpRight} value={stats?.best.pnl} format={(v) => formatCurrency(v)} colorBySign hint={stats ? `Episode ${stats.best.episode}` : undefined} />
          <KpiCard delay={40} label="Worst episode" icon={ArrowDownRight} value={stats?.worst.pnl} format={(v) => formatCurrency(v)} colorBySign hint={stats ? `Episode ${stats.worst.episode}` : undefined} />
          <KpiCard delay={80} label="Win rate" icon={Percent} value={winRate(eps) ?? undefined} format={(v) => formatPercent(v)} />
          <KpiCard delay={120} label="Avg PnL / episode" icon={Gauge} value={stats?.avg} format={(v) => formatCurrency(v)} colorBySign />
          <KpiCard delay={160} label="Max drawdown" icon={TrendingDown} value={maxDrawdown(eq) ?? undefined} format={(v) => formatCurrency(v)} hint="Peak-to-trough, from equity curve" />
        </div>
      )}
      <ChartCard title="Cumulative PnL" subtitle="Equity curve across episodes" loading={busy} empty={eq.length === 0} height={340}>
        <EquityChart data={eq} height={340} />
      </ChartCard>
      <ChartCard title="Episode PnL" subtitle="Profit (green) and loss (red) per episode" loading={busy} empty={eps.length === 0} height={260}>
        <div role="img" aria-label="PnL per episode" style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={eps} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="episode" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} minTickGap={16} />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={64} tickFormatter={(v) => formatCompact(v)} />
              <Tooltip cursor={{ fill: "var(--hover)" }} content={<ChartTooltip labelPrefix="Episode " formatter={(v) => formatCurrency(v)} />} />
              <ReferenceLine y={0} stroke="var(--text-3)" />
              <Bar dataKey="pnl" name="PnL" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {eps.map((e, i) => (
                  <Cell key={i} fill={e.pnl >= 0 ? "var(--green)" : "var(--red)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </>
  );
}
