import { useMemo } from "react";
import ResponsiveContainer from "./AutoSize";
import { CartesianGrid, Legend, Line, LineChart, ReferenceDot, Tooltip, XAxis, YAxis } from "recharts";
import { Trophy, Target, Flag } from "lucide-react";
import ChartCard from "./ChartCard";
import ChartTooltip from "./ChartTooltip";
import KpiCard from "./KpiCard";
import { SkeletonKpis } from "./Skeleton";
import EmptyState from "./EmptyState";
import NoRun from "./NoRun";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { getTrainingCurve } from "../api";
import { useAsync } from "../lib/hooks";
import { bestPoint, hasCurve, hasSeries, lastValue } from "../lib/training";
import { formatCompact, formatNumber } from "../lib/format";

function Curve({ title, subtitle, data, lines, best, bestKey, height = 260, fmt = (v) => formatNumber(v, 4), label }) {
  const has = lines.some((l) => hasSeries(data, l.key));
  return (
    <ChartCard title={title} subtitle={subtitle} empty={!has} emptyHint="This series was not logged for this run." height={height}>
      <div role="img" aria-label={label || title} style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="timesteps" type="number" domain={["dataMin", "dataMax"]} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} tickFormatter={(v) => formatCompact(v)} />
            <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={64} tickFormatter={(v) => formatNumber(v, 3)} />
            <Tooltip content={<ChartTooltip labelFormatter={(l) => `${formatNumber(l, 0)} timesteps`} formatter={fmt} />} isAnimationActive={false} />
            {lines.length > 1 && <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} />}
            {lines.map((l) => (
              <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={1.75} dot={{ r: 2 }} connectNulls isAnimationActive={false} />
            ))}
            {best && Number.isFinite(best[bestKey]) && (
              <ReferenceDot x={best.timesteps} y={best[bestKey]} r={6} fill="var(--amber)" stroke="var(--surface)" strokeWidth={2} ifOverflow="extendDomain" />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

export default function TrainingView({ runId, refreshKey }) {
  const { data, loading, error } = useAsync(async () => (runId ? getTrainingCurve(runId) : null), [runId, refreshKey]);
  const pts = useMemo(() => data || [], [data]);
  const best = useMemo(() => bestPoint(pts), [pts]);

  if (!runId) return <NoRun />;
  if (error) return <div className="card"><EmptyState hint={error} /></div>;
  if (loading && !data) return <><SkeletonKpis n={3} /><div className="card"><EmptyState compact hint="Loading…" /></div></>;
  if (!hasCurve(pts)) {
    return <div className="card"><EmptyState hint="No training curve was recorded for this run (baselines are not trained, and older runs logged none)." /></div>;
  }
  return (
    <>
      <div className="kpi-grid kpi-grid-4">
        <KpiCard delay={0} label="Evaluations logged" icon={Flag} value={pts.length} format={(v) => formatNumber(Math.round(v), 0)} />
        <KpiCard delay={40} label="Best checkpoint at" icon={Trophy} value={best?.timesteps} format={(v) => `${formatCompact(v)} steps`} hint="Highest mean validation reward" />
        <KpiCard delay={80} label="Best val reward" icon={Target} value={best?.val_reward_mean} colorBySign format={(v) => formatNumber(v, 4)} />
        <KpiCard delay={120} label="Last val PnL" icon={Target} value={lastValue(pts, "val_pnl_mean")} colorBySign format={(v) => formatNumber(v, 4)} />
      </div>
      <div className="grid-2">
        <Curve title="Reward: train vs validation" subtitle="Amber dot marks the selected (best) checkpoint" data={pts} best={best} bestKey="val_reward_mean"
          lines={[{ key: "train_reward_mean", name: "Train reward", color: "var(--accent)" }, { key: "val_reward_mean", name: "Validation reward", color: "var(--green)" }]} />
        <Curve title="Validation PnL" subtitle="Mean PnL on fixed validation windows" data={pts} best={best} bestKey="val_pnl_mean"
          lines={[{ key: "val_pnl_mean", name: "Val PnL", color: "var(--blue)" }]} />
      </div>
      <Curve title="Validation MAP" subtitle="Mean absolute position on validation windows (lower is less inventory risk)" data={pts} best={best} bestKey="val_map" height={220}
        lines={[{ key: "val_map", name: "Val MAP", color: "var(--amber)" }]} />
      <div className="small-multiples">
        <Curve title="Entropy coefficient" data={pts} height={190} lines={[{ key: "ent_coef", name: "ent_coef", color: "var(--accent-2)" }]} />
        <Curve title="Critic loss" data={pts} height={190} lines={[{ key: "critic_loss", name: "critic loss", color: "var(--red)" }]} />
        <Curve title="Actor loss" data={pts} height={190} lines={[{ key: "actor_loss", name: "actor loss", color: "var(--green)" }]} />
      </div>
    </>
  );
}
