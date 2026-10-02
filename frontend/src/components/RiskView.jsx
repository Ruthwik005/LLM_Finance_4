import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Layers, Maximize2, Scale, TrendingDown } from "lucide-react";
import ChartCard from "./ChartCard";
import ChartTooltip from "./ChartTooltip";
import KpiCard from "./KpiCard";
import { SkeletonKpis } from "./Skeleton";
import EmptyState from "./EmptyState";
import NoRun from "./NoRun";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { useRunData } from "../lib/runData";
import { formatCurrency, formatNumber } from "../lib/format";
import { inventoryPath, maxAbsInventory, maxDrawdown, meanAbsInventory, pnlPerMap } from "../lib/derive";

export default function RiskView({ runId, refreshKey }) {
  const { data, loading, error } = useRunData(runId, refreshKey, ["metrics", "fills", "equity"]);
  const path = useMemo(() => inventoryPath(data?.fills), [data]);
  if (!runId) return <NoRun />;
  if (error) return <div className="card"><EmptyState hint={error} /></div>;
  const busy = loading && !data;
  const m = data?.metrics || {};
  const map = Number.isFinite(m.map) ? m.map : meanAbsInventory(path);
  const ratio = pnlPerMap(m.mean_pnl, map);
  return (
    <>
      {busy ? (
        <SkeletonKpis n={4} />
      ) : (
        <div className="kpi-grid kpi-grid-4">
          <KpiCard delay={0} label="Mean abs position" icon={Layers} value={map ?? undefined} format={(v) => formatNumber(v, 3)} hint={Number.isFinite(m.map) ? "From metrics" : "Derived from fills"} />
          <KpiCard delay={40} label="Max |inventory|" icon={Maximize2} value={maxAbsInventory(path) ?? undefined} format={(v) => formatNumber(v, 3)} />
          <KpiCard delay={80} label="Max drawdown" icon={TrendingDown} value={maxDrawdown(data?.equity) ?? undefined} format={(v) => formatCurrency(v)} />
          <KpiCard delay={120} label="PnL / MAP" icon={Scale} value={ratio ?? undefined} format={(v) => formatNumber(v, 2)} colorBySign hint="Mean PnL ÷ mean abs position" />
        </div>
      )}
      <ChartCard
        title="Net inventory"
        subtitle="Derived from fills: bid = +volume, ask = −volume, reset each episode"
        loading={busy}
        empty={path.length === 0}
        height={360}
      >
        <div role="img" aria-label="Net inventory over fill sequence" style={{ width: "100%", height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={path} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="i" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} minTickGap={24} />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} />
              <Tooltip
                cursor={{ stroke: "var(--text-3)", strokeDasharray: "3 3" }}
                content={
                  <ChartTooltip
                    labelFormatter={(l, p) => `Fill #${l}${p?.[0]?.payload ? ` · episode ${p[0].payload.episode}` : ""}`}
                    formatter={(v) => formatNumber(v, 4)}
                  />
                }
              />
              <ReferenceLine y={0} stroke="var(--text-3)" />
              <Area type="stepAfter" dataKey="pos" name="Long" stroke="var(--green)" fill="var(--green)" fillOpacity={0.28} strokeWidth={1.5} isAnimationActive={false} />
              <Area type="stepAfter" dataKey="neg" name="Short" stroke="var(--red)" fill="var(--red)" fillOpacity={0.28} strokeWidth={1.5} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </>
  );
}
