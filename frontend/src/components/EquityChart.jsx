import { useId } from "react";
import ResponsiveContainer from "./AutoSize";
import { Area, AreaChart, CartesianGrid, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import ChartTooltip from "./ChartTooltip";
import { AXIS_LINE, AXIS_TICK, GRID } from "./chartTheme";
import { formatCurrency, formatCompact } from "../lib/format";

export default function EquityChart({ data, height = 320, compact = false }) {
  const id = useId().replace(/:/g, "");
  return (
    <div role="img" aria-label="Cumulative PnL by episode" style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 12, left: compact ? -24 : 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`eq-${id}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.42} />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
            </linearGradient>
          </defs>
          {!compact && <CartesianGrid {...GRID} />}
          <XAxis dataKey="episode" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} hide={compact} minTickGap={24} />
          <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={compact ? 44 : 64} tickFormatter={(v) => formatCompact(v)} hide={compact} />
          <Tooltip
            cursor={{ stroke: "var(--text-3)", strokeDasharray: "3 3" }}
            content={<ChartTooltip labelPrefix="Episode " formatter={(v) => formatCurrency(v)} />}
          />
          <ReferenceLine y={0} stroke="var(--text-3)" strokeDasharray="4 4" />
          <Area
            type="monotone"
            dataKey="cumulative_pnl"
            name="Cumulative PnL"
            stroke="var(--accent)"
            strokeWidth={2}
            fill={`url(#eq-${id})`}
            isAnimationActive={false}
            activeDot={{ r: 4, fill: "var(--accent)", stroke: "var(--surface)", strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
