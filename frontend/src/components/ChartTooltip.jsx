export default function ChartTooltip({ active, payload, label, labelPrefix = "", formatter, labelFormatter }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="tooltip">
      <div className="tooltip-label">{labelFormatter ? labelFormatter(label, payload) : `${labelPrefix}${label}`}</div>
      {payload
        .filter((p) => p.value !== undefined && p.value !== null)
        .map((p, i) => (
          <div className="tooltip-row" key={i}>
            <span className="dot" style={{ background: p.color || p.fill || p.stroke }} />
            <span className="tooltip-name">{p.name}</span>
            <span className="mono tooltip-val">{formatter ? formatter(p.value, p) : p.value}</span>
          </div>
        ))}
    </div>
  );
}
