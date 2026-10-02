import CountUp from "./CountUp";
import Sparkline from "./Sparkline";
import { DASH, signClass } from "../lib/format";

export default function KpiCard({ label, icon: Icon, value, format, colorBySign = false, spark, sparkTone, hint, delay = 0 }) {
  const has = Number.isFinite(value);
  const sign = colorBySign && has ? signClass(value) : "";
  const tone = sparkTone || (sign === "neg" ? "neg" : sign === "pos" ? "pos" : "accent");
  return (
    <div className="card card-hover kpi fade-up" style={{ animationDelay: `${delay}ms` }}>
      <div className="kpi-head">
        <span className="kpi-label">{label}</span>
        {Icon && (
          <span className="kpi-icon" aria-hidden="true">
            <Icon size={15} />
          </span>
        )}
      </div>
      <div className={`kpi-value mono ${sign}`}>{has ? <CountUp value={value} format={format} /> : DASH}</div>
      {spark !== undefined ? <Sparkline values={spark} tone={tone} /> : <div className="spark-gap" />}
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  );
}
