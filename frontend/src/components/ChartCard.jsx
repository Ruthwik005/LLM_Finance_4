import Skeleton from "./Skeleton";
import EmptyState from "./EmptyState";

export default function ChartCard({ title, subtitle, actions, loading, empty, emptyHint, height = 300, children, className = "" }) {
  return (
    <section className={`card chart-card fade-up ${className}`} aria-label={title}>
      <header className="card-head">
        <div>
          <h3 className="card-title">{title}</h3>
          {subtitle && <p className="card-sub">{subtitle}</p>}
        </div>
        {actions && <div className="card-actions">{actions}</div>}
      </header>
      <div className="chart-body" style={{ minHeight: height }}>
        {loading ? <Skeleton h={height} r={10} /> : empty ? <EmptyState hint={emptyHint} compact /> : children}
      </div>
    </section>
  );
}
