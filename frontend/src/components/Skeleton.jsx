export default function Skeleton({ h = 16, w = "100%", r, className = "", style }) {
  return (
    <span
      className={`skeleton ${className}`}
      aria-hidden="true"
      style={{ height: h, width: w, borderRadius: r, ...style }}
    />
  );
}

export function SkeletonCard({ h = 260 }) {
  return (
    <div className="card" role="status" aria-label="Loading">
      <Skeleton h={14} w="40%" />
      <Skeleton h={10} w="25%" style={{ marginTop: 10 }} />
      <Skeleton h={h} style={{ marginTop: 20 }} r={10} />
    </div>
  );
}

export function SkeletonKpis({ n = 6 }) {
  return (
    <div className="kpi-grid" role="status" aria-label="Loading metrics">
      {Array.from({ length: n }, (_, i) => (
        <div className="card kpi" key={i}>
          <Skeleton h={12} w="50%" />
          <Skeleton h={30} w="70%" style={{ marginTop: 14 }} />
          <Skeleton h={24} style={{ marginTop: 14 }} />
        </div>
      ))}
    </div>
  );
}
