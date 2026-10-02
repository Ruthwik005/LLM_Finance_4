import { Brain, Clock3, Database, LineChart as Trend } from "lucide-react";
import StatusChip from "./StatusChip";
import EmptyState from "./EmptyState";

const STATUS = "Computed offline · not wired into SAC_LOB_ONLY training";

const CARDS = [
  {
    icon: Brain,
    title: "FinBERT sentiment cache",
    text: "Headline-level sentiment scores (positive / neutral / negative) are computed offline with FinBERT and cached to disk for reuse.",
  },
  {
    icon: Trend,
    title: "Transformer trend probabilities",
    text: "A Transformer classifier emits up / flat / down trend probabilities from sliding windows of limit-order-book features. Outputs are stored offline and are not consumed by the agent.",
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

export default function SignalsView() {
  return (
    <>
      <section className="card notice fade-up" aria-label="Pipeline overview">
        <span className="notice-icon" aria-hidden="true"><Database size={18} /></span>
        <div>
          <h3 className="card-title">Offline NLP pipeline</h3>
          <p className="card-sub">
            The sentiment and temporal-signal components are validated separately from the market-making agent. They are shown here
            for transparency and do not influence the results on the other screens.
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

      <section className="card fade-up" aria-label="Signal outputs">
        <header className="card-head">
          <div>
            <h3 className="card-title">Signal outputs</h3>
            <p className="card-sub">Per-run sentiment and trend outputs are not exposed by the results API.</p>
          </div>
        </header>
        <EmptyState compact hint="No sentiment or trend values are displayed because none feed this run." />
      </section>
    </>
  );
}
