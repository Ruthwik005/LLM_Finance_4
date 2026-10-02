import { Moon, Sun } from "lucide-react";
import Logo from "./Logo";
import { VIEWS } from "./nav";

export default function Sidebar({ active, onChange, theme, onToggleTheme }) {
  const onKey = (e) => {
    const i = VIEWS.findIndex((v) => v.id === active);
    let n = null;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") n = (i + 1) % VIEWS.length;
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") n = (i - 1 + VIEWS.length) % VIEWS.length;
    if (n !== null) {
      e.preventDefault();
      onChange(VIEWS[n].id);
      document.getElementById(`tab-${VIEWS[n].id}`)?.focus();
    }
  };
  return (
    <aside className="sidebar" aria-label="Primary">
      <div className="brand">
        <Logo />
        <div className="brand-text">
          <span className="brand-name">MarketMaker Lab</span>
          <span className="brand-sub">Research dashboard</span>
        </div>
      </div>
      <nav className="nav" role="tablist" aria-orientation="vertical" aria-label="Dashboard views" onKeyDown={onKey}>
        {VIEWS.map((v) => {
          const Icon = v.icon;
          const on = v.id === active;
          return (
            <button
              key={v.id}
              id={`tab-${v.id}`}
              role="tab"
              aria-selected={on}
              aria-controls="main-panel"
              aria-label={v.label}
              tabIndex={on ? 0 : -1}
              className={`nav-item ${on ? "active" : ""}`}
              onClick={() => onChange(v.id)}
            >
              <Icon size={18} aria-hidden="true" />
              <span className="nav-label">{v.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="side-foot">
        <button
          type="button"
          className="icon-btn"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title="Toggle theme"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <span className="version mono">v0.1.0</span>
      </div>
    </aside>
  );
}
