import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import StatusChip from "./StatusChip";
import { runVariant } from "../lib/variant";

export default function RunSelect({ runs, value, onChange, variants, loading }) {
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const close = (e) => ref.current && !ref.current.contains(e.target) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const disabled = loading || !runs.length;
  const label = loading ? "Loading runs…" : runs.length ? value || "Select a run" : "No runs";

  const onKey = (e) => {
    if (disabled) return;
    if (e.key === "Escape") setOpen(false);
    else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) {
        setHi(Math.max(0, runs.indexOf(value)));
        setOpen(true);
      } else setHi((h) => Math.min(runs.length - 1, h + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHi((h) => Math.max(0, h - 1));
    } else if ((e.key === "Enter" || e.key === " ") && open) {
      e.preventDefault();
      onChange(runs[hi]);
      setOpen(false);
    }
  };

  return (
    <div className="runselect" ref={ref} onKeyDown={onKey}>
      <button
        type="button"
        className="runselect-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="Select experiment run"
        disabled={disabled}
        onClick={() => {
          setHi(Math.max(0, runs.indexOf(value)));
          setOpen((o) => !o);
        }}
      >
        <span className="runselect-eyebrow">Run</span>
        <span className="runselect-value mono">{label}</span>
        {value && variants[value] && <StatusChip tone="accent">{runVariant(variants[value])}</StatusChip>}
        <ChevronDown size={16} className={`chev ${open ? "open" : ""}`} aria-hidden="true" />
      </button>
      {open && (
        <ul className="runselect-menu" role="listbox" aria-label="Runs">
          {runs.map((r, i) => (
            <li
              key={r}
              role="option"
              aria-selected={r === value}
              className={`${i === hi ? "hi" : ""} ${r === value ? "sel" : ""}`}
              onMouseEnter={() => setHi(i)}
              onClick={() => {
                onChange(r);
                setOpen(false);
              }}
            >
              <span className="mono opt-id">{r}</span>
              {variants[r] && <StatusChip tone="accent">{runVariant(variants[r])}</StatusChip>}
              {r === value && <Check size={14} aria-hidden="true" />}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
