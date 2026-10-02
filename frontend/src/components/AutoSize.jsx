import { Children, cloneElement, isValidElement, useLayoutEffect, useRef, useState } from "react";

/**
 * Drop-in replacement for recharts' ResponsiveContainer: measures its own box and passes
 * explicit pixel width/height to the chart child, so charts never collapse to 0px.
 */
export default function ResponsiveContainer({ width = "100%", height = "100%", children }) {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const w = Math.round(r.width);
      const h = Math.round(r.height);
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const child = Children.only(children);
  const ready = size.w > 0 && size.h > 0 && isValidElement(child);
  return (
    <div ref={ref} style={{ width, height, minWidth: 0 }}>
      {ready ? cloneElement(child, { width: size.w, height: size.h }) : null}
    </div>
  );
}
