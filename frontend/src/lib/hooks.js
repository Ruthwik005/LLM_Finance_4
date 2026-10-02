import { useCallback, useEffect, useRef, useState } from "react";

/** Run an async loader whenever deps change. Returns {data, loading, error}. Stale results are ignored. */
export function useAsync(loader, deps) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  useEffect(() => {
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    Promise.resolve()
      .then(() => loaderRef.current())
      .then((data) => !cancelled && setState({ data, loading: false, error: null }))
      .catch((e) => !cancelled && setState({ data: null, loading: false, error: e?.message || "Request failed" }));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

function systemTheme() {
  try {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const t = localStorage.getItem("mml-theme");
      if (t === "light" || t === "dark") return t;
    } catch {
      /* ignore */
    }
    return systemTheme();
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  const toggle = useCallback(() => {
    setTheme((t) => {
      const next = t === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("mml-theme", next);
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);
  return [theme, toggle];
}

export function usePrefersReducedMotion() {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}
