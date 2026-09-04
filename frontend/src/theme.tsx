import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

/**
 * Light or dark, for the whole application.
 *
 * Three rules, and the middle one is the easy one to get wrong:
 *
 * 1. **The system setting is the default.** Someone arriving for the first time on a
 *    dark-set machine gets dark, without touching anything.
 * 2. **Nothing is stored until the button is actually pressed.** Writing the resolved
 *    theme on first render would silently freeze a preference the person never expressed,
 *    and they would stop following their own system setting from then on.
 * 3. **An explicit choice wins in both directions.** `[data-theme]` beats the media query
 *    in the stylesheet, so preferring light on a dark-set machine works — which a
 *    `prefers-color-scheme`-only implementation cannot express at all.
 *
 * While no choice has been made the attribute is left off entirely and the media query
 * decides, so the app keeps following the system live as it changes.
 */
export type Theme = "light" | "dark";

const STORAGE_KEY = "mda-theme";
const DARK_QUERY = "(prefers-color-scheme: dark)";

function systemTheme(): Theme {
  return typeof window !== "undefined" && window.matchMedia(DARK_QUERY).matches
    ? "dark"
    : "light";
}

function storedTheme(): Theme | null {
  // Reading storage throws outright in some privacy modes, so this is never assumed to work.
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

interface ThemeApi {
  /** What is actually on screen right now. */
  theme: Theme;
  /** False while the system setting is still deciding. */
  isExplicit: boolean;
  toggle: () => void;
  /** Forget the choice and follow the system again. */
  useSystem: () => void;
}

const Ctx = createContext<ThemeApi | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoice] = useState<Theme | null>(() => storedTheme());
  const [system, setSystem] = useState<Theme>(() => systemTheme());

  const theme = choice ?? system;

  // Follow the system while no explicit choice has been made.
  useEffect(() => {
    const media = window.matchMedia(DARK_QUERY);
    const onChange = (e: MediaQueryListEvent) => setSystem(e.matches ? "dark" : "light");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (choice) root.setAttribute("data-theme", choice);
    // Removed, not set to the resolved value: with no attribute the media query is back in
    // charge, which is exactly what "I have not chosen" should mean.
    else root.removeAttribute("data-theme");
  }, [choice]);

  const persist = useCallback((next: Theme | null) => {
    setChoice(next);
    try {
      if (next) localStorage.setItem(STORAGE_KEY, next);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // A blocked storage means the choice lasts for this tab only. Worth doing anyway.
    }
  }, []);

  const value = useMemo<ThemeApi>(
    () => ({
      theme,
      isExplicit: choice !== null,
      toggle: () => persist(theme === "dark" ? "light" : "dark"),
      useSystem: () => persist(null),
    }),
    [theme, choice, persist],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme(): ThemeApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTheme must be used inside <ThemeProvider>");
  return v;
}
