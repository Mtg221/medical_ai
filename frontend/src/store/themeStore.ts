import { create } from "zustand";

export type ThemeMode = "light" | "dark";

function readInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "light";
  try {
    const saved = window.localStorage.getItem("medresearch_theme");
    if (saved === "dark" || saved === "light") {
      document.documentElement.classList.toggle("dark", saved === "dark");
      return saved;
    }
    if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
      return "dark";
    }
  } catch {
    // ignore storage errors in headless/SSR environments
  }
  return "light";
}

interface ThemeState {
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  toggleTheme: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readInitialTheme(),
  setTheme: (theme) => {
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
    try {
      window.localStorage.setItem("medresearch_theme", theme);
    } catch {
      // ignore
    }
    set({ theme });
  },
  toggleTheme: () => {
    const next: ThemeMode = get().theme === "dark" ? "light" : "dark";
    get().setTheme(next);
  },
}));
