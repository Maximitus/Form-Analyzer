/**
 * Theme + accent presets — aligned with TOOLBOX_UI_SPEC.md (shared belt).
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

export const STORAGE_THEME_KEY = 'toolbox-theme';
export const STORAGE_ACCENT_KEY = 'toolbox-accent';

export type ThemeMode = 'light' | 'dark';

export type AccentPresetId = 'orange' | 'cyan' | 'violet' | 'green' | 'rose';

export type AccentPreset = {
  id: AccentPresetId;
  label: string;
  darkAccent: string;
  darkHover: string;
  lightAccent: string;
  lightHover: string;
};

export const ACCENT_PRESETS: readonly AccentPreset[] = [
  {
    id: 'orange',
    label: 'Orange',
    darkAccent: '#ff8800',
    darkHover: '#e67a00',
    lightAccent: '#ea5800',
    lightHover: '#c2410c',
  },
  {
    id: 'cyan',
    label: 'Cyan',
    darkAccent: '#22d3ee',
    darkHover: '#06b6d4',
    lightAccent: '#0891b2',
    lightHover: '#0e7490',
  },
  {
    id: 'violet',
    label: 'Violet',
    darkAccent: '#a78bfa',
    darkHover: '#8b5cf6',
    lightAccent: '#7c3aed',
    lightHover: '#6d28d9',
  },
  {
    id: 'green',
    label: 'Green',
    darkAccent: '#34d399',
    darkHover: '#10b981',
    lightAccent: '#059669',
    lightHover: '#047857',
  },
  {
    id: 'rose',
    label: 'Rose',
    darkAccent: '#fb7185',
    darkHover: '#f43f5e',
    lightAccent: '#e11d48',
    lightHover: '#be123c',
  },
] as const;

export function getAccentPreset(id: string): AccentPreset | undefined {
  return ACCENT_PRESETS.find(p => p.id === id);
}

export function applyDomTheme(theme: ThemeMode): void {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
}

export function applyAccentCssVars(theme: ThemeMode, accentId: string): void {
  const preset = getAccentPreset(accentId) ?? getAccentPreset('orange')!;
  const light = theme === 'light';
  document.documentElement.style.setProperty(
    '--color-accent',
    light ? preset.lightAccent : preset.darkAccent,
  );
  document.documentElement.style.setProperty(
    '--color-accent-hover',
    light ? preset.lightHover : preset.darkHover,
  );
}

type ThemeContextValue = {
  theme: ThemeMode;
  setTheme: (t: ThemeMode) => void;
  toggleTheme: () => void;
  accentId: AccentPresetId;
  setAccentId: (id: AccentPresetId) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

function readStoredTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(STORAGE_THEME_KEY);
    if (v === 'light') return 'light';
  } catch {
    /* ignore */
  }
  return 'dark';
}

function readStoredAccentId(): AccentPresetId {
  try {
    const v = localStorage.getItem(STORAGE_ACCENT_KEY);
    if (v && getAccentPreset(v)) return v as AccentPresetId;
  } catch {
    /* ignore */
  }
  return 'orange';
}

export function ThemeProvider({children}: {children: ReactNode}) {
  const [theme, setThemeState] = useState<ThemeMode>(() => readStoredTheme());
  const [accentId, setAccentIdState] = useState<AccentPresetId>(() => readStoredAccentId());

  const apply = useCallback((t: ThemeMode, accent: AccentPresetId) => {
    applyDomTheme(t);
    applyAccentCssVars(t, accent);
  }, []);

  useEffect(() => {
    apply(theme, accentId);
  }, [theme, accentId, apply]);

  const setTheme = useCallback((t: ThemeMode) => {
    setThemeState(t);
    try {
      localStorage.setItem(STORAGE_THEME_KEY, t);
    } catch {
      /* ignore */
    }
  }, []);

  const setAccentId = useCallback((id: AccentPresetId) => {
    setAccentIdState(id);
    try {
      localStorage.setItem(STORAGE_ACCENT_KEY, id);
    } catch {
      /* ignore */
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  }, [theme, setTheme]);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      toggleTheme,
      accentId,
      setAccentId,
    }),
    [theme, setTheme, toggleTheme, accentId, setAccentId],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return ctx;
}
