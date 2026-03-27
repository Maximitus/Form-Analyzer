/**
 * Raised settings cog + theme/accent modal — belt shell (TOOLBOX_UI_SPEC.md §6).
 */
import {useEffect, useState} from 'react';
import {Moon, Settings, Sun} from 'lucide-react';
import {ACCENT_PRESETS, type AccentPresetId, useTheme} from './theme';

export default function SettingsMenu() {
  const {theme, setTheme, accentId, setAccentId} = useTheme();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="settings-cog-trigger inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full outline-none transition active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/45 focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-bg-dark)]"
        aria-label="Open settings"
        title="Settings"
      >
        <Settings className="settings-cog h-7 w-7 sm:h-8 sm:w-8" strokeWidth={1.85} aria-hidden />
      </button>

      {open && (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 p-6 pt-[max(1rem,env(safe-area-inset-top))] pb-[max(1rem,env(safe-area-inset-bottom))]"
          role="dialog"
          aria-modal="true"
          aria-labelledby="settings-title"
          onClick={e => {
            if (e.target === e.currentTarget) setOpen(false);
          }}
        >
          <div
            className={`relative w-full max-w-md rounded-[1.25rem] border p-6 shadow-lg accent-glow ${
              theme === 'dark'
                ? 'border-[var(--color-accent)]/20 bg-[#2c3338]'
                : 'glass border-[var(--color-accent)]/15'
            }`}
            onClick={e => e.stopPropagation()}
          >
            <h2 id="settings-title" className="mb-6 text-lg font-semibold text-fg brand-font">
              Settings
            </h2>

            <div className="space-y-6">
              <div>
                <div className="flex rounded-full bg-[var(--color-surface)] p-1.5">
                  <button
                    type="button"
                    onClick={() => setTheme('dark')}
                    className={`flex flex-1 items-center justify-center gap-2 rounded-full py-2.5 text-sm font-medium transition ${
                      theme === 'dark'
                        ? 'bg-[var(--color-accent)] text-white shadow-sm'
                        : 'text-[var(--color-text-light)]'
                    }`}
                  >
                    <Moon className="h-4 w-4 shrink-0" aria-hidden />
                    Dark
                  </button>
                  <button
                    type="button"
                    onClick={() => setTheme('light')}
                    className={`flex flex-1 items-center justify-center gap-2 rounded-full py-2.5 text-sm font-medium transition ${
                      theme === 'light'
                        ? 'bg-[var(--color-accent)] text-white shadow-sm'
                        : 'text-[var(--color-text-light)]'
                    }`}
                  >
                    <Sun className="h-4 w-4 shrink-0" aria-hidden />
                    Light
                  </button>
                </div>
              </div>

              <div>
                <p className="mb-3 text-sm text-[#9ca3af]">Accent color</p>
                <div className="flex flex-wrap justify-center gap-2 sm:justify-between sm:gap-3">
                  {ACCENT_PRESETS.map(preset => {
                    const swatch = theme === 'light' ? preset.lightAccent : preset.darkAccent;
                    const selected = accentId === preset.id;
                    return (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => setAccentId(preset.id as AccentPresetId)}
                        className={`flex min-w-[4.25rem] flex-col items-center gap-2 rounded-xl px-2.5 py-2 transition ${
                          selected
                            ? 'border-2 border-[var(--color-accent)]'
                            : 'border-2 border-transparent hover:opacity-95'
                        }`}
                        title={preset.label}
                        aria-label={`Accent ${preset.label}`}
                        aria-pressed={selected}
                      >
                        <span
                          className="h-11 w-11 shrink-0 rounded-full shadow-inner"
                          style={{backgroundColor: swatch}}
                        />
                        <span className="text-center text-xs text-[#9ca3af]">{preset.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <button
              type="button"
              className={`mt-8 w-full rounded-full py-3.5 text-sm font-medium transition ${
                theme === 'dark'
                  ? 'bg-[var(--color-surface)] text-white hover:bg-[var(--color-panel-hover)]'
                  : 'bg-[var(--color-surface)] text-fg hover:bg-[var(--color-panel-hover)]'
              }`}
              onClick={() => setOpen(false)}
            >
              Done
            </button>
          </div>
        </div>
      )}
    </>
  );
}
