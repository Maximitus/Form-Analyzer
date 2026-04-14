/**
 * Header title + sister-app links — SISTER_APPS_UI_SPEC.md
 */
import {useEffect, useId, useRef, useState} from 'react';
import {ChevronDown} from 'lucide-react';

/** Canonical toolbox apps on maxmvs.com (exclude `currentSlug` in the menu). */
export const TOOLBOX_SISTER_APPS = [
  {slug: 'workout' as const, label: 'Workout', href: 'https://maxmvs.com/workout'},
  {slug: 'macrocounter' as const, label: 'Macro Counter', href: 'https://maxmvs.com/macrocounter'},
  {slug: 'formanalyzer' as const, label: 'Form Analyzer', href: 'https://maxmvs.com/formanalyzer'},
] as const;

type SisterSlug = (typeof TOOLBOX_SISTER_APPS)[number]['slug'];

type Props = {
  /** This app — omitted from the dropdown. */
  currentSlug: SisterSlug;
  /** Visible header title (matches product name). */
  title: string;
};

export default function SisterAppsMenu({currentSlug, title}: Props) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const baseId = useId();
  const triggerId = `sister-apps-trigger-${baseId}`;
  const menuId = `sister-apps-menu-${baseId}`;

  const items = TOOLBOX_SISTER_APPS.filter((a) => a.slug !== currentSlug);

  useEffect(() => {
    if (!open) return;
    const onDocPointerDown = (e: PointerEvent) => {
      const el = wrapRef.current;
      if (el && !el.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', onDocPointerDown);
    return () => document.removeEventListener('pointerdown', onDocPointerDown);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <div ref={wrapRef} className="relative min-w-0">
      <h1 className="min-w-0 text-2xl font-semibold leading-tight tracking-tight text-[var(--color-accent)] brand-font">
        <button
          type="button"
          id={triggerId}
          className="inline-flex max-w-full min-w-0 items-center gap-1 rounded-md text-left outline-none transition hover:opacity-90 focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/45 focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-chrome-bar)]"
          aria-expanded={open}
          aria-haspopup="menu"
          aria-controls={menuId}
          onClick={() => setOpen((v) => !v)}
        >
          <span className="min-w-0 truncate">{title}</span>
          <ChevronDown
            className={`h-6 w-6 shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
            aria-hidden
          />
        </button>
      </h1>

      {open ? (
        <div
          id={menuId}
          role="menu"
          aria-labelledby={triggerId}
          className="absolute left-0 top-full z-50 mt-1 min-w-[13rem] rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-chrome-bar)] py-1 shadow-lg ring-1 ring-black/15"
        >
          {items.map((app) => (
            <a
              key={app.slug}
              role="menuitem"
              href={app.href}
              target="_blank"
              rel="noopener noreferrer"
              className="block px-3 py-2.5 text-sm text-fg outline-none transition hover:bg-[var(--color-panel-hover)] focus-visible:bg-[var(--color-panel-hover)] focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-accent)]/35"
              onClick={() => setOpen(false)}
            >
              {app.label}
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}
