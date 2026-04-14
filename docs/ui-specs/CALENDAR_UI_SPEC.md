# Monthly calendar — UI spec (Macro Counter family)

Use this document when implementing the **calendar header button** and **monthly calendar modal** in a **sister project** so styling matches **Macro Counter** and other toolbox apps (see [TOOLBOX_UI_SPEC.md](./TOOLBOX_UI_SPEC.md)).

---

## 1. Prerequisites (shared design tokens)

The UI assumes the same **CSS variables** and utilities as the toolbox shell (`src/index.css` / `@theme`):

- `--color-bg-dark`, `--color-card-dark`, `--color-surface`, `--color-surface-deep`
- `--color-accent`, `--color-accent-hover`
- `--color-text-light`, `--color-fg`
- **`brand-font`** on headings (Space Grotesk via `.brand-font`)

**Required global classes** (define the same in the sister project):

```css
.glass {
  background: color-mix(in srgb, var(--color-card-dark) 88%, transparent);
  -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px);
  border: 1px solid color-mix(in srgb, var(--color-accent) 12%, transparent);
}

.accent-glow {
  box-shadow: 0 0 30px color-mix(in srgb, var(--color-accent) 15%, transparent);
}
```

The app root should use **`blueprint-bg`** (or equivalent) so the modal backdrop feels consistent with the rest of the shell.

---

## 2. Calendar trigger button (header)

Placed in the **Daily Totals** card header, **immediately to the left of** the goals/target button. Same affordance pattern as the target icon: **no filled tile**, accent icon + hover surface only.

| Element | Value |
|--------|--------|
| **Icon** | Lucide **`CalendarDays`**, **`h-5 w-5`** (20px) |
| **Button classes** | `rounded-lg p-1.5 text-[var(--color-accent)] transition hover:bg-[var(--color-surface)]` |
| **Accessibility** | `aria-label="Open monthly calendar"` |
| **Behavior** | `onClick` opens the modal (e.g. `setCalendarOpen(true)`) |

**Goals / target button** (pairing): use the **identical** button classes; swap the icon to **`Target`** and set an appropriate `aria-label` (e.g. “Set macro goals”).

Visual: orange (accent) icon on dark chrome; hover adds a subtle **surface** fill—**not** a separate bordered pill around the icon.

---

## 3. Modal shell (backdrop + panel)

**Backdrop** — full screen, click-to-dismiss:

```tsx
className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
onClick={onClose}
```

**Panel** — stop propagation so inner clicks do not close:

```tsx
className="glass w-full max-w-md rounded-2xl border border-[var(--color-accent)]/10 p-5 shadow-lg accent-glow"
onClick={(e) => e.stopPropagation()}
```

| Detail | Value |
|--------|--------|
| **z-index** | `z-50` (same layer as other modals in the app) |
| **Max width** | `max-w-md` |
| **Backdrop padding** | `p-4` on the overlay for small-screen edge breathing room |

**Body scroll:** While the modal is open, set `document.body.style.overflow = 'hidden'` and restore it when closed (match other modals).

---

## 4. Modal header (title + close)

```tsx
<div className="mb-4 flex items-center justify-between">
  <h2 className="text-lg font-semibold text-fg brand-font">Monthly Calendar</h2>
  <button
    type="button"
    className="rounded-full p-1.5 text-[var(--color-text-light)] transition hover:bg-[var(--color-surface)] hover:text-fg"
    onClick={onClose}
    aria-label="Close calendar"
  >
    <X className="h-5 w-5" />
  </button>
</div>
```

- Close control: Lucide **`X`**, `h-5 w-5`.

---

## 5. Macro filter strip (segmented control)

Outer container:

```tsx
className="mb-4 flex gap-1 rounded-full bg-[var(--color-surface)] p-1"
```

Per option (example when **selected**):

- Selected: `text-white shadow-sm` plus **inline** `style={{ backgroundColor: <per-metric color> }}` (Macro Counter uses distinct colors per macro: calories / protein / carbs / fat).
- Unselected: `text-[var(--color-text-light)] hover:text-fg`
- Button: `flex-1 rounded-full py-1.5 text-xs font-medium capitalize transition`

Sister apps may change labels or colors but should keep this **pill-in-pill** structure for visual parity.

---

## 6. Month navigation row

```tsx
<div className="mb-3 flex items-center justify-between">
  <button
    type="button"
    className="rounded-lg p-1.5 text-[var(--color-text-light)] transition hover:bg-[var(--color-surface)] hover:text-fg"
    aria-label="Previous month"
  >
    <ChevronLeft className="h-5 w-5" />
  </button>
  <span className="text-sm font-semibold text-fg">{/* e.g. January 2026 */}</span>
  <button
    type="button"
    className="rounded-lg p-1.5 text-[var(--color-text-light)] transition hover:bg-[var(--color-surface)] hover:text-fg"
    aria-label="Next month"
  >
    <ChevronRight className="h-5 w-5" />
  </button>
</div>
```

---

## 7. Weekday row + day grid

**Weekday labels** (7 columns):

```tsx
<div className="grid grid-cols-7 gap-px text-center text-xs">
  {/* each label */}
  className="py-1.5 font-semibold text-[var(--color-text-light)]"
```

**Day cells** (after leading empty cells for month start):

```tsx
className="flex flex-col items-center justify-center rounded-lg py-1.5 transition"
```

**“Today”** double ring (uses page bg + metric accent):

```tsx
style={{
  boxShadow: `0 0 0 2px var(--color-bg-dark), 0 0 0 4px ${macroColor}`,
  borderRadius: '0.5rem',
}}
```

**Day number:** `text-xs tabular-nums`; today: `font-bold text-fg`; other days: `text-[var(--color-text-light)]`.

**Status row under the number:** Lucide **`ChevronUp`** / **`ChevronDown`**, `h-4 w-4`, **`strokeWidth={3}`**. “Above goal” uses the **metric color**; “below” uses `text-[var(--color-text-light)]`. If no status, render an empty **`h-4 w-4`** spacer so rows stay aligned.

---

## 8. Legend (bottom of modal)

```tsx
<div className="mt-4 flex items-center justify-center gap-4 text-xs text-[var(--color-text-light)]">
  {/* Above: ChevronUp h-3.5 w-3.5, strokeWidth 3, metric color */}
  {/* Below: ChevronDown h-3.5 w-3.5, muted */}
</div>
```

---

## 9. Lucide notes

- Header row icons (`CalendarDays`, `Target`): default Lucide stroke.
- Small chevrons in the grid and legend: **`strokeWidth={3}`** so they stay legible at small sizes.

---

## 10. Quick parity checklist

| Item | Match |
|------|--------|
| `glass` + `accent-glow` | Same CSS as toolbox spec |
| Calendar button | `rounded-lg p-1.5` + `text-[var(--color-accent)]` + `hover:bg-[var(--color-surface)]` |
| Modal | `z-50`, `bg-black/50`, flex center, `glass` panel `max-w-md rounded-2xl p-5` |
| Title | `brand-font`, `text-fg` |
| Segmented filter | `rounded-full bg-[var(--color-surface)] p-1` |
| Month arrows | Same hover pattern as calendar button (rounded-lg icon buttons) |

---

## 11. Reference implementation

See **`MacroCounter-AI`** `src/App.tsx`:

- `MacroCalendar` component (modal markup and classes)
- Header: calendar + target buttons next to the D/W/M pill
- State: `calendarOpen` + `setCalendarOpen`; include `calendarOpen` in the same **body scroll lock** `useEffect` as other modals.
