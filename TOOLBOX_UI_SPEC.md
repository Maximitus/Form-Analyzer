# Toolbox app shell — UI spec for new tools

Use this document when you start a **new project** (or a new route) so it matches **Form Analyzer** and **Macro Counter**: same header size, page colors, glass cards, and grid background.

---

## 1. Stack expectations

| Piece | Version / notes |
|--------|------------------|
| **Tailwind CSS** | v4 with `@import "tailwindcss"` in CSS and **`@tailwindcss/vite`** in Vite |
| **Fonts** | Google Fonts: **Inter** (body) + **Space Grotesk** (headings / `.brand-font`) |
| **Framework** | React is what the reference apps use; the **CSS + HTML structure** is portable to other stacks if you map `className` to your templating |

`vite.config.ts` must include the Tailwind Vite plugin (see Form Analyzer repo).

---

## 2. Single source of truth: `index.css`

Copy the block below into **`src/index.css`** (or global styles) **verbatim**. All toolbox tokens and utilities live here.

```css
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;600;700&display=swap");

@import "tailwindcss";

@theme {
  --font-sans: "Inter", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-heading: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;

  /* Blueprint / glass palette (accent rgb(255,136,0), base #2a3439) */
  --color-bg-dark: #2a3439;
  --color-card-dark: #323d44;
  --color-surface: #252d33;
  /* Deepest panels / insets (modals, scroll track) */
  --color-surface-deep: #1c2328;
  /* Top header bar + footer bar — same as card-dark for layered toolbox UI */
  --color-chrome-bar: var(--color-card-dark);
  --color-accent: #ff8800;
  --color-accent-hover: #e67a00;
  --color-text-light: #b3c0c9;
  --color-panel-hover: #353e43;
}

html {
  font-family: var(--font-sans);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  @apply bg-[var(--color-bg-dark)] text-white font-sans overflow-x-hidden;
}

h1,
h2,
h3,
.brand-font {
  font-family: var(--font-heading), ui-sans-serif, system-ui, sans-serif;
}

.glass {
  background: color-mix(in srgb, var(--color-card-dark) 88%, transparent);
  -webkit-backdrop-filter: blur(16px);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 136, 0, 0.1);
}

.orange-glow {
  box-shadow: 0 0 30px rgba(255, 136, 0, 0.15);
}

.blueprint-bg {
  background-color: var(--color-bg-dark);
  background-image: linear-gradient(rgba(255, 136, 0, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 136, 0, 0.05) 1px, transparent 1px);
  background-size: 40px 40px;
}

* {
  scrollbar-width: thin;
  scrollbar-color: var(--color-accent) var(--color-surface-deep);
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--color-surface-deep);
}

::-webkit-scrollbar-thumb {
  background: var(--color-accent);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-accent-hover);
}
```

Import this file once from your app entry (e.g. `import './index.css'` in `main.tsx`).

---

## 3. Color tokens — when to use what

| Token | Role |
|--------|------|
| **`--color-bg-dark`** | Full page background + **inner wells** inside cards (nested panels, empty states) |
| **`--color-chrome-bar`** | **`<header>`** and optional **footer** strip (same hex family as card-dark) |
| **`--color-card-dark`** | Used inside `.glass` mix; don’t hard-code `#323d44` in new code — use tokens |
| **`--color-surface-deep`** | Scrollbar track, deepest modal / inset areas |
| **`--color-accent`** / **`--color-accent-hover`** | Primary actions, links, focus rings |
| **`--color-text-light`** | Secondary labels, helper text |
| **`--color-panel-hover`** | Hover state for controls sitting on `bg-dark` |

**Borders:** prefer `border-[#ff8800]/10` on outer glass, `border-[#ff8800]/20` on interactive chips/buttons (matches reference apps).

**Canvas / SVG accents:** `#ff8800` is the same as `--color-accent`.

---

## 4. Page structure (required DOM shape)

This is the **canonical shell**. Numbers map to the checklist in §7.

```
┌─────────────────────────────────────────────┐
│ ① Root: min-h-screen + bg-dark + blueprint-bg │
│ ┌─────────────────────────────────────────┐ │
│ │ ② Header: chrome-bar, px/py, mb-5, shadow │ │
│ │    h1: accent, brand-font, text-2xl …    │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │ ③ Main: grid gap-6, px/pt/pb (see below) │ │
│ │ ┌─────────────────────────────────────┐ │ │
│ │ │ ④ Section.card (glass + orange-glow) │ │ │
│ │ │   row: title mb-6 + optional actions │ │ │
│ │ │   …your tool content…                │ │ │
│ │ └─────────────────────────────────────┘ │ │
│ │   (repeat ④ for more cards)              │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### ① Root wrapper

```tsx
<div className="min-h-screen bg-[var(--color-bg-dark)] text-white font-sans blueprint-bg">
```

### ② App header (tool title)

Must match **horizontal padding** of the main column so the title lines up with cards.

```tsx
<header className="bg-[var(--color-chrome-bar)] px-4 py-4 md:px-8 mb-5 shadow-md border-b border-[var(--color-accent)]/20">
  <h1 className="text-2xl font-semibold tracking-tight text-[var(--color-accent)] brand-font leading-tight">
    Your Tool Name
  </h1>
</header>
```

- **`py-4`**: vertical size of the bar (not `py-2` — too shallow).
- **`mb-5`**: space before main content (do **not** use `mb-8` unless you intentionally want a looser toolbox).
- **`md:px-8`**: must match main content `md:px-8`.

### ③ Main content column

```tsx
<div className="grid gap-6 px-4 pt-3 pb-12 md:px-8 md:pt-5">
  {/* sections */}
</div>
```

- **`gap-6`**: vertical rhythm between stacked cards.
- **`pt-3` / `md:pt-5`**: tight band under header (paired with `mb-5` on header).

### ④ Primary card (one feature block)

```tsx
<section className="glass p-6 rounded-2xl border border-[#ff8800]/10 shadow-lg orange-glow">
  <div className="flex justify-between items-center mb-6">
    <h2 className="text-xl font-semibold text-white brand-font">Panel title</h2>
    {/* optional right-side actions */}
  </div>
  {/* body: use bg-[var(--color-bg-dark)] for nested wells */}
</section>
```

- Use **`<section>`** for each major card (semantics + consistency).
- **Section title row:** `mb-6` below the title row before the first body control.

### Nested panels inside a card (wells)

Same pattern as Macro “Daily Totals” inner tiles:

```tsx
<div className="rounded-xl border border-[#ff8800]/10 bg-[var(--color-bg-dark)] p-4 sm:p-5">
```

### Icon / circular controls (Form Analyzer pattern)

```tsx
<label className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20">
```

---

## 5. Typography rules

| Element | Classes |
|---------|---------|
| App title (`h1`) | `text-2xl font-semibold tracking-tight leading-tight text-[var(--color-accent)] brand-font` |
| Card title (`h2`) | `text-xl font-semibold text-white brand-font` |
| Subheading (`h3`) | `text-base font-semibold text-white brand-font` (if needed) |
| Body on dark | default `text-white`; muted → `text-[var(--color-text-light)]` |

---

## 6. Optional: secondary header (e.g. terms / back link)

Same chrome bar; often **`mb-0`** when the page is full-height with its own `main`:

```tsx
<header className="flex flex-wrap items-center gap-4 bg-[var(--color-chrome-bar)] px-4 py-4 md:px-8 mb-0 shadow-md border-b border-[var(--color-accent)]/20">
  {/* back link + h1 */}
</header>
```

Footer strip (disclaimer, etc.):

```tsx
<footer className="border-t border-[var(--color-accent)]/20 bg-[var(--color-chrome-bar)] px-6 py-4">
```

---

## 7. New project checklist

1. [ ] Add **Tailwind v4** + **`@tailwindcss/vite`**, wire plugin in `vite.config.ts`.
2. [ ] Paste **§2** into global CSS and import from entry.
3. [ ] Root div: **`min-h-screen bg-[var(--color-bg-dark)] text-white font-sans blueprint-bg`**.
4. [ ] Header: **`bg-[var(--color-chrome-bar)] px-4 py-4 md:px-8 mb-5 shadow-md border-b border-[var(--color-accent)]/20`** + **`h1`** classes from **§5**.
5. [ ] Main: **`grid gap-6 px-4 pt-3 pb-12 md:px-8 md:pt-5`**.
6. [ ] Each tool panel: **`glass p-6 rounded-2xl border border-[#ff8800]/10 shadow-lg orange-glow`** with title row **`mb-6`**.
7. [ ] Nested content blocks: **`bg-[var(--color-bg-dark)]`** + **`border-[#ff8800]/10`** (or `/20` for controls).
8. [ ] Do **not** invent new greys — use **§3** tokens only.
9. [ ] `index.html`: viewport meta + `lang="en"` (match reference `index.html`).

---

## 8. Reference implementation in this repo

| File | What to mirror |
|------|----------------|
| `src/index.css` | Canonical tokens + `.glass` / `.blueprint-bg` |
| `src/App.tsx` | Shell + first card structure |
| `desired_format/src/App.tsx` | Multi-card layout, inner wells, modals |
| `desired_format/src/TermsPage.tsx` | Secondary header + glass sections |
| `desired_format/src/Disclaimer.tsx` | Footer chrome + modal framing |

When in doubt, **diff your shell classes** against `src/App.tsx` lines **187–195** (root + header + main opener).

---

## 9. Version

Spec aligned with **Form Analyzer** repo layout as of the commit that introduced **`--color-chrome-bar`**, **`color-mix` glass**, and header spacing **`mb-5` / `pt-3` / `md:pt-5`**. If you change the reference apps, update **§2** and **§4** here to stay the single spec.
