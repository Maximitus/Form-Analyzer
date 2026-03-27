# Toolbox app shell — UI spec (“same belt”)

Use this document when you start or refactor a **tool** so it matches **Form Analyzer**, **Macro Counter**, and siblings: same header chrome, blueprint grid, glass cards, **theme** (dark/light), **accent palette**, and **settings** affordance. The goal is **one visual language**—tools from the **same belt**—not one-off styling per app.

---

## 1. Stack expectations

| Piece | Version / notes |
|--------|------------------|
| **Tailwind CSS** | v4 with `@import "tailwindcss"` in CSS and **`@tailwindcss/vite`** in Vite |
| **Fonts** | Google Fonts: **Inter** (body) + **Space Grotesk** (headings / `.brand-font`) |
| **Framework** | React in reference apps; **CSS tokens + `data-theme` + CSS variables** are portable to other stacks |

`vite.config.ts` must include the Tailwind Vite plugin (see reference repos).

---

## 2. Design rules (belt identity)

1. **Never hard-code the accent hex** in component classes for borders or glows. Use **`var(--color-accent)`** so the user’s accent choice applies everywhere. Prefer Tailwind arbitrary forms such as `border-[var(--color-accent)]/10` and `border-[var(--color-accent)]/20`.
2. **Primary text** on surfaces uses **`text-fg`** (token `--color-fg`), not raw `text-white`, so **light mode** stays readable. Reserve **`text-white`** for **primary buttons** that sit on **`bg-[var(--color-accent)]`** (white on saturated fill).
3. **Glass cards** use class **`glass`**; outer glow uses **`accent-glow`** (not `orange-glow`). Both derive tint from **`--color-accent`** via `color-mix`.
4. **Theme** is `html[data-theme="light"]` vs default (dark). Light mode **redefines** the same token names under `html[data-theme="light"] { ... }`—do not fork separate stylesheets per theme in components.
5. **Accent** at runtime: `--color-accent` and `--color-accent-hover` are set on **`document.documentElement`** from a small preset table (see §6). Defaults in `@theme` match the **orange** preset until JS runs.
6. **Settings** live behind a **raised cog** in the header (§7)—not a floating theme toggle outside the chrome.

---

## 3. Single source of truth: `index.css`

Copy the **entire** global stylesheet from a reference app’s **`src/index.css`** when bootstrapping a new tool. Below is the **canonical structure** Macro Counter uses; keep it in sync when the reference changes.

**Must include:**

- `@theme { ... }` with **`--color-fg`** and all surface tokens; default accent = orange.
- **`html[data-theme="light"] { ... }`** overrides for surfaces + `--color-fg` + light accent fallbacks.
- **`html[data-theme="light"] .glass`** and **`html[data-theme="light"] .blueprint-bg`** overrides (grid + glass borders track accent).
- **`body`**: `bg-[var(--color-bg-dark)] text-[var(--color-fg)]`.
- **`.settings-cog-trigger` / `.settings-cog`** (and light variants) for the header gear (§7).
- **`.glass`**, **`.accent-glow`**, **`.blueprint-bg`** using **`color-mix(..., var(--color-accent), ...)`**—not fixed `#ff8800` / `rgba(255,136,0,...)`.
- Scrollbar rules using **`var(--color-accent)`** and **`var(--color-accent-hover)`**.

**Do not** paste a shortened “legacy” block that still uses `orange-glow`, `text-white` on `body`, or hard-coded orange RGBA in `.glass` / `.blueprint-bg`.

---

## 4. Color tokens — when to use what

| Token | Role |
|--------|------|
| **`--color-bg-dark`** | Full page background + **inner wells** inside cards (nested panels, empty states). In light theme this is still the variable name but holds the **light** page fill. |
| **`--color-chrome-bar`** | **`<header>`** and **footer** strip |
| **`--color-card-dark`** | Mixed into `.glass`; avoid hard-coding `#323d44` |
| **`--color-surface`**, **`--color-surface-deep`** | Inputs, secondary buttons, scroll track, deep insets |
| **`--color-accent`**, **`--color-accent-hover`** | Primary actions, links, focus rings, scrollbar thumb; **set at runtime** from accent preset + theme (§6) |
| **`--color-text-light`** | Secondary labels, helper text |
| **`--color-panel-hover`** | Hover for controls on surfaces |
| **`--color-fg`** | Primary text and icons on backgrounds (use **`text-fg`** in Tailwind where `@theme` exposes it) |

**Borders:** `border-[var(--color-accent)]/10` on outer glass and nested wells; `border-[var(--color-accent)]/20` on interactive chips and inputs.

**SVG / charts:** use `var(--color-accent)` (or the macro ring pattern in Macro Counter) so accents follow the preset.

---

## 5. Theme & accent (behavior)

### 5.1 DOM contract

- **Light mode:** `document.documentElement.setAttribute('data-theme', 'light')`. Omit or use dark as default for **dark** UI.
- **Accent:** `document.documentElement.style.setProperty('--color-accent', '<hex>')` and same for `--color-accent-hover`, chosen from the active preset and current theme (each preset defines **dark** and **light** hex pairs for contrast).

### 5.2 Presets (fixed set)

All toolbox apps should expose the **same** five presets with the **same ids** so UX and copy stay predictable:

| `id` | Label | Purpose |
|------|--------|---------|
| `orange` | Orange | Default belt look |
| `cyan` | Cyan | |
| `violet` | Violet | |
| `green` | Green | |
| `rose` | Rose | |

Each preset row in code holds **four** hex values: **`darkAccent`**, **`darkHover`**, **`lightAccent`**, **`lightHover`**. Light theme uses the latter two; dark uses the former two.

### 5.3 localStorage (cross-app consistency)

Reference implementation (Macro Counter) uses:

| Key | Values |
|-----|--------|
| `macrocounter-theme` | `light` \| `dark` |
| `macrocounter-accent` | preset `id` (e.g. `orange`) |

For a **shared belt** across multiple apps in the same product, **prefer one pair of keys** everywhere (e.g. `toolbox-theme` / `toolbox-accent`) so preferences survive as users move between tools. If you keep app-prefixed keys, document them per repo and accept that settings won’t transfer.

### 5.4 React shell (reference pattern)

- **`ThemeProvider`** wraps the router (or root). It:
  - Reads / writes theme + accent id from `localStorage`.
  - On change, calls **`applyDomTheme`** + **`applyAccentCssVars(theme, accentId)`** (set `data-theme` and `--color-accent*` on `document.documentElement`).
- **`useTheme()`** returns `{ theme, setTheme, toggleTheme, accentId, setAccentId }`.
- Export **`ACCENT_PRESETS`**, **`getAccentPreset`**, and the small **`apply*`** helpers from a single module (e.g. `theme.tsx`) so **Settings** and tests stay aligned.

### 5.5 First paint (no flash)

Add a **synchronous** inline script in **`index.html`** `<head>` **before** CSS/JS loads that:

1. Reads stored theme; if `light`, sets `data-theme="light"`.
2. Reads stored accent id; maps id → four hex values; sets `--color-accent` and `--color-accent-hover` for the current theme branch.

The hex map in the script **must** stay in lockstep with **`ACCENT_PRESETS`** in TypeScript. When you add a preset, update **both**.

---

## 6. Settings UI (raised cog + modal)

### 6.1 Header placement

- **Primary shell:** `header` is **`flex items-center justify-between gap-4`** with the **tool title** on the left and **`SettingsMenu`** on the right. Same horizontal padding as before (`px-4 md:px-8`, `py-4`, `mb-5`, chrome bar, bottom border).
- **Secondary pages** (e.g. terms): left group = back link + title; **right** = **`SettingsMenu`** (same justify-between row).

Do **not** use a fixed-position theme control outside the header; it breaks alignment with the belt.

### 6.2 Raised cog (markup + CSS)

- Trigger is a **round** hit target (`h-11 w-11`, `rounded-full`), **no** filled tile, **no** border box.
- Icon: Lucide **`Settings`**, class **`settings-cog`**, larger than default (`h-7 w-7` / `sm:h-8 sm:w-8`), **`strokeWidth` ~ 1.85**.
- Parent button class **`settings-cog-trigger`**; CSS in `index.css` applies:
  - Stacked **drop-shadows** (top highlight + bottom shadow) for a **dimensional** gear.
  - **Hover:** color → `var(--color-accent)`, slightly stronger shadow.
  - **Active:** slight press + flatter shadow.
  - **`html[data-theme="light"]`** overrides for shadows (lighter UI chrome).

Focus: **`focus-visible:ring-2`** on accent with **`ring-offset-[var(--color-bg-dark)]`** (offset follows page background in both themes).

### 6.3 Modal contents

- **Backdrop** `z-[70]` (below app disclaimer if present at `z-[100]`).
- Sections: **Theme** (Dark / Light, two large buttons; selected uses `bg-[var(--color-accent)] text-white`).
- **Accent color:** one control per preset (swatch + label); selected state uses **ring** on `var(--color-accent)`.
- **Done** closes; backdrop click + **Escape** close; lock **body** scroll while open.

Copy **`SettingsMenu.tsx`** (or equivalent) from the reference app and only change strings if needed.

---

## 7. Page structure (required DOM shape)

```
┌─────────────────────────────────────────────────────────────┐
│ ① Root: min-h-screen + bg-dark + text-fg + blueprint-bg      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ② Header: chrome-bar, flex justify-between, title | ⚙    │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ③ Main: grid gap-6, px/pt/pb                            │ │
│ │ ┌───────────────────────────────────────────────────┐ │ │
│ │ │ ④ Section: glass + accent-glow + border accent/10 │ │ │
│ │ └───────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### ① Root wrapper

```tsx
<div className="min-h-screen bg-[var(--color-bg-dark)] text-fg font-sans blueprint-bg">
```

### ② App header (tool title + settings)

```tsx
<header className="mb-5 flex items-center justify-between gap-4 border-b border-[var(--color-accent)]/20 bg-[var(--color-chrome-bar)] px-4 py-4 shadow-md md:px-8">
  <h1 className="min-w-0 text-2xl font-semibold leading-tight tracking-tight text-[var(--color-accent)] brand-font">
    Your Tool Name
  </h1>
  <SettingsMenu />
</header>
```

### ③ Main content column

```tsx
<main className="grid gap-6 px-4 pt-3 pb-12 md:px-8 md:pt-5">
  {/* sections */}
</main>
```

### ④ Primary card

```tsx
<section className="glass rounded-2xl border border-[var(--color-accent)]/10 p-6 shadow-lg accent-glow">
  <div className="mb-6 flex items-center justify-between">
    <h2 className="text-xl font-semibold text-fg brand-font">Panel title</h2>
  </div>
  {/* nested wells: bg-[var(--color-bg-dark)] border-[var(--color-accent)]/10 */}
</section>
```

### Nested wells

```tsx
<div className="rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-4 sm:p-5">
```

---

## 8. Typography

| Element | Classes |
|---------|---------|
| App title (`h1`) | `text-2xl font-semibold tracking-tight leading-tight text-[var(--color-accent)] brand-font` |
| Card title (`h2`) | `text-xl font-semibold text-fg brand-font` |
| Subheading (`h3`) | `text-base font-semibold text-fg brand-font` |
| Muted | `text-[var(--color-text-light)]` |
| Primary buttons on accent fill | `bg-[var(--color-accent)] text-white` (and `hover:bg-[var(--color-accent-hover)]`) |

---

## 9. Optional: secondary header (terms / back)

```tsx
<header className="mb-0 flex flex-wrap items-center justify-between gap-4 border-b border-[var(--color-accent)]/20 bg-[var(--color-chrome-bar)] px-4 py-4 shadow-md md:px-8">
  <div className="flex min-w-0 flex-wrap items-center gap-4">
    {/* back link + h1 */}
  </div>
  <SettingsMenu />
</header>
```

Footer:

```tsx
<footer className="border-t border-[var(--color-accent)]/20 bg-[var(--color-chrome-bar)] px-6 py-4">
```

---

## 10. New project checklist (belt-aligned)

1. [ ] Tailwind v4 + `@tailwindcss/vite`, plugin in `vite.config.ts`.
2. [ ] Copy **`src/index.css`** from reference (full file: theme, light overrides, `.settings-cog*`, `.glass`, `.accent-glow`, `.blueprint-bg`, scrollbars).
3. [ ] Copy **`theme.tsx`** (or merge `ThemeProvider`, `ACCENT_PRESETS`, `applyAccentCssVars`, `applyDomTheme`).
4. [ ] Copy **`SettingsMenu.tsx`**; wrap app with **`ThemeProvider`** in `main.tsx`.
5. [ ] Add **`index.html`** inline script for theme + accent (§5.5); keep hex map in sync with **`ACCENT_PRESETS`**.
6. [ ] Root: **`min-h-screen bg-[var(--color-bg-dark)] text-fg font-sans blueprint-bg`**.
7. [ ] Header: **flex justify-between** + **`SettingsMenu`** + classes from §7.
8. [ ] Cards: **`glass`**, **`accent-glow`**, **`border-[var(--color-accent)]/10`**; no **`orange-glow`** or **`#ff8800`** in new code.
9. [ ] Use **`text-fg`** for primary copy; **`text-white`** only on accent-filled buttons.
10. [ ] Prefer **`toolbox-theme` / `toolbox-accent`** (or document shared keys) if multiple apps should share one user preference.

---

## 11. Reference files (Macro Counter)

| File | What to mirror |
|------|----------------|
| `src/index.css` | Tokens, light theme, glass/accent-glow/blueprint, settings cog CSS |
| `src/theme.tsx` | `ACCENT_PRESETS`, storage keys, `ThemeProvider`, `applyAccentCssVars` |
| `src/SettingsMenu.tsx` | Raised cog + modal (theme + accent) |
| `src/main.tsx` | `ThemeProvider` wrapping router |
| `index.html` | Inline first-paint script |
| `src/App.tsx` | Shell: root, header with `SettingsMenu`, main grid, cards |
| `src/TermsPage.tsx` | Secondary header + `SettingsMenu` |
| `src/Disclaimer.tsx` | Modal framing uses `accent-glow`, `border-[var(--color-accent)]/...` |

When in doubt, **diff class names** against `App.tsx` shell and **`index.css`** tokens.

---

## 12. Version

Spec updated for **belt identity**: dynamic **accent** (`--color-accent`), **light theme** (`data-theme="light"`, `--color-fg`), **`accent-glow`** + **`color-mix`** glass/grid, **settings** (raised cog + modal), and **no hard-coded orange** in component borders. Aligns with Macro Counter as of the commit that introduced **`SettingsMenu`**, **`theme.tsx`**, and the expanded **`index.css`**.

When reference apps change, update **§3**, **§11**, and the **checklist** so this file stays the single spec.
