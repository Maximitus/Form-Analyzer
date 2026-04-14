# Sister apps navigation — UI spec (maxmvs toolbox)

Use this document when adding or aligning **cross-links between toolbox apps** on **maxmvs.com** so each app exposes the same family navigation pattern and uses **canonical URLs** consistently.

---

## 1. Canonical URLs

| App | Path on maxmvs.com | Use when describing this app |
|-----|--------------------|------------------------------|
| **Workout** (this app) | **`/workout`** | **`maxmvs.com/workout`** (full URL: `https://maxmvs.com/workout`) |
| **Macro Counter** | `/macrocounter` | `maxmvs.com/macrocounter` |
| **Form Analyzer** | `/formanalyzer` | `maxmvs.com/formanalyzer` |

- Prefer **HTTPS** in implementation: `https://maxmvs.com/workout`, etc.
- In marketing or plain text, **maxmvs.com/workout** is the short form for this app.

---

## 2. Product behavior

1. **Primary app title** in the header (e.g. “Workout”) is the **anchor** for sister-app navigation—not a separate “Apps” icon.
2. Tapping the title opens a **dropdown menu** listing **other** toolbox apps (not the current one). Each row is a **link** that opens the target app in a **new tab** (`target="_blank"`, `rel="noopener noreferrer"`).
3. The menu **closes** when:
   - the user chooses a link,
   - presses **Escape**,
   - or **clicks outside** the dropdown (e.g. `pointerdown` on `document`, ignoring events inside the dropdown container).
4. Optional: disable conflicting gestures (e.g. horizontal tab swipe) while the menu is open.

---

## 3. Implementation notes (Workout reference)

Reference: `src/App.tsx` — header chrome when the session runner is not fullscreen (`!(screen === 'run' && active)`).

- Wrap the title + menu in a **relative** container with a **ref** for outside-click detection.
- Use a **heading** for accessibility: e.g. `<h1>` wrapping a **`<button type="button">`** that toggles the menu (`aria-expanded`, `aria-haspopup="menu"`, `aria-controls` pointing at the menu `id`).
- Menu panel: `role="menu"`, items `role="menuitem"`, **`aria-labelledby`** the trigger `id`.
- Show a **chevron** next to the title; rotate when open.
- Style the dropdown to match **TOOLBOX** chrome: bordered panel, surface hover, z-index above header/nav (e.g. `z-50`).

---

## 4. How sister apps should link back to Workout

When implementing the same pattern in **Macro Counter** or **Form Analyzer**:

- Include a menu row **“Workout”** (or the product name shown in the header) pointing to **`https://maxmvs.com/workout`**.
- Do **not** use a different path unless routing changes globally; **maxmvs.com/workout** is the canonical entry for this app.

---

## 5. Regression checklist

- [ ] Header title opens/closes the sister-apps menu.
- [ ] Menu lists Macro Counter and Form Analyzer with correct `https://maxmvs.com/...` URLs (and omits the current app when viewed from Workout).
- [ ] Links open in a new tab with `noopener` + `noreferrer`.
- [ ] Escape and outside click close the menu.
- [ ] Focus/screen reader: button has `aria-expanded`; menu has `role="menu"` and labelled trigger.

---

## 6. Related specs

- [TOOLBOX_UI_SPEC.md](./TOOLBOX_UI_SPEC.md) — shared header chrome, tokens, settings.
- [README.md](./README.md) — index of UI specs.
