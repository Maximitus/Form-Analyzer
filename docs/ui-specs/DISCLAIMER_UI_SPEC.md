# Disclaimer placement — UI spec (Macro Counter)

Use this document to keep legal/disclaimer UX consistent while avoiding bottom-chrome overlap on mobile devices.

---

## 1. Product decision

- Do **not** render a persistent legal/disclaimer footer on the primary capture screen.
- The capture screen already has a fixed bottom action bar ("chin"), so persistent footer copy is visually hidden or crowded.
- Keep legal coverage via:
  - **Mandatory first-use gate** (`DisclaimerGate`) before app usage.
  - **Full legal page** (`/terms`) containing terms + full disclaimer.
  - **In-app legal navigation** from Settings.

---

## 2. Required behavior

### 2.1 First-use gate (blocking modal)

- App must show the disclaimer gate until user explicitly accepts.
- Acceptance is persisted in `localStorage` (versioned key) so users are reprompted on material disclaimer updates.
- Gate modal should remain vertically centered and include safe-area-aware overlay padding.

Reference: `src/Disclaimer.tsx` (`DisclaimerGate`, `DisclaimerModal`, `DISCLAIMER_VERSION`).

### 2.2 Main app shell

- Do not mount `DisclaimerFooter` globally in `main.tsx`.
- Routes should render app content directly:
  - `/` -> `App`
  - `/terms` -> `TermsPage`

Reference: `src/main.tsx`.

### 2.3 Legal discoverability

- Settings modal must include a clear legal entry point linking to `/terms`.
- Place the legal link in the **settings modal header**, right-aligned opposite the "Settings" title.
- Style as **plain text link** (not a boxed/list-row button in the modal body).
- Recommended short copy: **"Legal"**.

Reference: `src/SettingsMenu.tsx`.

---

## 3. Placement rules

1. **Never attach persistent legal text to fixed bottom bars** used for core actions.
2. **Avoid duplicated legal footers** on screens with dense controls or gesture-nav overlap risk.
3. **Prefer explicit legal navigation** (Settings -> Terms) over always-on tiny footer text.
4. **Keep legal text readable**: full content belongs on `/terms`, not compressed into action chrome.

---

## 4. Accessibility + mobile constraints

- All legal links must be keyboard-focusable and screen-reader clear.
- Dialog overlays should account for safe areas:
  - `env(safe-area-inset-top)`
  - `env(safe-area-inset-bottom)`
- Do not rely on footer text that can sit under gesture indicators/home bars.

---

## 5. Regression checklist

- [ ] App opens with disclaimer gate for users without acceptance.
- [ ] Accepting gate stores acknowledgement and unlocks app.
- [ ] Capture screen has no persistent disclaimer footer.
- [ ] Settings includes link to `/terms`.
- [ ] `/terms` shows full disclaimer content.
- [ ] Mobile viewport: no legal UI hidden behind bottom "chin".

---

## 6. Sister app migration guide

Use this when applying the same decision to another toolbox app.

### 6.1 What to remove

- Remove any always-visible disclaimer/footer text mounted in the global app shell.
- Remove legal copy from fixed bottom action bars used for capture, submit, scan, or other primary actions.
- Remove duplicate legal snippets if a full terms/disclaimer page already exists.

### 6.2 What to add

- Keep (or add) a blocking first-use disclaimer gate with versioned acknowledgement storage.
- Add one explicit legal navigation path in settings/profile/help:
  - Preferred pattern: **header text link** in the settings modal (`Settings` on left, `Legal` link on right).
- Ensure there is a dedicated legal route/page (example: `/terms`) containing full terms + disclaimer.

### 6.3 File-by-file patch pattern (React apps)

1. **Router shell (`main.tsx` or equivalent)**
   - Remove global `DisclaimerFooter` mounting.
   - Keep routes for home + terms pages.
2. **Settings modal/menu**
   - Add a right-aligned `Legal` text link in the modal header that routes to terms.
   - Close modal on link click.
3. **Disclaimer module**
   - Keep `DisclaimerGate` + versioned `localStorage` acknowledgement.
   - Keep full disclaimer body component reusable by terms page.
4. **Terms page**
   - Confirm full disclaimer appears in readable body copy, not just a one-line summary.

### 6.4 Acceptance key naming

- Keep app-specific acknowledgement keys (example: `myapp_disclaimer_ack`) to avoid cross-app false acceptance.
- Bump disclaimer version when terms materially change so returning users are reprompted.

---

## 7. QA for sister apps

- [ ] No legal text is visually attached to a fixed bottom control bar.
- [ ] First-use flow blocks app interaction until acceptance.
- [ ] Legal link is reachable in <=2 taps/clicks from primary screen.
- [ ] Terms page is reachable directly by URL and from in-app navigation.
- [ ] iOS/Android gesture-nav devices do not hide legal affordances.
