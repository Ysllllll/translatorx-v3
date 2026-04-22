# translatorx-v3 — admin-ui sample frontend

Minimal React + Vite + TypeScript frontend demonstrating how to talk to
the translatorx-v3 REST + SSE API. This is a **sample**, not a product
UI — it's intentionally plain (no Tailwind, no component library) so
the code that matters is the API integration in `src/api.ts`.

## What you get

- One-page admin UI with simple hash routing (no react-router)
- `src/api.ts` — typed client for every endpoint (`/api/courses`,
  `/api/usage`, `/api/admin/*`, SSE event stream)
- User pages: **Submit** (new task), **My tasks** (live SSE progress +
  cancel + result download), **Usage** (own ledger + admin summaries)
- Admin pages: **All tasks**, **Users**, **Engines**, **Workers**,
  **Workspace**, **Terms**, **Errors**, **Config** (redacted)
- API key stored in `localStorage` under `trx.apiKey`, attached as
  `X-API-Key` on every request

## Run it

### 1. Start the backend with CORS enabled

Either add to your `app.yaml`:

```yaml
service:
  host: 0.0.0.0
  port: 8080
  cors_origins: ["http://localhost:5173"]
  api_keys:
    adm-dev: { user_id: admin, tier: admin }
    usr-dev: { user_id: alice, tier: paid }
```

…or skip CORS entirely and use the Vite dev proxy (recommended for
local dev):

```yaml
service:
  host: 0.0.0.0
  port: 8080
  # no cors_origins needed — Vite proxies /api to this backend
  api_keys:
    adm-dev: { user_id: admin, tier: admin }
```

Then:

```bash
translatorx-serve app.yaml
```

### 2. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

By default Vite proxies `/api`, `/metrics`, `/health`, `/ready` to
`http://localhost:8080`. Override with:

```bash
VITE_API_TARGET=http://backend:8080 npm run dev
```

### 3. Log in

Open <http://localhost:5173>. Paste an API key (e.g. `adm-dev`) into
the sidebar and click **Save**. Navigate pages via the sidebar.

## Production build

```bash
npm run build
# outputs dist/ — serve from any static host
```

For production deployments **either** serve the static bundle from
behind the same reverse proxy as the API (no CORS needed) **or** set
`service.cors_origins: ["https://admin.example.com"]` on the backend.

## Architecture notes

- **Hash routing**: the URL is `http://host/#/tasks`, not
  `/tasks` — lets the UI be served from any static host without
  server-side rewrites.
- **SSE** (`EventSource`): browsers can't send custom headers with
  `EventSource`, so the SSE endpoint must accept the API key via
  cookie or bearer query param in a real deployment. In this sample
  the Vite proxy forwards cookies transparently in dev.
- **No state library**: every page is self-contained with
  `useState` / `useEffect`. Swap in TanStack Query / Zustand when the
  app grows.
- **No auth flow**: API key is typed in directly. For production, add
  a real login page that swaps email+password for a key via your own
  `/auth/login` endpoint.

## Extending

Each admin endpoint has a one-liner in `src/api.ts` and a thin page in
`src/pages/`. Copy an existing one — `AdminWorkers.tsx` is the
simplest template.
