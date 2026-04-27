# browser_upload

Browser-based scenario for the production `/api/ws/streams` WebSocket
endpoint. Drag-drop an SRT file, watch bilingual translations stream
back in real time.

## Run

```bash
python demos/service/browser_upload/server.py
# then open http://127.0.0.1:8765
```

The page is served from this directory's `static/` folder. Default
credentials:

| field    | value         |
|----------|---------------|
| API key  | `demo-key`    |
| tenant   | `demo-tenant` |
| host     | `127.0.0.1`   |
| port     | `8765`        |

Pass `--host 0.0.0.0 --port 8080` to listen elsewhere.

## What you'll see

1. The browser parses SRT → segment frames.
2. JS opens a WebSocket to `/api/ws/streams?access_token=demo-key`.
3. It sends `start` then one `segment` frame per cue.
4. The server answers `started` / `progress` / `final` frames.
5. The bilingual table fills in as translations stream back.
6. Click **Abort** for a graceful shutdown (server sends `closed`).

The wire schema matches `demos/streaming/ws_client.py` exactly — same
frame shapes, same ordering. Swap the mock engine in `server.py` for a
real `OpenAICompatEngine` to translate against a live backend.

## Files

- `server.py` — FastAPI app + mock LLM + static mount.
- `static/index.html` — the page.
- `static/app.js` — vanilla JS WebSocket client + SRT parser.
- `static/app.css` — minimal dark theme.
- `_bootstrap.py` — sys.path shim so the demo is runnable as a script.
