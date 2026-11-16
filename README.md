# pi-analytical-memory

Pi extension that keeps a long investigation from going in circles: it injects a
short `<think>` discipline, logs every hypothesis to a **per-session SQLite**
database with an embedding, and steers the model away mid-turn when it starts
repeating tool calls.

## Install

```bash
pi install npm:pi-analytical-memory
# or a pinned git tag
pi install git:github.com/<you>/pi-analytical-memory@v0.1.0
```

Try it without touching your settings:

```bash
pi -e npm:pi-analytical-memory
```

## Usage

```text
/analytic on        # enable mode: system-prompt directive + warm the model
/analytic off
/analytic status    # mode, db path, counters, embedding state
/analytic steps     # last 12 logged hypotheses
/analytic find <text>
```

Once enabled, the `log_step` tool is exposed to the model. Each entry is hashed
into a 768-dim vector; a near-identical hypothesis shows up as
`⚠ hipótese quase idêntica já registrada`.

`log_step` arguments:

| field | required | meaning |
|---|---|---|
| `status` | yes | `ACTIVE`, `DEAD_END` or `CONFIRMED` |
| `hypothesis` | yes | one-sentence technical hypothesis |
| `layer` | no | target layer: binary, module, vtable, IOCTL, handler… |
| `evidence` | no | empirical proof: tool output, symbol, xref, offset |

## What it does

1. **`<think>` discipline** — when active, `before_agent_start` appends an
   `<analytical_protocol>` block: no guessing in the dark, absence of an expected
   symbol is evidence of indirection, trace backward through callers/lifecycle
   handlers, log every hypothesis.
2. **`log_step` + vector memory** — hypotheses are embedded
   (`nomic-embed-text-v1.5`, q8/CPU) and compared against the session's history.
3. **Context digest** — the last 8 steps plus recent dead ends are injected as a
   non-displayed custom message, so the model doesn't retry what already failed.
4. **Mid-turn watchdog** — three identical tool calls (or four consecutive
   errors) trigger a `steer` message that tells the model to stop and log the
   hypothesis as `DEAD_END` before changing approach (max 3 nudges per turn).

Everything is per session and lazy: no file, no model in RAM until `/analytic on`
or the first `log_step`.

## Storage

`<agent dir>/memory/<session id>.sqlite` (WAL), table:

```sql
CREATE TABLE steps (
  id INTEGER PRIMARY KEY, status TEXT, layer TEXT,
  hypothesis TEXT, evidence TEXT, vec BLOB, ts INTEGER
);
```

Default agent dir is `~/.pi/agent`. Deleting the database only loses the memory,
never the transcript.

## Requirements

- Pi with `pi.extensions` package support.
- Node **>= 22.13** (`node:sqlite` without a flag). If it is unavailable the
  extension degrades gracefully: `log_step` answers `stored: false` and the turn
  continues.
- ~560 MB of npm dependencies (`onnxruntime-node` native binaries) on install,
  plus ~132 MB of model weights on first activation, downloaded from
  Hugging Face (`nomic-ai/nomic-embed-text-v1.5`, `model_quantized.onnx`) into
  the cache. CPU only; no GPU needed.

## Environment overrides

| variable | default |
|---|---|
| `PI_CODING_AGENT_DIR` | read from Pi; sets where `<agent dir>` lives |
| `PI_ANALYTICAL_MEMORY_DIR` | `<agent dir>/memory` |
| `PI_ANALYTICAL_MEMORY_CACHE` | `~/.pi/models-cache` if it already has `nomic-ai/`, else `~/.cache/pi-analytical-memory` |
| `PI_ANALYTICAL_MEMORY_TRANSFORMERS_ROOT` | `~/.pi/tools/rag-book` (dev fallback only) |

## Troubleshooting

- `modo analítico ON, mas SQLite falhou` — your Node build lacks `node:sqlite`;
  upgrade Node.
- `embedding falhou: …` — model download blocked or offline. Steps are still
  stored, just without vectors (`vec NULL`), so dedupe/`find` skip them.
- `Tool "log_step" conflicts` — the extension is loaded twice (e.g. a local file
  plus a package). Keep one, or start with `pi -ne -e <package>`.

## Development

```bash
npm install
npm test
```

The smoke test drives the real module with a mocked `ExtensionAPI`: lazy init,
SQLite schema/WAL, 768-float blobs, duplicate detection, context digest,
directive injection and the watchdog.

## License

MIT
