# Changelog

## [0.1.0] - 2026-11-16

- Initial public release.
- `/analytic on|off|status|steps|find` command and `log_step` tool.
- Per-session SQLite memory (WAL, `node:sqlite`) with lazy init.
- `nomic-embed-text-v1.5` (q8/CPU) embeddings + cosine duplicate detection.
- Anti-loop context digest and mid-turn watchdog steer.
- `<analytical_protocol>` injected into the system prompt while the mode is on.
- Portable paths: `getAgentDir()`, cache reuse, env overrides.
