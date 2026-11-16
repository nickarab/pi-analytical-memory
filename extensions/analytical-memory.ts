/**
 * analytical-memory — extensão única do modo analítico.
 *
 * Une três coisas que compartilham o mesmo estado:
 *
 *  1. Diretriz XML de disciplina de <think>, anexada ao system prompt por
 *     `before_agent_start` quando o modo está ativo (`/analytic on`).
 *  2. Tool `log_step` — registra hipótese (ACTIVE / DEAD_END / CONFIRMED) e
 *     compara com a memória vetorial da sessão (anti-loop).
 *  3. Watchdog mid-turn em `tool_result` — detecta repetição e sequência de
 *     erros, e injeta steer para quebrar o loop antes de gastar mais turnos.
 *
 * Estado persistente: SQLite por sessão em `<agent dir>/memory/<session>.sqlite`
 * via `node:sqlite` (DatabaseSync) — criado só no primeiro `/analytic on` ou no
 * primeiro `log_step`. O embedding usa `@huggingface/transformers` +
 * `nomic-embed-text-v1.5` (q8/CPU). Cache do modelo: reusa `~/.pi/models-cache`
 * quando já existe, senão `~/.cache/pi-analytical-memory`.
 *
 * Overrides por ambiente:
 *   PI_ANALYTICAL_MEMORY_DIR                → diretório das bases por sessão
 *   PI_ANALYTICAL_MEMORY_CACHE              → cache do modelo ONNX
 *   PI_ANALYTICAL_MEMORY_TRANSFORMERS_ROOT  → instalação alternativa do transformers
 *
 * O modelo ONNX só entra na RAM na ativação do modo (ou se um `log_step`
 * chegar antes), nunca no load da extensão.
 */

import { existsSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// ────────────────────────────────────────────────────────────────────────────
// Constantes
// ────────────────────────────────────────────────────────────────────────────

const HOME = homedir();
const CONFIG_DIR_NAME = ".pi";

function expandHome(path: string): string {
	if (path === "~") return HOME;
	if (path.startsWith("~/")) return join(HOME, path.slice(2));
	return path;
}

/** Espelha `getAgentDir()` do Pi, sem importar o host no load da extensão. */
const AGENT_DIR = (() => {
	const envDir = process.env.PI_CODING_AGENT_DIR;
	return envDir ? expandHome(envDir) : join(HOME, CONFIG_DIR_NAME, "agent");
})();

/** Bases SQLite por sessão. */
const MEMORY_DIR = process.env.PI_ANALYTICAL_MEMORY_DIR || join(AGENT_DIR, "memory");

/** Cache do ONNX: reusa o cache já existente do rag-book/Pi quando presente. */
const LEGACY_CACHE = join(HOME, ".pi", "models-cache");
const MODEL_CACHE =
	process.env.PI_ANALYTICAL_MEMORY_CACHE ||
	(existsSync(join(LEGACY_CACHE, "nomic-ai")) ? LEGACY_CACHE : join(HOME, ".cache", "pi-analytical-memory"));

/** Instalação alternativa do transformers (dev: rag-book local). */
const RAG_ROOT = process.env.PI_ANALYTICAL_MEMORY_TRANSFORMERS_ROOT || join(HOME, ".pi", "tools", "rag-book");

const MODEL_ID = "nomic-ai/nomic-embed-text-v1.5";
const EMBED_DIM = 768;
const DOC_PREFIX = "search_document: ";
const QUERY_PREFIX = "search_query: ";

const MODE_ENTRY = "analytical-memory-mode";
const DIGEST_TYPE = "analytical-memory-digest";
const WATCHDOG_TYPE = "analytical-memory-watchdog";
const STATUS_KEY = "analytical";

/** Cosine mínimo para considerar duas hipóteses "a mesma tentativa". */
const DUP_COS = 0.86;
const TOPIC_COS = 0.72;

/** Watchdog: janela de assinaturas e gatilhos. */
const WINDOW = 12;
const WATCHDOG_REPEATS = 3;
const WATCHDOG_ERRORS = 4;
const WATCHDOG_MAX_NUDGES = 3;

const PROTOCOL = `<analytical_protocol>
- THINK_DISCIPLINE: Proibido ruminar ou adivinhar código no escuro. Use o <think> apenas para elencar 2-3 mecanismos técnicos, escolher a próxima verificação empírica e encerrar imediatamente para chamar a tool.
- ABSENCE_AS_EVIDENCE: A ausência de imports/símbolos esperados prova indireção (vtable, GetProcAddress, IOCTL ou camada abaixo), nunca um beco sem saída.
- BACKWARD_TRACING: Rastreie do efeito para os callers/xrefs ou handlers de ciclo de vida (init/open/close/destructor).
- STATE_LOGGING: Registre cada hipótese testada em \`log_step\` (ACTIVE, DEAD_END, CONFIRMED).
</analytical_protocol>`;

// ────────────────────────────────────────────────────────────────────────────
// SQLite lazy (node:sqlite)
// ────────────────────────────────────────────────────────────────────────────

interface Db {
	exec(sql: string): void;
	prepare(sql: string): {
		run(...params: unknown[]): { lastInsertRowid: number | bigint; changes: number | bigint };
		all(...params: unknown[]): unknown[];
		get(...params: unknown[]): unknown;
	};
	close(): void;
}

interface StepRow {
	id: number;
	status: string;
	layer: string | null;
	hypothesis: string;
	evidence: string | null;
	vec?: Uint8Array | null;
	ts: number;
}

let sqliteModule: Promise<Record<string, unknown>> | undefined;

async function openDbFile(file: string): Promise<Db> {
	sqliteModule ??= import("node:sqlite") as Promise<Record<string, unknown>>;
	const mod = await sqliteModule;
	const DatabaseSync = mod.DatabaseSync as (new (path: string) => Db) | undefined;
	if (!DatabaseSync) throw new Error("node:sqlite indisponível neste runtime");

	mkdirSync(MEMORY_DIR, { recursive: true });
	const db = new DatabaseSync(file);
	db.exec("PRAGMA journal_mode = WAL");
	db.exec("PRAGMA busy_timeout = 2000");
	db.exec(
		"CREATE TABLE IF NOT EXISTS steps (id INTEGER PRIMARY KEY, status TEXT, layer TEXT, hypothesis TEXT, evidence TEXT, vec BLOB, ts INTEGER)",
	);
	return db;
}

// ────────────────────────────────────────────────────────────────────────────
// Embeddings (pipeline local do rag-book, carregado sob demanda)
// ────────────────────────────────────────────────────────────────────────────

type Extractor = (texts: string[], options: { pooling: "mean"; normalize: boolean }) => Promise<{
	dims: number[];
	tolist(): number[][];
}>;

let extractorPromise: Promise<Extractor> | undefined;
let lastEmbedError: string | undefined;

function resolveTransformersEntry(): string {
	// 1) dependência do próprio pacote (instalação publicada)
	try {
		return createRequire(import.meta.url).resolve("@huggingface/transformers");
	} catch {
		/* sem node_modules local: tenta a instalação do rag-book */
	}

	// 2) fallback de dev: reaproveita o pipeline do rag-book, se existir
	const manifest = join(RAG_ROOT, "package.json");
	if (existsSync(manifest)) {
		try {
			return createRequire(manifest).resolve("@huggingface/transformers");
		} catch {
			/* segue para o erro explicativo */
		}
	}

	throw new Error(
		"@huggingface/transformers não encontrado — rode `npm install` no pacote ou aponte PI_ANALYTICAL_MEMORY_TRANSFORMERS_ROOT para uma instalação existente",
	);
}

async function loadTransformers(): Promise<Record<string, any>> {
	const entry = resolveTransformersEntry();
	const mod: any = await import(pathToFileURL(entry).href);
	const lib = mod?.env && mod?.pipeline ? mod : mod?.default;
	if (!lib?.pipeline) throw new Error(`pipeline ausente em ${entry}`);
	lib.env.cacheDir = MODEL_CACHE;
	lib.env.allowRemoteModels = true;
	lib.env.useBrowserCache = false;
	return lib;
}

function getExtractor(): Promise<Extractor> {
	extractorPromise ??= (async () => {
		const lib = await loadTransformers();
		return (await lib.pipeline("feature-extraction", MODEL_ID, {
			dtype: "q8",
			device: "cpu",
		})) as Extractor;
	})().catch((err: unknown) => {
		extractorPromise = undefined;
		throw err;
	});
	return extractorPromise;
}

/** Carrega o modelo na RAM. Chamado na ativação do modo. */
async function warmModel(): Promise<void> {
	try {
		await getExtractor();
		lastEmbedError = undefined;
	} catch (err) {
		lastEmbedError = err instanceof Error ? err.message : String(err);
		throw err;
	}
}

async function embed(text: string, prefix: string): Promise<Float32Array | null> {
	try {
		const extractor = await getExtractor();
		const tensor = await extractor([prefix + text], { pooling: "mean", normalize: true });
		const [rows, dim] = tensor.dims;
		if (rows !== 1 || dim !== EMBED_DIM) {
			throw new Error(`shape inesperado [${rows}, ${dim}], esperado [1, ${EMBED_DIM}]`);
		}
		lastEmbedError = undefined;
		return Float32Array.from(tensor.tolist()[0]);
	} catch (err) {
		lastEmbedError = err instanceof Error ? err.message : String(err);
		return null;
	}
}

function cosine(a: Float32Array, b: Float32Array): number {
	const n = Math.min(a.length, b.length);
	let dot = 0;
	for (let i = 0; i < n; i++) dot += a[i] * b[i];
	return dot; // vetores do nomic já saem normalizados
}

function toBlob(vec: Float32Array): Uint8Array {
	return new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength);
}

function fromBlob(blob: Uint8Array): Float32Array {
	const copy = new Uint8Array(blob.byteLength);
	copy.set(blob);
	return new Float32Array(copy.buffer, 0, copy.byteLength / 4);
}

// ────────────────────────────────────────────────────────────────────────────
// Estado por sessão
// ────────────────────────────────────────────────────────────────────────────

interface SessionState {
	sessionId: string;
	dbPath: string;
	active: boolean;
	db?: Db;
	dbError?: string;
	sigs: string[];
	errorStreak: number;
	nudges: number;
	lastNudgeSig?: string;
}

const sessions = new Map<string, SessionState>();
let fallbackSessionId = "unknown";

function stateFor(ctx: ExtensionContext): SessionState {
	let id = fallbackSessionId;
	try {
		id = ctx.sessionManager.getSessionId() || fallbackSessionId;
	} catch {
		/* sessão sem manager: usa o fallback */
	}
	fallbackSessionId = id;

	const file = id.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
	let state = sessions.get(file);
	if (!state) {
		state = {
			sessionId: id,
			dbPath: join(MEMORY_DIR, `${file}.sqlite`),
			active: false,
			sigs: [],
			errorStreak: 0,
			nudges: 0,
		};
		sessions.set(file, state);
	}
	return state;
}

/** Abre a base; `create: false` nunca cria arquivo (mantém o lazy init). */
async function ensureDb(state: SessionState, create: boolean): Promise<Db | undefined> {
	if (state.db) return state.db;
	if (!create && !existsSync(state.dbPath)) return undefined;
	try {
		state.db = await openDbFile(state.dbPath);
		state.dbError = undefined;
		return state.db;
	} catch (err) {
		state.dbError = err instanceof Error ? err.message : String(err);
		return undefined;
	}
}

function recentSteps(db: Db, limit: number): StepRow[] {
	return (db.prepare("SELECT id, status, layer, hypothesis, evidence, ts FROM steps ORDER BY id DESC LIMIT ?").all(limit) as StepRow[]).reverse();
}

function statusCounts(db: Db): Record<string, number> {
	const rows = db.prepare("SELECT status, COUNT(*) AS c FROM steps GROUP BY status").all() as Array<{ status: string; c: number }>;
	const counts: Record<string, number> = { ACTIVE: 0, DEAD_END: 0, CONFIRMED: 0 };
	for (const row of rows) counts[row.status] = Number(row.c);
	return counts;
}

interface Match extends StepRow {
	cos: number;
}

function similarSteps(db: Db, vec: Float32Array, limit = 512): Match[] {
	const rows = db
		.prepare("SELECT id, status, layer, hypothesis, evidence, vec, ts FROM steps WHERE vec IS NOT NULL ORDER BY id DESC LIMIT ?")
		.all(limit) as StepRow[];
	const out: Match[] = [];
	for (const row of rows) {
		if (!row.vec) continue;
		const stored = fromBlob(row.vec);
		if (stored.length !== EMBED_DIM) continue;
		out.push({ ...row, cos: cosine(vec, stored) });
	}
	return out.sort((a, b) => b.cos - a.cos);
}

function short(text: string, max = 150): string {
	const flat = text.replace(/\s+/g, " ").trim();
	return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

function stepLine(row: StepRow, cos?: number): string {
	const extra = cos === undefined ? "" : ` cos=${cos.toFixed(2)}`;
	const layer = row.layer ? ` [${row.layer}]` : "";
	return `#${row.id} ${row.status}${layer}${extra} — ${short(row.hypothesis)}`;
}

// ────────────────────────────────────────────────────────────────────────────
// Tool `log_step`
// ────────────────────────────────────────────────────────────────────────────

const LogStepParams = Type.Object({
	status: Type.Union([Type.Literal("ACTIVE"), Type.Literal("DEAD_END"), Type.Literal("CONFIRMED")], {
		description: "ACTIVE = em teste; DEAD_END = descartada com evidência; CONFIRMED = comprovada",
	}),
	hypothesis: Type.String({ description: "Hipótese técnica testada, em uma frase" }),
	layer: Type.Optional(Type.String({ description: "Camada/alvo: binário, módulo, vtable, IOCTL, handler…" })),
	evidence: Type.Optional(Type.String({ description: "Evidência empírica: saída de tool, símbolo, xref, offset" })),
});

// ────────────────────────────────────────────────────────────────────────────
// Textos injetados
// ────────────────────────────────────────────────────────────────────────────

function digest(steps: StepRow[], counts: Record<string, number>): string {
	const total = counts.ACTIVE + counts.DEAD_END + counts.CONFIRMED;
	const lines = [
		"<analytical_memory>",
		`steps: ${total} (ACTIVE ${counts.ACTIVE} | DEAD_END ${counts.DEAD_END} | CONFIRMED ${counts.CONFIRMED}) — últimas ${steps.length}:`,
		...steps.map((row) => stepLine(row)),
	];
	const deadEnds = steps.filter((row) => row.status === "DEAD_END").slice(-3);
	if (deadEnds.length > 0) {
		lines.push("Não repita estes becos sem saída; mude de camada ou rastreie backward:");
		lines.push(...deadEnds.map((row) => `- ${short(row.hypothesis, 120)}`));
	}
	if (counts.ACTIVE >= 3) {
		lines.push("Feche as hipóteses ACTIVE (DEAD_END/CONFIRMED) antes de abrir outra.");
	}
	lines.push("</analytical_memory>");
	return lines.join("\n");
}

function watchdogText(kind: "repeat" | "errors", detail: string, repeats: number): string {
	const head =
		kind === "repeat"
			? `Sinal de loop: a mesma chamada foi repetida ${repeats}× sem progresso (${short(detail, 100)}).`
			: `Sinal de loop: ${WATCHDOG_ERRORS} erros consecutivos (${short(detail, 100)}).`;
	return [
		`<analytical_watchdog signal="${kind}">`,
		head,
		"Pare agora. Em vez de repetir:",
		"1. Encerre o <think>: 2-3 mecanismos possíveis e uma única verificação empírica.",
		"2. A ausência do resultado esperado é evidência de indireção (vtable, GetProcAddress, IOCTL, camada abaixo) — não é beco sem saída.",
		"3. Registre a hipótese atual em log_step como DEAD_END antes de mudar de abordagem.",
		"4. Rastreie backward: callers/xrefs ou handlers init/open/close/destructor.",
		"</analytical_watchdog>",
	].join("\n");
}

// ────────────────────────────────────────────────────────────────────────────
// Extensão
// ────────────────────────────────────────────────────────────────────────────

export default function analyticalMemory(pi: ExtensionAPI) {
	const setToolActive = (active: boolean) => {
		try {
			const current = pi.getActiveTools();
			if (active && !current.includes("log_step")) pi.setActiveTools([...current, "log_step"]);
			if (!active && current.includes("log_step")) pi.setActiveTools(current.filter((name) => name !== "log_step"));
		} catch {
			/* loadout indisponível: a tool continua registrada */
		}
	};

	const setFooter = (ctx: ExtensionContext, active: boolean) => {
		try {
			ctx.ui.setStatus(STATUS_KEY, active ? "analytic:on" : undefined);
		} catch {
			/* modo sem UI */
		}
	};

	const setMode = (ctx: ExtensionContext, state: SessionState, active: boolean) => {
		state.active = active;
		setToolActive(active);
		setFooter(ctx, active);
		pi.appendEntry(MODE_ENTRY, { active });
	};

	// ── Tool: log_step ───────────────────────────────────────────────────────
	pi.registerTool({
		name: "log_step",
		label: "Log step",
		description:
			"Registra uma hipótese do modo analítico (ACTIVE, DEAD_END, CONFIRMED) e compara com a memória vetorial da sessão, avisando sobre hipóteses quase idênticas já testadas.",
		promptSnippet: "log_step: registra hipótese analítica e checa repetição contra a memória da sessão",
		promptGuidelines: [
			"Use log_step ao abrir uma hipótese, ao descartá-la (DEAD_END) e ao confirmá-la (CONFIRMED) — é o registro que impede repetir becos sem saída.",
		],
		defaultActive: false,
		parameters: LogStepParams,
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const state = stateFor(ctx);
			const db = await ensureDb(state, true);
			if (!db) {
				return {
					content: [
						{
							type: "text" as const,
							text: `log_step: memória indisponível (${state.dbError ?? "sqlite não abriu"}). Hipótese registrada apenas no transcript: [${params.status}] ${params.hypothesis}`,
						},
					],
					details: { stored: false, error: state.dbError },
				};
			}

			const text = [params.hypothesis, params.evidence].filter((part): part is string => Boolean(part)).join("\n");
			const vec = await embed(text, DOC_PREFIX);

			const dupes = vec
				? similarSteps(db, vec)
						.filter((match) => match.cos >= TOPIC_COS)
						.slice(0, 3)
				: [];

			const info = db
				.prepare("INSERT INTO steps (status, layer, hypothesis, evidence, vec, ts) VALUES (?, ?, ?, ?, ?, ?)")
				.run(
					params.status,
					params.layer ?? null,
					params.hypothesis,
					params.evidence ?? null,
					vec ? toBlob(vec) : null,
					Date.now(),
				);
			const id = Number(info.lastInsertRowid);
			const counts = statusCounts(db);

			const lines = [
				`#${id} ${params.status}${params.layer ? ` [${params.layer}]` : ""} registrado.`,
				`memória: ${counts.ACTIVE} ACTIVE | ${counts.DEAD_END} DEAD_END | ${counts.CONFIRMED} CONFIRMED`,
			];
			const repeats = dupes.filter((match) => match.cos >= DUP_COS);
			if (repeats.length > 0) {
				lines.push("⚠ hipótese quase idêntica já registrada — não repita a mesma tentativa:");
				for (const match of repeats) lines.push(`- ${stepLine(match, match.cos)}`);
			} else if (dupes.length > 0 && params.status === "ACTIVE") {
				lines.push("relacionadas:");
				for (const match of dupes) lines.push(`- ${stepLine(match, match.cos)}`);
			}
			if (!vec) lines.push(`embedding indisponível (${lastEmbedError ?? "modelo não carregado"}) — passo salvo sem vetor.`);
			if (!state.active) lines.push("modo analítico inativo: use /analytic on para carregar a diretriz.");

			const reply = lines.join("\n");
			return {
				content: [{ type: "text" as const, text: reply }],
				details: {
					id,
					status: params.status,
					embedding: Boolean(vec),
					duplicates: repeats.map((match) => ({ id: match.id, status: match.status, cos: Number(match.cos.toFixed(3)) })),
				},
			};
		},
	});

	// ── Comando: /analytic [on|off|status|steps|find] ────────────────────────
	pi.registerCommand("analytic", {
		description: "Modo analítico: diretriz de <think>, log_step e memória vetorial da sessão",
		getArgumentCompletions: (prefix) => {
			const items = ["on", "off", "status", "steps", "find"];
			const filtered = items.filter((item) => item.startsWith(prefix.trim()));
			return filtered.length > 0 ? filtered.map((value) => ({ value, label: value })) : null;
		},
		handler: async (args, ctx) => {
			const state = stateFor(ctx);
			const [verb = "status", ...rest] = args.trim().split(/\s+/);

			if (verb === "on") {
				const db = await ensureDb(state, true);
				setMode(ctx, state, true);
				if (!db) {
					ctx.ui.notify(`modo analítico ON, mas SQLite falhou: ${state.dbError ?? "desconhecido"}`, "warning");
					return;
				}
				try {
					await warmModel();
					const counts = statusCounts(db);
					ctx.ui.notify(
						`modo analítico ON — ${state.dbPath} (${counts.ACTIVE} ACTIVE | ${counts.DEAD_END} DEAD_END | ${counts.CONFIRMED} CONFIRMED), modelo pronto`,
						"info",
					);
				} catch (err) {
					ctx.ui.notify(
						`modo analítico ON — SQLite pronto; embedding falhou: ${err instanceof Error ? err.message : String(err)}`,
						"warning",
					);
				}
				return;
			}

			if (verb === "off") {
				setMode(ctx, state, false);
				ctx.ui.notify("modo analítico OFF", "info");
				return;
			}

			const db = await ensureDb(state, false);
			const counts = db ? statusCounts(db) : { ACTIVE: 0, DEAD_END: 0, CONFIRMED: 0 };

			if (verb === "status") {
				const total = counts.ACTIVE + counts.DEAD_END + counts.CONFIRMED;
				ctx.ui.notify(
					[
						`modo: ${state.active ? "ON" : "OFF"}`,
						`db: ${db ? state.dbPath : "(ainda não criado)"}`,
						`steps: ${total} (${counts.ACTIVE} ACTIVE | ${counts.DEAD_END} DEAD_END | ${counts.CONFIRMED} CONFIRMED)`,
						`embedding: ${lastEmbedError ? `erro (${lastEmbedError})` : extractorPromise ? "carregado" : "não carregado"}`,
					].join("\n"),
					"info",
				);
				return;
			}

			if (!db) {
				ctx.ui.notify("memória ainda não criada nesta sessão (use /analytic on)", "info");
				return;
			}

			if (verb === "steps") {
				const rows = recentSteps(db, 12);
				if (rows.length === 0) {
					ctx.ui.notify("nenhum step registrado", "info");
					return;
				}
				const lines = rows.map((row) => stepLine(row));
				if (ctx.hasUI) await ctx.ui.select("Memória analítica (últimos 12)", lines);
				else ctx.ui.notify(lines.join("\n"), "info");
				return;
			}

			if (verb === "find") {
				const query = rest.join(" ").trim();
				if (!query) {
					ctx.ui.notify("uso: /analytic find <texto>", "warning");
					return;
				}
				const vec = await embed(query, QUERY_PREFIX);
				if (!vec) {
					ctx.ui.notify(`embedding falhou: ${lastEmbedError ?? "desconhecido"}`, "error");
					return;
				}
				const matches = similarSteps(db, vec).slice(0, 5);
				if (matches.length === 0) {
					ctx.ui.notify("nenhum step com vetor", "info");
					return;
				}
				const lines = matches.map((match) => stepLine(match, match.cos));
				if (ctx.hasUI) await ctx.ui.select(`Memória: "${short(query, 40)}"`, lines);
				else ctx.ui.notify(lines.join("\n"), "info");
				return;
			}

			ctx.ui.notify("uso: /analytic [on|off|status|steps|find <texto>]", "warning");
		},
	});

	// ── System prompt: diretriz quando ativo ─────────────────────────────────
	pi.on("before_agent_start", (event, ctx) => {
		const state = stateFor(ctx);
		if (!state.active) return;
		if (event.systemPrompt.includes("<analytical_protocol>")) return;
		return { systemPrompt: `${event.systemPrompt}\n\n${PROTOCOL}` };
	});

	// ── Memória vetorial no contexto (anti-loop) ─────────────────────────────
	pi.on("context", async (event, ctx) => {
		const state = stateFor(ctx);
		if (!state.active) return;
		const db = await ensureDb(state, false); // nunca cria arquivo aqui
		if (!db) return;
		let rows: StepRow[] = [];
		try {
			rows = recentSteps(db, 8);
		} catch {
			return;
		}
		if (rows.length === 0) return;

		const text = digest(rows, statusCounts(db));
		const messages = event.messages.filter(
			(message) => !(message.role === "custom" && (message as { customType?: string }).customType === DIGEST_TYPE),
		);
		messages.push({
			role: "custom",
			customType: DIGEST_TYPE,
			content: text,
			display: false,
			timestamp: Date.now(),
		} as (typeof messages)[number]);
		return { messages };
	});

	// ── Watchdog mid-turn ────────────────────────────────────────────────────
	pi.on("tool_result", (event, ctx) => {
		const state = stateFor(ctx);
		if (!state.active) return;

		const sig = `${event.toolName}:${short(JSON.stringify(event.input ?? {}), 240)}`;
		state.sigs.push(sig);
		if (state.sigs.length > WINDOW) state.sigs.shift();
		state.errorStreak = event.isError ? state.errorStreak + 1 : 0;

		const repeats = state.sigs.filter((item) => item === sig).length;
		const looping = repeats >= WATCHDOG_REPEATS;
		const failing = state.errorStreak >= WATCHDOG_ERRORS;
		if (!looping && !failing) return;
		if (state.nudges >= WATCHDOG_MAX_NUDGES) return;
		if (state.lastNudgeSig === sig && looping && repeats === WATCHDOG_REPEATS) return;

		state.nudges += 1;
		state.lastNudgeSig = sig;
		state.sigs = [];
		state.errorStreak = 0;

		pi.sendMessage(
			{
				customType: WATCHDOG_TYPE,
				content: watchdogText(looping ? "repeat" : "errors", event.toolName, repeats),
				display: true,
				details: { kind: looping ? "repeat" : "errors", tool: event.toolName, repeats },
			},
			{ deliverAs: "steer", triggerTurn: true },
		);
	});

	// ── Ciclo de vida ────────────────────────────────────────────────────────
	pi.on("session_start", (_event, ctx) => {
		const state = stateFor(ctx);
		try {
			for (const entry of ctx.sessionManager.getEntries()) {
				if (entry.type === "custom" && entry.customType === MODE_ENTRY) {
					state.active = Boolean((entry.data as { active?: boolean } | undefined)?.active);
				}
			}
		} catch {
			/* sem histórico legível */
		}
		setToolActive(state.active);
		setFooter(ctx, state.active);
	});

	pi.on("agent_start", (_event, ctx) => {
		const state = stateFor(ctx);
		state.sigs = [];
		state.errorStreak = 0;
		state.nudges = 0;
		state.lastNudgeSig = undefined;
	});

	pi.on("session_shutdown", () => {
		for (const state of sessions.values()) {
			try {
				state.db?.close();
			} catch {
				/* já fechado */
			}
			state.db = undefined;
		}
		sessions.clear();
	});
}
