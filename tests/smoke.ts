/**
 * Smoke test do pacote: exercita o módulo real com uma ExtensionAPI mockada.
 *
 *   npm test
 *
 * Usa um diretório temporário para as bases (`PI_ANALYTICAL_MEMORY_DIR`) e
 * apaga tudo no fim. O embedding roda no modelo local (q8/CPU).
 */

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

const WORK = mkdtempSync(join(tmpdir(), "pam-smoke-"));
process.env.PI_ANALYTICAL_MEMORY_DIR = join(WORK, "memory");

const SID = `smoke-${Date.now()}`;
const DB = join(WORK, "memory", `${SID}.sqlite`);

const { default: analyticalMemory } = (await import("../extensions/analytical-memory.ts")) as {
	default: (pi: any) => void;
};

const handlers = new Map<string, ((...args: any[]) => any)[]>();
let toolDef: any;
let command: any;
let sentMessage: any;
const notes: string[] = [];

const pi: any = {
	registerTool: (definition: any) => {
		toolDef = definition;
	},
	registerCommand: (_name: string, definition: any) => {
		command = definition;
	},
	on: (event: string, handler: any) => {
		handlers.set(event, [...(handlers.get(event) ?? []), handler]);
	},
	getActiveTools: () => ["read"],
	setActiveTools: () => {},
	appendEntry: () => {},
	sendMessage: (message: any) => {
		sentMessage = message;
	},
};

const ctx: any = {
	sessionManager: { getSessionId: () => SID, getEntries: () => [] },
	ui: { setStatus() {}, notify: (text: string) => notes.push(text), select: async (_t: string, items: string[]) => items[0] },
	hasUI: false,
};

const checks: string[] = [];
function check(name: string, fn: () => void | Promise<void>) {
	return { name, fn };
}
function pass(name: string) {
	checks.push(name);
	console.log(`ok - ${name}`);
}

const tests = [
	check("registra log_step, /analytic e os 6 eventos", () => {
		analyticalMemory(pi);
		assert.equal(toolDef?.name, "log_step");
		assert.ok(command, "comando /analytic ausente");
		for (const event of ["before_agent_start", "context", "tool_result", "session_start", "agent_start", "session_shutdown"]) {
			assert.ok(handlers.has(event), `evento ${event} não registrado`);
		}
	}),
	check("/analytic status antes da ativação não cria arquivo (lazy)", async () => {
		await command.handler("status", ctx);
		assert.match(notes.at(-1)!, /ainda não criado/);
		assert.equal(existsSync(DB), false, "arquivo sqlite criado cedo demais");
	}),
	check("/analytic on cria a base e carrega o modelo", async () => {
		await command.handler("on", ctx);
		assert.ok(existsSync(DB), "base não criada");
		assert.match(notes.at(-1)!, /modo analítico ON/);
	}),
	check("log_step grava 768 floats e detecta hipótese repetida", async () => {
		const one = await toolDef.execute("c1", {
			status: "ACTIVE",
			hypothesis: "wrapper usa vtable para despachar IOCTL",
			layer: "binário",
			evidence: "xref em 0x401a20",
		}, undefined, undefined, ctx);
		assert.equal(one.details.embedding, true);

		const two = await toolDef.execute("c2", {
			status: "DEAD_END",
			hypothesis: "wrapper usa vtable para despachar IOCTL pro driver",
			layer: "binário",
			evidence: "sem xref",
		}, undefined, undefined, ctx);
		assert.equal(two.details.duplicates.length, 1);
		assert.ok(two.details.duplicates[0].cos >= 0.86, `cos inesperado: ${two.details.duplicates[0].cos}`);

		const db = new DatabaseSync(DB, { readOnly: true });
		const rows = db.prepare("SELECT status, length(vec) AS n FROM steps ORDER BY id").all() as any[];
		db.close();
		assert.deepEqual(rows.map((row) => row.n), [3072, 3072]);
	}),
	check("digest anti-loop entra no contexto e deduplica", async () => {
		const handler = handlers.get("context")![0];
		const result = await handler({ messages: [{ role: "user", content: "oi" }] }, ctx);
		const digest = result.messages.at(-1);
		assert.equal(digest.customType, "analytical-memory-digest");
		assert.match(digest.content, /DEAD_END 1/);

		const again = await handler({ messages: result.messages }, ctx);
		assert.equal(again.messages.filter((m: any) => m.customType === "analytical-memory-digest").length, 1);
	}),
	check("/analytic find ranqueia por similaridade", async () => {
		await command.handler("find vtable IOCTL", ctx);
		const [first] = (notes.at(-1) ?? "").split("\n");
		assert.match(first, /^#1 ACTIVE/);
		assert.match(first, /cos=0\.\d+/);
	}),
	check("diretriz XML anexada só com modo ativo", async () => {
		const handler = handlers.get("before_agent_start")![0];
		const result = await handler({ systemPrompt: "base" }, ctx);
		assert.match(result.systemPrompt, /<analytical_protocol>/);
		await command.handler("off", ctx);
		const off = await handler({ systemPrompt: "base" }, ctx);
		assert.equal(off, undefined);
	}),
	check("watchdog dispara steer após repetição", async () => {
		await command.handler("on", ctx);
		const handler = handlers.get("tool_result")![0];
		for (let i = 0; i < 3; i++) await handler({ toolName: "read", input: { path: "/x" }, isError: false }, ctx);
		assert.equal(sentMessage?.customType, "analytical-memory-watchdog");
	}),
	check("base em WAL e íntegra", async () => {
		const db = new DatabaseSync(DB, { readOnly: true });
		const integrity = db.prepare("PRAGMA integrity_check").get() as any;
		const journal = db.prepare("PRAGMA journal_mode").get() as any;
		const schema = db.prepare("SELECT sql FROM sqlite_master WHERE name = 'steps'").get() as any;
		db.close();
		assert.equal(integrity.integrity_check, "ok");
		assert.equal(journal.journal_mode, "wal");
		assert.match(schema.sql, /CREATE TABLE steps \(id INTEGER PRIMARY KEY/);
		assert.ok(readFileSync(DB).byteLength > 0);
	}),
];

let failed = 0;
try {
	for (const test of tests) {
		try {
			await test.fn();
			pass(test.name);
		} catch (error) {
			failed += 1;
			console.error(`FAIL - ${test.name}`);
			console.error(error instanceof Error ? error.message : error);
		}
	}
} finally {
	rmSync(WORK, { recursive: true, force: true });
}

console.log(`\n${checks.length}/${tests.length} checks ok`);
process.exit(failed === 0 ? 0 : 1);
