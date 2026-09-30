/**
 * Reachability guard for the ext: call-time veto (issue #125).
 * Pi's Agent hook dispatches the child loader's bound tool_call handlers.
 * Exercise that real dispatch to ensure an out-of-scope call is blocked and
 * an in-scope call reaches the other extension handlers. Nested ctx.executeTool
 * dispatch is covered separately in pi-0991.test.ts.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runAgent } from "../../src/agent-runner.js";
import { registerAgents } from "../../src/agent-types.js";
import type { AgentConfig } from "../../src/types.js";
import { fauxModelBackend } from "../helpers/faux-model-backend.js";
import { registerFauxProvider } from "../helpers/pi-ai.js";

// Real pi-mono (loader + dynamic extension import + session construction).
vi.setConfig({ testTimeout: 30_000 });

/** Registers `alpha_read` / `alpha_write`; reused so no new fixture is needed. */
const ALPHA = resolve(fileURLToPath(new URL("../fixtures/ext-alpha.mjs", import.meta.url)));
/** Registers `beta_tool` — loaded but NOT selected by the `ext:` selector below. */
const BETA = resolve(fileURLToPath(new URL("../fixtures/ext-beta.mjs", import.meta.url)));

function makePi() {
  return { exec: async () => ({ code: 1, stdout: "", stderr: "" }) } as any;
}

describe("tool veto reachability against real pi-mono", () => {
  let cwd: string;
  let faux: ReturnType<typeof registerFauxProvider>;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "subagents-veto-"));
    faux = registerFauxProvider({
      provider: "faux",
      models: [{ id: "faux-1", contextWindow: 200_000 }],
    });
  });
  afterEach(() => {
    faux.unregister();
    rmSync(cwd, { recursive: true, force: true });
  });

  it("Pi dispatches the bound guard and blocks out-of-scope tools", async () => {
    registerAgents(
      new Map([
        [
          "veto",
          {
            name: "veto",
            description: "veto guard",
            builtinToolNames: ["read"],
            // Select alpha only — beta loads (its handlers run) but is muted.
            extensions: [ALPHA, BETA],
            extSelectors: ["ext:ext-alpha.mjs"],
            skills: false,
            systemPrompt: "You are veto.",
            promptMode: "replace",
            inheritContext: false,
            runInBackground: false,
            isolated: false,
            persistSession: false,
          } as AgentConfig,
        ],
      ]),
    );

    const model = faux.getModel();
    const modelRegistry: any = {
      runtime: fauxModelBackend(model).modelRuntime,
      find: () => model,
      getAll: () => [model],
      getAvailable: () => [model],
      hasConfiguredAuth: () => true,
      isUsingOAuth: () => false,
      getApiKeyAndHeaders: async () => ({ apiKey: "faux", headers: {} }),
      registerProvider: () => {},
      unregisterProvider: () => {},
    };
    const ctx: any = { cwd, getSystemPrompt: () => "PARENT", model, modelRegistry };

    let priorIsFunction: boolean | undefined;
    let session: any;
    try {
      await runAgent(ctx, "veto", "go", {
        pi: makePi(),
        model,
        onSessionCreated: (s: any) => {
          session = s;
          // Pi retains ownership of the hook and dispatches the bound guard.
          priorIsFunction = typeof s.agent?.beforeToolCall === "function";
        },
      });
    } catch (error) {
      if (!session) throw error;
      // A faux-model turn may not complete; the veto is fixed at construction.
    }

    expect(priorIsFunction).toBe(true);

    // Out of scope: beta loaded but the ext: flip did not select it.
    await expect(
      session.agent.beforeToolCall({ toolCall: { name: "beta_tool" }, args: {} }),
    ).resolves.toMatchObject({ block: true, reason: expect.any(String) });

    // In-scope tools pass Pi's bound event dispatch.
    await expect(
      session.agent.beforeToolCall({ toolCall: { name: "alpha_read" }, args: {} }),
    ).resolves.toSatisfy((r: any) => !r?.block);
  });
});
