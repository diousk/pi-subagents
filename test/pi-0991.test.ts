import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { fauxToolCall, type Model, type ThinkingLevel } from "@earendil-works/pi-ai";
import { createAgentSession, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { extensionCanonicalNames } from "../src/agent-runner.js";
import { fauxModelBackend } from "./helpers/faux-model-backend.js";
import { currentTools, getModel } from "./helpers/pi-ai.js";
import { agentCall, agentToolResults, runPrintMode } from "./helpers/print-mode-runner.js";

describe("synthetic extension identities", () => {
  it("matches built-in and named inline extensions by their logical names", () => {
    expect(extensionCanonicalNames("builtin:MCP")).toEqual(["mcp"]);
    expect(extensionCanonicalNames("builtin:codemode")).toEqual(["codemode"]);
    expect(extensionCanonicalNames("<inline:Scope>")).toEqual(["scope"]);
  });
});

describe("Pi 0.99.1 integration", () => {
  it("blocks an out-of-scope deferred tool reached through ctx.executeTool", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "pi-nested-scope-"));
    const agentDir = join(cwd, ".pi", "agents");
    mkdirSync(agentDir, { recursive: true });
    const fixture = fileURLToPath(new URL("./fixtures/nested-scope.mjs", import.meta.url));
    writeFileSync(join(agentDir, "scoped.md"), [
      "---",
      "tools: ext:nested-scope.mjs/bridge, ext:nested-scope.mjs/allowed",
      `extensions: ${JSON.stringify([fixture])}`,
      "skills: false",
      "persist_session: false",
      "---",
      "Call bridge once.",
    ].join("\n"));
    let run: Awaited<ReturnType<typeof runPrintMode>> | undefined;
    let childTools: string[] = [];
    try {
      run = await runPrintMode({
        cwd,
        prompt: "root",
        live: false,
        respond: (context) => {
          const user = context.messages.find((message) => message.role === "user");
          const isChild = JSON.stringify(user).includes("scoped-child");
          const toolResult = context.messages.find((message) => message.role === "toolResult");
          if (isChild) {
            childTools = currentTools(context).map((tool) => tool.name);
            return toolResult ? JSON.stringify(toolResult) : fauxToolCall("bridge", {});
          }
          return toolResult ? "done" : agentCall({
            subagent_type: "scoped", description: "scope probe", prompt: "scoped-child", run_in_background: false,
          });
        },
      });
      const result = agentToolResults(run.parentSession).join("\n");
      expect(result).toContain("allowed EXECUTED");
      expect(result).toContain('Tool \\"denied\\" is not available to this subagent.');
      expect(result).not.toContain("denied EXECUTED");
      expect(childTools).toContain("bridge");
      expect(childTools).not.toContain("allowed");
      expect(childTools).not.toContain("denied");
      const message = run.parentSession.messages.find((entry) => entry.role === "toolResult");
      const details = (message as { details?: { toolUses?: number } }).details;
      expect(details?.toolUses).toBe(3); // bridge plus both nested attempts
    } finally {
      await run?.dispose();
      rmSync(cwd, { recursive: true, force: true });
    }
  }, 30_000);

  it.each(["off", "minimal", "low", "medium", "high", "xhigh", "max"] as ThinkingLevel[])(
    "resolves Sol reasoning request %s using Pi's model capabilities",
    async (requested) => {
      const model = (getModel as (provider: string, id: string) => Model<string> | undefined)("openai-codex", "gpt-6.1-sol");
      expect(model).toBeDefined();
      if (!model) throw new Error("Sol missing from Pi's catalog");
      expect(model.api).toBe("openai-codex-responses");
      const cwd = mkdtempSync(join(tmpdir(), "pi-sol-reasoning-"));
      const { session } = await createAgentSession({
        cwd,
        model,
        ...fauxModelBackend(model),
        thinkingLevel: requested,
        sessionManager: SessionManager.inMemory(cwd),
        settingsManager: SettingsManager.inMemory(),
        tools: [],
      });
      try {
        expect(session.thinkingLevel).toBe(requested === "off" ? "minimal" : requested);
        if (requested === "minimal") expect(model.thinkingLevelMap?.minimal).toBe("low");
      } finally {
        session.dispose();
        rmSync(cwd, { recursive: true, force: true });
      }
    },
  );
});
