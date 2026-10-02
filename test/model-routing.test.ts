import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Api, ClassifierApi, ClassifierModel, ClassifierResult, Model, Usage } from "@earendil-works/pi-ai";
import type { AgentSession, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentManager } from "../src/agent-manager.js";
import * as runner from "../src/agent-runner.js";
import { registerAgents } from "../src/agent-types.js";
import { loadCustomAgents } from "../src/custom-agents.js";
import extension from "../src/index.js";
import { loadRoutingPolicy, ModelRouter, type RoutingPolicy, routingGuidance } from "../src/model-routing.js";
import { createNestedSubagentTools } from "../src/nested-tools.js";
import { parseJevConfig } from "../src/routing-config.js";
import { SubagentScheduler } from "../src/schedule.js";
import { ScheduleStore } from "../src/schedule-store.js";
import { loadSettings, projectRoutingSettings, publicSettings, saveAndEmitChanged } from "../src/settings.js";
import { maskedApiKey, showRoutingMenu } from "../src/ui/model-routing-menu.js";
import { createWorkflowHost } from "../src/workflow/host.js";
import { ctx, flush, type Hermetic, hermeticDir, makePi } from "./helpers/boot-extension.js";

vi.mock("../src/agent-runner.js", async () => {
  const actual = await vi.importActual<typeof runner>("../src/agent-runner.js");
  return { ...actual, runAgent: vi.fn(), resumeAgent: vi.fn() };
});

const baseline: Model<Api> = {
  id: "baseline", provider: "test", name: "Baseline", api: "anthropic-messages", baseUrl: "https://example.invalid",
  reasoning: true, input: ["text"], cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 }, contextWindow: 64000, maxTokens: 4096,
};
const selected: Model<Api> = { ...baseline, id: "selected", name: "Selected" };
const classifier: ClassifierModel<ClassifierApi> = {
  type: "classifier", id: "jev-latest", provider: "typesafe", name: "Jev", api: "typesafe-system-one",
  baseUrl: "https://api.typesafe.ai/v1/", input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 64000,
};
const usage: Usage = {
  input: 50, output: 2, cacheRead: 0, cacheWrite: 0, totalTokens: 52,
  cost: { input: 0.01, output: 0.001, cacheRead: 0, cacheWrite: 0, total: 0.011 },
};
const jev = { TYPESAFE_API_KEY: "private-route-key", models: [{ model: "test/selected", description: "Complex coding tasks" }] };
const policy: RoutingPolicy = { mode: "auto", source: "jev", jev };
function result(choice = "route_0", confidence = 0.9): ClassifierResult {
  return { api: "typesafe-system-one", provider: "typesafe", model: "jev-latest", timestamp: 0, stopReason: "stop", usage,
    answers: { route: { type: "choice", choice, confidence, probabilities: { [choice]: 1 } } } };
}

let hermetic: Hermetic;
let classify: ReturnType<typeof vi.fn<ExtensionContext["modelRegistry"]["classify"]>>;
let context: ExtensionContext;
let manager: AgentManager | undefined;
let shutdown: (() => Promise<void>) | undefined;

beforeEach(() => {
  hermetic = hermeticDir({ settings: { jev, outputTranscript: false, rememberAgents: false } });
  classify = vi.fn<ExtensionContext["modelRegistry"]["classify"]>().mockResolvedValue(result());
  context = ctx({ model: baseline, scopedModels: [], modelRegistry: {
    getAvailable: () => [baseline, selected],
    find: (provider: string, id: string) => [baseline, selected].find(model => model.provider === provider && model.id === id),
    findOfType: () => classifier, getAvailableOfType: vi.fn().mockResolvedValue([classifier]), classify,
  } });
  registerAgents(new Map());
  vi.mocked(runner.runAgent).mockReset();
  vi.mocked(runner.runAgent).mockImplementation(async (_ctx, _type, _prompt, options) => {
    const session = { model: options.model ?? baseline, thinkingLevel: "low", dispose: vi.fn() } as unknown as AgentSession;
    await Promise.resolve();
    options.onSessionCreated?.(session);
    return { responseText: "done", session, aborted: false, steered: false };
  });
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(async () => {
  await shutdown?.();
  shutdown = undefined;
  await manager?.dispose();
  manager = undefined;
  registerAgents(new Map());
  vi.useRealTimers();
  vi.restoreAllMocks();
  hermetic.restore();
});

function settings(value: unknown): void {
  writeFileSync(join(hermetic.dir, ".pi", "subagents.json"), JSON.stringify(value));
}
function agent(name: string, content = "---\ndescription: Review code\n---\nReview."): void {
  mkdirSync(join(hermetic.dir, ".pi", "agents"), { recursive: true });
  writeFileSync(join(hermetic.dir, ".pi", "agents", name), content);
}

describe("routing configuration and priority", () => {
  it.each([
    [false, false, "jev"], [true, false, "agents"], [false, true, "guideline"], [true, true, "agents"],
  ] as const)("agents=%s, guideline=%s selects %s", (hasAgent, hasGuideline, source) => {
    if (hasAgent) agent("reviewer.md");
    if (hasGuideline) agent("custom-route.md", "Use test/selected.");
    settings({ jev, ...(hasGuideline ? { customGuideline: "agents/custom-route.md" } : {}) });
    expect(loadRoutingPolicy(hermetic.dir).source).toBe(source);
    expect(loadCustomAgents(hermetic.dir).has("custom-route")).toBe(false);
  });

  it("excludes an arbitrary configured guideline path and disabled agents", () => {
    agent("policy.md", "Use the small model.");
    agent("disabled.md", "---\nenabled: false\n---\nDisabled.");
    settings({ customGuideline: "agents/policy.md", jev });
    expect([...loadCustomAgents(hermetic.dir).keys()]).toEqual(["disabled"]);
    expect(loadRoutingPolicy(hermetic.dir).source).toBe("guideline");
  });

  it("uses the default global agents directory and respects PI_CODING_AGENT_DIR", () => {
    const root = process.env.PI_CODING_AGENT_DIR!;
    mkdirSync(join(root, "agents"));
    writeFileSync(join(root, "agents", "reviewer.md"), "---\ndescription: Global reviewer\n---\nReview.");
    expect(loadRoutingPolicy(hermetic.dir).source).toBe("agents");
  });

  it("resolves relative guideline paths from the supplying global or project file and records the original content hash", () => {
    const root = process.env.PI_CODING_AGENT_DIR!;
    mkdirSync(join(root, "agents"));
    const markdown = "    Preserve this code block.\n\nUse the small model.\n";
    writeFileSync(join(root, "agents", "policy.md"), markdown);
    writeFileSync(join(root, "subagents.json"), JSON.stringify({ customGuideline: "agents/policy.md" }));
    settings({ jev });
    const globalPolicy = loadRoutingPolicy(hermetic.dir);
    expect(globalPolicy).toMatchObject({ source: "guideline", guideline: markdown, guidelinePath: join(root, "agents", "policy.md"), guidelineHash: expect.stringMatching(/^[a-f0-9]{64}$/) });
    expect(loadCustomAgents(hermetic.dir).has("policy")).toBe(false);
    writeFileSync(join(root, "agents", "policy.md"), "Updated rule.");
    expect(loadRoutingPolicy(hermetic.dir).guidelineHash).not.toBe(globalPolicy.guidelineHash);
    // A project-relative file now overrides the global file.
    unlinkSync(join(root, "agents", "policy.md"));
    agent("custom-route.md", "Project rule.");
    settings({ customGuideline: "agents/custom-route.md", jev });
    expect(loadRoutingPolicy(hermetic.dir)).toMatchObject({ guideline: "Project rule.", guidelinePath: join(hermetic.dir, ".pi", "agents", "custom-route.md") });
  });

  it("rejects oversized guidelines without falling through to Jev", () => {
    agent("custom-route.md", "x".repeat(256001));
    settings({ customGuideline: "agents/custom-route.md", jev });
    expect(loadRoutingPolicy(hermetic.dir)).toMatchObject({ source: "guideline", diagnostic: expect.any(String) });
  });

  it("keeps a broken configured guideline above Jev, and false removes it", () => {
    settings({ customGuideline: "missing.md", jev });
    expect(loadRoutingPolicy(hermetic.dir)).toMatchObject({ source: "guideline", diagnostic: expect.stringContaining("unreadable") });
    settings({ customGuideline: false, jev });
    expect(loadRoutingPolicy(hermetic.dir).source).toBe("jev");
    settings({ customGuideline: false, jev: false });
    expect(loadRoutingPolicy(hermetic.dir).source).toBe("baseline");
  });

  it("replaces the entire global Jev block without inheriting a secret", () => {
    writeFileSync(join(process.env.PI_CODING_AGENT_DIR!, "subagents.json"), JSON.stringify({ jev }));
    settings({ jev: { models: [{ model: "test/baseline", description: "Easy tasks" }] } });
    expect(loadSettings(hermetic.dir).jev).toEqual({ models: [{ model: "test/baseline", description: "Easy tasks" }] });
    settings({ jev: { models: [] } });
    expect(loadSettings(hermetic.dir).jev).toBe(false);
    settings({});
    expect(loadSettings(hermetic.dir).jev).toEqual(jev);
    expect(projectRoutingSettings(hermetic.dir).jev).toBeUndefined();
  });

  it("malformed project JSON cannot revive a paid global route or leak its contents", () => {
    writeFileSync(join(process.env.PI_CODING_AGENT_DIR!, "subagents.json"), JSON.stringify({ jev }));
    writeFileSync(join(hermetic.dir, ".pi", "subagents.json"), '{"TYPESAFE_API_KEY":"private-route-key"');
    expect(loadRoutingPolicy(hermetic.dir)).toMatchObject({ mode: "off", source: "baseline" });
    expect(JSON.stringify(vi.mocked(console.warn).mock.calls)).not.toContain("private-route-key");
  });

  it.each([null, [], "private-route-key", 42])("an invalid project root cannot reactivate global Jev %#", raw => {
    writeFileSync(join(process.env.PI_CODING_AGENT_DIR!, "subagents.json"), JSON.stringify({ jev }));
    settings(raw);
    expect(loadSettings(hermetic.dir).jev).toBe(false);
    expect(JSON.stringify(vi.mocked(console.warn).mock.calls)).not.toContain("private-route-key");
  });

  it.each([{}, { models: [] }, { models: [{ model: "fuzzy", description: "task" }] },
    { models: [jev.models[0], jev.models[0]] }, { models: [{ model: "test/a", description: " " }] },
    { models: [{ model: "test/a", description: "x".repeat(4001) }] }, { ...jev, mode: "auto" },
    { models: Array.from({ length: 255 }, (_, index) => ({ model: `test/${index}`, description: "task" })) },
  ])("rejects invalid Jev configuration %#", value => {
    expect(() => parseJevConfig(value)).toThrow();
  });

  it("omits keys from events and public settings while persisting the configured key", () => {
    const emit = vi.fn();
    saveAndEmitChanged({ jev }, "Saved", emit, hermetic.dir);
    expect(JSON.stringify(emit.mock.calls)).not.toContain("private-route-key");
    expect(JSON.stringify(publicSettings({ jev }))).not.toContain("private-route-key");
    expect(readFileSync(join(hermetic.dir, ".pi", "subagents.json"), "utf-8")).toContain("private-route-key");
  });
});

describe("routing modes", () => {
  it.each(["auto", "shadow", "jev", "off"] as const)("inherits %s and preserves omission when saving another setting", mode => {
    writeFileSync(join(process.env.PI_CODING_AGENT_DIR!, "subagents.json"), JSON.stringify({ routingMode: mode, jev }));
    settings({ outputTranscript: false });
    expect(loadRoutingPolicy(hermetic.dir).mode).toBe(mode);
    saveAndEmitChanged({ ...projectRoutingSettings(hermetic.dir), showCost: true }, "Saved", vi.fn(), hermetic.dir);
    expect(loadSettings(hermetic.dir).routingMode).toBe(mode);
    expect(readFileSync(join(hermetic.dir, ".pi", "subagents.json"), "utf-8")).not.toContain("routingMode");
    settings({ routingMode: "off" });
    expect(loadRoutingPolicy(hermetic.dir)).toEqual({ mode: "off", source: "baseline" });
  });

  it.each(["unknown", null, false, {}, "private-route-key"])("invalid mode %# cannot inherit a paid global mode or leak its value", mode => {
    writeFileSync(join(process.env.PI_CODING_AGENT_DIR!, "subagents.json"), JSON.stringify({ routingMode: "jev", jev }));
    settings({ routingMode: mode });
    expect(loadRoutingPolicy(hermetic.dir)).toEqual({ mode: "off", source: "baseline" });
    expect(JSON.stringify(vi.mocked(console.warn).mock.calls)).not.toContain("private-route-key");
  });

  it.each((["auto", "shadow", "jev", "off"] as const).flatMap(mode =>
    (["agents", "guideline", "baseline"] as const).map(source => ({ mode, source }))))("$mode with $source uses the correct actual model", async ({ mode, source }) => {
    if (source === "agents") agent("reviewer.md", "---\ndescription: Review\nmodel: test/baseline\nthinking: high\n---\nReview.");
    if (source === "guideline") agent("custom-route.md", "Choose test/baseline for review.");
    settings({ routingMode: mode, jev, ...(source === "guideline" ? { customGuideline: "agents/custom-route.md" } : {}) });
    const agents = loadCustomAgents(hermetic.dir);
    registerAgents(agents);
    const onUsage = vi.fn();
    manager = new AgentManager(undefined, undefined, undefined, undefined, onUsage);
    const { record } = await manager.spawnAndWait(makePi().pi, context, source === "agents" ? "reviewer" : "general-purpose", "task", {
      model: baseline, thinkingLevel: "high", description: "task",
      routing: { policy: loadRoutingPolicy(hermetic.dir, agents), modelExplicit: true, thinkingExplicit: true, entrypoint: "agent" },
    });
    expect(classify).toHaveBeenCalledTimes(mode === "jev" || mode === "shadow" ? 1 : 0);
    expect(record.invocation?.modelId).toBe(mode === "jev" ? "test/selected" : "test/baseline");
    expect(record.routing?.mode).toBe(mode);
    if (mode === "shadow") {
      expect(record.routing).toMatchObject({ code: "shadow", suggestedModel: "test/selected", fallbackSource: source, confidence: 0.9 });
      expect(record.routing?.model).toBeUndefined();
      expect(record.routingUsage).toEqual(usage);
      expect(onUsage).toHaveBeenCalledTimes(1);
    }
    if (mode === "off") expect(routingGuidance(loadRoutingPolicy(hermetic.dir))).toBe("");
  });

  it.each(["agents", "guideline", "baseline"] as const)("Jev low confidence retains the %s choice without classifying twice", async source => {
    if (source === "agents") agent("reviewer.md");
    if (source === "guideline") agent("custom-route.md", "Use test/baseline.");
    settings({ routingMode: "jev", jev, ...(source === "guideline" ? { customGuideline: "agents/custom-route.md" } : {}) });
    classify.mockResolvedValue(result("route_0", 0.59));
    manager = new AgentManager();
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", { description: "task", model: baseline });
    expect(record.invocation?.modelId).toBe("test/baseline");
    expect(record.routing).toMatchObject({ mode: "jev", code: "abstained", fallbackSource: source });
    expect(classify).toHaveBeenCalledTimes(1);
  });

  it.each([undefined, false, { ...jev, models: [] }, { ...jev, TYPESAFE_API_KEY: "" },
    { ...jev, TYPESAFE_API_KEY: "private-route-key\ninvalid" }, { ...jev, TYPESAFE_API_KEY: 42 },
  ])("Jev-first invalid or absent config %# retains agent guidance without spending", async config => {
    agent("reviewer.md");
    settings({ routingMode: "jev", jev: config });
    const policy = loadRoutingPolicy(hermetic.dir);
    expect(routingGuidance(policy)).toContain("reviewer: Review code");
    manager = new AgentManager();
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", { model: baseline });
    expect(record.routing).toMatchObject({ code: "config_unavailable", fallbackSource: "agents" });
    expect(record.invocation?.modelId).toBe("test/baseline");
    expect(classify).not.toHaveBeenCalled();
    expect(JSON.stringify(vi.mocked(console.warn).mock.calls)).not.toContain("private-route-key");
  });

  it.each(["jev", "shadow"] as const)("%s validates Pi credentials before sending a request", async mode => {
    settings({ routingMode: mode, jev: { models: jev.models } });
    vi.mocked(context.modelRegistry.getAvailableOfType).mockResolvedValue([]);
    manager = new AgentManager();
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", { model: baseline });
    expect(record.routing).toMatchObject({ mode, code: "credentials_unavailable", fallbackSource: "baseline" });
    expect(classify).not.toHaveBeenCalled();
    expect(record.routingUsage).toBeUndefined();
    expect(record.invocation?.modelId).toBe("test/baseline");
  });

  it("retains low-confidence shadow suggestions for comparison without applying them", async () => {
    settings({ routingMode: "shadow", jev }); classify.mockResolvedValue(result("route_0", 0.59));
    manager = new AgentManager();
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", { description: "task" });
    expect(record.routing).toMatchObject({ mode: "shadow", code: "abstained", suggestedModel: "test/selected", confidence: 0.59 });
    expect(record.routing?.model).toBeUndefined();
    expect(record.invocation?.modelId).toBe("test/baseline");
  });

  it("Jev-first still runs with an unreadable guideline and preserves its diagnostic fallback", async () => {
    settings({ routingMode: "jev", customGuideline: "missing.md", jev });
    manager = new AgentManager();
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", { model: baseline });
    expect(record.routing).toMatchObject({ code: "selected", mode: "jev", model: "test/selected" });
    expect(record.invocation?.modelId).toBe("test/selected");
    expect(classify).toHaveBeenCalledTimes(1);
  });

  it.each(["jev", "shadow", "off"] as const)("%s never classifies resumes or internal helpers", async mode => {
    settings({ routingMode: mode, jev });
    manager = new AgentManager();
    await manager.spawnAndWait(makePi().pi, context, "general-purpose", "internal", {
      routing: { modelExplicit: false, thinkingExplicit: false, entrypoint: "internal" },
    });
    await manager.spawnAndWait(makePi().pi, context, "general-purpose", "resume", { resumeSessionFile: "/recorded/session.jsonl" });
    expect(classify).not.toHaveBeenCalled();
  });

  it.each(["jev", "shadow"] as const)("%s uses the fallback guidance on the main turn", async mode => {
    agent("custom-route.md", "Choose test/baseline for review.");
    settings({ routingMode: mode, customGuideline: "agents/custom-route.md", jev });
    const boot = makePi(); extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    const turn = await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context);
    expect(turn.systemPrompt).toContain(`Routing mode: ${mode}`);
    expect(turn.systemPrompt).toContain("Choose test/baseline for review.");
  });

  it.each(["agents", "guideline"] as const)("off removes %s selection guidance from custom tool descriptions and main turns", async source => {
    agent("reviewer.md", "---\ndescription: Inspect race conditions\n---\nReview.");
    agent("custom-route.md", "Custom routing marker.");
    if (source === "guideline") unlinkSync(join(hermetic.dir, ".pi", "agents", "reviewer.md"));
    settings({ routingMode: "off", customGuideline: "agents/custom-route.md", jev, toolDescriptionMode: "custom" });
    writeFileSync(join(hermetic.dir, ".pi", "agent-tool-description.md"), "Delegate a task.");
    const boot = makePi(); extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    expect(boot.tools.get("Agent").description.trim()).toBe("Delegate a task.");
    expect(await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context)).toBeUndefined();
    expect(classify).not.toHaveBeenCalled();
  });

  it.each(["auto", "shadow", "jev", "off"] as const)("the routing menu saves %s as a scalar override without copying credentials", async mode => {
    const select = vi.fn<ExtensionContext["ui"]["select"]>().mockResolvedValueOnce("Routing mode")
      .mockImplementationOnce(async (_title, options) => options.find(option => option.startsWith(`${mode} —`)));
    const patch = await showRoutingMenu({ ...context, ui: { ...context.ui, select } } as Parameters<typeof showRoutingMenu>[0]);
    expect(patch).toEqual({ routingMode: mode });
    expect(select.mock.calls[1][0]).toContain("current: auto");
  });

  it("saves a mode from the registered command and applies it on the next turn", async () => {
    const boot = makePi(); extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    const select = vi.fn<ExtensionContext["ui"]["select"]>().mockResolvedValueOnce("Model routing")
      .mockResolvedValueOnce("Routing mode").mockImplementationOnce(async (_title, options) => options.find(option => option.startsWith("off —")))
      .mockResolvedValueOnce(undefined);
    await boot.commands.get("agents").handler("", { ...context, ui: { ...context.ui, select } });
    expect(loadSettings(hermetic.dir).routingMode).toBe("off");
    expect(loadSettings(hermetic.dir).jev).toEqual(jev);
    expect(await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context)).toBeUndefined();
    expect(boot.tools.get("Agent").description).not.toContain("Routing:");
    expect(JSON.stringify(boot.pi.events.emit.mock.calls)).not.toContain("private-route-key");
  });

  it("removes a previously injected guideline from the tool description after switching off", async () => {
    agent("custom-route.md", "Private custom routing marker.");
    settings({ customGuideline: "agents/custom-route.md", jev });
    const boot = makePi(); extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    expect(boot.tools.get("Agent").description).toContain("Private custom routing marker.");
    settings({ routingMode: "off", customGuideline: "agents/custom-route.md", jev });
    expect(await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context)).toBeUndefined();
    expect(boot.tools.get("Agent").description).not.toContain("Private custom routing marker.");
    settings({ routingMode: "jev", customGuideline: "agents/custom-route.md", jev });
    await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context);
    expect(boot.tools.get("Agent").description).toContain("Routing mode: jev");
    expect(boot.tools.get("Agent").description).toContain("Private custom routing marker.");
  });

  it("shows shadow suggestions in results while preserving actual model metadata and usage", async () => {
    settings({ routingMode: "shadow", jev, reportUsage: true, outputTranscript: false });
    const boot = makePi(); extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    const response = await boot.tools.get("Agent").execute("shadow", {
      subagent_type: "general-purpose", model: "test/baseline", thinking: "high", prompt: "task", description: "task", run_in_background: false,
    }, undefined, undefined, context);
    expect(classify).toHaveBeenCalledTimes(1);
    expect(response.details.modelName).toBe("baseline");
    expect(response.details.tags).toContain("Jev shadow");
    expect(response.usage).toMatchObject({ totalTokens: 52, cost: { total: 0.011 } });
    expect(boot.pi.appendEntry).toHaveBeenCalledWith("subagents:record", expect.objectContaining({
      routing: expect.objectContaining({ mode: "shadow", suggestedModel: "test/selected" }),
      routingUsage: usage,
    }));
  });

  it.each(["shadow", "jev", "off"] as const)("%s applies to explicit workflow options", async mode => {
    agent("reviewer.md"); settings({ routingMode: mode, jev });
    manager = new AgentManager();
    const host = createWorkflowHost({ pi: makePi().pi, ctx: context, manager, workflowId: "wf_mode" });
    await host.spawnAgent({ prompt: "task", label: "task", agentType: "general-purpose", model: "test/baseline", effort: "high" });
    expect(classify).toHaveBeenCalledTimes(mode === "off" ? 0 : 1);
    expect(runner.runAgent).toHaveBeenCalledWith(context, "general-purpose", "task", expect.objectContaining({ model: mode === "jev" ? selected : baseline }));
  });

  it("reads an updated routing mode after dequeue, including with explicit pins", async () => {
    let finish!: () => void;
    vi.mocked(runner.runAgent).mockImplementationOnce(() => new Promise(resolve => {
      finish = () => resolve({ responseText: "done", session: {} as AgentSession, aborted: false, steered: false });
    }));
    settings({ routingMode: "off", jev });
    manager = new AgentManager(undefined, 1);
    const pi = makePi().pi;
    const first = manager.spawn(pi, context, "general-purpose", "first", { description: "first", model: baseline, isBackground: true });
    const next = manager.spawn(pi, context, "general-purpose", "next", { description: "next", model: baseline, isBackground: true });
    settings({ routingMode: "jev", jev }); agent("reviewer.md");
    finish(); await manager.getRecord(first)?.promise; await manager.waitForAll();
    expect(classify).toHaveBeenCalledTimes(1);
    expect(manager.getRecord(next)?.routing).toMatchObject({ mode: "jev", model: "test/selected" });
  });

  it.each(["shadow", "jev", "off"] as const)("%s is applied at schedule fire time despite custom agents and explicit inputs", async mode => {
    vi.useFakeTimers(); settings({ routingMode: "off", jev });
    manager = new AgentManager();
    const scheduler = new SubagentScheduler();
    const store = new ScheduleStore(join(hermetic.dir, ".pi", "schedules.json"));
    scheduler.start(makePi().pi, context, manager, store);
    try {
      const job = scheduler.addJob({ name: "mode", schedule: "+1s", subagent_type: "general-purpose", prompt: "task", description: "task", model: "test/baseline", thinking: "high" });
      settings({ routingMode: mode, jev }); agent("reviewer.md");
      await vi.advanceTimersByTimeAsync(1000); await manager.waitForAll();
      expect(store.get(job.id)?.lastStatus).toBe("success");
      expect(classify).toHaveBeenCalledTimes(mode === "off" ? 0 : 1);
      expect(runner.runAgent).toHaveBeenCalledWith(context, "general-purpose", "task", expect.objectContaining({ model: mode === "jev" ? selected : baseline }));
    } finally { scheduler.stop(); }
  });

  it.each(["shadow", "jev", "off"] as const)("RPC uses %s from settings and ignores forged routing policy", async mode => {
    settings({ routingMode: mode, jev }); agent("reviewer.md");
    const boot = makePi(); extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    await boot.lifecycle.get("session_start")({}, context);
    const rpc = boot.pi.events.on.mock.calls.find((call: unknown[]) => call[0] === "subagents:rpc:spawn")?.[1] as (data: unknown) => Promise<void>;
    await rpc({ requestId: mode, type: "general-purpose", prompt: "task", options: {
      model: baseline, thinkingLevel: "high", routing: { policy: { mode: "off", source: "baseline" } },
    } });
    await flush();
    expect(classify).toHaveBeenCalledTimes(mode === "off" ? 0 : 1);
    expect(runner.runAgent).toHaveBeenCalledWith(context, "general-purpose", "task", expect.objectContaining({ model: mode === "jev" ? selected : baseline }));
    expect(boot.pi.events.emit).toHaveBeenCalledWith("subagents:completed", expect.objectContaining({ routing: expect.objectContaining({ mode }) }));
  });

  it("nested calls use branch modes despite a different parent mode and agent pins", async () => {
    settings({ routingMode: "off", jev });
    manager = new AgentManager();
    const pi = makePi().pi;
    let finishParent!: () => void;
    vi.mocked(runner.runAgent).mockImplementationOnce(() => new Promise(resolve => {
      finishParent = () => resolve({ responseText: "done", session: {} as AgentSession, aborted: false, steered: false });
    }));
    const parent = manager.spawn(pi, context, "general-purpose", "parent", { description: "parent", model: baseline });
    const branch = join(hermetic.dir, "branch");
    mkdirSync(join(branch, ".pi", "agents"), { recursive: true });
    writeFileSync(join(branch, ".pi", "agents", "reviewer.md"), "---\ndescription: Review\nmodel: test/baseline\n---\nReview.");
    const nested = createNestedSubagentTools({ manager, pi, parentAgentId: parent, depth: 1, maxSubagentDepth: 3, allowedSubagents: "all", configCwd: branch })[0];
    for (const mode of ["jev", "shadow", "off"] as const) {
      writeFileSync(join(branch, ".pi", "subagents.json"), JSON.stringify({ routingMode: mode, jev: { ...jev, TYPESAFE_API_KEY: "branch-mode-key" } }));
      await nested.execute(mode, { subagent_type: "reviewer", prompt: "child", description: "child" }, undefined, undefined, context);
      expect(runner.runAgent).toHaveBeenLastCalledWith(context, "reviewer", "child", expect.objectContaining({ model: mode === "jev" ? selected : baseline }));
    }
    expect(classify).toHaveBeenCalledTimes(2);
    expect(classify.mock.calls.map(call => call[2]?.apiKey)).toEqual(["branch-mode-key", "branch-mode-key"]);
    expect(manager.getRecord(parent)?.lifetimeUsage.cost).toBeCloseTo(0.022);
    finishParent(); await manager.waitForAll();
  });
});

describe("native classifier routing", () => {
  it("uses one request-scoped key and opaque choices, without exposing model IDs as answers", async () => {
    const before = process.env.TYPESAFE_API_KEY;
    const chosen = await new ModelRouter().choose(context, policy, "debug a race", "debug", baseline, new AbortController().signal, vi.fn());
    expect(chosen.model).toBe(selected);
    expect(chosen.decision).toMatchObject({ model: "test/selected", confidence: 0.9, unpriced: true });
    expect(classify).toHaveBeenCalledTimes(1);
    const call = classify.mock.calls[0];
    expect(call[0]).toBe(classifier);
    expect(call[1].questions.route).toMatchObject({ type: "choice", criteria: { route_0: "Complex coding tasks" } });
    expect(JSON.stringify(call[1])).not.toContain("private-route-key");
    expect(call[2]).toMatchObject({ apiKey: "private-route-key", maxRetries: 0 });
    expect(process.env.TYPESAFE_API_KEY).toBe(before);
  });

  it("leaves credential resolution to Pi when no literal key is configured", async () => {
    await new ModelRouter().choose(context, { mode: "auto", source: "jev", jev: { models: jev.models } }, "task", "task", baseline, new AbortController().signal, vi.fn());
    expect(classify.mock.calls[0][2]?.apiKey).toBeUndefined();
  });

  it("cannot select unavailable or out-of-scope candidates", async () => {
    context.scopedModels = [{ model: baseline, thinkingLevel: "low" }];
    const chosen = await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn());
    expect(chosen.model).toBeUndefined();
    expect(classify).not.toHaveBeenCalled();
  });

  it("rejects stale registry entries and virtual models before spending", async () => {
    vi.spyOn(context.modelRegistry, "find").mockReturnValue(undefined);
    await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn());
    expect(classify).not.toHaveBeenCalled();
    vi.mocked(context.modelRegistry.find).mockReturnValue({ ...selected, api: "pi-virtual" });
    await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn());
    expect(classify).not.toHaveBeenCalled();
  });

  it("keeps the baseline when the classifier is absent", async () => {
    vi.spyOn(context.modelRegistry, "findOfType").mockReturnValue(undefined);
    expect((await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn())).model).toBeUndefined();
    expect(classify).not.toHaveBeenCalled();
  });

  it.each([{ route_0: -0.1, keep_baseline: 1.1 }, { route_0: Number.NaN },
    { route_0: 0.1 }, { route_0: 0.9, unknown: 0.1 }, { keep_baseline: 1 },
  ])("rejects malformed probability maps %# while counting billed usage", async probabilities => {
    classify.mockResolvedValue({ ...result(), answers: { route: { type: "choice", choice: "route_0", confidence: 0.9, probabilities } } });
    const onUsage = vi.fn();
    expect((await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, onUsage)).model).toBeUndefined();
    expect(onUsage).toHaveBeenCalledExactlyOnceWith(usage);
  });

  it("bounds the independent pool and cancels a waiting request without spending", async () => {
    const controllers = Array.from({ length: 6 }, () => new AbortController());
    const finishers: (() => void)[] = [];
    classify.mockImplementation(() => new Promise(resolve => { finishers.push(() => resolve(result())); }));
    const router = new ModelRouter();
    const pending = controllers.map(controller => router.choose(context, policy, "task", "task", baseline, controller.signal, vi.fn()));
    await flush();
    expect(classify).toHaveBeenCalledTimes(4);
    controllers[4].abort();
    expect((await pending[4]).decision.reason).toBe("Cancelled");
    finishers[0]();
    await flush();
    expect(classify).toHaveBeenCalledTimes(5);
    for (const finish of finishers.slice(1)) finish();
    await Promise.all(pending);
    expect(classify).toHaveBeenCalledTimes(5);
  });

  it("isolates concurrent request credentials and never mutates provider registration", async () => {
    const router = new ModelRouter();
    await Promise.all(["key-a", "key-b"].map(key => router.choose(context, { mode: "auto", source: "jev", jev: { ...jev, TYPESAFE_API_KEY: key } }, "task", "task", baseline, new AbortController().signal, vi.fn())));
    expect(classify.mock.calls.map(call => call[2]?.apiKey)).toEqual(["key-a", "key-b"]);
    expect(JSON.stringify(classify.mock.calls.map(call => call[1]))).not.toContain("key-a");
  });

  it("rechecks availability after classification", async () => {
    classify.mockImplementation(async () => {
      vi.spyOn(context.modelRegistry, "getAvailable").mockReturnValue([baseline]);
      return result();
    });
    const chosen = await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn());
    expect(chosen.model).toBeUndefined();
  });

  it.each(["keep_baseline", "invented/model", "route_999"])("keeps baseline for %s", async choice => {
    classify.mockResolvedValue(result(choice));
    const onUsage = vi.fn();
    const chosen = await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, onUsage);
    expect(chosen.model).toBeUndefined();
    expect(onUsage).toHaveBeenCalledExactlyOnceWith(usage);
  });

  it.each([0.59, -1, 2, Number.NaN])("rejects low/invalid confidence %s", async confidence => {
    classify.mockResolvedValue(result("route_0", confidence));
    expect((await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn())).model).toBeUndefined();
  });

  it("counts billed usage on an error without exposing provider error text", async () => {
    classify.mockResolvedValue({ ...result(), stopReason: "error", errorMessage: "private-route-key" });
    const onUsage = vi.fn();
    const chosen = await new ModelRouter().choose(context, policy, "task", "task", baseline, new AbortController().signal, onUsage);
    expect(chosen.model).toBeUndefined();
    expect(onUsage).toHaveBeenCalledExactlyOnceWith(usage);
    expect(JSON.stringify(chosen)).not.toContain("private-route-key");
  });

  it("aborts a stuck classifier at two seconds and releases its pool slot", async () => {
    vi.useFakeTimers();
    classify.mockImplementationOnce(() => new Promise(() => {}));
    const router = new ModelRouter();
    const pending = router.choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn());
    await vi.advanceTimersByTimeAsync(2000);
    expect((await pending).decision.reason).toContain("timed out");
    expect(classify.mock.calls[0][2]?.signal?.aborted).toBe(true);
    expect((await router.choose(context, policy, "task", "task", baseline, new AbortController().signal, vi.fn())).model).toBe(selected);
  });
});

describe("startup wiring and lifecycle", () => {
  it("routes an inherited model before session creation and uses actual session metadata", async () => {
    const onUsage = vi.fn();
    manager = new AgentManager(undefined, undefined, undefined, undefined, onUsage);
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", {
      description: "task", model: baseline, routing: { modelExplicit: false, thinkingExplicit: false, entrypoint: "agent" },
    });
    expect(runner.runAgent).toHaveBeenCalledWith(context, "general-purpose", "task", expect.objectContaining({ model: selected }));
    expect(record.invocation).toMatchObject({ modelId: "test/selected", thinking: "low" });
    expect(record.lifetimeUsage).toMatchObject({ input: 0, output: 0, cost: 0.011 });
    expect(record.routingUsage).toEqual(usage);
    expect(onUsage).toHaveBeenCalledTimes(1);
  });

  it.each([{ model: baseline }, { thinkingLevel: "high" as const }, { resumeSessionFile: "/recorded/session.jsonl" },
    { routing: { modelExplicit: false, thinkingExplicit: false, entrypoint: "internal" as const } },
  ])("does not classify explicit choices, reopened sessions or internal helpers %#", async options => {
    manager = new AgentManager();
    await manager.spawnAndWait(makePi().pi, context, "general-purpose", "task", { description: "task", ...options });
    expect(classify).not.toHaveBeenCalled();
  });

  it("stopping during classification launches no fallback and frees the agent slot", async () => {
    classify.mockImplementationOnce(() => new Promise(() => {}));
    manager = new AgentManager(undefined, 1);
    const pi = makePi().pi;
    const id = manager.spawn(pi, context, "general-purpose", "task", { description: "task", isBackground: true });
    await flush();
    const queued = manager.spawn(pi, context, "general-purpose", "next", { description: "next", isBackground: true });
    expect(manager.getRecord(queued)?.status).toBe("queued");
    manager.abort(id);
    await manager.awaitStartup(id);
    await manager.waitForAll();
    expect(manager.getRecord(id)?.status).toBe("stopped");
    expect(runner.runAgent).toHaveBeenCalledTimes(1);
    expect(runner.runAgent.mock.calls[0][2]).toBe("next");
  });

  it("does not classify a pre-cancelled request or launch a session during shutdown", async () => {
    manager = new AgentManager();
    const controller = new AbortController();
    controller.abort();
    const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "cancelled", { description: "cancelled", signal: controller.signal });
    expect(record.status).toBe("stopped");
    expect(classify).not.toHaveBeenCalled();
    expect(runner.runAgent).not.toHaveBeenCalled();
    classify.mockImplementationOnce(() => new Promise(() => {}));
    manager.spawn(makePi().pi, context, "general-purpose", "shutdown", { description: "shutdown" });
    await flush();
    await manager.dispose();
    expect(classify.mock.calls[0][2]?.signal?.aborted).toBe(true);
    expect(runner.runAgent).not.toHaveBeenCalled();
  });

  it("never classifies a cancelled queued agent and honors newly added higher-priority files", async () => {
    let finish!: () => void;
    vi.mocked(runner.runAgent).mockImplementationOnce(() => new Promise(resolve => {
      finish = () => resolve({ responseText: "done", session: {} as AgentSession, aborted: false, steered: false });
    }));
    manager = new AgentManager(undefined, 1);
    const pi = makePi().pi;
    const first = manager.spawn(pi, context, "general-purpose", "first", { description: "first", model: baseline, isBackground: true });
    const cancelled = manager.spawn(pi, context, "general-purpose", "cancelled", { description: "cancelled", isBackground: true });
    const next = manager.spawn(pi, context, "general-purpose", "next", { description: "next", isBackground: true });
    manager.abort(cancelled);
    agent("reviewer.md");
    finish();
    await manager.getRecord(first)?.promise;
    await manager.waitForAll();
    expect(classify).not.toHaveBeenCalled();
    expect(manager.getRecord(next)?.routing?.source).toBe("agents");
  });

  it.each(["full", "compact", "custom"])("includes the guideline in %s mode and the main turn", async mode => {
    agent("custom-route.md", "Route debugging to test/selected.");
    settings({ jev, customGuideline: "agents/custom-route.md", toolDescriptionMode: mode, outputTranscript: false });
    writeFileSync(join(hermetic.dir, ".pi", "agent-tool-description.md"), "My short Agent description.");
    const boot = makePi();
    extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    expect(boot.tools.get("Agent").description).toContain("Route debugging to test/selected.");
    const turn = await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context);
    expect(turn.systemPrompt).toContain("Route debugging to test/selected.");
    await boot.tools.get("Agent").execute("tc", { subagent_type: "general-purpose", prompt: "task", description: "task", run_in_background: false }, undefined, undefined, context);
    expect(classify).not.toHaveBeenCalled();
  });

  it("advertises custom agent descriptions even when a custom tool description omits the roster", async () => {
    agent("reviewer.md", "---\ndescription: Inspect race conditions carefully\n---\nReview.");
    settings({ jev, toolDescriptionMode: "custom", outputTranscript: false });
    writeFileSync(join(hermetic.dir, ".pi", "agent-tool-description.md"), "Delegate a task.");
    const boot = makePi();
    extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    expect(boot.tools.get("Agent").description).toContain("reviewer: Inspect race conditions carefully");
    const turn = await boot.lifecycle.get("before_agent_start")({ systemPrompt: "parent" }, context);
    expect(turn.systemPrompt).toContain("reviewer: Inspect race conditions carefully");
  });

  it("routes the real Agent tool once and reports classifier spend once to the parent", async () => {
    settings({ jev, reportUsage: true, outputTranscript: false });
    const boot = makePi();
    extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    const tool = boot.tools.get("Agent");
    const first = await tool.execute("tc", { subagent_type: "general-purpose", prompt: "task", description: "task", run_in_background: false }, undefined, undefined, context);
    expect(classify).toHaveBeenCalledTimes(1);
    expect(first.usage).toMatchObject({ totalTokens: 52, cost: { total: 0.011 } });
    expect(first.details.tags).toContain("Jev price unavailable");
    expect(boot.pi.appendEntry).toHaveBeenCalledWith("subagents:record", expect.objectContaining({ routing: expect.objectContaining({ model: "test/selected" }), routingUsage: usage }));
    const second = await tool.execute("tc2", { subagent_type: "general-purpose", prompt: "task", description: "task", thinking: "high", run_in_background: false }, undefined, undefined, context);
    expect(second.usage).toBeUndefined();
    expect(JSON.stringify(boot.pi.events.emit.mock.calls)).not.toContain("private-route-key");
  });

  it("routes a workflow's inherited model without making it an explicit pin", async () => {
    manager = new AgentManager();
    const host = createWorkflowHost({ pi: makePi().pi, ctx: context, manager, workflowId: "wf_test" });
    await host.spawnAgent({ prompt: "task", label: "task", agentType: "general-purpose" });
    expect(classify).toHaveBeenCalledTimes(1);
    expect(runner.runAgent).toHaveBeenCalledWith(context, "general-purpose", "task", expect.objectContaining({ model: selected }));
  });

  it("does not let RPC callers forge provenance or bypass higher-priority policy", async () => {
    const boot = makePi();
    extension(boot.pi);
    shutdown = () => boot.lifecycle.get("session_shutdown")({}, context);
    await boot.lifecycle.get("session_start")({}, context);
    const rpc = boot.pi.events.on.mock.calls.find((call: unknown[]) => call[0] === "subagents:rpc:spawn")?.[1] as (data: unknown) => Promise<void>;
    expect(rpc).toBeTypeOf("function");
    await rpc({ requestId: "pin", type: "general-purpose", prompt: "task", options: { model: baseline, routing: { modelExplicit: false, thinkingExplicit: false, policy } } });
    await flush();
    expect(classify).not.toHaveBeenCalled();
    await rpc({ requestId: "unpinned", type: "general-purpose", prompt: "task" });
    await flush();
    expect(classify).toHaveBeenCalledTimes(1);
    expect(classify.mock.calls[0][1].state.description).toBe("general-purpose");
    agent("reviewer.md");
    await rpc({ requestId: "agents", type: "general-purpose", prompt: "task", options: { routing: { modelExplicit: false, thinkingExplicit: false, policy }, agentConfig: { model: "test/selected" } } });
    await flush();
    expect(classify).toHaveBeenCalledTimes(1);
    expect(boot.pi.events.emit).toHaveBeenCalledWith("subagents:completed", expect.objectContaining({ routing: expect.objectContaining({ source: "agents" }) }));
  });

  it("routes nested agents with branch settings and rolls classifier cost into ancestors once", async () => {
    manager = new AgentManager(undefined, 1);
    const pi = makePi().pi;
    let finishParent!: () => void;
    vi.mocked(runner.runAgent).mockImplementationOnce(() => new Promise(resolve => { finishParent = () => resolve({ responseText: "done", session: {} as AgentSession, aborted: false, steered: false }); }));
    const parent = manager.spawn(pi, context, "general-purpose", "parent", { model: baseline, description: "parent", isBackground: true });
    const branch = join(hermetic.dir, "branch");
    mkdirSync(join(branch, ".pi"), { recursive: true });
    writeFileSync(join(branch, ".pi", "subagents.json"), JSON.stringify({ jev: { ...jev, TYPESAFE_API_KEY: "branch-key" }, outputTranscript: false }));
    agent("main-reviewer.md");
    const nested = createNestedSubagentTools({ manager, pi, parentAgentId: parent, depth: 1, maxSubagentDepth: 3, allowedSubagents: "all", configCwd: branch })[0];
    await nested.execute("child", { subagent_type: "general-purpose", prompt: "child", description: "child" }, undefined, undefined, context);
    expect(classify).toHaveBeenCalledTimes(1);
    expect(classify.mock.calls[0][2]?.apiKey).toBe("branch-key");
    expect(manager.getRecord(parent)?.lifetimeUsage).toMatchObject({ input: 0, output: 0, cost: 0.011 });
    expect(runner.runAgent).toHaveBeenLastCalledWith(context, "general-purpose", "child", expect.objectContaining({ model: selected, configCwd: branch }));
    writeFileSync(join(branch, ".pi", "custom-route.md"), "Use the baseline.");
    writeFileSync(join(branch, ".pi", "subagents.json"), JSON.stringify({ customGuideline: "custom-route.md", jev }));
    await nested.execute("child2", { subagent_type: "general-purpose", prompt: "child2", description: "child2" }, undefined, undefined, context);
    expect(classify).toHaveBeenCalledTimes(1);
    finishParent();
    await manager.waitForAll();
  });

  it.each(["jev", "agents", "guideline", "explicit"])("reads schedule-time settings and preserves %s provenance", async source => {
    vi.useFakeTimers();
    manager = new AgentManager();
    const scheduler = new SubagentScheduler();
    const store = new ScheduleStore(join(hermetic.dir, ".pi", "schedules.json"));
    scheduler.start(makePi().pi, context, manager, store);
    try {
      const job = scheduler.addJob({ name: "job", schedule: "+1s", subagent_type: "general-purpose", prompt: "scheduled", description: "scheduled", ...(source === "explicit" ? { thinking: "high" as const } : {}) });
      if (source === "agents") agent("reviewer.md");
      if (source === "guideline") settings({ customGuideline: "missing.md", jev });
      if (source === "jev") settings({ jev: { ...jev, TYPESAFE_API_KEY: "fire-key" } });
      await vi.advanceTimersByTimeAsync(1000);
      await manager.waitForAll();
      expect(store.get(job.id)?.lastStatus).toBe("success");
      expect(classify).toHaveBeenCalledTimes(source === "jev" ? 1 : 0);
      if (source === "jev") expect(classify.mock.calls[0][2]?.apiKey).toBe("fire-key");
      expect(runner.runAgent).toHaveBeenCalledWith(context, "general-purpose", "scheduled", expect.objectContaining({ model: source === "jev" ? selected : undefined }));
    } finally { scheduler.stop(); }
  });

  it("masks API-key paste and editing without calling the plaintext renderer", async () => {
    const frames: string[][] = [];
    const ui = { custom: vi.fn(async (factory: Parameters<ExtensionContext["ui"]["custom"]>[0]) => {
      let submitted: unknown;
      const component = factory({} as never, {} as never, {} as never, value => { submitted = value; });
      if (component instanceof Promise) throw new Error("unexpected async UI");
      component.handleInput?.("secret-key");
      frames.push(component.render(80));
      component.handleInput?.("\r");
      return submitted;
    }) };
    const key = await maskedApiKey({ ui } as unknown as Parameters<typeof maskedApiKey>[0]);
    expect(key).toBe("secret-key");
    expect(JSON.stringify(frames)).not.toContain("secret-key");
    expect(JSON.stringify(frames)).toContain("**********");
  });
});
