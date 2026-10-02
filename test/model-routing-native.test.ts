import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { InMemoryCredentialStore, type Usage } from "@earendil-works/pi-ai";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { type AgentSession, createAgentSession, createCodemodeExtension, DefaultResourceLoader, type ExtensionContext, ModelRegistry, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AgentManager } from "../src/agent-manager.js";
import { registerAgents } from "../src/agent-types.js";
import { loadCustomAgents } from "../src/custom-agents.js";
import extension from "../src/index.js";
import { ModelRouter } from "../src/model-routing.js";
import { ctx, type Hermetic, hermeticDir, makePi } from "./helpers/boot-extension.js";

let hermetic: Hermetic;
let manager: AgentManager | undefined;
let parent: AgentSession | undefined;
beforeEach(() => { hermetic = hermeticDir(); });
afterEach(async () => {
  await parent?.extensionRunner?.emit({ type: "session_shutdown", reason: "quit" });
  parent?.dispose(); parent = undefined;
  await manager?.dispose(); manager = undefined;
  registerAgents(new Map());
  vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); hermetic.restore();
});

it.each(["selected", "uncertain", "rejected_key", "missing_key"] as const)("Jev-first %s handles a real agent-file model pin with native authentication", async outcome => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockImplementation(async () => outcome === "rejected_key"
    ? new Response(JSON.stringify({ error: "invalid key private-native-key" }), { status: 401 })
    : new Response(JSON.stringify({
      answers: { route: { type: "choice", choice: "route_0", confidence: outcome === "uncertain" ? 0.59 : 0.9,
        probabilities: { route_0: 0.9, keep_baseline: 0.1 } } }, usage: { input_tokens: 50, output_tokens: 2 },
    })));
  vi.stubGlobal("fetch", fetch);
  vi.stubEnv("TYPESAFE_API_KEY", "");
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
    modelsStorePath: join(hermetic.dir, "models-cache.json"), allowModelNetwork: false, refreshOnCreate: false });
  const faux = fauxProvider({ provider: "pinned-faux", models: [{ id: "parent" }, { id: "pinned" }, { id: "selected" }], tokensPerSecond: 0 });
  faux.setResponses([fauxAssistantMessage("pinned reply")]);
  runtime.registerNativeProvider(faux.provider); await runtime.refresh({ allowNetwork: false });
  const registry = new ModelRegistry(runtime);
  mkdirSync(join(hermetic.dir, ".pi", "agents"));
  writeFileSync(join(hermetic.dir, ".pi", "agents", "reviewer.md"), "---\ndescription: Review\nmodel: pinned-faux/pinned\n---\nReview.");
  writeFileSync(join(hermetic.dir, ".pi", "subagents.json"), JSON.stringify({ routingMode: "jev", jev: {
    ...(outcome === "missing_key" ? {} : { TYPESAFE_API_KEY: "private-native-key" }),
    models: [{ model: "pinned-faux/selected", description: "Coding" }],
  }, rememberAgents: false, outputTranscript: false }));
  registerAgents(loadCustomAgents(hermetic.dir));
  manager = new AgentManager();
  const context: ExtensionContext = ctx({ model: registry.find("pinned-faux", "parent"), modelRegistry: registry, scopedModels: [] });
  const { record } = await manager.spawnAndWait(makePi().pi, context, "reviewer", "Review", { description: "review", isolated: true });
  expect(record.status).toBe("completed");
  expect(record.session?.model?.id).toBe(outcome === "selected" ? "selected" : "pinned");
  expect(record.routing?.code).toBe(outcome === "selected" ? "selected" : outcome === "uncertain" ? "abstained" : outcome === "missing_key" ? "credentials_unavailable" : "classifier_error");
  expect(record.routing?.fallbackSource).toBe(outcome === "selected" ? undefined : "agents");
  expect(fetch).toHaveBeenCalledTimes(outcome === "missing_key" ? 0 : 1);
  if (outcome !== "missing_key") expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).state.baseline).toBe("pinned-faux/pinned");
  expect(JSON.stringify(record.routing)).not.toContain("private-native-key");
});

it.each(["literal", "environment"])("uses Pi 1.0's real registry, authentication and TypeSafe transport with %s credentials", async credentialSource => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify({
    answers: { route: { type: "choice", choice: "route_0", confidence: 0.9, probabilities: { route_0: 0.9, keep_baseline: 0.1 } } },
    usage: { input_tokens: 50, output_tokens: 2 },
  }), { headers: { "content-type": "application/json" } }));
  vi.stubGlobal("fetch", fetch);
  vi.stubEnv("TYPESAFE_API_KEY", credentialSource === "environment" ? "environment-key" : "different-environment-key");
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
    modelsStorePath: join(hermetic.dir, "models-cache.json"), allowModelNetwork: false, refreshOnCreate: false });
  await runtime.setRuntimeApiKey("anthropic", "coding-key");
  const registry = new ModelRegistry(runtime);
  const model = registry.getAvailable().find(entry => entry.provider === "anthropic");
  expect(model).toBeDefined();
  const context: ExtensionContext = ctx({ model, modelRegistry: registry, scopedModels: [] });
  const onUsage = vi.fn<(usage: Usage) => void>();
  const key = credentialSource === "literal" ? { TYPESAFE_API_KEY: "literal-key" } : {};
  const chosen = await new ModelRouter().choose(context, { mode: "auto", source: "jev", jev: { ...key,
    models: [{ model: `${model!.provider}/${model!.id}`, description: "Coding tasks" }] } }, "Review this function", "review", model, new AbortController().signal, onUsage);
  expect(chosen.model?.id).toBe(model!.id);
  expect(chosen.decision).toMatchObject({ source: "jev", unpriced: true });
  expect(fetch).toHaveBeenCalledTimes(1);
  const [url, request] = fetch.mock.calls[0];
  expect(String(url)).toBe("https://api.typesafe.ai/v1/systemone");
  expect(new Headers(request?.headers).get("authorization")).toBe(`Bearer ${credentialSource === "literal" ? "literal-key" : "environment-key"}`);
  const payload = JSON.parse(request!.body as string);
  expect(payload).toMatchObject({ model: "jev-latest", state: { task: "Review this function" },
    questions: { route: { type: "choice", criteria: { route_0: "Coding tasks" } } } });
  expect(JSON.stringify(payload)).not.toContain("literal-key");
  expect(onUsage).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ input: 50, output: 2, totalTokens: 52 }));
  expect(await runtime.listCredentials()).toEqual([expect.objectContaining({ providerId: "anthropic" })]);
});

it.each(["auto", "shadow", "jev", "off"] as const)("creates a real child session in %s mode", async mode => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify({
    answers: { route: { type: "choice", choice: "route_0", confidence: 0.9, probabilities: { route_0: 0.9, keep_baseline: 0.1 } } },
    usage: { input_tokens: 50, output_tokens: 2 },
  })));
  vi.stubGlobal("fetch", fetch);
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
    modelsStorePath: join(hermetic.dir, "models-cache.json"), allowModelNetwork: false, refreshOnCreate: false });
  const faux = fauxProvider({ provider: "routing-faux", models: [{ id: "baseline" }, { id: "selected" }], tokensPerSecond: 0 });
  faux.setResponses([fauxAssistantMessage("routed reply")]);
  runtime.registerNativeProvider(faux.provider);
  // Registration starts an asynchronous catalog refresh; await a complete
  // refresh before relying on the registry's synchronous availability snapshot.
  await runtime.refresh({ allowNetwork: false });
  const registry = new ModelRegistry(runtime);
  const baseline = registry.find("routing-faux", "baseline")!;
  expect(registry.getAvailable().some(model => model.provider === "routing-faux" && model.id === "selected")).toBe(true);
  writeFileSync(join(hermetic.dir, ".pi", "subagents.json"), JSON.stringify({ routingMode: mode, jev: { TYPESAFE_API_KEY: "native-key",
    models: [{ model: "routing-faux/selected", description: "Coding" }] }, rememberAgents: false, outputTranscript: false }));
  manager = new AgentManager();
  const context: ExtensionContext = ctx({ model: baseline, modelRegistry: registry, scopedModels: [] });
  const { record } = await manager.spawnAndWait(makePi().pi, context, "general-purpose", "Say routed reply", {
    description: "native session", model: baseline, isolated: true,
    routing: { modelExplicit: false, thinkingExplicit: false, entrypoint: "agent" },
  });
  expect(record.status).toBe("completed");
  expect(record.result).toBe("routed reply");
  const applied = mode === "auto" || mode === "jev";
  expect(record.routing?.mode).toBe(mode);
  expect(record.routing?.model).toBe(applied ? "routing-faux/selected" : undefined);
  if (mode === "shadow") expect(record.routing?.suggestedModel).toBe("routing-faux/selected");
  expect(record.session?.model?.id).toBe(applied ? "selected" : "baseline");
  expect(record.invocation?.modelId).toBe(`routing-faux/${applied ? "selected" : "baseline"}`);
  expect(record.routingUsage?.totalTokens).toBe(mode === "off" ? undefined : 52);
  expect(fetch).toHaveBeenCalledTimes(mode === "off" ? 0 : 1);
  expect(faux.state.callCount).toBe(1);
});

it.each(["auto", "shadow", "jev", "off"] as const)("runs %s routing through Pi's real codemode sandbox", async mode => {
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify({
    answers: { route: { type: "choice", choice: "route_0", confidence: 0.9, probabilities: { route_0: 0.9, keep_baseline: 0.1 } } },
    usage: { input_tokens: 50, output_tokens: 2 },
  })));
  vi.stubGlobal("fetch", fetch);
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
    modelsStorePath: join(hermetic.dir, "models-cache.json"), allowModelNetwork: false, refreshOnCreate: false });
  const faux = fauxProvider({ provider: "codemode-faux", models: [{ id: "baseline" }, { id: "selected" }], tokensPerSecond: 0 });
  faux.setResponses([
    fauxAssistantMessage(fauxToolCall("codemode", { code: 'text(await tools.Agent({subagent_type:"general-purpose",prompt:"Say child reply",description:"child",run_in_background:false,isolated:true}));' }), { stopReason: "toolUse" }),
    fauxAssistantMessage("child reply"), fauxAssistantMessage("parent done"),
  ]);
  runtime.registerNativeProvider(faux.provider);
  await runtime.refresh({ allowNetwork: false });
  writeFileSync(join(hermetic.dir, ".pi", "subagents.json"), JSON.stringify({ routingMode: mode, jev: { TYPESAFE_API_KEY: "codemode-key",
    models: [{ model: "codemode-faux/selected", description: "Coding" }] }, rememberAgents: false, outputTranscript: false }));
  const loader = new DefaultResourceLoader({ cwd: hermetic.dir, agentDir: process.env.PI_CODING_AGENT_DIR!,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noContextFiles: true,
    extensionFactories: [extension, createCodemodeExtension({ mode: "only", models: false })] });
  await loader.reload();
  const stream = vi.spyOn(runtime, "streamSimple");
  const { session } = await createAgentSession({ cwd: hermetic.dir, agentDir: process.env.PI_CODING_AGENT_DIR!,
    modelRuntime: runtime, model: runtime.getModel("codemode-faux", "baseline"), resourceLoader: loader,
    sessionManager: SessionManager.inMemory(hermetic.dir),
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }) });
  parent = session;
  await session.bindExtensions({});
  session.setActiveToolsByName(["Agent", "codemode"]);
  await session.prompt("Delegate through codemode.");
  expect(fetch).toHaveBeenCalledTimes(mode === "off" ? 0 : 1);
  expect(stream.mock.calls.map(call => call[0].id)).toEqual(["baseline", mode === "auto" || mode === "jev" ? "selected" : "baseline", "baseline"]);
  const codeResult = session.messages.find(message => message.role === "toolResult" && message.toolName === "codemode");
  expect(JSON.stringify(codeResult)).toContain("Script completed");
  expect(JSON.stringify(codeResult)).toContain("child reply");
  expect(JSON.stringify(session.messages)).not.toContain("codemode-key");
  if (mode === "jev") {
    expect(session.getAllTools().find(tool => tool.name === "Agent")?.description).toContain("Routing mode: jev");
    writeFileSync(join(hermetic.dir, ".pi", "subagents.json"), JSON.stringify({ routingMode: "off", rememberAgents: false, outputTranscript: false }));
    faux.setResponses([fauxAssistantMessage("mode changed")]);
    await session.prompt("Continue after changing the routing mode.");
    expect(session.getAllTools().find(tool => tool.name === "Agent")?.description).not.toContain("Routing mode: jev");
    expect(fetch).toHaveBeenCalledTimes(1);
  }
});
