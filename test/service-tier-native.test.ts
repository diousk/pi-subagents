import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { zstdDecompressSync } from "node:zlib";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { type AgentSession, type ExtensionContext, ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { resumeAgent, runAgent } from "../src/agent-runner.js";
import { loadCustomAgents } from "../src/custom-agents.js";
import type { LifetimeUsage } from "../src/usage.js";
import { ctx, type Hermetic, hermeticDir, makePi } from "./helpers/boot-extension.js";

let hermetic: Hermetic;
let session: AgentSession | undefined;
beforeEach(() => { hermetic = hermeticDir({ settings: { rememberAgents: false, outputTranscript: false } }); });
afterEach(() => {
  session?.dispose(); session = undefined;
  vi.restoreAllMocks(); vi.unstubAllGlobals(); hermetic.restore();
});

const cases = ["openai", "openai-codex"].flatMap(provider => ["gpt-6.1-sol", "gpt-6-luna"].flatMap(modelId => [
  { provider, modelId, requested: "fast", returned: "fast" },
  { provider, modelId, requested: "priority", returned: "fast" },
  { provider, modelId, requested: "priority", returned: "priority" },
  { provider, modelId, requested: "fast", returned: "default" },
  { provider, modelId, requested: undefined, returned: "default" },
]));

it.each(cases)("$provider/$modelId sends $requested with returned $returned on initial and resumed turns", async ({ provider, modelId, requested, returned }) => {
  const requests: { url: string; payload: Record<string, unknown> }[] = [];
  // Fake only the HTTP boundary: Pi's catalog, authentication, session,
  // payload hook, provider serialization and response parser stay native.
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockImplementation(async (input, init) => {
    const request = new Request(input, init);
    const bytes = Buffer.from(await request.arrayBuffer());
    const body = request.headers.get("content-encoding") === "zstd" ? zstdDecompressSync(bytes) : bytes;
    requests.push({ url: request.url, payload: JSON.parse(body.toString("utf8")) });
    const item = { type: "message", id: `msg_${requests.length}`, role: "assistant", status: "completed",
      content: [{ type: "output_text", text: "OK", annotations: [] }] };
    const events = [
      { type: "response.output_item.added", output_index: 0, item: { ...item, status: "in_progress", content: [] } },
      { type: "response.output_item.done", output_index: 0, item },
      { type: "response.completed", response: { id: `resp_${requests.length}`, status: "completed", output: [item], service_tier: returned,
        usage: { input_tokens: 10, output_tokens: 1, total_tokens: 11, input_tokens_details: { cached_tokens: 0 } } } },
    ];
    return new Response(events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""),
      { headers: { "content-type": "text/event-stream" } });
  }));
  const claims = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "test-account" } })).toString("base64url");
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(provider, async () => provider === "openai"
    ? { type: "api_key", key: "sk-test-only" }
    : { type: "oauth", access: `test.${claims}.signature`, refresh: "test-refresh", expires: Date.now() + 3_600_000 });
  const runtime = await ModelRuntime.create({ credentials, modelsPath: null,
    modelsStorePath: join(hermetic.dir, "models-cache.json"), allowModelNetwork: false, refreshOnCreate: false });
  await runtime.refresh({ allowNetwork: false });
  const registry = new ModelRegistry(runtime);
  const model = registry.find(provider, modelId);
  expect(model).toBeDefined();
  if (!model) throw new Error(`${provider}/${modelId} missing from Pi's catalog`);
  expect(model.api).toBe(provider === "openai" ? "openai-responses" : "openai-codex-responses");
  // Force HTTP so this test cannot open a real Codex WebSocket connection.
  const streamSimple = runtime.streamSimple.bind(runtime);
  vi.spyOn(runtime, "streamSimple").mockImplementation((requestModel, context, options) =>
    streamSimple(requestModel, context, { ...options, transport: "sse", maxRetries: 0 }));
  mkdirSync(join(hermetic.dir, ".pi", "agents"));
  writeFileSync(join(hermetic.dir, ".pi", "agents", "tier-worker.md"), [
    "---", "description: Service tier probe", `model: ${provider}/${modelId}`,
    ...(requested === undefined ? [] : [`service_tier: ${requested}`]),
    "thinking: low", "tools: none", "extensions: false", "skills: false", "persist_session: false", "---", "Reply OK.",
  ].join("\n"));
  const agentConfig = loadCustomAgents(hermetic.dir).get("tier-worker");
  expect(agentConfig?.serviceTier).toBe(requested);
  const context: ExtensionContext = ctx({ model, modelRegistry: registry, scopedModels: [] });
  const usage = vi.fn<(usage: LifetimeUsage) => void>();
  const result = await runAgent(context, "tier-worker", "Reply OK", { pi: makePi().pi, agentConfig, isolated: true, onAssistantUsage: usage,
    onSessionCreated: created => { session = created; } });
  session = result.session;
  expect(result.failure).toBeUndefined();
  expect(result.responseText).toBe("OK");
  expect(session.model?.id).toBe(modelId);
  expect(await resumeAgent(session, "Reply again", { onAssistantUsage: usage })).toEqual({ text: "OK", failure: undefined });
  expect(requests).toHaveLength(2);
  for (const request of requests) {
    expect(request.url).toBe(provider === "openai" ? "https://api.openai.com/v1/responses" : "https://chatgpt.com/backend-api/codex/responses");
    expect(request.payload.model).toBe(modelId);
    if (requested === undefined) expect(request.payload).not.toHaveProperty("service_tier");
    else expect(request.payload.service_tier).toBe(requested);
  }
  expect(usage).toHaveBeenCalledTimes(2);
  for (const [turn] of usage.mock.calls) {
    expect(turn).toMatchObject({ input: 10, output: 1, cacheRead: 0, cacheWrite: 0 });
    // Pi 1.0.0's Codex adapter omits the fast pricing multiplier. Do not
    // encode that bug as the expected price or calculate our own replacement.
    if (provider === "openai-codex" && returned === "fast") continue;
    const expectedCost = (10 * model.cost.input + model.cost.output) / 1_000_000 * (returned === "fast" || returned === "priority" ? 2 : 1);
    expect(turn.cost).toBeCloseTo(expectedCost, 12);
  }
});
