import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { fauxProvider } from "@earendil-works/pi-ai/providers/faux";
import { type ExtensionContext, ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ModelRouter } from "../src/model-routing.js";
import { setScopeModelsEnabled } from "../src/model-scope.js";
import { ctx, type Hermetic, hermeticDir } from "./helpers/boot-extension.js";

let hermetic: Hermetic;
beforeEach(() => { hermetic = hermeticDir(); setScopeModelsEnabled(true); });
afterEach(() => { setScopeModelsEnabled(false); vi.restoreAllMocks(); vi.unstubAllGlobals(); hermetic.restore(); });

it.each([
  { mode: "auto", scenario: "blocked" }, { mode: "jev", scenario: "blocked" }, { mode: "shadow", scenario: "blocked" },
  { mode: "jev", scenario: "allowed" }, { mode: "jev", scenario: "scope_off" }, { mode: "jev", scenario: "scope_changed" },
] as const)("native $mode routing respects enabledModels when $scenario", async ({ mode, scenario }) => {
  const settingsPath = join(hermetic.dir, ".pi", "settings.json");
  writeFileSync(settingsPath, JSON.stringify({ enabledModels: ["scope-faux/baseline"] }));
  if (scenario === "scope_off") setScopeModelsEnabled(false);
  const fetch = vi.fn<typeof globalThis.fetch>().mockImplementation(async () => {
    if (scenario === "scope_changed") writeFileSync(settingsPath, JSON.stringify({ enabledModels: ["scope-faux/selected"] }));
    return new Response(JSON.stringify({ answers: { route: { type: "choice", choice: "route_0", confidence: 0.9,
      probabilities: { route_0: 0.9, keep_baseline: 0.1 } } }, usage: { input_tokens: 50, output_tokens: 2 } }));
  });
  vi.stubGlobal("fetch", fetch);
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
    modelsStorePath: join(hermetic.dir, "models-cache.json"), allowModelNetwork: false, refreshOnCreate: false });
  runtime.registerNativeProvider(fauxProvider({ provider: "scope-faux", models: [{ id: "baseline" }, { id: "selected" }] }).provider);
  await runtime.refresh({ allowNetwork: false });
  const registry = new ModelRegistry(runtime);
  const baseline = registry.find("scope-faux", "baseline")!;
  const context: ExtensionContext = ctx({ model: baseline, modelRegistry: registry, scopedModels: [] });
  const modelId = scenario === "allowed" || scenario === "scope_changed" ? "baseline" : "selected";
  const selected = await new ModelRouter().choose(context, { mode, source: "jev", jev: { TYPESAFE_API_KEY: "scope-test-key",
    models: [{ model: `scope-faux/${modelId}`, description: "Coding" }] } }, "Review", "Review", baseline, new AbortController().signal, vi.fn());
  const blocked = scenario === "blocked";
  expect(fetch).toHaveBeenCalledTimes(blocked ? 0 : 1);
  expect(selected.decision.code).toBe(blocked ? "no_candidates" : scenario === "scope_changed" ? "unavailable_choice" : "selected");
  expect(selected.model?.id).toBe(blocked || scenario === "scope_changed" ? undefined : modelId);
});
