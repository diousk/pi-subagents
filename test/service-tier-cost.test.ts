import { type Api, type AssistantMessage, calculateCost, type Model } from "@earendil-works/pi-ai";
import { expect, it } from "vitest";
import { serviceTierCostExtension } from "../src/service-tier-cost.js";
import { ctx, makePi } from "./helpers/boot-extension.js";
import { getModel } from "./helpers/pi-ai.js";

function boot(model: Model<Api>) {
  const setup = makePi();
  serviceTierCostExtension(setup.pi);
  const context = ctx({ model, modelRegistry: { find: () => model }, isIdle: () => true });
  return {
    context,
    lifecycle: setup.lifecycle,
    record(type: string, responseId: string, tier = "fast") {
      setup.lifecycle.get("provider_stream_event")({
        type: "provider_stream_event", provider: model.provider, api: model.api, model: model.id,
        data: { type, response: { id: responseId, service_tier: tier } },
      }, context);
    },
    complete(message: AssistantMessage) {
      return setup.lifecycle.get("message_end")({ type: "message_end", message }, context);
    },
  };
}

function message(model: Model<Api>, responseId = "response-1"): AssistantMessage {
  const usage = { input: 9, output: 4, cacheRead: 7, cacheWrite: 2, totalTokens: 22,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  calculateCost(model, usage);
  return { role: "assistant", api: model.api, provider: model.provider, model: model.id,
    responseId, content: [], stopReason: "stop", timestamp: 0, usage };
}

it.each(["response.completed", "response.done", "response.incomplete"])("corrects %s Fast costs, including cached tokens", (type) => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const setup = boot(model);
  const original = message(model);
  if (type === "response.incomplete") original.stopReason = "length";
  setup.record(type, original.responseId!);
  const corrected = setup.complete(original).message as AssistantMessage;
  expect(corrected.usage).toMatchObject({ input: 9, output: 4, cacheRead: 7, cacheWrite: 2 });
  for (const key of ["input", "output", "cacheRead", "cacheWrite", "total"] as const) {
    expect(corrected.usage.cost[key]).toBeCloseTo(original.usage.cost[key] * 2, 12);
  }
  expect(setup.complete(original)).toBeUndefined();
});

it("uses GPT-5.5's 2.5x Fast multiplier", () => {
  const model = getModel("openai-codex", "gpt-5.5");
  const setup = boot(model);
  const original = message(model);
  setup.record("response.completed", original.responseId!);
  expect(setup.complete(original).message.usage.cost.total).toBeCloseTo(original.usage.cost.total * 2.5, 12);
});

it.each([1.5, 2, 3])("preserves a cost already adjusted by a factor of %s", (factor) => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const setup = boot(model);
  const priced = message(model);
  for (const key of ["input", "output", "cacheRead", "cacheWrite", "total"] as const) priced.usage.cost[key] *= factor;
  const expected = structuredClone(priced);
  setup.record("response.completed", priced.responseId!);
  expect(setup.complete(priced)).toBeUndefined();
  expect(priced).toEqual(expected);
});

it.each(["before_provider_request", "before_agent_start", "agent_settled", "session_shutdown"])("discards pending Fast responses on %s", (event) => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const setup = boot(model);
  const original = message(model);
  setup.record("response.completed", original.responseId!);
  setup.lifecycle.get(event)({ type: event }, setup.context);
  expect(setup.complete(original)).toBeUndefined();
});

it("keeps identical response IDs isolated between child sessions", () => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const first = boot(model);
  const second = boot(model);
  const original = message(model);
  first.record("response.completed", original.responseId!);
  expect(second.complete(original)).toBeUndefined();
  expect(first.complete(original).message.usage.cost.total).toBeCloseTo(original.usage.cost.total * 2, 12);
});

it("leaves standard fallbacks and other provider adapters unchanged", () => {
  for (const provider of ["openai-codex", "openai"] as const) {
    const model = getModel(provider, "gpt-6.1-sol");
    const setup = boot(model);
    const original = message(model);
    setup.record("response.completed", original.responseId!, provider === "openai" ? "fast" : "default");
    expect(setup.complete(original)).toBeUndefined();
  }
});

it.each(["model", "provider", "api"])("does not price a response with a mismatched %s", (property) => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const setup = boot(model);
  const unrelated = { ...message(model), [property]: "other-value" };
  setup.record("response.completed", unrelated.responseId!);
  expect(setup.complete(unrelated)).toBeUndefined();
});

it("retains the pending price while another request starts in a busy session", () => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const setup = boot(model);
  const original = message(model);
  setup.record("response.completed", original.responseId!);
  setup.context.isIdle = () => false;
  setup.lifecycle.get("before_provider_request")({ type: "before_provider_request" }, setup.context);
  expect(setup.complete(original).message.usage.cost.total).toBeCloseTo(original.usage.cost.total * 2, 12);
});

it("prices a physical model even when the session now selects a virtual model", () => {
  const model = getModel("openai-codex", "gpt-6.1-sol");
  const setup = boot(model);
  setup.context.model = { ...model, id: "virtual-model" };
  const original = message(model);
  setup.record("response.completed", original.responseId!);
  expect(setup.complete(original).message.usage.cost.total).toBeCloseTo(original.usage.cost.total * 2, 12);
});
