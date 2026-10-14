import { type Api, calculateCost, type Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Repair Codex's unscaled Fast estimate before usage reporting and persistence. */
export function serviceTierCostExtension(pi: ExtensionAPI): void {
  const fastResponses = new Map<string, Model<Api>>();

  // Idle warming must not discard a completed response still being delivered.
  pi.on("before_provider_request", (_event, ctx) => { if (ctx.isIdle()) fastResponses.clear(); });
  pi.on("before_agent_start", () => { fastResponses.clear(); });
  pi.on("agent_settled", () => { fastResponses.clear(); });
  pi.on("session_shutdown", () => { fastResponses.clear(); });
  pi.on("provider_stream_event", (event, ctx) => {
    if (event.api !== "openai-codex-responses" || !isObject(event.data)) return;
    if (event.data.type !== "response.completed" && event.data.type !== "response.done" && event.data.type !== "response.incomplete") return;
    const response = event.data.response;
    if (!isObject(response) || typeof response.id !== "string" || response.service_tier !== "fast") return;
    const model = ctx.model?.provider === event.provider && ctx.model.id === event.model
      ? ctx.model : ctx.modelRegistry.find(event.provider, event.model);
    if (model?.api === event.api) fastResponses.set(response.id, model);
  });
  pi.on("message_end", (event) => {
    const message = event.message;
    if (message.role !== "assistant" || !message.responseId) return;
    const model = fastResponses.get(message.responseId);
    fastResponses.delete(message.responseId);
    if (!model || message.api !== model.api || message.provider !== model.provider || message.model !== model.id) return;

    const usage = structuredClone(message.usage);
    calculateCost(model, usage);
    const costKeys = ["input", "output", "cacheRead", "cacheWrite", "total"] as const;
    // Preserve pricing already applied upstream or by another extension.
    if (costKeys.some(key => usage.cost[key] !== message.usage.cost[key])) return;
    const multiplier = model.id === "gpt-5.5" ? 2.5 : 2;
    for (const key of costKeys) usage.cost[key] *= multiplier;
    return { message: { ...message, usage } };
  });
}
