import type { ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { matchesKey, Text } from "@earendil-works/pi-tui";
import { eligibleModels, loadRoutingPolicy, routingCandidates } from "../model-routing.js";
import { isScopeModelsEnabled } from "../model-scope.js";
import { loadRoutingSettings, projectRoutingSettings } from "../settings.js";
import type { AgentRecord } from "../types.js";

/** Read current configuration and retained decisions without making a paid classifier request. */
export async function routingStatusText(ctx: ExtensionCommandContext, records: readonly AgentRecord[]): Promise<string> {
  const { settings, guidelineFile } = loadRoutingSettings(ctx.cwd);
  const local = projectRoutingSettings(ctx.cwd);
  const policy = loadRoutingPolicy(ctx.cwd);
  const config = settings.jev === false ? false : policy.jev ?? settings.jev;
  const apiKey = config ? config.TYPESAFE_API_KEY : undefined;
  const available = eligibleModels(ctx);
  const labels = { agents: "Custom agents", guideline: "Custom guideline", jev: "Jev", baseline: "Existing model" };
  const lines = [
    `Directory: ${ctx.cwd}`,
    `Mode: ${policy.mode} (${local.routingMode !== undefined ? "project" : settings.routingMode !== undefined ? "global" : "default"})`,
    `Routing source: ${labels[policy.source]}`,
    `Fallback source: ${labels[policy.fallbackSource ?? (policy.source === "jev" ? "baseline" : policy.source)]}`,
    `Parent model: ${ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "Pi default"}`,
    `Custom agents: ${policy.agents?.map(agent => agent.name).join(", ") || "none active in this policy"}`,
    `Guideline: ${guidelineFile ?? "not configured"}${policy.source !== "guideline" && policy.fallbackSource !== "guideline" ? " (inactive)" : ""}`,
    `Jev config: ${config ? "valid" : config === false ? "disabled or invalid; check subagents.json" : "not configured"} (${local.jev !== undefined ? "project" : settings.jev !== undefined ? "global" : "default"})`,
    `Scope: Pi scoped models ${ctx.scopedModels?.length ? "active" : "unrestricted"}; enabledModels filter ${isScopeModelsEnabled() ? "on" : "off"}`,
  ];
  if (policy.diagnostic) lines.push(`Diagnostic: ${policy.diagnostic}`);
  const classifier = ctx.modelRegistry.findOfType("classifier", "typesafe", "jev-latest");
  lines.push(`Jev classifier: ${classifier ? "available" : "unavailable in Pi registry"}`);
  if (apiKey) {
    lines.push("Credentials: configured key present (acceptance checked on next route)");
  } else {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const availableCredentials = await Promise.race([
        ctx.modelRegistry.getAvailableOfType("classifier", "typesafe", { signal: controller.signal }),
        new Promise<undefined>(resolve => { timer = setTimeout(() => { controller.abort(); resolve(undefined); }, 2000); }),
      ]);
      lines.push(`Credentials: ${availableCredentials === undefined ? "check timed out" : availableCredentials.some(model => model.id === "jev-latest") ? "Pi/environment credentials available (acceptance checked on next route)" : "missing Pi/environment credentials"}`);
    } catch {
      lines.push("Credentials: availability check failed");
    } finally { clearTimeout(timer); }
  }
  lines.push("", "Configured profiles:");
  if (config) {
    for (const profile of config.models) {
      lines.push(`${profile.model} | thinking: ${profile.thinkingLevel} | ${available.has(profile.model) ? "eligible" : "unavailable or excluded by scope"}`, `  ${profile.description}`);
    }
  } else lines.push("None. Every profile requires model, description and thinkingLevel.");
  if (policy.mode === "jev" || policy.mode === "shadow") {
    const candidates = routingCandidates(ctx, policy, ctx.model);
    lines.push("", `Combined eligible profiles: ${candidates.length} (maximum 254)`);
    for (const agent of policy.agentProfiles ?? []) {
      const candidate = candidates.find(entry => entry.agentConfig?.name === agent.name);
      lines.push(`Agent: ${agent.name} | ${candidate ? `${candidate.model} | thinking: ${candidate.thinkingLevel ?? "inherited"} | eligible` : "unavailable or excluded by scope"}`, `  ${agent.description}`);
    }
  }
  lines.push("", "Recent retained agents (latest 10; includes nested/workflow agents):");
  const recent = [...records].filter(record => record.routing).sort((a, b) => b.startedAt - a.startedAt).slice(0, 10);
  if (!recent.length) lines.push("No routing decisions yet. Start a fresh subagent; resumed agents do not reroute.");
  for (const record of recent) {
    const route = record.routing!;
    lines.push(`${new Date(record.startedAt).toISOString()} | ${record.id} | ${record.status}`,
      `  Mode: ${route.mode} | Source: ${route.source} | Result: ${route.code}`,
      `  ${route.reason}`);
    if (route.model) lines.push(`  Selected: ${route.model} | requested thinking: ${route.thinkingLevel ?? "inherited"}`);
    if (route.agent) lines.push(`  Selected agent: ${route.agent}`);
    if (route.suggestedAgent) lines.push(`  Shadow agent suggestion: ${route.suggestedAgent}`);
    if (route.suggestedModel) lines.push(`  Shadow suggestion: ${route.suggestedModel} | thinking: ${route.suggestedThinkingLevel ?? "inherited"}`);
    if (record.invocation?.modelId) lines.push(`  Actual: ${record.invocation.modelId} | effective thinking: ${record.invocation.thinking ?? "unknown"}`);
    if (route.confidence !== undefined) lines.push(`  Confidence: ${(route.confidence * 100).toFixed(1)}%`);
    if (record.routingUsage) lines.push(`  Classifier tokens: ${record.routingUsage.totalTokens} | ${route.unpriced ? "price unavailable" : `reported cost: $${record.routingUsage.cost.total}`}`);
  }
  lines.push("", "Snapshot only. Refresh to reload settings and retained decisions. Opening this page does not classify a task.");
  const text = lines.join("\n");
  return apiKey ? text.replaceAll(apiKey, "[redacted]") : text;
}

export async function showRoutingStatus(ctx: ExtensionCommandContext, records: () => AgentRecord[]): Promise<void> {
  let refresh = true;
  while (refresh) {
    const content = await routingStatusText(ctx, records());
    refresh = await ctx.ui.custom<boolean>((tui, _theme, _kb, done) => {
      let offset = 0;
      let maxOffset = 0;
      return {
        render(width: number) {
          const rows = new Text(content, 0, 0).render(width);
          const height = Math.max(1, tui.terminal.rows - 6);
          maxOffset = Math.max(0, rows.length - height);
          offset = Math.min(offset, maxOffset);
          return [...new Text("Routing runtime status | Up/Down: scroll | r: refresh | Esc: back", 0, 0).render(width), ...rows.slice(offset, offset + height)];
        },
        invalidate() {},
        handleInput(data: string) {
          if (matchesKey(data, "escape") || matchesKey(data, "q")) { done(false); return; }
          if (matchesKey(data, "r")) { done(true); return; }
          if (matchesKey(data, "up")) offset = Math.max(0, offset - 1);
          if (matchesKey(data, "down")) offset = Math.min(maxOffset, offset + 1);
          tui.requestRender();
        },
      };
    });
  }
}
