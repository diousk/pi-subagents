import type { ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { Input, Text } from "@earendil-works/pi-tui";
import { loadRoutingPolicy } from "../model-routing.js";
import { parseJevConfig, type RoutingMode } from "../routing-config.js";
import { loadSettings, projectRoutingSettings, type SubagentsSettings } from "../settings.js";

type RoutingSettings = Pick<SubagentsSettings, "routingMode" | "customGuideline" | "jev">;

/** Input handles paste/editing, but its unmasked renderer is never called. */
export async function maskedApiKey(ctx: ExtensionCommandContext): Promise<string | undefined> {
  return ctx.ui.custom<string | undefined>((_tui, _theme, _kb, done) => {
    const input = new Input();
    input.onSubmit = value => done(value.trim());
    input.onEscape = () => done(undefined);
    return {
      render: (width: number) => new Text(`Typesafe API key: ${"*".repeat(input.getValue().length)}\nEnter to save; Esc to cancel`, 0, 0).render(width),
      invalidate: () => {},
      handleInput: (data: string) => input.handleInput(data),
    };
  });
}

/** Simple mode, guideline, models and credentials; saves project overrides. */
export async function showRoutingMenu(ctx: ExtensionCommandContext): Promise<RoutingSettings | undefined> {
  const settings = loadSettings(ctx.cwd);
  const local = projectRoutingSettings(ctx.cwd);
  const policy = loadRoutingPolicy(ctx.cwd);
  const labels = { agents: "Custom agents", guideline: "Custom guideline", jev: "Jev", baseline: "Existing model" };
  const inactive = policy.source === "agents" ? "Guideline and Jev are inactive while custom agents exist."
    : policy.source === "guideline" ? "Jev is inactive while a custom guideline is configured." : "";
  const note = policy.mode === "off" ? "No routing guidance or Jev requests."
    : policy.mode === "shadow" ? `Observe Jev; actual choice: ${labels[policy.source]}. Jev calls may incur charges.`
    : policy.mode === "jev" ? `Jev first; fallback: ${labels[policy.fallbackSource ?? "baseline"]}.${policy.jev ? "" : " Configure Jev models and credentials to use Jev."}`
    : policy.diagnostic ?? inactive;
  const choice = await ctx.ui.select(`Model routing: ${policy.mode} — ${note || labels[policy.source]}`, [
    "Routing mode", "Custom guideline path", "Jev models and descriptions", "Typesafe API key", "Use Pi/environment credentials", "Disable custom guideline", "Disable Jev", "Back",
  ]);
  if (!choice || choice === "Back") return;
  if (choice === "Routing mode") {
    const modes: { mode: RoutingMode; label: string }[] = [
      { mode: "auto", label: "auto — default priority: agents, guideline, Jev, existing model" },
      { mode: "shadow", label: "shadow — record Jev suggestions; keep the default model (may incur charges)" },
      { mode: "jev", label: "jev — Jev first; default-priority fallback on uncertainty or failure" },
      { mode: "off", label: "off — no routing guidance or Jev requests" },
    ];
    const selected = await ctx.ui.select(`Routing mode (current: ${policy.mode})`, modes.map(entry => entry.label));
    const mode = modes.find(entry => entry.label === selected)?.mode;
    return mode ? { routingMode: mode } : undefined;
  }
  if (choice === "Disable custom guideline") return { customGuideline: false };
  if (choice === "Disable Jev") return { jev: false };
  if (choice === "Custom guideline path") {
    const path = await ctx.ui.input("Routing guideline Markdown path", typeof settings.customGuideline === "string" ? settings.customGuideline : "~/.pi/agent/agents/custom-route.md");
    return path?.trim() ? { customGuideline: path.trim() } : undefined;
  }
  const models = settings.jev ? settings.jev.models.map(entry => ({ ...entry })) : [];
  // Creating a project block must not silently copy a global credential.
  const localKey = local.jev ? local.jev.TYPESAFE_API_KEY : undefined;
  if (choice === "Use Pi/environment credentials") return models.length ? { jev: { models } } : undefined;
  if (choice === "Typesafe API key") {
    if (!models.length) { ctx.ui.notify("Configure Jev models first.", "info"); return; }
    const key = await maskedApiKey(ctx);
    if (!key) return;
    try { return { jev: parseJevConfig({ models, TYPESAFE_API_KEY: key }) }; }
    catch (err) { ctx.ui.notify(err instanceof Error ? err.message : "Invalid Jev settings", "warning"); return; }
  }
  for (;;) {
    const options = [...models.map((entry, index) => `${index + 1}. ${entry.model}`), "Add model", "Save", "Cancel"];
    const action = await ctx.ui.select("Jev models — describe which tasks each model should handle", options);
    if (!action || action === "Cancel") return;
    if (action === "Save") {
      try { return { jev: parseJevConfig({ models, ...(localKey ? { TYPESAFE_API_KEY: localKey } : {}) }) }; }
      catch (err) { ctx.ui.notify(err instanceof Error ? err.message : "Invalid Jev models", "warning"); continue; }
    }
    const index = options.indexOf(action);
    const current = action === "Add model" ? undefined : models[index];
    if (current) {
      const edit = await ctx.ui.select(current.model, ["Edit", "Remove", "Back"]);
      if (edit === "Remove") { models.splice(index, 1); continue; }
      if (edit !== "Edit") continue;
    }
    const model = await ctx.ui.input("Exact provider/model-id", current?.model);
    if (!model) continue;
    const description = await ctx.ui.editor("Tasks this model is suitable for", current?.description ?? "");
    if (!description?.trim()) continue;
    const entry = { model: model.trim(), description: description.trim() };
    if (current) models[index] = entry;
    else models.push(entry);
  }
}
