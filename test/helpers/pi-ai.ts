/** Shared transcript replay and faux-provider helpers for Pi 0.99.1. */
import { getCurrentSystemPrompt, getCurrentTools, type TranscriptContext } from "@earendil-works/pi-ai";

export function currentTools(context: TranscriptContext) {
  return getCurrentTools(context.messages);
}

export function currentSystemPrompt(context: TranscriptContext): string {
  return getCurrentSystemPrompt(context.messages);
}

export { getModel, registerFauxProvider, streamSimple } from "@earendil-works/pi-ai/compat";
