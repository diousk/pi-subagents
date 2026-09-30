/**
 * faux-model-backend.ts — the model/auth plumbing a faux-provider session needs,
 * in one place.
 *
 * `registerFauxProvider` scripts the *responses*, but a session still has to get
 * past model lookup and auth before it streams anything, and where that check
 * lives in ModelRuntime: auth via `getAuth()`/`hasConfiguredAuth()`, and the
 * turn itself streams through `modelRuntime.streamSimple`.
 *
 * Structural fakes (not real instances) keep the suites hermetic —
 * no auth.json, no network, no local login state.
 */
import type { Model } from "@earendil-works/pi-ai";
import type { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { streamSimple } from "./pi-ai.js";

/** Runtime option for `createAgentSession`, for the given faux model. */
export function fauxModelBackend(model: Model<string>): {
  modelRuntime: ModelRuntime;
} {
  return {
    modelRuntime: {
      getModel: () => model,
      getPhysicalModel: () => model,
      getModels: () => [model],
      getProvider: () => undefined,
      getProviders: () => [],
      getAvailable: async () => [model],
      getAvailableSnapshot: () => [model],
      getError: () => undefined,
      hasConfiguredAuth: () => true,
      checkAuth: async () => ({ ok: true }),
      isUsingOAuth: () => false,
      isUsingSubscription: () => false,
      // Shape mirrors ModelRuntime.getAuth: the session reads `auth.apiKey` /
      // `auth.headers` and throws "No API key found" when both are absent.
      getAuth: async () => ({ auth: { apiKey: "faux", headers: {} } }),
      getProviderAuthStatus: () => "configured",
      getCompatibilityRequestConfig: () => ({}),
      getRegisteredProviderIds: () => [],
      getRegisteredProviderConfig: () => undefined,
      getRegisteredNativeProvider: () => undefined,
      registerProvider: () => {},
      registerNativeProvider: () => {},
      unregisterProvider: () => {},
      refresh: async () => ({}),
      // The faux provider registers itself in pi-ai's global api-provider
      // registry, so compat's dispatcher reaches it by `model.api`.
      stream: streamSimple,
      streamSimple,
    } as unknown as ModelRuntime,
  };
}
