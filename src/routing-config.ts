export type RoutingMode = "auto" | "shadow" | "jev" | "off";

export interface JevConfig {
  TYPESAFE_API_KEY?: string;
  models: { model: string; description: string }[];
}

/** Validate the whole block; never merge candidate lists or credentials. */
export function parseJevConfig(raw: unknown): JevConfig | false {
  if (raw === false) return false;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("jev must be an object or false");
  const value = raw as Record<string, unknown>;
  if (Object.keys(value).some(key => key !== "models" && key !== "TYPESAFE_API_KEY")) {
    throw new Error("jev accepts only TYPESAFE_API_KEY and models");
  }
  if (value.TYPESAFE_API_KEY !== undefined &&
    (typeof value.TYPESAFE_API_KEY !== "string" || !value.TYPESAFE_API_KEY || /\s|[\x00-\x1f\x7f]/.test(value.TYPESAFE_API_KEY))) {
    throw new Error("TYPESAFE_API_KEY must be a nonempty token without whitespace or control characters");
  }
  if (!Array.isArray(value.models) || value.models.length < 1 || value.models.length > 254) {
    throw new Error("jev.models must contain 1–254 models");
  }
  const seen = new Set<string>();
  const models = value.models.map((entry: unknown) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error("Each Jev model needs model and description");
    const candidate = entry as Record<string, unknown>;
    if (Object.keys(candidate).some(key => key !== "model" && key !== "description") ||
      typeof candidate.model !== "string" || !/^[^\s/]+\/\S+$/.test(candidate.model) ||
      typeof candidate.description !== "string" || !candidate.description.trim() || candidate.description.length > 4000) {
      throw new Error("Each Jev model needs an exact provider/model-id and a description of 1–4000 characters");
    }
    if (seen.has(candidate.model)) throw new Error("jev.models contains duplicate models");
    seen.add(candidate.model);
    return { model: candidate.model, description: candidate.description.trim() };
  });
  return { models, ...(typeof value.TYPESAFE_API_KEY === "string" ? { TYPESAFE_API_KEY: value.TYPESAFE_API_KEY } : {}) };
}
