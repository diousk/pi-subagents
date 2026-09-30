import { Type } from "@sinclair/typebox";

export default function (pi) {
  for (const name of ["allowed", "denied"]) {
    pi.registerTool({
      name,
      label: name,
      description: name,
      exposure: "deferred",
      parameters: Type.Object({}),
      async execute() {
        return { content: [{ type: "text", text: `${name} EXECUTED` }] };
      },
    });
  }
  pi.registerTool({
    name: "bridge",
    label: "bridge",
    description: "Call deferred tools through the nested execution API.",
    parameters: Type.Object({}),
    async execute(_id, _args, _signal, _update, ctx) {
      const allowed = await ctx.executeTool("allowed", {});
      const denied = await ctx.executeTool("denied", {});
      return { content: [{ type: "text", text: JSON.stringify({ allowed, denied }) }] };
    },
  });
}
