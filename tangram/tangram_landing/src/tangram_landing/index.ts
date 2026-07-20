import type { PluginContext } from "@open-aviation/tangram-core/api";
import AircraftLanding from "./AircraftLanding.vue";

export function install(ctx: PluginContext) {
  ctx.api.ui.registerWidget(
    "aircraft-align-widget",
    "MapOverlay",
    AircraftLanding,
    {
      pluginId: ctx.id,
    },
  );
}
