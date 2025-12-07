import type { TangramApi } from "@open-aviation/tangram-core/api";
import AircraftLanding from "./AircraftLanding.vue";

export function install(api: TangramApi) {
  api.ui.registerWidget("aircraft-align-widget", "MapOverlay", AircraftLanding);
}
