import { defineConfig } from "vite";
// @ts-expect-error TODO upstream does not have a vite-plugin-tangram.d.ts
// that exports the types for mjs
import { tangramPlugin } from "@open-aviation/tangram-core/vite-plugin";

export default defineConfig({
  plugins: [tangramPlugin()],
});
