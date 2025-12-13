import { globalIgnores } from "eslint/config";
import {
  defineConfigWithVueTs,
  vueTsConfigs
} from "@vue/eslint-config-typescript";
import pluginVue from "eslint-plugin-vue";
import skipFormatting from "@vue/eslint-config-prettier/skip-formatting";

export default defineConfigWithVueTs(
  {
    name: "app/files-to-lint",
    files: ["**/*.{ts,mts,js,mjs,vue}"]
  },
  globalIgnores(["**/dist/**", "**/dist-frontend/**", ".venv"]),
  pluginVue.configs["flat/recommended"],
  vueTsConfigs.recommended,
  skipFormatting  // use prettier for code formatting, eslint for code quality
);
