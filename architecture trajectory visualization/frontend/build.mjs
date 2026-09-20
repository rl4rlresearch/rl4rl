import { build } from "esbuild";
await build({
  entryPoints: ["src/viewer.js"],
  bundle: true,
  format: "esm",
  target: "es2022",
  outdir: "dist",
  minify: true,
  sourcemap: true,
  legalComments: "linked",
});
