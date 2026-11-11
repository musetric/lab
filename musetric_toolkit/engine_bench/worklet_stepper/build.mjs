import { build } from 'esbuild';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMusetricAlias } from '../shared/musetric-alias.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const musetricAlias = createMusetricAlias(here);

const entries = process.argv.slice(2);
for (const entry of entries.length ? entries : ['run', 'loops']) {
  await build({
    entryPoints: [path.join(here, 'src', `${entry}.ts`)],
    outfile: path.join(here, 'dist', `${entry}.mjs`),
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node24',
    sourcemap: true,
    keepNames: true,
    minify: false,
    plugins: [musetricAlias],
    define: { 'import.meta.env': '{"DEV":false,"MODE":"production","PROD":true}' },
    banner: {
      js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);",
    },
    logLevel: 'warning',
  });
}
console.log('built');
