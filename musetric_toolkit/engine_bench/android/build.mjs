import { build } from 'esbuild';
import path from 'node:path';
import { createMusetricAlias } from '../shared/musetric-alias.mjs';

const here = import.meta.dirname;

await build({
  entryPoints: [path.join(here, 'src/stepper-worker.ts')],
  outfile: path.join(here, 'dist/stepper-worker.js'),
  bundle: true,
  platform: 'browser',
  format: 'esm',
  target: 'es2022',
  sourcemap: false,
  keepNames: true,
  minify: false,
  plugins: [createMusetricAlias(here)],
  define: {
    'import.meta.env': '{"DEV":false,"MODE":"production","PROD":true}',
    'process.env.TRACK_BYTES': '"0"',
  },
  nodePaths: [path.join(here, 'node_modules')],
  logLevel: 'warning',
});
console.log('built');
