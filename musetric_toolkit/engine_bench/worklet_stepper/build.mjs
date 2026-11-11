import { build } from 'esbuild';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const musetricRoot = process.env.MUSETRIC_ROOT;
if (!musetricRoot) {
  throw new Error('set MUSETRIC_ROOT to a musetric checkout');
}
const pkgs = path.join(path.resolve(musetricRoot), 'packages');

const musetricAlias = {
  name: 'musetric-alias',
  setup(b) {
    b.onResolve({ filter: /^musetric-root\// }, (args) =>
      b.resolve('./' + args.path.slice('musetric-root/'.length), { resolveDir: path.resolve(musetricRoot), kind: args.kind }),
    );
    b.onResolve({ filter: /^@musetric\/[^/]+(\/.*)?$/ }, (args) => {
      const [, pkg, sub] = args.path.match(/^@musetric\/([^/]+)(?:\/(.*))?$/);
      const manifest = JSON.parse(readFileSync(path.join(pkgs, pkg, 'package.json'), 'utf8'));
      const exact = sub ? `./${sub}` : '.';
      let target;
      for (const [key, value] of Object.entries(manifest.exports)) {
        const mono = value?.monorepo;
        if (!mono) continue;
        if (key === exact) {
          target = mono;
          break;
        }
        if (key.includes('*')) {
          const [pre, post] = key.split('*');
          if (exact.startsWith(pre) && exact.endsWith(post)) {
            target = mono.replace('*', exact.slice(pre.length, exact.length - post.length));
            break;
          }
        }
      }
      if (!target) throw new Error(`no monorepo export for ${args.path}`);
      return { path: path.join(pkgs, pkg, target) };
    });
    b.onResolve({ filter: /\?raw$/ }, (args) => {
      const name = args.path.replace(/\?raw$/, '');
      const file = path.join(here, 'node_modules', name, 'SignalsmithStretch.mjs');
      if (!existsSync(file)) throw new Error(`raw: ${file} missing`);
      return { path: file, namespace: 'raw' };
    });
    b.onLoad({ filter: /.*/, namespace: 'raw' }, (args) => ({
      contents: `export default ${JSON.stringify(readFileSync(args.path, 'utf8'))};`,
      loader: 'js',
    }));
  },
};

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
