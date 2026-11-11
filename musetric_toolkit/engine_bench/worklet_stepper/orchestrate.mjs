import { spawnSync } from 'node:child_process';
const [mode, rounds = '3', ids = 'S0,S1,S2,S3,S4,S4b,S5,S6,S7a,S7b'] = process.argv.slice(2);
const flags = mode === 'alloc' ? ['--expose-gc', '--min-semi-space-size=128'] : ['--expose-gc'];
for (let r = 1; r <= Number(rounds); r += 1) {
  for (const id of ids.split(',')) {
    const res = spawnSync('node', [...flags, 'dist/run.mjs', mode, id, '--seconds', process.env.SECONDS ?? '60'], {
      env: { ...process.env, TAG: `${id}_r${r}`, TRACK_BYTES: mode === 'alloc' ? '1' : '0', GC_EVERY: '1000000000' },
      encoding: 'utf8',
    });
    process.stdout.write(res.stdout.split('\n').filter((l) => l.startsWith(id)).join('\n') + '\n');
    if (res.status !== 0) console.log(res.stderr.slice(0, 500));
  }
}
