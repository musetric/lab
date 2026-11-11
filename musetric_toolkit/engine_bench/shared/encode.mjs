import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const [dataDirArg, ...flags] = process.argv.slice(2);
if (!dataDirArg) {
  console.error('usage: node encode.mjs <dataDir> [--flac]');
  process.exit(2);
}
const dataDir = path.resolve(dataDirArg);
const withFlac = flags.includes('--flac');
const mediaBin = process.env.MEDIA_BIN;
const musetricRoot = process.env.MUSETRIC_ROOT;
if (!mediaBin && !musetricRoot) {
  console.error('set MEDIA_BIN (a built media example binary) or MUSETRIC_ROOT (a musetric checkout)');
  process.exit(2);
}

const runMedia = (args) => {
  const [command, cwd] = mediaBin
    ? [[mediaBin, ...args], process.cwd()]
    : [['cargo', 'run', '--locked', '--quiet', '-p', 'musetric-media', '--example', 'media', '--', ...args], path.join(musetricRoot, 'packages', 'server')];
  const result = spawnSync(command[0], command.slice(1), { cwd, stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`media ${args[0]} failed with status ${result.status}`);
  }
};

for (const file of fs.readdirSync(dataDir).filter((name) => name.endsWith('.wav'))) {
  const stem = path.join(dataDir, file.slice(0, -4));
  runMedia(['fmp4', '--from', `${stem}.wav`, '--to', `${stem}.mp4`, '--sample-rate', '48000']);
  if (withFlac) {
    runMedia(['flac', '--from', `${stem}.wav`, '--to', `${stem}.flac`, '--sample-rate', '48000']);
  }
  console.log('encoded', file);
}
