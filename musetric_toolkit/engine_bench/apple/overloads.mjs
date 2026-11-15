import { execFileSync } from 'node:child_process';
import { isPhone } from './device.mjs';

const macLines = (from, to) => {
  const stamp = (t) => new Date(t).toISOString().replace('T', ' ').replace(/\.\d+Z$/, '+0000');
  const text = execFileSync(
    '/usr/bin/log',
    [
      'show',
      '--start', stamp(from),
      '--end', stamp(to + 1000),
      '--predicate', 'process == "coreaudiod" AND (eventMessage CONTAINS "Audio IO Overload" OR eventMessage CONTAINS "overload_type")',
      '--style', 'compact',
    ],
    { encoding: 'utf8', maxBuffer: 1 << 26 },
  );
  return text.split('\n');
};

export class Overloads {
  constructor(target) {
    this.target = target;
  }

  count(from, to) {
    if (isPhone(this.target)) {
      return null;
    }
    const lines = macLines(from, to);
    const reports = lines.filter((l) => l.includes('Audio IO Overload')).length;
    const clients = {};
    for (const l of lines) {
      const m = /HostApplicationDisplayID": Optional\(([^)]+)\)/.exec(l);
      if (m) {
        clients[m[1]] = (clients[m[1]] ?? 0) + 1;
      }
    }
    return { reports, clients };
  }

  stop() {}
}
