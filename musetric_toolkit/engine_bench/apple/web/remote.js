(() => {
  if (window.__remote) {
    return;
  }
  window.__remote = true;
  const client = crypto.randomUUID?.() ?? String(Math.random()).slice(2);
  const describe = (error) =>
    error && typeof error === 'object' ? `${error.name ?? 'Error'}: ${error.message ?? ''}\n${error.stack ?? ''}` : String(error);
  const post = (path, body) => fetch(path, { method: 'POST', body: JSON.stringify(body) }).catch(() => {});
  addEventListener('error', (event) => post('/__remote/log', { client, kind: 'error', text: describe(event.error ?? event.message) }));
  addEventListener('unhandledrejection', (event) => post('/__remote/log', { client, kind: 'rejection', text: describe(event.reason) }));
  const consoleError = console.error.bind(console);
  console.error = (...args) => {
    post('/__remote/log', { client, kind: 'console.error', text: args.map(String).join(' ') });
    consoleError(...args);
  };
  const run = async (command) => {
    let text;
    try {
      const fn = (0, eval)(`(${command.fn})`);
      const value = await fn(command.arg);
      text = JSON.stringify({ id: command.id, value });
    } catch (error) {
      text = JSON.stringify({ id: command.id, error: describe(error) });
    }
    await fetch('/__remote/result', { method: 'POST', body: text });
  };
  const page = encodeURIComponent(location.pathname + location.search);
  (async () => {
    for (;;) {
      try {
        const response = await fetch(`/__remote/next?client=${client}&page=${page}`, { cache: 'no-store' });
        const command = await response.json();
        if (command) {
          run(command);
        }
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  })();
})();

(() => {
  if (typeof AudioContext !== 'function' || AudioContext.prototype.__remoteResume) {
    return;
  }
  const waiting = new Set();
  const resume = AudioContext.prototype.resume;
  AudioContext.prototype.__remoteResume = resume;
  AudioContext.prototype.resume = function () {
    if (this.state === 'suspended') {
      waiting.add(this);
    }
    return resume.call(this);
  };
  const activate = () => {
    for (const context of waiting) {
      if (context.state === 'suspended') {
        resume.call(context);
      }
      if (context.state !== 'suspended') {
        waiting.delete(context);
      }
    }
  };
  for (const type of ['touchend', 'pointerup', 'click', 'keydown']) {
    addEventListener(type, activate, true);
  }
  window.__remoteSuspended = () => [...waiting].filter((context) => context.state === 'suspended').length;
})();
