import { webkit } from 'playwright'; import { startServer } from './server.mjs';
const { server, port } = await startServer();
const b = await webkit.launch({ headless: true }); const p = await b.newPage(); await p.goto(`http://127.0.0.1:${port}/web/index.html`);
const r = await p.evaluate(async () => ({ ua: navigator.userAgent, win: { AudioDecoder: typeof AudioDecoder, VideoDecoder: typeof VideoDecoder, EncodedAudioChunk: typeof EncodedAudioChunk, AudioData: typeof AudioData, decodeAudioData: typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined' } }));
const w = await p.evaluate(() => new Promise((res) => { const w = new Worker(URL.createObjectURL(new Blob(['postMessage({AudioDecoder:typeof AudioDecoder,VideoDecoder:typeof VideoDecoder,EncodedAudioChunk: typeof EncodedAudioChunk})'])) ); w.onmessage = (e) => res(e.data); }));
console.log(JSON.stringify({ r, w }, null, 1)); await b.close(); server.close(); process.exit(0);
