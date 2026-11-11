# ios

`server.mjs` is an HTTPS static server with range requests for a phone on the
LAN. `www/index.html` reports what the browser has (secure context,
`crossOriginIsolated`, `SharedArrayBuffer`, AudioWorklet, AudioDecoder with AAC
and FLAC, `playbackStats`, render size hint) and posts it to `/report`.

```sh
mkdir certs && cd certs
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 7 -keyout key.pem -out cert.pem -subj "/CN=lab" -addext "subjectAltName=IP:<lan-ip>"
cd .. && node server.mjs [rootDir] [port]
```

- `rootDir` defaults to `www`; give a stand folder to serve it (the server maps
  `/mediabunny.mjs` and `/vendor/SignalsmithStretch.mjs` to the stand's
  `node_modules`). `CERT_DIR` overrides `certs`.
- Reports are appended to `reports.jsonl`. The certificate is made per machine
  and never committed.
- The procedure on a phone: [Running on an iPhone from a Mac](../readme.md#running-on-an-iphone-from-a-mac).
