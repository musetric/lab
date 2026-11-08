import { createRequire } from "node:module";
import { join } from "node:path";

const [base, shots, playwrightHome] = process.argv.slice(2);
const require = createRequire(join(playwrightHome, "package.json"));
const { chromium } = require("playwright");

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name} ${detail}`);
  if (!ok) failures.push(name);
};

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
const errors = [];
const auditions = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("request", (r) => { if (r.url().includes("/api/audition")) auditions.push(r.url()); });

await page.goto(base);
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
const messages = await page.evaluate(() => fetch("/messages.ru.json").then((r) => r.json()));
check("every message is filled", await page.evaluate(() => !/\{\w+\}/.test(document.body.innerText)));
await page.waitForTimeout(300);
await page.screenshot({ path: `${shots}/better-1-open.png` });

const form = await page.evaluate(() => ({
  stem: state.stem,
  loop: state.loop,
  toolHidden: $("tool").hidden,
  cards: $("cards").children.length,
  credit: $("credit").textContent,
  title: $("songTitle").textContent,
  stems: [...document.querySelectorAll("#stems button")].length,
  choices: [...document.querySelectorAll("#present [data-choice]")].map((b) => b.dataset.choice).join(),
}));
check("the form asks which side is better", form.stem === "mix" && form.loop && form.toolHidden && form.cards === 0 && form.credit === "" && form.title === messages.betterTitle && form.stems === 7 && form.choices === "a,b,same", JSON.stringify(form));

const firstClip = await page.evaluate(() => clip().id);
await page.keyboard.press("Enter");
await page.waitForTimeout(100);
const refused = await page.evaluate(() => ({ clip: clip().id, done: label().done, hint: $("doneHint").textContent }));
check("Enter refuses without an answer", refused.clip === firstClip && !refused.done && refused.hint === messages.markBetter, JSON.stringify(refused));

await page.keyboard.press("Digit1");
const a = await page.evaluate(() => ({ better: label().better, on: [...document.querySelectorAll("#present [data-choice].on")].map((b) => b.dataset.choice).join(), hint: $("doneHint").textContent }));
check("1 says A is better", a.better === "a" && a.on === "a" && a.hint === "", JSON.stringify(a));
await page.keyboard.press("Digit1");
const cleared = await page.evaluate(() => label().better);
check("the same key takes the answer back", cleared === null, String(cleared));
await page.keyboard.press("Digit2");
const b = await page.evaluate(() => label().better);
check("2 says B is better", b === "b", String(b));
await page.keyboard.press("Digit4");
const ignored = await page.evaluate(() => label().better);
check("a key past the choices does nothing", ignored === "b", String(ignored));
await page.locator('#present [data-choice="same"]').click();
const same = await page.evaluate(() => label().better);
check("a click says they are the same", same === "same", String(same));
await page.keyboard.press("Control+KeyZ");
await page.waitForTimeout(50);
const undone = await page.evaluate(() => label().better);
check("undo takes the last answer back", undone === "b", String(undone));
await page.locator("#comment").fill("check");
await page.keyboard.press("Escape");
await page.waitForTimeout(500);

await page.reload();
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
const saved = await page.evaluate(() => label());
check("the answer survives a reload", saved.better === "b" && saved.comment === "check" && !saved.done, JSON.stringify({ better: saved.better }));

const box = await page.locator("#canvas").boundingBox();
await page.mouse.move(box.x + 300, box.y + 200);
await page.mouse.down();
await page.mouse.move(box.x + 500, box.y + 300, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(100);
const dragged = await page.evaluate(() => ({ pending: state.pending, regions: label().regions.length, position: state.position }));
check("a drag only moves the cursor", dragged.pending === null && dragged.regions === 0 && dragged.position > 0, JSON.stringify(dragged));
await page.keyboard.press("KeyS");
await page.keyboard.press("KeyD");
await page.keyboard.press("KeyF");
await page.waitForTimeout(100);
check("the outline keys do nothing", auditions.length === 0 && (await page.evaluate(() => state.tool)) === "lasso");

const third = await page.evaluate(() => state.manifest.stems[2].id);
await page.keyboard.press("KeyE");
const moved = await page.evaluate(() => state.stem);
check("a stem hotkey plays a stem of B", moved === third && third.endsWith("_b"), moved);
await page.keyboard.press("Space");
await page.waitForTimeout(300);
await page.evaluate(() => document.activeElement.blur());
await page.keyboard.press("Enter");
await page.waitForFunction((id) => clip().id !== id && window.standReady === clip().id, firstClip, { timeout: 30000 });
await page.waitForTimeout(500);
const next = await page.evaluate(() => ({
  clip: clip().id,
  stem: state.stem,
  playing: state.playing,
  at: currentPosition(),
  gain: Number(audio.gains.mix.gain.value.toFixed(2)),
}));
check("Enter plays the next track from the mix", next.clip !== firstClip && next.stem === "mix" && next.playing && next.at > 0 && next.gain === 1, JSON.stringify(next));
const stored = await page.evaluate(async (id) => (await (await fetch("/api/round")).json()).labels[id], firstClip);
check("the finished track is stored as done", stored.done === true && stored.better === "b", JSON.stringify({ done: stored.done, better: stored.better }));
await page.keyboard.press("Digit3");
await page.keyboard.press("Enter");
await page.waitForTimeout(500);
await page.keyboard.press("Space");
check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));

const post = (body) => page.evaluate((text) => fetch("/api/save", { method: "POST", body: text }).then((r) => r.status), JSON.stringify(body));
const empty = await page.evaluate(() => emptyLabel());
const server = {
  noAnswer: await post({ clip: firstClip, data: { ...empty, done: true } }),
  unknown: await post({ clip: firstClip, data: { ...empty, better: "unknown", done: true } }),
  word: await post({ clip: firstClip, data: { ...empty, better: "c" } }),
  number: await post({ clip: firstClip, data: { ...empty, better: 1 } }),
};
check("the server refuses a track without a valid answer", Object.values(server).every((code) => code === 400), JSON.stringify(server));
const taken = {
  a: await post({ clip: firstClip, data: { ...empty, better: "a", done: true } }),
  same: await post({ clip: firstClip, data: { ...empty, better: "same", done: true } }),
  open: await post({ clip: firstClip, data: { ...empty, better: null } }),
};
check("the server takes A, B or the same without ticks", Object.values(taken).every((code) => code === 200), JSON.stringify(taken));

const hidden = {
  key: await page.evaluate(() => fetch("/key.json").then((r) => r.status)),
  selection: await page.evaluate(() => fetch("/selection.json").then((r) => r.status)),
};
check("the key and the selection are not served", hidden.key === 404 && hidden.selection === 404, JSON.stringify(hidden));

await page.locator("#side").evaluate((node) => { node.scrollTop = 0; });
await page.screenshot({ path: `${shots}/better-2-next.png` });
await browser.close();
console.log(failures.length ? `FAILED: ${failures.join(", ")}` : "all passed");
process.exit(failures.length ? 1 : 0);
