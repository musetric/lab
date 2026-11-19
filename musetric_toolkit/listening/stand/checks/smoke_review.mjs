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
await page.screenshot({ path: `${shots}/review-1-open.png` });

const form = await page.evaluate(() => ({
  stem: state.stem,
  loop: state.loop,
  toolHidden: $("tool").hidden,
  cards: $("cards").children.length,
  credit: $("credit").textContent,
  title: $("songTitle").textContent,
  stems: [...document.querySelectorAll("#stems button")].map((b) => b.textContent),
  scores: document.querySelectorAll("#present [data-score]").length,
  drop: document.querySelectorAll("#present [data-drop]").length,
}));
check("the form asks for a score or a drop", form.stem === "mix" && form.loop && form.toolHidden && form.cards === 0 && form.credit === "" && form.title === messages.review && form.stems.length === 4 && form.scores === 11 && form.drop === 1, JSON.stringify(form));

const firstClip = await page.evaluate(() => clip().id);
await page.keyboard.press("Enter");
await page.waitForTimeout(100);
const refused = await page.evaluate(() => ({ clip: clip().id, done: label().done, hint: $("doneHint").textContent }));
check("Enter refuses without an answer", refused.clip === firstClip && !refused.done && refused.hint === messages.markScore, JSON.stringify(refused));

await page.keyboard.press("Digit7");
const seven = await page.evaluate(() => ({ score: label().score, on: [...document.querySelectorAll("#present [data-score].on")].map((b) => b.dataset.score), hint: $("doneHint").textContent }));
check("a digit sets the score", seven.score === 7 && seven.on.join() === "7" && seven.hint === "", JSON.stringify(seven));
await page.keyboard.press("Digit0");
const ten = await page.evaluate(() => label().score);
check("zero sets ten", ten === 10, String(ten));
await page.keyboard.press("Digit0");
const cleared = await page.evaluate(() => label().score);
check("the same digit takes the score back", cleared === null, String(cleared));
await page.locator('#present [data-score="0"]').click();
const zero = await page.evaluate(() => label().score);
check("a click sets zero", zero === 0, String(zero));
await page.keyboard.press("KeyX");
const dropped = await page.evaluate(() => ({ drop: label().drop, on: document.querySelector("[data-drop]").className }));
check("X marks the track for dropping", dropped.drop === true && dropped.on === "on", JSON.stringify(dropped));
await page.keyboard.press("Control+KeyZ");
await page.waitForTimeout(50);
const undone = await page.evaluate(() => ({ drop: label().drop, score: label().score }));
check("undo takes the last answer back", !undone.drop && undone.score === 0, JSON.stringify(undone));
await page.keyboard.press("Digit6");
await page.locator("#comment").fill("check");
await page.keyboard.press("Escape");
await page.waitForTimeout(500);

await page.reload();
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
const saved = await page.evaluate(() => label());
check("the answer survives a reload", saved.score === 6 && !saved.drop && saved.comment === "check" && !saved.done, JSON.stringify({ score: saved.score, drop: saved.drop }));

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

await page.keyboard.press("KeyE");
const moved = await page.evaluate(() => state.stem);
check("a stem hotkey plays a stem", moved === "backing_a", moved);
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
  other: Number(audio.gains.backing_a.gain.value.toFixed(2)),
}));
check("Enter plays the next track from the mix", next.clip !== firstClip && next.stem === "mix" && next.playing && next.at > 0 && next.gain === 1 && next.other === 0, JSON.stringify(next));
const stored = await page.evaluate(async (id) => (await (await fetch("/api/round")).json()).labels[id], firstClip);
check("the finished track is stored as done", stored.done === true && stored.score === 6 && !stored.drop, JSON.stringify({ done: stored.done, score: stored.score }));
await page.keyboard.press("KeyX");
await page.keyboard.press("Enter");
await page.waitForTimeout(500);
await page.keyboard.press("Space");
check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));

const post = (body) => page.evaluate((text) => fetch("/api/save", { method: "POST", body: text }).then((r) => r.status), JSON.stringify(body));
const empty = await page.evaluate(() => emptyLabel());
const server = {
  noAnswer: await post({ clip: firstClip, data: { ...empty, done: true } }),
  noDrop: await post({ clip: firstClip, data: { ...empty, drop: false, done: true } }),
  overTen: await post({ clip: firstClip, data: { ...empty, score: 11 } }),
  below: await post({ clip: firstClip, data: { ...empty, score: -1 } }),
  text: await post({ clip: firstClip, data: { ...empty, score: "5" } }),
  fraction: await post({ clip: firstClip, data: { ...empty, score: 5.5 } }),
  flag: await post({ clip: firstClip, data: { ...empty, score: true } }),
  wrongDrop: await post({ clip: firstClip, data: { ...empty, drop: "yes" } }),
};
check("the server refuses a track without a valid answer", Object.values(server).every((code) => code === 400), JSON.stringify(server));
const taken = {
  zero: await post({ clip: firstClip, data: { ...empty, score: 0, done: true } }),
  drop: await post({ clip: firstClip, data: { ...empty, drop: true, done: true } }),
  both: await post({ clip: firstClip, data: { ...empty, score: 6, drop: true, done: true } }),
};
check("the server takes a score or a drop without ticks", Object.values(taken).every((code) => code === 200), JSON.stringify(taken));

const hidden = {
  key: await page.evaluate(() => fetch("/key.json").then((r) => r.status)),
  selection: await page.evaluate(() => fetch("/selection.json").then((r) => r.status)),
};
check("the key and the selection are not served", hidden.key === 404 && hidden.selection === 404, JSON.stringify(hidden));

await page.locator("#side").evaluate((node) => { node.scrollTop = 0; });
await page.screenshot({ path: `${shots}/review-2-next.png` });
await browser.close();
console.log(failures.length ? `FAILED: ${failures.join(", ")}` : "all passed");
process.exit(failures.length ? 1 : 0);
