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
await page.screenshot({ path: `${shots}/present-1-open.png` });

const form = await page.evaluate(() => ({
  stem: state.stem,
  loop: state.loop,
  toolHidden: $("tool").hidden,
  cards: $("cards").children.length,
  credit: $("credit").textContent,
  answers: [...document.querySelectorAll("#present button")].map((b) => b.textContent),
  hints: $("hints").tagName,
}));
check("the form asks only about the song", form.stem === "mix" && form.loop && form.toolHidden && form.cards === 0 && form.credit === "" && form.answers.length === 4, JSON.stringify(form));

const firstClip = await page.evaluate(() => clip().id);
await page.keyboard.press("Enter");
await page.waitForTimeout(100);
const refused = await page.evaluate(() => ({ clip: clip().id, done: label().done, hint: $("doneHint").textContent }));
check("Enter refuses without an answer", refused.clip === firstClip && !refused.done && refused.hint.includes(messages.better.unknown), JSON.stringify(refused));

await page.keyboard.press("Digit1");
await page.keyboard.press("Digit3");
const ticked = await page.evaluate(() => ({ present: label().present, on: [...document.querySelectorAll("#present button.on")].map((b) => b.dataset.present), hint: $("doneHint").textContent }));
check("digits tick the song", ticked.present.join() === "lead,instrumental" && ticked.on.join() === "lead,instrumental" && ticked.hint === "", JSON.stringify(ticked));
await page.keyboard.press("Digit1");
const untick = await page.evaluate(() => label().present);
check("a digit takes a tick back", untick.join() === "instrumental", JSON.stringify(untick));
await page.keyboard.press("Digit4");
const unsure = await page.evaluate(() => ({ present: label().present, unsure: label().unsure, on: document.querySelector("[data-unsure]").className }));
check("unsure clears the ticks", unsure.present.length === 0 && unsure.unsure && unsure.on === "on", JSON.stringify(unsure));
await page.keyboard.press("Digit2");
const leaves = await page.evaluate(() => ({ present: label().present, unsure: label().unsure }));
check("a tick clears unsure", leaves.present.join() === "backing" && !leaves.unsure, JSON.stringify(leaves));
await page.keyboard.press("Control+KeyZ");
await page.waitForTimeout(50);
const undone = await page.evaluate(() => ({ present: label().present, unsure: label().unsure, on: document.querySelector("[data-unsure]").className }));
check("undo takes the last answer back", undone.present.length === 0 && undone.unsure && undone.on === "on", JSON.stringify(undone));
await page.keyboard.press("Digit4");
await page.keyboard.press("Digit3");
await page.locator("#comment").fill("check");
await page.keyboard.press("Escape");
await page.waitForTimeout(500);

await page.reload();
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
const saved = await page.evaluate(() => label());
check("the answer survives a reload", saved.present.join() === "instrumental" && saved.comment === "check" && !saved.done, JSON.stringify(saved.present));

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
check("a stem hotkey still plays a stem", moved === "lead_b", moved);
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
  other: Number(audio.gains.lead_b.gain.value.toFixed(2)),
}));
check("Enter plays the next clip from the mix", next.clip !== firstClip && next.stem === "mix" && next.playing && next.at > 0 && next.gain === 1 && next.other === 0, JSON.stringify(next));
const stored = await page.evaluate(async (id) => (await (await fetch("/api/round")).json()).labels[id], firstClip);
check("the finished clip is stored as done", stored.done === true && stored.present.join() === "instrumental" && stored.stems.lead.better === null, JSON.stringify({ done: stored.done, present: stored.present }));
await page.keyboard.press("Space");
check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));

const post = (body) => page.evaluate((text) => fetch("/api/save", { method: "POST", body: text }).then((r) => r.status), JSON.stringify(body));
const empty = await page.evaluate(() => emptyLabel());
const server = {
  noAnswer: await post({ clip: firstClip, data: { ...empty, done: true } }),
  clearedAnswer: await post({ clip: firstClip, data: { ...empty, present: [], done: true } }),
  wrongUnsure: await post({ clip: firstClip, data: { ...empty, unsure: "maybe" } }),
  wrongPresent: await post({ clip: firstClip, data: { ...empty, present: ["voice"] } }),
};
check("the server refuses a clip without an answer", Object.values(server).every((code) => code === 400), JSON.stringify(server));
const taken = {
  unsure: await post({ clip: firstClip, data: { ...empty, present: [], unsure: true, done: true } }),
  ticks: await post({ clip: firstClip, data: { ...empty, present: ["instrumental"], done: true } }),
};
check("the server takes an answer without comparisons", taken.unsure === 200 && taken.ticks === 200, JSON.stringify(taken));

const hidden = {
  key: await page.evaluate(() => fetch("/key.json").then((r) => r.status)),
  selection: await page.evaluate(() => fetch("/selection.json").then((r) => r.status)),
};
check("the key and the selection are not served", hidden.key === 404 && hidden.selection === 404, JSON.stringify(hidden));

const list = await page.evaluate(() => {
  const real = state.manifest.clips;
  const index = state.index;
  state.manifest.clips = Array.from({ length: 110 }, (_, i) => ({ id: `row${i}` }));
  state.index = 100;
  renderList();
  const host = $("list");
  const row = host.children[100].getBoundingClientRect();
  const frame = host.getBoundingClientRect();
  const first = host.scrollTop;
  state.index = 101;
  renderList();
  const second = host.scrollTop;
  const seen = { first, second, visible: row.top >= frame.top && row.bottom <= frame.bottom, position: $("position").textContent };
  state.manifest.clips = real;
  state.index = index;
  renderList();
  return seen;
});
check("the long list follows the current clip and keeps its place", list.first > 0 && list.second >= list.first && list.visible && list.position.includes("110"), JSON.stringify(list));

await page.locator("#side").evaluate((node) => { node.scrollTop = 0; });
await page.screenshot({ path: `${shots}/present-2-next.png` });
await browser.close();
console.log(failures.length ? `FAILED: ${failures.join(", ")}` : "all passed");
process.exit(failures.length ? 1 : 0);
