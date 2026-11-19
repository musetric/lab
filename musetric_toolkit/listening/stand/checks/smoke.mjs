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
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

await page.goto(base);
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
const messages = await page.evaluate(() => fetch("/messages.ru.json").then((r) => r.json()));
check("every message is filled", await page.evaluate(() => !/\{\w+\}/.test(document.body.innerText)));
await page.evaluate(() => fetch("/api/save", { method: "POST", body: JSON.stringify({ clip: clip().id, data: emptyLabel() }) }));
await page.reload();
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
check("clip loaded", true, await page.evaluate(() => window.standReady));
await page.waitForTimeout(300);
await page.screenshot({ path: `${shots}/1-open.png` });

const box = await page.locator("#canvas").boundingBox();
const at = async (tx, fy) => {
  const point = await page.evaluate(([t, f]) => ({ x: toX(t), y: toY(f) }), [tx, fy]);
  return { x: box.x + point.x, y: box.y + point.y };
};

const a = await at(3.75, 190);
const b = await at(4.05, 100);
await page.keyboard.press("KeyF");
const toolName = await page.locator("#tool").innerText();
check("tool toggles to box", toolName.startsWith(messages.toolBox), toolName);
await page.mouse.move(a.x, a.y);
await page.mouse.down();
await page.mouse.move(b.x, b.y, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(100);
const pending = await page.evaluate(() => state.pending);
check("box gives a pending region", !!pending && pending.polygon.length === 4, JSON.stringify(pending?.polygon));

const onlyResponse = page.waitForResponse((r) => r.url().includes("/api/audition"));
await page.keyboard.press("KeyS");
const only = await onlyResponse;
check("audition only answers", only.status() === 200, `boost ${only.headers()["x-boost-db"]} start ${only.headers()["x-start-s"]}`);
await page.waitForTimeout(250);
const auditionPosition = await page.evaluate(() => currentPosition());
check("audition moves the playhead", auditionPosition > 3.5 && auditionPosition < 4.4, String(auditionPosition));
const withoutResponse = page.waitForResponse((r) => r.url().includes("/api/audition"));
await page.keyboard.press("KeyD");
check("audition without answers", (await withoutResponse).status() === 200);

await page.keyboard.press("Digit2");
await page.waitForTimeout(500);
const regions = await page.evaluate(() => label().regions.map((r) => r.cls));
check("digit assigns echo", regions.length === 1 && regions[0] === "echo", JSON.stringify(regions));
await page.screenshot({ path: `${shots}/2-region.png` });

await page.keyboard.press("Escape");
await page.keyboard.press("KeyF");
const c = await at(5.0, 700);
const d = await at(5.6, 500);
const e = await at(5.2, 400);
await page.mouse.move(c.x, c.y);
await page.mouse.down();
await page.mouse.move(d.x, d.y, { steps: 8 });
await page.mouse.move(e.x, e.y, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(100);
const lasso = await page.evaluate(() => state.pending?.polygon.length);
check("lasso gives many points", lasso > 6, String(lasso));
await page.keyboard.press("Digit1");

await page.keyboard.press("KeyE");
const stem = await page.evaluate(() => state.stem);
check("stem hotkey switches stem", stem === "lead_b", stem);
await page.keyboard.press("Space");
const stopped = await page.evaluate(() => ({ playing: state.playing, audition: !!audio.auditionSource }));
check("space stops a running audition", !stopped.playing && !stopped.audition, JSON.stringify(stopped));
await page.keyboard.press("Space");
await page.waitForTimeout(600);
const playing = await page.evaluate(() => ({ playing: state.playing, at: currentPosition(), run: audio.ctx.state }));
check("playback advances", playing.playing && playing.at > 0.2 && playing.run === "running", JSON.stringify(playing));
await page.keyboard.press("KeyR");
await page.waitForTimeout(200);
const gains = await page.evaluate(() => Object.fromEntries(Object.entries(audio.gains).map(([k, g]) => [k, Number(g.gain.value.toFixed(2))])));
check("only the chosen stem sounds", gains.backing_a > 0.5 && gains.lead_b < 0.5, JSON.stringify(gains));
await page.keyboard.press("Space");

const card = (group) => page.locator(`.card[data-stem="${group}"]`);
await page.evaluate(() => document.activeElement.blur());
const clipBefore = await page.evaluate(() => clip().id);
await page.keyboard.press("Enter");
await page.waitForTimeout(100);
const untouched = await page.evaluate(() => ({ clip: clip().id, done: label().done, cards: document.querySelectorAll(".card[data-stem]").length, hint: $("doneHint").textContent }));
check("Enter refuses an empty song and asks no comparison yet", untouched.clip === clipBefore && !untouched.done && untouched.cards === 0 && untouched.hint === messages.markOrUnsure, JSON.stringify(untouched));
await page.locator('[data-present="instrumental"]').check();
const single = await page.evaluate(() => ({ cards: document.querySelectorAll(".card[data-stem]").length, hint: $("doneHint").textContent }));
check("one stem in the song asks no comparison", single.cards === 0 && single.hint === messages.idealKnown, JSON.stringify(single));
await page.locator('[data-unsure]').check();
const unsure = await page.evaluate(() => ({ cards: document.querySelectorAll(".card[data-stem]").length, present: label().present.length }));
check("an unsure song clears the ticks and asks the comparison", unsure.cards === 3 && unsure.present === 0, JSON.stringify(unsure));
await page.locator('[data-present="lead"]').check();
await page.locator('[data-present="instrumental"]').check();
const cleared = await page.evaluate(() => ({ unsure: label().unsure, box: document.querySelector("[data-unsure]").checked, cards: document.querySelectorAll(".card[data-stem]").length }));
check("two stems clear unsure and ask the comparison", !cleared.unsure && !cleared.box && cleared.cards === 3, JSON.stringify(cleared));
await page.evaluate(() => document.activeElement.blur());
await page.keyboard.press("Enter");
await page.waitForTimeout(100);
const refused = await page.evaluate(() => ({ clip: clip().id, done: label().done, flagged: document.querySelectorAll(".card.missing").length }));
check("Enter refuses two stems without the three comparisons", refused.clip === clipBefore && !refused.done && refused.flagged === 3, JSON.stringify(refused));
await card("lead").locator('[data-better="b"]').click();
await card("backing").locator('[data-better="unknown"]').click();
await page.keyboard.press("Enter");
await page.waitForTimeout(100);
const partly = await page.evaluate(() => ({ done: label().done, flagged: [...document.querySelectorAll(".card.missing")].map((c) => c.dataset.stem), hint: $("doneHint").textContent }));
check("only the unanswered stem stays flagged", !partly.done && partly.flagged.join() === "instrumental" && partly.hint.includes(messages.stemGroups.instrumental.title.toLowerCase()), JSON.stringify(partly));
await card("instrumental").locator('[data-listen="b"]').click();
const heard = await page.evaluate(() => ({ stem: state.stem, listening: [...document.querySelectorAll(".card.listening")].map((c) => c.dataset.stem), tag: $("plotStem").textContent }));
check("a card plays its own stem and shows it", heard.stem === "instrumental_b" && heard.listening.join() === "instrumental" && heard.tag.includes("B"), JSON.stringify(heard));
await card("instrumental").locator('[data-better="same"]').click();
await card("lead").locator('[data-side="a"][data-foreign="backing"]').check();
await card("instrumental").locator('[data-side="b"][data-foreign="lead"]').check();
await card("backing").locator("[data-comment]").fill("little voice in the backing");
await card("lead").locator("[data-comment]").fill("echo in the lead");
await page.locator('[data-present="lead"]').check();
await page.locator('[data-present="instrumental"]').check();
await page.locator('[data-present="backing"]').check();
await page.locator('[data-present="backing"]').uncheck();
await page.keyboard.press("Space");
await page.waitForTimeout(100);
await page.keyboard.press("Space");
const spaced = await page.evaluate(() => ({ present: label().present, box: document.querySelector('[data-present="backing"]').checked }));
check("space over a song box plays and keeps the answer", spaced.present.join() === "lead,instrumental" && !spaced.box, JSON.stringify(spaced));
await page.locator("#comment").fill("check: clear echo");
await page.keyboard.press("Escape");
await page.waitForTimeout(500);
await page.screenshot({ path: `${shots}/3-form.png` });

await page.reload();
await page.waitForFunction(() => window.standReady, null, { timeout: 60000 });
const saved = await page.evaluate(() => label());
check(
  "labels survive a reload",
  saved.regions.length === 2 && saved.present.join() === "lead,instrumental" && saved.comment.includes("echo")
    && saved.stems.lead.better === "b" && saved.stems.backing.better === "unknown" && saved.stems.instrumental.better === "same"
    && saved.stems.lead.foreign.a.join() === "backing" && saved.stems.lead.foreign.b.length === 0
    && saved.stems.instrumental.foreign.b.join() === "lead"
    && saved.stems.backing.comment.includes("little") && saved.stems.lead.comment.includes("echo"),
  JSON.stringify(saved.stems),
);
const shown = await page.evaluate(() => ({
  lead: document.querySelector('.card[data-stem="lead"] [data-comment]').value,
  boxA: document.querySelector('.card[data-stem="lead"] [data-side="a"][data-foreign="backing"]').checked,
  boxB: document.querySelector('.card[data-stem="lead"] [data-side="b"][data-foreign="backing"]').checked,
  better: document.querySelector('.card[data-stem="backing"] .on[data-better]').dataset.better,
  song: [...document.querySelectorAll("[data-present]")].filter((box) => box.checked).map((box) => box.dataset.present).join(),
}));
check("the form shows what was saved", shown.lead === "echo in the lead" && shown.boxA && !shown.boxB && shown.better === "unknown" && shown.song === "lead,instrumental", JSON.stringify(shown));
const old = await page.evaluate(() => normalizeLabel({ regions: [], content: "one", better_lead: "a", better_backing: "same", comment: "old", done: true }));
check("an old label keeps its answers and reopens", old.stems.lead.better === "a" && old.stems.backing.better === "same" && old.stems.instrumental.better === null && !old.done && old.comment === "old" && !("better_lead" in old) && !("present" in old) && old.content === "one", JSON.stringify(old.stems.lead));
const oldTicks = await page.evaluate(() => [{ content: "none" }, { content: "one" }, { content: "lead_backing" }, { content: null }, { content: "one", present: [] }].map((data) => presentOf(data).join()));
check("an old song answer shows as ticks, a cleared one stays clear", oldTicks.join("|") === "instrumental|lead|lead,backing||", oldTicks.join("|"));
await page.evaluate(() => { const data = label(); data.present = undefined; data.content = "one"; renderForm(); });
await page.locator('[data-present="backing"]').check();
const merged = await page.evaluate(() => ({ present: label().present, content: label().content }));
check("a tick on an old answer keeps the rest and the old answer", merged.present.join() === "lead,backing" && merged.content === "one", JSON.stringify(merged));
await page.keyboard.press("Control+KeyZ");
await page.waitForTimeout(100);
const undone = await page.evaluate(() => ({ present: label().present ?? null, boxes: [...document.querySelectorAll("[data-present]")].filter((box) => box.checked).map((box) => box.dataset.present).join() }));
check("undo takes a song tick back", undone.present === null && undone.boxes === "lead", JSON.stringify(undone));

await page.evaluate(() => { state.selected = 0; renderSelection(); });
await page.keyboard.press("Delete");
await page.waitForTimeout(100);
await page.keyboard.press("Control+KeyZ");
await page.waitForTimeout(100);
check("delete and undo restore", (await page.evaluate(() => label().regions.length)) === 2);

check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
await page.evaluate(() => document.activeElement.blur());
const firstClip = await page.evaluate(() => clip().id);
await page.keyboard.press("Enter");
await page.waitForFunction((id) => clip().id !== id && window.standReady === clip().id, firstClip, { timeout: 30000 });
check("a complete clip goes to the next one", true, await page.evaluate(() => clip().id));
await page.waitForTimeout(400);
check("the finished clip is stored as done", await page.evaluate(async (id) => (await (await fetch("/api/round")).json()).labels[id].done === true, firstClip));
const post = (body) => page.evaluate((text) => fetch("/api/save", { method: "POST", body: text }).then((r) => r.status), JSON.stringify(body));
const incomplete = await page.evaluate(() => { const data = emptyLabel(); data.done = true; data.stems.lead.better = "a"; data.stems.backing.better = "b"; return data; });
const wrongAnswer = await page.evaluate(() => { const data = emptyLabel(); data.stems.lead.better = "maybe"; return data; });
const wrongForeign = await page.evaluate(() => { const data = emptyLabel(); data.stems.lead.foreign.a = ["lead"]; return data; });
const server2 = {
  incomplete: await post({ clip: firstClip, data: incomplete }),
  wrongAnswer: await post({ clip: firstClip, data: wrongAnswer }),
  wrongForeign: await post({ clip: firstClip, data: wrongForeign }),
  noStems: await post({ clip: firstClip, data: { regions: [], done: true } }),
  wrongPresent: await post({ clip: firstClip, data: { ...wrongForeign, stems: (await page.evaluate(() => emptyLabel())).stems, present: ["voice"] } }),
  notList: await post({ clip: firstClip, data: { ...wrongForeign, stems: (await page.evaluate(() => emptyLabel())).stems, present: "lead" } }),
};
check("the server refuses an incomplete clip", Object.values(server2).every((code) => code === 400), JSON.stringify(server2));
const oldPage = await post({ clip: firstClip, data: { regions: [], content: "one", stems: (await page.evaluate(() => emptyLabel())).stems, comment: "", done: false } });
check("the server still takes a page opened before the ticks", oldPage === 200, String(oldPage));
const ticked = (present) => page.evaluate((list) => ({ ...emptyLabel(), present: list, done: true }), present);
const oneStem = await post({ clip: firstClip, data: await ticked(["instrumental"]) });
check("the server takes one stem without a comparison", oneStem === 200, String(oneStem));
const twoStems = await post({ clip: firstClip, data: await ticked(["lead", "instrumental"]) });
check("the server refuses two stems without a comparison", twoStems === 400, String(twoStems));
const server = {
  badClip: await page.evaluate(() => fetch("/api/save", { method: "POST", body: JSON.stringify({ clip: "nope", data: { regions: [] } }) }).then((r) => r.status)),
  badClass: await page.evaluate(() => fetch("/api/save", { method: "POST", body: JSON.stringify({ clip: clip().id, data: { ...emptyLabel(), regions: [{ cls: "nope", stem: "mix", polygon: [[0, 100], [1, 100], [1, 200]] }] } }) }).then((r) => r.status)),
  nan: await page.evaluate(() => fetch("/api/audition", { method: "POST", body: `{"clip":"${clip().id}","stem":"mix","mode":"only","polygon":[[0,NaN],[1,100],[1,200]]}` }).then((r) => r.status)),
  badMode: await page.evaluate(() => fetch("/api/audition", { method: "POST", body: JSON.stringify({ clip: clip().id, stem: "mix", mode: "x", polygon: [[0, 100], [1, 100], [1, 200]] }) }).then((r) => r.status)),
  traversal: await page.evaluate(() => fetch("/clips/..%2F..%2Fkey/mix.wav").then((r) => r.status)),
  key: await page.evaluate(() => fetch("/key.json").then((r) => r.status)),
  selection: await page.evaluate(() => fetch("/selection.json").then((r) => r.status)),
};
check("server refuses bad input", server.badClip === 400 && server.badClass === 400 && server.nan === 400 && server.badMode === 400 && server.traversal === 404 && server.key === 404 && server.selection === 404, JSON.stringify(server));
await browser.close();
console.log(failures.length ? `FAILED: ${failures.join(", ")}` : "all passed");
process.exit(failures.length ? 1 : 0);
