import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import path from "node:path";

import {parseReplay, serializeReplay, encodeReplayWord} from "../../metsu-static/replay.mjs";
import {parseReplay as parseNormal, serializeReplay as serializeNormal} from "../../static/replay.mjs";
import {BrowserGame} from "../../metsu-static/exact-runtime.js";
import {CONTROL_WORD, EXACT_LIBRARY_SHA256, decodeHello, decodeStep,
  validateMetsuIdentity} from "../../metsu-static/exact-codec.mjs";
import {metsuWallRects, isExBody} from "../../metsu-static/presentation.mjs";
import {palette, activatedPalette, colorFor} from "../../metsu-static/colors.mjs";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const metadata = {seed: 0xfedcba98, highestLevel: 75, finalScore: 12345, highestChain: 9,
  words: [encodeReplayWord(3, 320, 420), encodeReplayWord(0, 320, 420)]};
const identity = Object.freeze({prepared: true, mode: 1, config_hash: "13197304445783894243",
  backend: "exact-msvc9-r58-metsu-forward", exact_library_sha256: EXACT_LIBRARY_SHA256});

function hello(backend = identity.backend, hash = identity.config_hash) {
  const strings = [backend, "gcc", EXACT_LIBRARY_SHA256].map(value => new TextEncoder().encode(value));
  const bytes = new Uint8Array(32 + strings.reduce((sum, value) => sum + value.length + 2, 0));
  const data = new DataView(bytes.buffer);
  data.setUint32(0, 1, true); data.setUint32(4, 32, true); data.setUint32(8, 196, true);
  data.setBigUint64(16, BigInt(hash), true); data.setUint32(24, CONTROL_WORD, true);
  data.setUint32(28, 1, true);
  let offset = 32;
  for (const string of strings) {
    data.setUint16(offset, string.length, true); bytes.set(string, offset + 2);
    offset += string.length + 2;
  }
  return bytes;
}

test("Metsu replay round trips mode 1 and both clients reject the other mode", () => {
  const bytes = serializeReplay(metadata);
  const replay = parseReplay(bytes);
  assert.equal(replay.mode, 1);
  assert.equal(replay.seed, metadata.seed);
  assert.deepEqual([...replay.words], metadata.words);
  assert.throws(() => parseNormal(bytes), /mode 1 is unsupported/);
  assert.throws(() => parseReplay(serializeNormal(metadata)), /supports Metsu mode 1/);
  assert.throws(() => serializeReplay({...metadata, mode: 0}), /only Metsu-mode/);
});

test("Metsu runtime records mode 1 and rejects normal replay objects before resetting", async () => {
  let resets = 0;
  const game = new BrowserGame({reset() { resets++; }}, () => {}, {seed: 41});
  game.recordedMetadata = {...metadata};
  game.recordedWords = [...metadata.words];
  assert.equal(parseReplay(game.replayBytes()).mode, 1);
  await assert.rejects(game.loadReplay({...metadata, mode: 0}), /mode-1 replay/);
  assert.equal(resets, 0);
  assert.equal(game.mode, "live");
});

test("Metsu Hello requires the separate backend and exact generated configuration", () => {
  assert.equal(decodeHello(hello(), identity).config_hash, identity.config_hash);
  assert.throws(() => decodeHello(hello("exact-msvc9-r58-multiworld-forward"), identity), /identity or ABI/);
  assert.throws(() => decodeHello(hello(identity.backend, "17009678407634462320"), identity), /identity or ABI/);
  assert.throws(() => validateMetsuIdentity({...identity, mode: 0}), /mode 1/);
  assert.throws(() => validateMetsuIdentity({...identity, prepared: false}), /mode 1/);
  assert.throws(() => validateMetsuIdentity({...identity, config_hash: "17009678407634462320"}), /mode 1/);
  assert.throws(() => decodeHello(hello()), /identity is missing/);
});

test("Metsu step diagnostics reject a normal configuration after a valid Hello", () => {
  const bytes = new Uint8Array(112 + 84);
  const data = new DataView(bytes.buffer);
  data.setBigUint64(112 + 16, BigInt(identity.config_hash), true);
  assert.equal(decodeStep(bytes, null, identity).diagnostics.config_hash, identity.config_hash);
  data.setBigUint64(112 + 16, 17009678407634462320n, true);
  assert.throws(() => decodeStep(bytes, null, identity), /Metsu mode/);
});

test("Metsu wall rendering follows the larger field's four recovered fixtures", () => {
  assert.deepEqual(metsuWallRects({x: 94, y: 20, width: 420, height: 370}), [
    {x: 94, y: 20, width: 16, height: 370},
    {x: 522, y: 20, width: 16, height: 370},
    {x: 94, y: 420, width: 452, height: 400},
    {x: 95, y: -290, width: 450, height: 300},
  ]);
  assert.equal(isExBody({kind: "piece", shape: "circle"}), true);
  assert.equal(isExBody({kind: "bonus", shape: "circle"}), false);
  assert.equal(isExBody({kind: "projectile", shape: "box"}), false);
});

test("all nine Metsu color IDs remain distinct before and after activation", () => {
  assert.equal(palette.length, 9); assert.equal(activatedPalette.length, 9);
  for (const lifecycle of ["scripted_falling", "dynamic_fresh"]) {
    const colors = Array.from({length: 9}, (_, color) => colorFor({kind: "piece", color, lifecycle}));
    assert.equal(new Set(colors).size, 9);
  }
});

test("Metsu worker uses an independent cache keyed by packaged runtime identity", () => {
  const source = readFileSync(path.join(web, "metsu-static/exact-worker.js"), "utf8");
  assert.match(source, /RUNTIME_CACHE_PREFIX = "irisu-metsu-exact-runtime-"/);
  assert.match(source, /RUNTIME_VERSION = identity.runtime_version/);
  assert.match(source, /guestFiles = identity.guest_files/);
  assert.doesNotMatch(source, /RUNTIME_CACHE_PREFIX = "irisu-exact-runtime-"/);
});
