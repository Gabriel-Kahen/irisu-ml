import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../static/spotify-player.js", import.meta.url), "utf8");
const first = "spotify:track:69Syxg5Ujwse0b2Cbn7zw4";
const second = "spotify:track:6eoGq11wAa2DXTs5Qenszg";
const third = "spotify:track:69ocfwfs4udHqebi4e2ouO";
const gameOver = "spotify:track:4DDJ7CfXUhSDf2Gpp06szh";

function setup(search = `?uri=${first}&cycle=1`) {
  const messages = [], loads = [], frames = [], timers = new Map(), events = {};
  let timerId = 0, plays = 0, restarts = 0, destroyed = 0, listener;
  const parent = {postMessage: (message, origin) => messages.push({message, origin})};
  const controller = {
    addListener: (name, handler) => { events[name] = handler; },
    setIframeDimensions() {},
    loadEntity: uri => loads.push(uri),
    play: () => { plays++; },
    restart: () => { restarts++; },
    destroy: () => { destroyed++; },
  };
  const window = {addEventListener: (_, handler) => { listener = handler; }};
  const document = {
    getElementById: () => ({}),
    createElement: tag => ({tag, setAttribute() {}, addEventListener() {}}),
    head: {append() {}},
    body: {replaceChildren: frame => frames.push(frame)},
  };
  vm.runInNewContext(source, {
    window, document, parent, URLSearchParams, location: {origin: "https://irisu.test", search},
    setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: id => timers.delete(id),
  });
  return {
    messages, loads, frames,
    start(overrides = {}) {
      listener({source: parent, origin: "https://irisu.test", data: {type: "irisu:music-start"}, ...overrides});
    },
    get plays() { return plays; },
    get restarts() { return restarts; },
    get destroyed() { return destroyed; },
    init() {
      window.onSpotifyIframeApiReady({createController: (_, options, handlers) => {
        assert.equal(options.height, 80);
        loads.push(options.uri);
        // The SDK adds a throwing default error listener unless onError is supplied here.
        assert.equal(typeof handlers.events.onError, "function");
        events.error = handlers.events.onError;
        handlers.onCreateCallback(controller);
      }});
    },
    emit: (name, data) => events[name]({data}),
    update(uri = first, extra = {}) {
      events.playback_update({data: {
        playingURI: uri, isPaused: false, isBuffering: false,
        duration: 216367, position: 1000, ...extra,
      }});
    },
    select(uri, cycle = true, overrides = {}, autoplay = false) {
      listener({source: parent, origin: "https://irisu.test", data: {type: "irisu:music", uri, cycle, autoplay}, ...overrides});
    },
    timeout() { for (const callback of [...timers.values()]) callback(); },
  };
}

test("handshake accepts the latest mode before API startup, without autoplay", () => {
  const player = setup();
  assert.equal(player.messages[0].message.type, "irisu:music-ready");
  assert.equal(player.messages[0].origin, "https://irisu.test");
  player.select(second);
  player.init();
  player.emit("ready");
  assert.deepEqual(player.loads, [second]);
  assert.equal(player.plays, 0);
});

test("game startup autoplays the preloaded track before or after player readiness", () => {
  for (const early of [true, false]) {
    const player = setup();
    if (early) player.select(first, true, {}, true);
    player.init();
    player.emit("ready");
    if (!early) player.select(first, true, {}, true);
    assert.equal(player.plays, 1);
    assert.deepEqual(player.loads, [first]);
  }
});

test("loss cue autoplays even after a paused song or an exhausted preview", () => {
  for (const previewEnded of [true, false]) {
    const player = setup();
    player.init();
    player.emit("ready");
    player.update();
    player.update(first, {isPaused: true, duration: 30000, position: previewEnded ? 30000 : 1000});
    player.select(gameOver, false, {}, true);
    assert.equal(player.plays, 0);
    player.emit("ready");
    assert.equal(player.plays, 1);
    assert.equal(player.loads.at(-1), gameOver);
  }
});

test("a blocked startup autoplay can retry on the first gameplay gesture", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.select(first, true, {}, true);
  player.update(first, {isPaused: true, position: 0});
  player.start();
  assert.equal(player.plays, 2);
});

test("only the same-origin parent may change mode or request playback", () => {
  const player = setup("?uri=javascript:alert(1)");
  player.init();
  player.emit("ready");
  player.select(second, true, {origin: "https://evil.test"});
  player.select(second, true, {source: {}});
  player.select("spotify:track:0000000000000000000000");
  player.select("spotify:album:72x9BgVuSuOGqmeGgTbAOV");
  player.select(second, "true");
  player.start({origin: "https://evil.test"});
  player.start({source: {}});
  assert.deepEqual(player.loads, [first]);
  assert.equal(player.plays, 0);
  player.select(second);
  assert.deepEqual(player.loads, [first, second]);
});

test("an early interaction queues playback until ready; a later interaction plays immediately", () => {
  for (const early of [true, false]) {
    const player = setup();
    if (early) player.start();
    player.init();
    assert.equal(player.plays, 0);
    player.emit("ready");
    if (!early) player.start();
    assert.equal(player.plays, 1);
    player.start(); // Parent may resend after the bridge handshake.
    assert.equal(player.plays, 1);
    assert.equal(player.messages.at(-1).message.message, "");
  }
});

test("first game interaction respects music already played and manually paused", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.update();
  player.update(first, {isPaused: true});
  player.start();
  assert.equal(player.plays, 0);
  player.select(gameOver, false);
  player.emit("ready");
  assert.equal(player.plays, 0);
});

test("rapid song changes preserve playback intent, while manual pause is respected", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.update();
  player.select(second);
  player.select(third);
  player.update(first, {isPaused: true}); // Stale old-frame update while loading.
  player.emit("ready");
  assert.equal(player.plays, 1);
  player.update(third);
  player.update(third, {isPaused: true});
  player.select(first);
  player.emit("ready");
  assert.equal(player.plays, 1);
});

test("completed full gameplay songs advance through the first three and wrap once", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  for (const [uri, duration] of [[first, 216367], [second, 219481], [third, 231038]]) {
    player.update(uri, {duration});
    player.update(uri, {duration, isPaused: true, position: duration});
    player.update(uri, {duration, isPaused: true, position: duration}); // Duplicate old-song event.
    player.emit("ready");
  }
  assert.deepEqual(player.loads, [first, second, third, first]);
  assert.equal(player.plays, 3);
  assert.equal(player.restarts, 0);
});

test("manual pauses, buffering, and completed previews never advance or restart", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.update();
  player.update(first, {isPaused: true, position: 200000});
  player.update();
  player.update(first, {isPaused: true, isBuffering: true, position: 216367});
  player.update(first, {duration: 30000});
  player.update(first, {duration: 30000, position: 30000, isPaused: true});
  assert.deepEqual(player.loads, [first]);
  assert.equal(player.restarts, 0);
  player.select(gameOver, false);
  player.emit("ready");
  assert.equal(player.plays, 0);
});

test("a completed game-over jingle plays once and retains intent for the next run", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.update();
  player.select(gameOver, false);
  player.emit("ready");
  assert.equal(player.plays, 1);
  player.update(gameOver, {duration: 12000});
  player.update(gameOver, {duration: 12000, position: 12000, isPaused: true});
  player.update(gameOver, {duration: 12000, position: 12000, isPaused: true});
  assert.deepEqual(player.loads, [first, gameOver]);
  assert.equal(player.restarts, 0);
  player.select(first);
  player.emit("ready");
  assert.equal(player.plays, 2);
});

test("blocked API falls back to a usable native Spotify iframe and follows selections", () => {
  const player = setup();
  player.timeout();
  assert.equal(player.frames[0].src, "https://open.spotify.com/embed/track/69Syxg5Ujwse0b2Cbn7zw4?theme=0");
  assert.match(player.messages.at(-1).message.message, /controls unavailable/);
  player.select(gameOver, false);
  assert.equal(player.frames.at(-1).src, "https://open.spotify.com/embed/track/4DDJ7CfXUhSDf2Gpp06szh?theme=0");
  player.init(); // A late API callback must not replace a player the user may already be using.
  assert.deepEqual(player.loads, []);
});

test("controller errors before or after readiness and readiness timeouts use the same fallback", () => {
  for (const failure of [
    player => player.emit("error"),
    player => { player.emit("ready"); player.emit("error"); },
    player => player.timeout(),
  ]) {
    const player = setup();
    player.init();
    failure(player);
    assert.equal(player.destroyed, 1);
    assert.equal(player.frames.length, 1);
    player.emit("error");
    assert.equal(player.frames.length, 1);
  }
});
