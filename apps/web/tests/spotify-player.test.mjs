import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../static/spotify-player.js", import.meta.url), "utf8");
const first = "spotify:track:69Syxg5Ujwse0b2Cbn7zw4";
const second = "spotify:track:6eoGq11wAa2DXTs5Qenszg";
const third = "spotify:track:69ocfwfs4udHqebi4e2ouO";
const album = "spotify:album:72x9BgVuSuOGqmeGgTbAOV";

function setup(search = `?uri=${first}&loop=1`) {
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
    get plays() { return plays; },
    get restarts() { return restarts; },
    get destroyed() { return destroyed; },
    init() {
      window.onSpotifyIframeApiReady({createController: (_, options, handlers) => {
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
    select(uri, loop = true, overrides = {}) {
      listener({source: parent, origin: "https://irisu.test", data: {type: "irisu:music", uri, loop}, ...overrides});
    },
    timeout() { for (const callback of [...timers.values()]) callback(); },
  };
}

test("handshake accepts the latest selection before API startup, without autoplay", () => {
  const player = setup();
  assert.equal(player.messages[0].message.type, "irisu:music-ready");
  assert.equal(player.messages[0].origin, "https://irisu.test");
  player.select(second);
  player.init();
  player.emit("ready");
  assert.deepEqual(player.loads, [second]);
  assert.equal(player.plays, 0);
});

test("only the same-origin parent may choose a whitelisted soundtrack URI", () => {
  const player = setup("?uri=javascript:alert(1)");
  player.init();
  player.select(first, true, {origin: "https://evil.test"});
  player.select(first, true, {source: {}});
  player.select("spotify:track:0000000000000000000000");
  player.select(first, "true");
  assert.deepEqual(player.loads, [album]);
  player.select(first);
  assert.deepEqual(player.loads, [album, first]);
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

test("completed full tracks loop once, but manual pauses and previews never loop", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.update();
  player.update(first, {isPaused: true, position: 216367});
  player.update(first, {isPaused: true, position: 216367});
  assert.equal(player.restarts, 1);
  player.update();
  player.update(first, {isPaused: true, position: 200000});
  assert.equal(player.restarts, 1);
  player.update(first, {duration: 30000});
  player.update(first, {duration: 30000, position: 30000, isPaused: true});
  assert.equal(player.restarts, 1);
});

test("non-looping game-over songs and album playback never restart", () => {
  const player = setup();
  player.init();
  player.emit("ready");
  player.select(first, false);
  player.update();
  player.update(first, {isPaused: true, position: 216367});
  assert.equal(player.restarts, 0);
  player.select(album, true);
  player.emit("ready");
  player.update();
  player.update(first, {isPaused: true, position: 216367});
  assert.equal(player.restarts, 0);
});

test("a completed game-over jingle retains intent for the next run, but a preview does not", () => {
  for (const duration of [216367, 30000]) {
    const player = setup();
    player.init();
    player.emit("ready");
    player.select(first, false);
    player.update(first, {duration});
    player.update(first, {duration, position: duration, isPaused: true});
    player.update(first, {duration, position: duration, isPaused: true});
    player.select(second);
    player.emit("ready");
    assert.equal(player.plays, duration === 216367 ? 1 : 0);
  }
});

test("blocked API falls back to a usable native Spotify iframe and follows selections", () => {
  const player = setup();
  player.timeout();
  assert.equal(player.frames[0].src, "https://open.spotify.com/embed/track/69Syxg5Ujwse0b2Cbn7zw4?theme=0");
  assert.match(player.messages.at(-1).message.message, /automatic control is unavailable/);
  player.select(album, false);
  assert.equal(player.frames.at(-1).src, "https://open.spotify.com/embed/album/72x9BgVuSuOGqmeGgTbAOV?theme=0");
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
