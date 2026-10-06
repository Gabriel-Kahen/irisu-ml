import assert from "node:assert/strict";
import test from "node:test";
import {createSoundtrack, DEFAULT_GAME_TRACK, GAME_OVER_TRACK, GAME_TRACKS, soundtrackFor} from "../static/music.mjs";

const loss = {observation: {terminated: true}, terminal_reason: "game_over"};

test("gameplay starts with the first song; only a loss interrupts the automatic rotation", () => {
  assert.equal(DEFAULT_GAME_TRACK, GAME_TRACKS[0]);
  for (const terminal_reason of [null, "replay_exhausted", "time_limit", "level_completed"]) {
    assert.deepEqual(soundtrackFor({observation: {score: 100000, terminated: true}, terminal_reason}),
      {uri: DEFAULT_GAME_TRACK, cycle: true});
  }
  assert.deepEqual(soundtrackFor(loss), {uri: GAME_OVER_TRACK, cycle: false});
  assert.deepEqual(soundtrackFor({observation: {score: 0}}), {uri: DEFAULT_GAME_TRACK, cycle: true});
});

function setup() {
  const sent = [], events = {};
  const frame = {contentWindow: {postMessage: (...args) => sent.push(args)}};
  const status = {hidden: true};
  const window = {
    location: {origin: "https://irisu.online"},
    addEventListener: (name, callback) => { events[name] = callback; },
  };
  const music = createSoundtrack({querySelector: id => id === "#spotifyPlayer" ? frame : status}, window);
  const message = data => events.message({origin: window.location.origin, source: frame.contentWindow, data});
  return {sent, events, frame, status, window, music, message};
}

test("snapshots don't reset the ongoing playlist; terminal changes and handshake resync do", () => {
  const {music, sent, message} = setup();
  music.update({observation: {score: 10}});
  music.update({observation: {score: 20000}});
  assert.equal(sent.length, 1);
  music.update(loss);
  music.update(loss);
  assert.equal(sent.length, 2);
  assert.equal(sent[1][0].uri, GAME_OVER_TRACK);
  message({type: "irisu:music-ready"});
  assert.equal(sent[2][0].uri, GAME_OVER_TRACK);
  music.update({observation: {score: 0}});
  assert.equal(sent[3][0].uri, DEFAULT_GAME_TRACK);
});

test("first gameplay interaction requests play once, including before the bridge loads", () => {
  const {music, sent, message} = setup();
  music.start();
  music.start();
  assert.equal(sent.filter(([data]) => data.type === "irisu:music-start").length, 1);
  message({type: "irisu:music-ready"});
  assert.equal(sent.at(-1)[0].type, "irisu:music-start");
});

test("only the trusted player can report errors; normal status occupies no space", () => {
  const {events, frame, status, message} = setup();
  const data = {type: "irisu:music-status", message: "Spotify unavailable"};
  events.message({origin: "https://other.example", source: frame.contentWindow, data});
  events.message({origin: "https://irisu.online", source: {}, data});
  assert.equal(status.hidden, true);
  message(data);
  assert.equal(status.textContent, data.message);
  assert.equal(status.hidden, false);
  message({type: "irisu:music-status", message: ""});
  assert.equal(status.hidden, true);
});
