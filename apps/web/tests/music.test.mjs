import assert from "node:assert/strict";
import test from "node:test";
import {
  createSoundtrack, DEFAULT_GAME_TRACK, GAME_OVER_TRACK, GAME_TRACKS,
  SOUNDTRACK_ALBUM, soundtrackFor,
} from "../static/music.mjs";

test("normal music uses the fresh-save default and keeps the selected song at every score", () => {
  assert.equal(DEFAULT_GAME_TRACK, GAME_TRACKS[2]);
  assert.deepEqual(soundtrackFor(null), {uri: DEFAULT_GAME_TRACK, loop: true});
  for (const uri of GAME_TRACKS) {
    for (const score of [0, 9999, 10000, 20000, 40000, 100000, 4393101]) {
      assert.deepEqual(soundtrackFor({observation: {score}}, uri), {uri, loop: true});
    }
  }
});

test("terminal gameplay switches to the normal jingle; restarting or seeking back restores selection", () => {
  for (const score of [0, 20000, 100000]) {
    assert.deepEqual(soundtrackFor({observation: {score, terminated: true}}, GAME_TRACKS[0]),
      {uri: GAME_OVER_TRACK, loop: false});
  }
  assert.deepEqual(soundtrackFor({observation: {score: 0}}, GAME_TRACKS[0]),
    {uri: GAME_TRACKS[0], loop: true});
  for (const terminal_reason of ["replay_exhausted", "time_limit"]) {
    assert.equal(soundtrackFor({terminal_reason, observation: {truncated: true}}).uri,
      DEFAULT_GAME_TRACK);
  }
});

test("full-album listening is independent of the game", () => {
  assert.deepEqual(soundtrackFor({observation: {terminated: true}}, SOUNDTRACK_ALBUM),
    {uri: SOUNDTRACK_ALBUM, loop: false});
  assert.equal(soundtrackFor(null, "untrusted-uri").uri, DEFAULT_GAME_TRACK);
});

test("embed bridge resends latest state on ready, avoids repeated reloads, and checks message source", () => {
  const sent = [];
  const events = {};
  const frame = {contentWindow: {postMessage: (...args) => sent.push(args)}};
  const select = {value: DEFAULT_GAME_TRACK, addEventListener: (_, callback) => { events.change = callback; }};
  const status = {};
  const elements = {"#spotifyPlayer": frame, "#musicSelection": select, "#musicStatus": status};
  const window = {location: {origin: "https://irisu.online"}, addEventListener: (_, callback) => { events.message = callback; }};
  const music = createSoundtrack({querySelector: id => elements[id]}, window);
  const ended = {observation: {terminated: true}};
  music.update(ended);
  music.update(ended);
  assert.equal(sent.length, 2);
  assert.equal(sent[1][0].uri, GAME_OVER_TRACK);
  const event = {origin: window.location.origin, source: frame.contentWindow, data: {type: "irisu:music-ready"}};
  events.message({...event, origin: "https://other.example"});
  events.message({...event, source: {}});
  assert.equal(sent.length, 2);
  events.message(event);
  assert.equal(sent.length, 3);
  assert.equal(sent[2][0].uri, GAME_OVER_TRACK);
  select.value = SOUNDTRACK_ALBUM;
  events.change();
  assert.equal(frame.height, "352");
  assert.equal(sent[3][0].uri, SOUNDTRACK_ALBUM);
  events.message({...event, data: {type: "irisu:music-status", message: "Press play in Spotify."}});
  assert.equal(status.textContent, "Press play in Spotify.");
});

test("a saved music choice restores without autoplay and invalid stored URIs are ignored", () => {
  for (const saved of [GAME_TRACKS[0], SOUNDTRACK_ALBUM, "spotify:track:invalid"]) {
    const sent = [];
    const handlers = {};
    const select = {value: DEFAULT_GAME_TRACK, addEventListener: (name, fn) => { handlers[name] = fn; }};
    const frame = {contentWindow: {postMessage: value => sent.push(value)}};
    const storage = {getItem: () => saved, setItem: (key, value) => { storage.value = value; }};
    createSoundtrack({querySelector: id => id === "#spotifyPlayer" ? frame : select}, {
      location: {origin: "https://irisu.online"}, localStorage: storage, addEventListener() {},
    });
    assert.equal(sent[0].uri, saved.includes("invalid") ? DEFAULT_GAME_TRACK : saved);
    select.value = GAME_TRACKS[1];
    handlers.change();
    assert.equal(storage.value, GAME_TRACKS[1]);
  }
});
