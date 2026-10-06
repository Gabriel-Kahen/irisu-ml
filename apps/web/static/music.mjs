export const GAME_TRACKS = [
  "spotify:track:69Syxg5Ujwse0b2Cbn7zw4",
  "spotify:track:6eoGq11wAa2DXTs5Qenszg",
  "spotify:track:69ocfwfs4udHqebi4e2ouO",
];
export const GAME_OVER_TRACK = "spotify:track:4DDJ7CfXUhSDf2Gpp06szh";
export const DEFAULT_GAME_TRACK = GAME_TRACKS[0];

// The web soundtrack cycles the three gameplay songs; only a loss interrupts it.
export function soundtrackFor(snapshot, gameTrack = DEFAULT_GAME_TRACK) {
  const lost = snapshot?.observation?.terminated && snapshot.terminal_reason === "game_over";
  return {uri: lost ? GAME_OVER_TRACK : gameTrack, cycle: !lost};
}

export function createSoundtrack(document, window, random = Math.random) {
  const frame = document.querySelector("#spotifyPlayer");
  const status = document.querySelector("#musicStatus");
  const chooseTrack = () => GAME_TRACKS[Math.floor(random() * GAME_TRACKS.length)];
  let gameTrack = chooseTrack();
  let snapshot = null;
  let current = null;
  let startRequested = false;
  let runStarted = false;

  function start() {
    startRequested = true;
    frame.contentWindow?.postMessage({type: "irisu:music-start"}, window.location.origin);
  }

  function update(next = snapshot, force = false) {
    const newRun = next && (!snapshot || next.seed !== snapshot.seed || next.mode !== snapshot.mode);
    if (newRun) {
      if (snapshot) gameTrack = chooseTrack();
      runStarted = false;
    }
    const gameStarting = Boolean(next?.running && !runStarted);
    if (gameStarting) runStarted = true;
    snapshot = next;
    const track = soundtrackFor(snapshot, gameTrack);
    if (!force && !gameStarting && current?.uri === track.uri && current.cycle === track.cycle) return;
    current = track;
    const autoplay = Boolean(snapshot && (snapshot.running || !track.cycle));
    frame.contentWindow?.postMessage({type: "irisu:music", ...track, autoplay}, window.location.origin);
  }

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin || event.source !== frame.contentWindow) return;
    if (event.data?.type === "irisu:music-ready") {
      update(snapshot, true);
      if (startRequested) frame.contentWindow?.postMessage({type: "irisu:music-start"}, window.location.origin);
    }
    if (event.data?.type === "irisu:music-status" && typeof event.data.message === "string") {
      status.textContent = event.data.message;
      status.hidden = !event.data.message;
    }
  });
  update();
  return {update, start};
}
