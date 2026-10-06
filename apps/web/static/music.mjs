export const SOUNDTRACK_ALBUM = "spotify:album:72x9BgVuSuOGqmeGgTbAOV";
export const GAME_TRACKS = [
  "spotify:track:69Syxg5Ujwse0b2Cbn7zw4",
  "spotify:track:6eoGq11wAa2DXTs5Qenszg",
  "spotify:track:69ocfwfs4udHqebi4e2ouO",
];
export const GAME_OVER_TRACK = "spotify:track:4DDJ7CfXUhSDf2Gpp06szh";
export const DEFAULT_GAME_TRACK = GAME_TRACKS[2];

// v2.03 normal mode keeps the chosen BGM at every score, then uses irisu_gos1.
// See reference/music-rules.md for the original-game evidence and Spotify IDs.
export function soundtrackFor(snapshot, selection = DEFAULT_GAME_TRACK) {
  if (selection === SOUNDTRACK_ALBUM) return {uri: selection, loop: false};
  if (snapshot?.observation?.terminated) return {uri: GAME_OVER_TRACK, loop: false};
  return {uri: GAME_TRACKS.includes(selection) ? selection : DEFAULT_GAME_TRACK, loop: true};
}

export function createSoundtrack(document, window) {
  const frame = document.querySelector("#spotifyPlayer");
  const select = document.querySelector("#musicSelection");
  const status = document.querySelector("#musicStatus");
  const preferenceKey = "irisu-soundtrack";
  try {
    const saved = window.localStorage.getItem(preferenceKey);
    if (saved === SOUNDTRACK_ALBUM || GAME_TRACKS.includes(saved)) select.value = saved;
  } catch { /* Music still works when browser storage is unavailable. */ }
  let snapshot = null;
  let current = null;

  function update(next = snapshot, force = false) {
    snapshot = next;
    const track = soundtrackFor(snapshot, select.value);
    if (!force && current?.uri === track.uri && current.loop === track.loop) return;
    current = track;
    frame.height = track.uri === SOUNDTRACK_ALBUM ? "352" : "152";
    frame.contentWindow?.postMessage({type: "irisu:music", ...track}, window.location.origin);
  }

  select.addEventListener("change", () => {
    try { window.localStorage.setItem(preferenceKey, select.value); } catch { /* Optional preference. */ }
    update();
  });
  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin || event.source !== frame.contentWindow) return;
    if (event.data?.type === "irisu:music-ready") update(snapshot, true);
    if (event.data?.type === "irisu:music-status" && typeof event.data.message === "string") {
      status.textContent = event.data.message;
    }
  });
  update();
  return {update};
}
