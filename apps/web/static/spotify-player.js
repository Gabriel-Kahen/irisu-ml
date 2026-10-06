(() => {
  "use strict";

  const album = "spotify:album:72x9BgVuSuOGqmeGgTbAOV";
  // Official album durations (milliseconds), also used to avoid looping restricted previews.
  const durations = new Map(Object.entries({
    "69Syxg5Ujwse0b2Cbn7zw4": 216367, "6eoGq11wAa2DXTs5Qenszg": 219481,
    "69ocfwfs4udHqebi4e2ouO": 231038, "0dnz5asD7Hw6XMfDY4POHH": 77142,
    "4AfzYBClNpA56oxLt3t3jF": 17706, "1JPd2pBFL4FrTSo0kie8oJ": 196320,
    "2OLorxEZPSFnqyjAho1Myh": 246577, "6Cq6P6KrTEwCnHURQoOnqZ": 141773,
    "0LI9qyuCQfWTd1d5bMwxmO": 153669, "4DDJ7CfXUhSDf2Gpp06szh": 12000,
    "7fZKm6CdYTpQxskbIc6YBN": 12005, "02EOHQWJKcElhq01dG6chM": 9724,
    "0M3Z7PHTyjdkfJJaaiCYlO": 11619, "2PBbGnPwSAlCJi2zJoAZbt": 10018,
    "7IsiZa1n5orY0H6DhpdZQm": 23250, "717W41hhLpf4pbPwNyvydH": 135306,
    "4RF8mnbbath89EmF8h7YmP": 226253, "4G2Cno8LLVDhsJAMwJ2Z8D": 237353,
    "41hPcp8oADmP90nIE3v6x8": 305218, "1tiEzW7dOGPbPueYQDH97L": 247449,
  }).map(([id, duration]) => [`spotify:track:${id}`, duration]));
  const validUri = uri => typeof uri === "string" && (uri === album || durations.has(uri));
  const params = new URLSearchParams(location.search);
  let uri = validUri(params.get("uri")) ? params.get("uri") : album;
  let loop = params.get("loop") === "1";
  let controller = null;
  let fallback = false;
  let ready = false;
  let playing = false;
  let endedPlaying = false;
  let resume = false;
  let loopPending = false;
  let timer;
  const send = data => parent.postMessage(data, location.origin);
  const status = message => send({type: "irisu:music-status", message});
  const height = () => uri === album ? 352 : 152;
  const nativeUrl = () => `https://open.spotify.com/embed/${uri.split(":").slice(1).join("/")}?theme=0`;

  function nativeEmbed() {
    const frame = document.createElement("iframe");
    frame.src = nativeUrl();
    frame.title = "Irisu Syndrome! Original Soundtrack by watson on Spotify";
    frame.allow = "autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture";
    frame.setAttribute("allowfullscreen", "");
    document.body.replaceChildren(frame);
  }

  function useFallback() {
    if (fallback) return;
    fallback = true;
    clearTimeout(timer);
    try { controller?.destroy(); } catch { /* The native embed still works without the API. */ }
    controller = null;
    nativeEmbed();
    status("Spotify automatic control is unavailable. Use play in the embed; song changes may need another click.");
  }

  function watchReady() {
    clearTimeout(timer);
    timer = setTimeout(useFallback, 15000);
  }

  function select(nextUri, nextLoop) {
    loop = nextLoop;
    if (nextUri === uri) return;
    uri = nextUri;
    resume = playing || endedPlaying || resume;
    playing = false;
    endedPlaying = false;
    ready = false;
    loopPending = false;
    if (fallback) {
      nativeEmbed();
    } else if (controller) {
      try {
        watchReady();
        controller.setIframeDimensions("100%", height());
        (controller.loadEntity || controller.loadUri).call(controller, uri);
      } catch { useFallback(); }
    }
  }

  window.addEventListener("message", event => {
    if (event.source !== parent || event.origin !== location.origin) return;
    const data = event.data;
    if (data?.type !== "irisu:music" || !validUri(data.uri) || typeof data.loop !== "boolean") return;
    select(data.uri, data.loop);
  });

  window.onSpotifyIframeApiReady = api => {
    if (fallback || controller) return;
    try {
      api.createController(document.getElementById("spotify-player"), {
        uri, width: "100%", height: height(), theme: "dark",
      }, {events: {onError: useFallback}, onCreateCallback: created => {
        if (fallback) { created.destroy(); return; }
        controller = created;
        controller.addListener("ready", () => {
          if (fallback) return;
          clearTimeout(timer);
          ready = true;
          status("Use Spotify’s play button to start. Playback availability is controlled by Spotify.");
          if (resume) {
            resume = false;
            try { controller.play(); } catch { useFallback(); }
          }
        });
        controller.addListener("playback_update", event => {
          const data = event.data;
          if (fallback || !ready || !data || (uri !== album && data.playingURI !== uri)) return;
          if (typeof data.isPaused !== "boolean") return;
          const wasPlaying = playing;
          playing = !data.isPaused;
          const duration = durations.get(uri);
          const fullTrack = duration && Number.isFinite(data.duration) && Math.abs(data.duration - duration) < 1000;
          const completed = fullTrack && data.isPaused && !data.isBuffering &&
            data.position >= data.duration - 100;
          endedPlaying = Boolean(completed && (wasPlaying || endedPlaying));
          if (data.position < data.duration - 1000) loopPending = false;
          // Only a completed full track can loop; never restart a 30-second preview.
          if (loop && completed && wasPlaying && !loopPending) {
            loopPending = true;
            try { controller.restart(); } catch { useFallback(); }
          }
        });
      }});
    } catch { useFallback(); }
  };

  watchReady();
  send({type: "irisu:music-ready"});
  const script = document.createElement("script");
  script.src = "https://open.spotify.com/embed/iframe-api/v1";
  script.async = true;
  script.addEventListener("error", useFallback);
  document.head.append(script);
})();
