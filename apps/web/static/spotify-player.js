(() => {
  "use strict";

  const tracks = [
    "spotify:track:69Syxg5Ujwse0b2Cbn7zw4",
    "spotify:track:6eoGq11wAa2DXTs5Qenszg",
    "spotify:track:69ocfwfs4udHqebi4e2ouO",
  ];
  // Official full-track durations distinguish completion from restricted previews.
  const durations = new Map([
    [tracks[0], 216367], [tracks[1], 219481], [tracks[2], 231038],
    ["spotify:track:4DDJ7CfXUhSDf2Gpp06szh", 12000],
  ]);
  const validUri = uri => typeof uri === "string" && durations.has(uri);
  const params = new URLSearchParams(location.search);
  let uri = validUri(params.get("uri")) ? params.get("uri") : tracks[0];
  let cycle = params.get("cycle") !== "0";
  let controller = null;
  let fallback = false;
  let ready = false;
  let playing = false;
  let endedPlaying = false;
  let resume = false;
  let startRequested = false;
  let hasPlayed = false;
  let timer;
  const send = data => parent.postMessage(data, location.origin);
  const status = message => send({type: "irisu:music-status", message});
  const nativeUrl = () => `https://open.spotify.com/embed/${uri.split(":").slice(1).join("/")}?theme=0`;

  function nativeEmbed() {
    const frame = document.createElement("iframe");
    frame.src = nativeUrl();
    frame.width = "100%";
    frame.height = "80";
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
    status("Spotify controls unavailable; press play in Spotify.");
  }

  function watchReady() {
    clearTimeout(timer);
    timer = setTimeout(useFallback, 15000);
  }

  function select(nextUri, nextCycle, autoplay = false) {
    cycle = nextCycle;
    if (nextUri === uri) {
      if (autoplay && !fallback) {
        if (!ready) resume = true;
        else {
          try { controller.play(); } catch { useFallback(); }
        }
      }
      return;
    }
    uri = nextUri;
    resume = autoplay || playing || endedPlaying || resume;
    playing = false;
    endedPlaying = false;
    ready = false;
    if (fallback) {
      nativeEmbed();
    } else if (controller) {
      try {
        watchReady();
        controller.setIframeDimensions("100%", 80);
        (controller.loadEntity || controller.loadUri).call(controller, uri);
      } catch { useFallback(); }
    }
  }

  window.addEventListener("message", event => {
    if (event.source !== parent || event.origin !== location.origin) return;
    const data = event.data;
    if (data?.type === "irisu:music-start") {
      if (fallback || startRequested || hasPlayed) return;
      startRequested = true;
      if (!ready) resume = true;
      else {
        try { controller.play(); } catch { useFallback(); }
      }
      return;
    }
    if (data?.type !== "irisu:music" || !validUri(data.uri) || typeof data.cycle !== "boolean" ||
        (data.autoplay !== undefined && typeof data.autoplay !== "boolean")) return;
    select(data.uri, data.cycle, data.autoplay === true);
  });

  window.onSpotifyIframeApiReady = api => {
    if (fallback || controller) return;
    try {
      api.createController(document.getElementById("spotify-player"), {
        uri, width: "100%", height: 80, theme: "dark",
      }, {events: {onError: useFallback}, onCreateCallback: created => {
        if (fallback) { created.destroy(); return; }
        controller = created;
        controller.addListener("ready", () => {
          if (fallback) return;
          clearTimeout(timer);
          ready = true;
          status("");
          if (resume) {
            resume = false;
            try { controller.play(); } catch { useFallback(); }
          }
        });
        controller.addListener("playback_update", event => {
          const data = event.data;
          if (fallback || !ready || !data || data.playingURI !== uri) return;
          if (typeof data.isPaused !== "boolean") return;
          const wasPlaying = playing;
          playing = !data.isPaused;
          hasPlayed ||= playing;
          const duration = durations.get(uri);
          const fullTrack = duration && Number.isFinite(data.duration) && Math.abs(data.duration - duration) < 1000;
          const completed = fullTrack && data.isPaused && !data.isBuffering &&
            data.position >= data.duration - 100;
          endedPlaying = Boolean(completed && (wasPlaying || endedPlaying));
          // Advance full songs only; never loop or advance Spotify's restricted previews.
          if (cycle && completed && wasPlaying && tracks.includes(uri)) {
            select(tracks[(tracks.indexOf(uri) + 1) % tracks.length], true);
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
