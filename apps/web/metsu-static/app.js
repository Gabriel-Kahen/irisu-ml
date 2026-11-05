import {BrowserGame} from "./exact-runtime.js?v=metsu-20261103d";
import {
  activatedTrailAlphas, colorFor, hasActivatedTrail,
} from "./colors.mjs?v=metsu-20261103d";
import {
  clampReplayScrubFrame, parseReplay, REPLAY_TICK_MS,
} from "./replay.mjs?v=metsu-20261103d";
import {RestartGate} from "./restart-gate.mjs?v=metsu-20261103d";
import {drawMetsuWalls, isExBody} from "./presentation.mjs?v=metsu-20261103d";

const canvas = document.querySelector("#game");
const ctx = canvas.getContext("2d");
const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
canvas.width = 640 * pixelRatio;
canvas.height = 480 * pixelRatio;
ctx.scale(pixelRatio, pixelRatio);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const $ = (selector) => document.querySelector(selector);
const ui = {
  over: $("#gameOver"), again: $("#againButton"), finalTitle: $("#finalTitle"),
  finalScore: $("#finalScore"),
  paused: $("#paused"), pause: $("#pauseButton"), restart: $("#restartButton"),
  app: $(".app"), openReplay: $("#openReplayButton"),
  replayFile: $("#replayFileInput"), saveReplay: $("#saveReplayButton"),
  replayControls: $("#replayControls"), replayName: $("#replayName"),
  replayStatus: $("#replayStatus"), replayError: $("#replayError"),
  replayAnnouncement: $("#replayAnnouncement"),
  replayPlay: $("#replayPlayButton"), replayBack: $("#replayBackButton"),
  replayForward: $("#replayForwardButton"), replayScrubber: $("#replayScrubber"),
  replaySpeed: $("#replaySpeed"), replayPosition: $("#replayPosition"),
  exitReplay: $("#exitReplayButton"),
  runtimeLoading: $("#runtimeLoading"),
  appError: $("#appError"),
  toast: $("#toast"),
};

let snapshot = null;
let previousObservation = null;
let snapshotTime = performance.now();
let aim = {x: 320, y: 390, visible: false};
let started = false;
let lastEvent = -1;
let toastTimer;
let fastForwardTimer;
let game = null;
let trailTick = -1;
let replayLoadEpoch = 0;
let lastReplayAnnouncement = "";
let replayScrubbing = false;
let replayScrubTarget = null;
const bodyTrails = new Map();
const bodyPositions = new Map();
const pendingScores = [];
const scorePopups = [];
const clearBursts = [];
const soundtrack = {start() {}, update() {}};

const restartGate = new RestartGate((pending) => {
  if (ui.runtimeLoading) ui.runtimeLoading.hidden = !pending;
  ui.restart.disabled = pending;
  ui.again.disabled = pending;
  ui.openReplay.disabled = pending || !game;
});

const fastForwardIdleMs = 160;
const replaySkipFrames = 5_000 / REPLAY_TICK_MS;

function stopFastForward() {
  clearTimeout(fastForwardTimer);
  fastForwardTimer = 0;
  game?.setFastForward(false);
}

function startFastForward() {
  if (!snapshot?.running) return;
  clearTimeout(fastForwardTimer);
  fastForwardTimer = 0;
  game?.setFastForward(true);
}

function continueFastForward() {
  startFastForward();
  fastForwardTimer = setTimeout(stopFastForward, fastForwardIdleMs);
}

function showToast(text) {
  ui.toast.textContent = text;
  ui.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 1200);
}

function showPersistentError(text = "") {
  ui.appError.textContent = text;
  ui.appError.hidden = !text;
}

function announceReplay(key, text) {
  if (key === lastReplayAnnouncement) return;
  lastReplayAnnouncement = key;
  ui.replayAnnouncement.textContent = text;
}

function acceptSnapshot(next, force = false) {
  force ||= Boolean(snapshot && next?.mode !== snapshot.mode);
  force ||= next?.mode === "replay" && snapshot?.mode === "replay" &&
    next.replay.frame < snapshot.replay.frame;
  if (!force && snapshot && next.seed === snapshot.seed &&
      next.observation.tick < snapshot.observation.tick) return;
  if (force || !snapshot || next.seed !== snapshot.seed ||
      next.observation.tick < trailTick) {
    bodyTrails.clear();
    bodyPositions.clear();
    pendingScores.length = 0;
    scorePopups.length = 0;
    clearBursts.length = 0;
    trailTick = -1;
  }
  if (next.observation.tick !== trailTick) {
    const active = new Set();
    for (const body of next.observation.bodies) {
      bodyPositions.set(body.id, {x: body.x, y: body.y, color: colorFor(body, snapshotTime)});
      if (!hasActivatedTrail(body)) continue;
      active.add(body.id);
      const trail = bodyTrails.get(body.id) || [];
      trail.push({x: body.x, y: body.y, angle: body.angle || 0});
      if (trail.length > activatedTrailAlphas.length + 1) trail.shift();
      bodyTrails.set(body.id, trail);
    }
    for (const id of bodyTrails.keys()) {
      if (!active.has(id)) bodyTrails.delete(id);
    }
    trailTick = next.observation.tick;
  }
  previousObservation = force ? null : snapshot?.observation || next.observation;
  snapshot = next;
  snapshotTime = performance.now();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(640, (event.clientX - rect.left) * 640 / rect.width)),
    y: Math.max(0, Math.min(480, (event.clientY - rect.top) * 480 / rect.height)),
  };
}

function shoot(kind = "weak") {
  if (!snapshot || snapshot.observation.terminated || snapshot.observation.truncated) return;
  soundtrack.start();
  started = true;
  try { game?.shoot(kind, aim.x, aim.y); }
  catch (error) { showToast(error.message); }
}

function setRunning(running) {
  if (!game) return;
  if (running) soundtrack.start();
  if (!running) stopFastForward();
  game.setRunning(running);
  started ||= running;
  syncUi();
}

async function restart() {
  if (!game) return false;
  return restartGate.run(async () => {
    replayLoadEpoch++;
    replayScrubbing = false;
    replayScrubTarget = null;
    stopFastForward();
    ui.replayError.hidden = true;
    showPersistentError();
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const restarted = await game.restart(seed);
    if (restarted) {
      started = true;
      lastEvent = -1;
    }
    return restarted;
  });
}

function replayFilename(state) {
  const now = new Date();
  const part = value => String(value).padStart(2, "0");
  const stamp = `${now.getFullYear()}${part(now.getMonth() + 1)}${part(now.getDate())}_` +
    `${part(now.getHours())}${part(now.getMinutes())}${part(now.getSeconds())}`;
  const score = Math.max(0, state.score).toString().padStart(8, "0").slice(-8);
  return `irisu_metsu_${score}_${stamp}_1.rpy`;
}

function saveReplay() {
  if (!game || !snapshot?.can_save_replay) return;
  try {
    const data = game.replayBytes();
    const url = URL.createObjectURL(new Blob([data], {type: "application/octet-stream"}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = replayFilename(snapshot.observation);
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
    showToast(`saved ${anchor.download}`);
  } catch (error) {
    showToast(error.message);
  }
}

async function openReplayFile(file) {
  const epoch = ++replayLoadEpoch;
  replayScrubbing = false;
  replayScrubTarget = null;
  stopFastForward();
  ui.replayError.hidden = true;
  showPersistentError();
  ui.openReplay.disabled = true;
  ui.replayStatus.textContent = "Loading exact replay…";
  announceReplay("loading", "Loading exact replay");
  try {
    const replay = parseReplay(await file.arrayBuffer());
    if (epoch !== replayLoadEpoch) return;
    const loaded = await game.loadReplay(replay, file.name || "replay.rpy");
    if (!loaded || epoch !== replayLoadEpoch) return;
    showPersistentError();
    started = true;
    lastEvent = Number.MAX_SAFE_INTEGER;
    syncUi();
  } catch (error) {
    if (epoch !== replayLoadEpoch) return;
    ui.replayError.textContent = error.message;
    ui.replayError.hidden = false;
    ui.replayControls.hidden = false;
    showPersistentError(error.message);
    showToast(error.message);
  } finally {
    if (epoch === replayLoadEpoch) ui.openReplay.disabled = !game;
  }
}

function bodyPath(body, size) {
  ctx.beginPath();
  if (body.shape === "circle") ctx.arc(0, 0, size / 2, 0, Math.PI * 2);
  else if (body.shape === "triangle") {
    ctx.moveTo(-size / 2, -size / 2);
    ctx.lineTo(-size / 2, size / 2);
    ctx.lineTo(size / 2, size / 2);
    ctx.closePath();
  } else ctx.rect(-size / 2, -size / 2, size, size);
}

function fillBody(body, size, color, pose, alpha = 1, detailed = false) {
  ctx.save();
  ctx.translate(pose.x, pose.y);
  ctx.rotate(pose.angle || 0);
  ctx.globalAlpha = alpha;
  bodyPath(body, size);
  ctx.fillStyle = color;
  if (detailed) {
    ctx.shadowColor = "#00000050";
    ctx.shadowBlur = 7;
    ctx.shadowOffsetY = 3;
  }
  ctx.fill();
  if (detailed) {
    ctx.shadowColor = "transparent";
    const face = ctx.createLinearGradient(-size / 2, -size / 2, size / 2, size / 2);
    face.addColorStop(0, "#ffffff24");
    face.addColorStop(.45, "#ffffff00");
    face.addColorStop(1, "#00000026");
    ctx.fillStyle = face;
    ctx.fill();
    ctx.strokeStyle = "#ffffff40";
    ctx.lineWidth = .8;
    ctx.stroke();
    if (body.lifecycle === "rotten") {
      ctx.clip();
      ctx.strokeStyle = "#10171355";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = -size; x < size; x += 9) {
        ctx.moveTo(x, -size / 2);
        ctx.lineTo(x + size, size / 2);
      }
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawBody(body, now) {
  const size = Math.max(2, body.size);
  const color = colorFor(body, now);
  if (hasActivatedTrail(body) && !reducedMotion.matches) {
    const trail = bodyTrails.get(body.id) || [];
    trail.slice(0, -1).forEach((pose, index, echoes) => {
      const alphaOffset = activatedTrailAlphas.length - echoes.length;
      fillBody(body, size, color, pose,
        activatedTrailAlphas[alphaOffset + index] * .4);
    });
  }
  fillBody(body, size, color, body,
    body.lifecycle === "scripted_falling" ? .88 : 1, true);
  if (isExBody(body)) {
    ctx.save();
    ctx.translate(body.x, body.y);
    const active = body.lifecycle === "confirmed";
    ctx.strokeStyle = active ? "#fff1bf" : "#d6b46ca0";
    ctx.lineWidth = active ? 1.6 : 1;
    ctx.beginPath();
    ctx.arc(0, 0, size / 2 + 3, 0, Math.PI * 2);
    ctx.stroke();
    const rotation = (reducedMotion.matches || !snapshot?.running ? 0 : now / 1800) + body.id * 1.7;
    ctx.beginPath();
    ctx.arc(0, 0, size / 2 + 3, rotation, rotation + Math.PI * .55);
    ctx.strokeStyle = "#fff0c9";
    ctx.lineCap = "round";
    ctx.shadowColor = "#e0bf7450";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.restore();
  }
}

function drawBackdrop() {
  ctx.fillStyle = "#0c1517";
  ctx.fillRect(0, 0, 640, 480);
}

function outlinedText(text, x, y, size, align = "left") {
  ctx.save();
  ctx.textAlign = align;
  ctx.font = `italic 900 ${size}px "Trebuchet MS", sans-serif`;
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#55152c";
  ctx.lineWidth = 5;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = "#f0e2a6";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawHud(state) {
  outlinedText("Level", 21, 428, 28);
  outlinedText(String(state.level), 52, 458, 25, "center");

  const trackX = 151, trackY = 437, trackW = 312, trackH = 15;
  ctx.fillStyle = "#33161eaa";
  ctx.fillRect(trackX, trackY, trackW, trackH);
  const ratio = Math.max(0, Math.min(1, state.gauge / state.gauge_max));
  const gaugeGradient = ctx.createLinearGradient(trackX, 0, trackX + trackW, 0);
  gaugeGradient.addColorStop(0, "#7c1b31");
  gaugeGradient.addColorStop(1, "#b02a3f");
  ctx.fillStyle = gaugeGradient;
  ctx.fillRect(trackX, trackY, trackW * ratio, trackH);
  ctx.fillStyle = "#ffffff18";
  ctx.fillRect(trackX, trackY, trackW, 3);

  const digits = String(state.score).padStart(8, "0");
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "900 26px Georgia, serif";
  ctx.strokeStyle = "#681a38";
  ctx.lineWidth = 5;
  ctx.strokeText(digits, 320, 462);
  ctx.fillStyle = "#eee0a4";
  ctx.fillText(digits, 320, 462);
  ctx.restore();
}

function drawClearBursts(now) {
  for (let i = clearBursts.length - 1; i >= 0; i--) {
    const burst = clearBursts[i];
    const progress = (now - burst.startedAt) / 550;
    if (progress >= 1) { clearBursts.splice(i, 1); continue; }
    const radius = 6 + 24 * progress;
    ctx.save();
    ctx.globalAlpha = (1 - progress) * .65;
    ctx.strokeStyle = burst.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(burst.x, burst.y, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = burst.color;
    for (let n = 0; n < 6; n++) {
      const angle = n * Math.PI / 3;
      ctx.fillRect(burst.x + Math.cos(angle) * radius * 1.3 - 1,
        burst.y + Math.sin(angle) * radius * 1.3 - 1, 2, 2);
    }
    ctx.restore();
  }
}

function drawScorePopups(now) {
  for (let index = scorePopups.length - 1; index >= 0; index--) {
    const popup = scorePopups[index];
    const progress = Math.min(1, (now - popup.startedAt) / 900);
    if (progress >= 1) {
      scorePopups.splice(index, 1);
      continue;
    }
    const rise = 42 * (1 - (1 - progress) ** 2);
    ctx.save();
    ctx.globalAlpha = 1 - progress;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "900 26px Georgia, serif";
    ctx.strokeStyle = "#681a38";
    ctx.lineWidth = 5;
    ctx.lineJoin = "round";
    ctx.strokeText(String(popup.value), popup.x, popup.y - rise);
    ctx.fillStyle = "#eee0a4";
    ctx.fillText(String(popup.value), popup.x, popup.y - rise);
    ctx.restore();
  }
}

function drawWalls(state) {
  drawMetsuWalls(ctx, state.field);
}

function interpolatedBodies(now) {
  const current = snapshot.observation;
  if (!previousObservation || previousObservation.tick === current.tick) return current.bodies;
  const tickGap = current.tick - previousObservation.tick;
  if (tickGap <= 0 || tickGap > 4) return current.bodies;
  const alpha = Math.min(1, (now - snapshotTime) / Math.min(60, Math.max(20, tickGap * 20)));
  const oldBodies = new Map(previousObservation.bodies.map((body) => [body.id, body]));
  return current.bodies.map((body) => {
    const old = oldBodies.get(body.id);
    if (!old) return body;
    let angleDelta = (body.angle - old.angle) % (Math.PI * 2);
    if (angleDelta > Math.PI) angleDelta -= Math.PI * 2;
    if (angleDelta < -Math.PI) angleDelta += Math.PI * 2;
    return {
      ...body,
      x: old.x + (body.x - old.x) * alpha,
      y: old.y + (body.y - old.y) * alpha,
      angle: old.angle + angleDelta * alpha,
    };
  });
}

function draw(now) {
  drawBackdrop();
  if (snapshot) {
    const state = snapshot.observation;
    drawWalls(state);
    [...interpolatedBodies(now)].sort((a, b) => a.id - b.id)
      .forEach((body) => drawBody(body, now));
    drawClearBursts(now);
    drawScorePopups(now);
    drawHud(state);
  }
  if (aim.visible) {
    ctx.save();
    ctx.strokeStyle = "#f6edcebb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(aim.x, aim.y, 9, 0, Math.PI * 2);
    ctx.moveTo(aim.x - 15, aim.y); ctx.lineTo(aim.x - 5, aim.y);
    ctx.moveTo(aim.x + 5, aim.y); ctx.lineTo(aim.x + 15, aim.y);
    ctx.moveTo(aim.x, aim.y - 15); ctx.lineTo(aim.x, aim.y - 5);
    ctx.moveTo(aim.x, aim.y + 5); ctx.lineTo(aim.x, aim.y + 15);
    ctx.stroke();
    ctx.restore();
  }
  requestAnimationFrame(draw);
}

function processEvents(events) {
  for (const event of events) {
    if (event.sequence <= lastEvent) continue;
    lastEvent = event.sequence;
    if (event.kind_name === "cleared" && !reducedMotion.matches) {
      const position = bodyPositions.get(event.a);
      if (position) {
        clearBursts.push({...position, startedAt: performance.now()});
        if (clearBursts.length > 48) clearBursts.shift();
      }
    }
    if (event.kind_name === "score_changed" && event.value > 0) {
      pendingScores.push(event);
    } else if (event.kind_name === "cleared" &&
               /burst actor teardown$/.test(event.detail)) {
      const index = pendingScores.findIndex(score => score.tick === event.tick);
      if (index >= 0) {
        const [score] = pendingScores.splice(index, 1);
        const position = bodyPositions.get(event.a) || {x: 320, y: 240};
        scorePopups.push({...position, value: score.value, startedAt: performance.now()});
      }
    }
    if (event.kind_name === "level_changed") showToast(`LEVEL ${event.value}`);
  }
}

function showReplayPosition(frame, totalFrames) {
  const seconds = (frame * REPLAY_TICK_MS / 1000).toFixed(1);
  const totalSeconds = (totalFrames * REPLAY_TICK_MS / 1000).toFixed(1);
  ui.replayScrubber.setAttribute("aria-valuetext",
    `frame ${frame} of ${totalFrames}`);
  ui.replayPosition.value = `frame ${frame} / ${totalFrames}`;
  ui.replayPosition.textContent =
    `${frame} / ${totalFrames} · ${seconds}s / ${totalSeconds}s`;
}

function syncUi() {
  if (!snapshot) return;
  soundtrack.update(snapshot);
  const state = snapshot.observation;
  const replay = snapshot.replay;
  const replayMode = snapshot.mode === "replay";
  started ||= snapshot.running || state.tick > 0;
  ui.pause.firstChild.textContent = snapshot.running ? "pause " : "resume ";
  ui.pause.disabled = replayMode && replay.frame >= replay.total_frames;
  ui.paused.hidden = !started || snapshot.running || state.terminated || state.truncated;
  ui.over.hidden = replayMode ?
    !(replay.complete && replay.frame >= replay.total_frames) :
    !(state.terminated || state.truncated);
  if (!ui.over.hidden) {
    ui.finalTitle.textContent = replayMode ?
      snapshot.terminal_reason === "replay_exhausted" ? "Replay exhausted" : "Replay complete" :
      snapshot.terminal_reason === "level_completed" ? "Level 100 complete" :
        snapshot.terminal_reason === "time_limit" ? "Time limit" : "Game over";
    ui.finalScore.textContent = `${String(state.score).padStart(8, "0")} · level ${state.level}`;
  }
  ui.saveReplay.hidden = replayMode || !snapshot.can_save_replay;
  ui.again.textContent = replayMode ? "new game" : "try again";
  ui.replayControls.hidden = !replayMode;
  ui.app.classList.toggle("replay-mode", replayMode);
  document.documentElement.dataset.mode = snapshot.mode;
  if (replayMode) {
    if (replay.cursor) aim = {x: replay.cursor.x, y: replay.cursor.y, visible: true};
    else aim.visible = false;
    ui.replayName.textContent = replay.name;
    ui.replayScrubber.max = String(replay.total_frames);
    const scrubTarget = replayScrubTarget === null ? null :
      Math.min(replayScrubTarget, replay.total_frames);
    if (scrubTarget === replay.frame) replayScrubTarget = null;
    if (!replayScrubbing && replayScrubTarget === null) {
      ui.replayScrubber.value = String(replay.frame);
      showReplayPosition(replay.frame, replay.total_frames);
    }
    ui.replayPlay.textContent = snapshot.running ? "pause" : "play";
    ui.replaySpeed.value = String(replay.speed);
    ui.replayPlay.disabled = replay.frame >= replay.total_frames;
    ui.replayBack.disabled = replay.frame <= 0;
    ui.replayForward.disabled = replay.frame >= replay.total_frames;
    ui.replayStatus.textContent = replay.buffering ?
      `Buffering ${replay.buffered_frames} / ${replay.total_frames}…` :
      replay.complete ? (replay.warning || "Exact replay ready") :
        `Prepared ${replay.buffered_frames} / ${replay.total_frames}`;
    if (replay.complete) announceReplay("complete", "Replay is ready");
    else if (replay.buffering) announceReplay("buffering", "Buffering replay");
    ui.replayError.textContent = replay.warning;
    ui.replayError.hidden = !replay.warning;
    document.documentElement.dataset.replayFrame = String(replay.frame);
    document.documentElement.dataset.replayBuffered = String(replay.buffered_frames);
  } else {
    replayScrubbing = false;
    replayScrubTarget = null;
    lastReplayAnnouncement = "";
    ui.replayError.hidden = true;
    delete document.documentElement.dataset.replayFrame;
    delete document.documentElement.dataset.replayBuffered;
    processEvents(snapshot.events);
  }
}

function receiveSnapshot(next, error) {
  if (error) {
    document.documentElement.dataset.ready = "false";
    document.documentElement.dataset.error = error.message;
    ui.replayError.textContent = error.message;
    ui.replayError.hidden = false;
    showPersistentError(error.message);
    if (ui.runtimeLoading) ui.runtimeLoading.hidden = true;
    showToast(error.message);
    return;
  }
  acceptSnapshot(next);
  if (ui.runtimeLoading) ui.runtimeLoading.hidden = true;
  document.documentElement.dataset.ready = "true";
  delete document.documentElement.dataset.error;
  document.documentElement.dataset.tick = String(next.observation.tick);
  document.documentElement.dataset.seed = String(next.seed);
  syncUi();
}

canvas.addEventListener("pointermove", (event) => {
  if (snapshot?.mode === "replay") return;
  aim = {...canvasPoint(event), visible: true};
  game?.setAim(aim.x, aim.y);
});
canvas.addEventListener("pointerleave", () => {
  if (snapshot?.mode !== "replay") aim.visible = false;
});
canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  canvas.focus({preventScroll: true});
  if (snapshot?.mode === "replay") return;
  aim = {...canvasPoint(event), visible: true};
  game?.setAim(aim.x, aim.y);
  shoot(event.shiftKey ? "both" : event.button === 2 ? "strong" : "weak");
});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (snapshot?.mode === "replay") return;
  if (event.deltaY > 0) continueFastForward();
  else if (event.deltaY < 0) stopFastForward();
}, {passive: false});
ui.pause.addEventListener("click", () => setRunning(!snapshot?.running));
ui.restart.addEventListener("click", () => { void restart(); });
ui.again.addEventListener("click", () => { void restart(); });
ui.saveReplay.addEventListener("click", saveReplay);
ui.openReplay.addEventListener("click", () => {
  if (ui.replayFile.showPicker) ui.replayFile.showPicker();
  else ui.replayFile.click();
});
ui.replayFile.addEventListener("change", () => {
  const file = ui.replayFile.files?.[0];
  ui.replayFile.value = "";
  if (file) void openReplayFile(file);
});
ui.replayPlay.addEventListener("click", () => setRunning(!snapshot?.running));
ui.replayBack.addEventListener("click", () => game?.stepReplay(-replaySkipFrames));
ui.replayForward.addEventListener("click", () => game?.stepReplay(replaySkipFrames));
ui.replaySpeed.addEventListener("change", () => game?.setReplaySpeed(
  Number(ui.replaySpeed.value)));
ui.replayScrubber.addEventListener("input", () => {
  replayScrubbing = true;
  replayScrubTarget = null;
  const frame = Number(ui.replayScrubber.value);
  showReplayPosition(frame, Number(ui.replayScrubber.max));
});
ui.replayScrubber.addEventListener("change", () => {
  const frame = clampReplayScrubFrame(
    ui.replayScrubber.value, snapshot?.replay?.buffered_frames,
  );
  replayScrubbing = false;
  replayScrubTarget = frame;
  ui.replayScrubber.value = String(frame);
  showReplayPosition(frame, Number(ui.replayScrubber.max));
  game?.seekReplay(frame, {preserveRunning: true});
});
ui.exitReplay.addEventListener("click", () => { void restart(); });
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement ||
      event.target instanceof HTMLSelectElement || event.target?.isContentEditable) return;
  if (snapshot?.mode === "replay" &&
      ["Enter", "ArrowLeft", "ArrowRight"].includes(event.code)) {
    event.preventDefault();
    if (event.repeat && event.code === "Enter") return;
    if (event.code === "Enter") setRunning(!snapshot.running);
    else game?.stepReplay(event.code === "ArrowLeft" ?
      -replaySkipFrames : replaySkipFrames);
    return;
  }
  if (event.code === "Enter") { event.preventDefault(); setRunning(!snapshot?.running); }
  if (event.code === "Space") {
    event.preventDefault();
    if (!event.repeat && snapshot?.mode !== "replay") startFastForward();
  }
  if (event.key.toLowerCase() === "r") void restart();
  if (snapshot?.mode !== "replay" && event.key.toLowerCase() === "w") shoot("weak");
  if (snapshot?.mode !== "replay" && event.key.toLowerCase() === "s") shoot("strong");
});
window.addEventListener("keyup", (event) => {
  if (event.code === "Space") stopFastForward();
});
window.addEventListener("blur", stopFastForward);

draw();
BrowserGame.create(receiveSnapshot).then((instance) => {
  game = instance;
  game.setAim(aim.x, aim.y);
  document.documentElement.dataset.backend = "exact-v86";
  document.documentElement.dataset.ready = "true";
  ui.openReplay.disabled = false;
})
  .catch((error) => receiveSnapshot(null, error));
