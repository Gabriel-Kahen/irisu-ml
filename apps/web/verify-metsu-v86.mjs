#!/usr/bin/env node
// Compare a packaged i386 worker under v86 with its native execution.
import fs from "node:fs";
import path from "node:path";
import {createHash} from "node:crypto";
import {createRequire} from "node:module";
import {spawn} from "node:child_process";

const options = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => {
  if (index % 2 === 0) pairs.push([value.replace(/^--/, ""), all[index + 1]]);
  return pairs;
}, []));
if (!options.runtime || !options.worker) {
  throw new Error("usage: verify-metsu-v86.mjs --runtime DIR --worker ELF [--steps N] [--seed N] [--output JSON]");
}
const runtime = path.resolve(options.runtime);
const workerPath = path.resolve(options.worker);
const manifest = JSON.parse(fs.readFileSync(path.join(runtime, "metsu-runtime.json"), "utf8"));
const seeds = (options.seed || "0,41,123").split(",").map(Number);
const steps = Number(options.steps || 800);
const policy = options.policy || "scripted";
if (!Number.isInteger(steps) || steps < 1 || steps > 100000 ||
    !["scripted", "noinput"].includes(policy) || !seeds.length ||
    seeds.some(seed => !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)) {
  throw new Error("invalid verification seeds or step count");
}
const {V86} = createRequire(import.meta.url)(path.join(runtime, "libv86.js"));
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
const environment = Object.fromEntries(Object.entries(process.env)
  .filter(([name]) => !name.startsWith("LD_") && name !== "GLIBC_TUNABLES"));
environment.IRISU_EXACT_CW = "0x027f";

class Frames {
  constructor(send) { this.send = send; this.buffer = Buffer.alloc(0); this.next = 1; this.pending = new Map(); }
  receive(bytes) {
    this.buffer = Buffer.concat([this.buffer, bytes]);
    while (this.buffer.length >= 16) {
      const input = this.buffer;
      if (input.readUInt32LE(0) !== 0x43505249 || input.readUInt16LE(4) !== 1) {
        throw new Error("invalid response framing");
      }
      const size = input.readUInt32LE(12);
      if (size < 4 || size > 4 * 1024 * 1024) throw new Error("invalid response size");
      if (input.length < 16 + size) return;
      const pending = this.pending.get(input.readUInt32LE(8));
      if (!pending || pending.opcode !== input.readUInt16LE(6)) throw new Error("unexpected response");
      clearTimeout(pending.timer);
      this.pending.delete(input.readUInt32LE(8));
      const status = input.readInt32LE(16);
      const payload = Buffer.from(input.subarray(20, 16 + size));
      this.buffer = input.subarray(16 + size);
      if (status) pending.reject(new Error(`worker rejected RPC: ${payload}`));
      else pending.resolve(payload);
    }
  }
  fail(error) {
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pending.clear();
  }
  call(opcode, payload = Buffer.alloc(0)) {
    const id = this.next++;
    const frame = Buffer.alloc(16 + payload.length);
    frame.writeUInt32LE(0x43505249, 0); frame.writeUInt16LE(1, 4);
    frame.writeUInt16LE(opcode, 6); frame.writeUInt32LE(id, 8);
    frame.writeUInt32LE(payload.length, 12); payload.copy(frame, 16);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`RPC ${opcode} timed out`)); }, 60000);
      this.pending.set(id, {opcode, resolve, reject, timer});
      this.send(frame);
    });
  }
}

async function compareSeed(seed) {
  let serial = "";
  let stderr = "";
  let emulator;
  const native = spawn(workerPath, [], {cwd: "/", env: environment, stdio: ["pipe", "pipe", "pipe"]});
  const host = new Frames(bytes => native.stdin.write(bytes));
  const guest = new Frames(bytes => emulator.bus.send("virtio-console0-input-bytes", bytes));
  native.stdout.on("data", bytes => { try { host.receive(bytes); } catch (error) { host.fail(error); } });
  native.stderr.on("data", bytes => stderr = (stderr + bytes).slice(-8192));
  native.on("error", error => host.fail(error));
  native.on("exit", code => { if (code) host.fail(new Error(`native worker exited ${code}: ${stderr}`)); });
  const transcript = createHash("sha256");
  let comparedBytes = 0;
  let eventCount = 0;
  let exObservations = 0;
  let eventDetails = new Set();
  const exIds = new Set();
  const exEventCounts = {};
  let finalState;
  let completed = 0;
  let terminalTail = 0;
  const compare = async (opcode, payload, label, ignorePid = false) => {
    const [left, right] = await Promise.all([host.call(opcode, payload), guest.call(opcode, payload)]);
    if (ignorePid) { left.fill(0, 12, 16); right.fill(0, 12, 16); }
    if (!left.equals(right)) {
      const offset = left.findIndex((byte, index) => byte !== right[index]);
      throw new Error(`${label}: native/v86 mismatch at byte ${offset}; lengths ${left.length}/${right.length}`);
    }
    const length = Buffer.alloc(4); length.writeUInt32LE(left.length);
    transcript.update(length); transcript.update(left); comparedBytes += left.length;
    return left;
  };
  try {
    emulator = new V86({
      wasm_path: path.join(runtime, "v86.wasm"), memory_size: 128 << 20,
      vga_memory_size: 2 << 20,
      bios: {url: path.join(runtime, "seabios.bin")},
      vga_bios: {url: path.join(runtime, "vgabios.bin")},
      bzimage: {url: path.join(runtime, "buildroot-bzimage68.bin"), async: false},
      filesystem: {}, autostart: false, fastboot: true,
      disable_keyboard: true, disable_mouse: true, disable_speaker: true,
      virtio_console: true,
      cmdline: "rdinit=/irisu-direct-init console=ttyS0 quiet loglevel=0 tsc=reliable mitigations=off random.trust_cpu=on",
    });
    emulator.add_listener("serial0-output-byte", byte => serial = (serial + String.fromCharCode(byte)).slice(-8192));
    emulator.add_listener("virtio-console0-output-bytes", bytes => {
      try { guest.receive(bytes); } catch (error) { guest.fail(error); }
    });
    await new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
    for (const name of manifest.guest_files) {
      await emulator.create_file(`/${name}`, new Uint8Array(fs.readFileSync(path.join(runtime, "guest", name))));
    }
    emulator.run();
    await new Promise((resolve, reject) => {
      const started = Date.now();
      const timer = setInterval(() => {
        if (serial.includes("__IRISU_RPC_READY__")) { clearInterval(timer); resolve(); }
        else if (serial.includes("__IRISU_GUEST_ERROR__") || Date.now() - started > 60000) {
          clearInterval(timer); reject(new Error(`v86 boot failed: ${serial}`));
        }
      }, 20);
    });
    await compare(1, Buffer.alloc(0), "Hello", true);
    const reset = Buffer.alloc(8); reset.writeBigUInt64LE(BigInt(seed));
    await compare(2, reset, "Reset");
    for (let tick = 0; tick < steps; tick++) {
      const action = Buffer.alloc(28);
      action.writeUInt32LE(policy === "noinput" ? 0 : tick % 9 === 0 ? 3 : tick % 5 === 0 ? 1 : 0);
      action.writeDoubleLE(110 + ((tick * 37 + seed) % 390), 4);
      action.writeDoubleLE(270 + ((tick * 13) % 150), 12);
      action.writeUInt32LE(1, 20);
      const response = await compare(3, action, `seed ${seed} tick ${tick + 1}`);
      const count = response.readUInt32LE(104);
      for (let index = 0; index < count; index++) {
        const body = 112 + index * 100;
        if (response[body + 96] === 0 && response[body + 97] === 0) {
          exObservations++; exIds.add(response.readUInt32LE(body + 80));
        }
      }
      const transition = 112 + count * 100;
      const events = Number(response.readBigUInt64LE(transition + 8));
      eventCount += events;
      let offset = transition + 84;
      for (let index = 0; index < events; index++) {
        const size = response.readUInt16LE(offset + 32);
        const detail = response.subarray(offset + 36, offset + 36 + size).toString();
        if (/\bex\b/i.test(detail)) eventDetails.add(detail);
        if (exIds.has(response.readUInt32LE(offset + 24)) || exIds.has(response.readUInt32LE(offset + 28))) {
          const kind = response[offset + 34];
          const names = {3: "activated", 5: "confirmed", 6: "chain_joined", 7: "cleared", 8: "rotten", 10: "destroyed"};
          if (names[kind]) exEventCounts[names[kind]] = (exEventCounts[names[kind]] || 0) + 1;
        }
        offset += 36 + size;
      }
      completed = tick + 1;
      finalState = {tick: Number(response.readBigUInt64LE(0)), score: Number(response.readBigInt64LE(8)),
        gauge: Number(response.readBigInt64LE(16)), level: response.readUInt32LE(88),
        terminated: Boolean(response[108]), truncated: Boolean(response[109]),
        terminal_metadata_recorded: Boolean(response[transition + 82]),
        recorded_final_score: Number(response.readBigInt64LE(transition + 32)),
        recorded_final_level: response.readUInt32LE(transition + 68)};
      if (finalState.terminated || finalState.truncated) {
        if (++terminalTail >= 4) break;
      }
      if ((tick + 1) % 1000 === 0) process.stderr.write(`v86 parity seed ${seed}: ${tick + 1} ticks matched\n`);
    }
    process.stderr.write(`v86 parity seed ${seed}: ${completed} responses matched\n`);
    return {seed, requested_steps: steps, steps: completed, policy, final_state: finalState,
      compared_payload_bytes: comparedBytes, event_count: eventCount,
      ex_body_observations: exObservations, ex_event_details: [...eventDetails].sort(),
      ex_event_counts: exEventCounts,
      transcript_sha256: transcript.digest("hex"), mismatches: 0};
  } catch (error) {
    throw new Error(`${error.message}\nserial: ${serial}\nnative stderr: ${stderr}`);
  } finally {
    native.stdin.end(); native.kill(); emulator?.destroy();
  }
}

const started = Date.now();
const results = [];
for (const seed of seeds) results.push(await compareSeed(seed));
const report = {schema: "irisu-metsu-native-v86-parity-v1", mode: manifest.mode,
  worker_sha256: sha256(fs.readFileSync(workerPath)), runtime_version: manifest.runtime_version,
  guest_file_sha256: manifest.guest_file_sha256,
  runtime_asset_sha256: manifest.runtime_asset_sha256,
  seed_results: results, elapsed_seconds: (Date.now() - started) / 1000, mismatches: 0};
const encoded = JSON.stringify(report, null, 2) + "\n";
if (options.output) fs.writeFileSync(options.output, encoded);
process.stdout.write(encoded);
