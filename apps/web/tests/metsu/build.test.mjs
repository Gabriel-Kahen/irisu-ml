import assert from "node:assert/strict";
import test from "node:test";
import {execFileSync} from "node:child_process";
import {mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import {createHash} from "node:crypto";
import {fileURLToPath} from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const build = path.join(web, "build-metsu-static.sh");
const digest = value => createHash("sha256").update(value).digest("hex");

function runtime(directory, changes = {}) {
  const output = path.join(directory, "runtime"); mkdirSync(output);
  const manifest = {mode: 1, prepared: true, config_hash: "13197304445783894243",
    backend: "exact-msvc9-r58-metsu-forward", runtime_version: "fixture-digest",
    exact_library_sha256: "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5",
    guest_files: ["irisu-exact-worker"], ...changes};
  const content = JSON.stringify(manifest);
  writeFileSync(path.join(output, "metsu-runtime.json"), content);
  writeFileSync(path.join(output, ".irisu-metsu-runtime"), "fixture");
  writeFileSync(path.join(output, "SHA256SUMS"), `${digest(content)}  metsu-runtime.json\n`);
  return output;
}

test("Metsu static build stages a separate site and generated strict mode identity", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "metsu-static-"));
  try {
    const prepared = runtime(directory);
    const site = path.join(directory, "site");
    execFileSync(build, [site], {env: {...process.env, IRISU_METSU_RUNTIME_DIR: prepared}, stdio: "pipe"});
    assert.equal(existsSync(path.join(site, ".irisu-metsu-static-site")), true);
    assert.equal(existsSync(path.join(site, "CNAME")), false);
    const module = readFileSync(path.join(site, "mode-identity.js"), "utf8");
    assert.match(module, /"mode": 1/); assert.match(module, /fixture-digest/);
    assert.match(readFileSync(path.join(site, "index.html"), "utf8"), /Metsu/);
  } finally { rmSync(directory, {recursive: true, force: true}); }
});

test("Metsu static build refuses both normal source and a normal-mode runtime", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "metsu-reject-"));
  try {
    const prepared = runtime(directory, {mode: 0});
    const settings = {env: {...process.env, IRISU_METSU_RUNTIME_DIR: prepared}, stdio: "pipe", encoding: "utf8"};
    assert.throws(() => execFileSync(build, [path.join(web, "static")], settings),
      error => /unsafe or overlapping/.test(error.stdout + error.stderr));
    assert.throws(() => execFileSync(build, [path.join(directory, "site")], settings),
      error => /does not identify verified Metsu/.test(error.stdout + error.stderr));
    assert.equal(existsSync(path.join(directory, "site")), false);
  } finally { rmSync(directory, {recursive: true, force: true}); }
});

test("Metsu static build detects changed prepared runtime contents", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "metsu-checksum-"));
  try {
    const prepared = runtime(directory);
    writeFileSync(path.join(prepared, "metsu-runtime.json"), "changed");
    assert.throws(() => execFileSync(build, [path.join(directory, "site")],
      {env: {...process.env, IRISU_METSU_RUNTIME_DIR: prepared}, stdio: "pipe"}));
  } finally { rmSync(directory, {recursive: true, force: true}); }
});

test("Metsu runtime cache identity covers launcher, guest libraries, and pinned emulator bytes", () => {
  const script = `
import runpy,sys,tempfile
from pathlib import Path
module=runpy.run_path(sys.argv[1])
with tempfile.TemporaryDirectory() as directory:
    runtime=Path(directory); (runtime/'guest').mkdir()
    for name in module['RUNTIME_ASSETS']: (runtime/name).write_bytes(name.encode())
    for name in ['irisu-exact-worker','libc.so.6']: (runtime/'guest'/name).write_bytes(name.encode())
    identity={'mode':1,'prepared':False,'guest_files':['irisu-exact-worker','libc.so.6']}
    stamp=module['stamp_identity']; stamp(identity,runtime)
    original=identity['runtime_version']
    identity['prepared']=True; stamp(identity,runtime)
    assert identity['runtime_version']==original
    (runtime/'guest/irisu-exact-worker').write_bytes(b'new launcher')
    stamp(identity,runtime); launcher=identity['runtime_version']; assert launcher!=original
    (runtime/'guest/libc.so.6').write_bytes(b'new library')
    stamp(identity,runtime); library=identity['runtime_version']; assert library!=launcher
    (runtime/'v86.wasm').write_bytes(b'new emulator')
    stamp(identity,runtime); assert identity['runtime_version']!=library
`;
  execFileSync("python3", ["-c", script, path.join(web, "prepare-metsu-runtime.py")], {stdio: "pipe"});
});
