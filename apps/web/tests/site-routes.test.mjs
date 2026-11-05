import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createHash} from "node:crypto";
import {existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import {fileURLToPath} from "node:url";
import test from "node:test";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const build = path.join(web, "build-site.sh");
const sha = value => createHash("sha256").update(value).digest("hex");

function fixture(directory) {
  const normal = path.join(directory, "normal-runtime");
  const metsu = path.join(directory, "metsu-runtime");
  for (const target of [normal, metsu]) mkdirSync(target);
  writeFileSync(path.join(normal, ".irisu-exact-runtime"), "fixture");
  writeFileSync(path.join(normal, "normal-worker"), "unchanged normal bytes");
  writeFileSync(path.join(normal, "SHA256SUMS"), `${sha("unchanged normal bytes")}  normal-worker\n`);
  const manifest = JSON.stringify({mode: 1, prepared: true,
    config_hash: "1958987580749614460", backend: "exact-msvc9-r58-metsu-forward"});
  writeFileSync(path.join(metsu, ".irisu-metsu-runtime"), "fixture");
  writeFileSync(path.join(metsu, "metsu-runtime.json"), manifest);
  writeFileSync(path.join(metsu, "SHA256SUMS"), `${sha(manifest)}  metsu-runtime.json\n`);
  return {normal, metsu, env: {...process.env,
    IRISU_EXACT_RUNTIME_DIR: normal, IRISU_METSU_RUNTIME_DIR: metsu}};
}

test("combined site serves normal at root and alias with a separate Metsu entry", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "irisu-routes-"));
  try {
    const {env} = fixture(directory);
    const site = path.join(directory, "site");
    execFileSync(build, [site], {env, stdio: "pipe"});
    const normal = readFileSync(path.join(site, "index.html"), "utf8");
    const alias = readFileSync(path.join(site, "normal/index.html"), "utf8");
    const metsu = readFileSync(path.join(site, "metsu/index.html"), "utf8");
    assert.equal(alias.replace('  <base href="../">\n', ""), normal);
    assert.match(normal, /href="\.\/normal\/" aria-current="page"/);
    assert.match(normal, /href="\.\/metsu\/"/);
    assert.match(metsu, /href="\.\.\/normal\/"/);
    assert.match(metsu, /href="\.\/" aria-current="page"/);
    assert.doesNotMatch(metsu, /<base /);
    assert.equal(readFileSync(path.join(site, "exact-runtime/normal-worker"), "utf8"), "unchanged normal bytes");
    assert.equal(JSON.parse(readFileSync(path.join(site, "metsu/exact-runtime/metsu-runtime.json"))).mode, 1);
    assert.equal(existsSync(path.join(site, "normal/exact-runtime")), false);
    assert.equal(readFileSync(path.join(site, "CNAME"), "utf8").trim(), "irisu.online");
    // Resolve both menu links and normal assets from root and project-path hosts.
    for (const prefix of ["https://irisu.online/", "https://example.test/project/"]) {
      const base = new URL("../", new URL("normal/", prefix));
      assert.equal(new URL("./app.js", base).href, new URL("app.js", prefix).href);
      assert.equal(new URL("./metsu/", base).href, new URL("metsu/", prefix).href);
      assert.equal(new URL("../normal/", new URL("metsu/", prefix)).href, new URL("normal/", prefix).href);
    }
  } finally { rmSync(directory, {recursive: true, force: true}); }
});

test("failed Metsu validation preserves an existing normal site", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "irisu-routes-failure-"));
  try {
    const {env, metsu} = fixture(directory);
    const site = path.join(directory, "site"); mkdirSync(site);
    writeFileSync(path.join(site, ".irisu-static-site"), "existing build");
    writeFileSync(path.join(site, "index.html"), "previous working site");
    writeFileSync(path.join(metsu, "metsu-runtime.json"), "corrupted runtime");
    assert.throws(() => execFileSync(build, [site], {env, stdio: "pipe"}));
    assert.equal(readFileSync(path.join(site, "index.html"), "utf8"), "previous working site");
    assert.equal(existsSync(path.join(site, "metsu")), false);
  } finally { rmSync(directory, {recursive: true, force: true}); }
});

test("combined build rejects source/runtime overlap and unmarked outputs", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "irisu-routes-paths-"));
  try {
    const {env, normal, metsu} = fixture(directory);
    const unmarked = path.join(directory, "unmarked"); mkdirSync(unmarked);
    writeFileSync(path.join(unmarked, "keep"), "keep");
    for (const target of [normal, metsu, path.join(web, "static"), unmarked]) {
      assert.throws(() => execFileSync(build, [target], {env, stdio: "pipe"}));
    }
    assert.equal(readFileSync(path.join(unmarked, "keep"), "utf8"), "keep");
  } finally { rmSync(directory, {recursive: true, force: true}); }
});
