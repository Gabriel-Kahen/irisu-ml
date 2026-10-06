import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import path from "node:path";
import test from "node:test";
import {fileURLToPath} from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = readFileSync(path.join(web, "static/app.js"), "utf8");

test("score events rise and fade from the cleared block", () => {
  assert.match(app, /event\.kind_name === "score_changed" && event\.value > 0/);
  assert.match(app, /event\.detail === "normal burst actor teardown"/);
  assert.match(app, /bodyPositions\.get\(event\.a\)/);
  assert.match(app, /ctx\.globalAlpha = 1 - progress/);
  assert.match(app, /popup\.y - rise/);
  assert.match(app, /ctx\.font = "900 26px Georgia, serif"/);
  assert.match(app, /String\(popup\.value\)/);
  assert.doesNotMatch(app, /`\+\$\{popup\.value\}`/);
});
