import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import path from "node:path";
import test from "node:test";
import {fileURLToPath} from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const index = readFileSync(path.join(web, "static/index.html"), "utf8");
const analytics = readFileSync(path.join(web, "static/analytics.js"), "utf8");

test("GA4 tag is present and allowed by the content security policy", () => {
  assert.match(index, /googletagmanager\.com\/gtag\/js\?id=G-DCS6EVRYKH/);
  assert.match(index, /src="\.\/analytics\.js"/);
  assert.match(index, /script-src[^;]+https:\/\/www\.googletagmanager\.com/);
  assert.match(index, /connect-src[^;]+https:\/\/\*\.google-analytics\.com/);
  assert.match(index, /connect-src[^;]+https:\/\/\*\.analytics\.google\.com/);
  assert.match(analytics, /gtag\("config", "G-DCS6EVRYKH"\)/);
});
