import assert from "node:assert/strict";
import test from "node:test";

import {
  activatedPalette, activatedTrailAlphas, bonusColorIntervalMs,
  bonusPalette, colorFor, hasActivatedTrail, palette,
} from "../static/colors.mjs";

test("custom block slots use their requested colors in both states", () => {
  assert.equal(palette[3], "#b227b5");
  assert.equal(activatedPalette[3], "#b227b5");
  assert.equal(palette[4], "#52aba7");
  assert.equal(activatedPalette[4], "#52aba7");
  assert.equal(palette[5], "#5b729c");
  assert.equal(activatedPalette[5], "#5b729c");
});

test("the bonus ball cycles through exactly five block colors", () => {
  assert.deepEqual(bonusPalette, [
    "#e44717", "#2945ff", "#eee116", "#b227b5", "#52aba7",
  ]);
  const bonus = {kind: "bonus"};
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5].map((step) => colorFor(bonus, step * bonusColorIntervalMs)),
    [...bonusPalette, bonusPalette[0]],
  );
});

test("only confirmed pieces receive six pronounced motion echoes", () => {
  assert.deepEqual(activatedTrailAlphas, [.1, .16, .24, .34, .47, .62]);
  assert.equal(hasActivatedTrail({kind: "piece", lifecycle: "confirmed"}), true);
  assert.equal(hasActivatedTrail({kind: "piece", lifecycle: "dynamic_fresh"}), false);
  assert.equal(hasActivatedTrail({kind: "piece", lifecycle: "scripted_falling"}), false);
  assert.equal(hasActivatedTrail({kind: "piece", lifecycle: "rotten"}), false);
  assert.equal(hasActivatedTrail({kind: "bonus", lifecycle: "confirmed"}), false);
  assert.equal(hasActivatedTrail({kind: "projectile", lifecycle: "confirmed"}), false);
});
