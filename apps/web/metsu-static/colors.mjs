// Nine stable hues: falling, active, and EX bodies keep their color identity.
export const palette = [
  "#db7768", "#7395dc", "#d2bc62", "#b68acb",
  "#68b5a2", "#a4bbd1", "#57a8be", "#cf87a4", "#d59a61",
];

export const activatedPalette = [
  "#ff9988", "#a1b9ff", "#f4df89", "#deb0f5",
  "#8bdac5", "#d0e4f4", "#87d8ea", "#f4b1cf", "#f6c18d",
];

export const bonusPalette = [
  activatedPalette[0], activatedPalette[1], activatedPalette[2],
  activatedPalette[3], activatedPalette[4],
];

export const bonusColorIntervalMs = 400;
export const activatedTrailAlphas = [.1, .16, .24, .34, .47, .62];

function isActivatedBlock(body) {
  return body.kind === "piece" &&
    (body.lifecycle === "dynamic_fresh" || body.lifecycle === "confirmed");
}

export function hasActivatedTrail(body) {
  return body.kind === "piece" && body.lifecycle === "confirmed";
}

export function colorFor(body, now = 0) {
  if (body.kind === "projectile") return "#f4eedb";
  if (body.kind === "bonus") {
    return bonusPalette[Math.floor(now / bonusColorIntervalMs) % bonusPalette.length];
  }
  const index = ((body.color % palette.length) + palette.length) % palette.length;
  return (isActivatedBlock(body) ? activatedPalette : palette)[index];
}
