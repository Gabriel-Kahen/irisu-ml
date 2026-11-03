export const METSU_GEOMETRY = Object.freeze({
  thickness: 16, blank: 30, bottomHeight: 400, topWidth: 450,
  topY: -140, topHeight: 300,
});

export function isExBody(body) {
  return body.ex === true || (body.kind === "piece" && body.shape === "circle");
}

export function metsuWallRects(field) {
  const {thickness: thick, blank, bottomHeight, topWidth, topY, topHeight} = METSU_GEOMETRY;
  const centerX = field.x + Math.trunc(field.width / 2) + thick;
  return [
    {x: field.x, y: field.y, width: thick, height: field.height},
    {x: field.x + field.width + thick / 2, y: field.y, width: thick, height: field.height},
    {x: field.x, y: field.y + field.height + blank,
      width: field.width + 2 * thick, height: bottomHeight},
    {x: centerX - topWidth / 2, y: topY - topHeight / 2,
      width: topWidth, height: topHeight},
  ];
}

export function drawMetsuWalls(ctx, field) {
  ctx.fillStyle = "#f3f3ef";
  for (const rect of metsuWallRects(field)) {
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  }
  ctx.fillStyle = "#cad0cd55";
  ctx.fillRect(field.x, field.y, 3, field.height);
  ctx.fillRect(field.x + field.width + 8, field.y, 3, field.height);
}
