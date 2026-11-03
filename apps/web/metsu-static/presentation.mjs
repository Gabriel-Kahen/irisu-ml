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
  const walls = metsuWallRects(field);
  ctx.save();
  for (const rect of walls) {
    ctx.fillStyle = "#29372e";
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.strokeStyle = "#5e70574d";
    ctx.lineWidth = 1;
    ctx.strokeRect(rect.x + .5, rect.y + .5, rect.width - 1, rect.height - 1);
  }
  const floor = walls[2];
  ctx.fillStyle = "#d6b46c80";
  ctx.fillRect(floor.x, floor.y, floor.width, 1);
  ctx.fillStyle = "#cad9b51c";
  ctx.fillRect(field.x + 15, field.y, 1, field.height);
  ctx.fillRect(field.x + field.width + 8, field.y, 1, field.height);
  // Small contact-line markers give the floor an edge without hiding EX bodies.
  ctx.fillStyle = "#d6b46c66";
  for (let x = floor.x + 10; x < floor.x + floor.width - 6; x += 12) {
    ctx.fillRect(x, floor.y + 5, 3, 1);
  }
  ctx.restore();
}
