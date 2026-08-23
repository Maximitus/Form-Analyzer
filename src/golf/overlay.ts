import {formatSignedAngle, scoredKeypoint} from './kinematics';
import {COCO_INDEX} from './metrics';
import type {GolfFrontalMetrics, Keypoint2D} from './types';
import {GOLF_PHASE_LABEL} from './types';

function dashed(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function solid(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width = 3,
): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function chip(ctx: CanvasRenderingContext2D, text: string, x: number, y: number): void {
  ctx.save();
  ctx.font = 'bold 13px "Space Grotesk", "Segoe UI", system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  const width = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
  ctx.fillRect(x - 6, y - 11, width + 12, 22);
  ctx.fillStyle = '#f8fafc';
  ctx.fillText(text, x, y);
  ctx.restore();
}

export function drawGolfOverlay(
  ctx: CanvasRenderingContext2D,
  keypoints: readonly Keypoint2D[],
  metrics: GolfFrontalMetrics | null,
): void {
  if (!metrics) return;

  const leftShoulder = scoredKeypoint(keypoints, COCO_INDEX.leftShoulder, 0.3);
  const rightShoulder = scoredKeypoint(keypoints, COCO_INDEX.rightShoulder, 0.3);
  const leftHip = scoredKeypoint(keypoints, COCO_INDEX.leftHip, 0.3);
  const rightHip = scoredKeypoint(keypoints, COCO_INDEX.rightHip, 0.3);

  const {midShoulder, midHip, midAnkle, head} = metrics.landmarks;
  const height = ctx.canvas.height;
  const midlineX = midAnkle?.x ?? midHip?.x ?? null;
  const faceOn = metrics.camera === 'face-on';

  if (midlineX !== null) {
    dashed(ctx, midlineX, 8, midlineX, height - 8, 'rgba(250, 250, 250, 0.7)');
    chip(ctx, faceOn ? 'Frontal midline' : 'Ball line', midlineX + 8, 22);
  }

  if (leftShoulder && rightShoulder) {
    solid(ctx, leftShoulder.x, leftShoulder.y, rightShoulder.x, rightShoulder.y, '#38bdf8');
    const labelAt = midShoulder ?? leftShoulder;
    chip(ctx, `Shoulders ${formatSignedAngle(metrics.shoulderTiltDeg)}`, labelAt.x + 12, labelAt.y - 22);
  }

  if (leftHip && rightHip) {
    solid(ctx, leftHip.x, leftHip.y, rightHip.x, rightHip.y, '#fbbf24');
    const labelAt = midHip ?? leftHip;
    chip(ctx, `Pelvis ${formatSignedAngle(metrics.pelvicTiltDeg)}`, labelAt.x + 12, labelAt.y + 22);
  }

  if (midShoulder && midHip) {
    solid(ctx, midHip.x, midHip.y, midShoulder.x, midShoulder.y, '#a78bfa', 2);
  }

  if (head && midlineX !== null) {
    ctx.save();
    ctx.strokeStyle = '#f472b6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(head.x, head.y, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    dashed(ctx, head.x, head.y, midlineX, head.y, 'rgba(244, 114, 182, 0.85)');
  }

  const spineLabel = faceOn
    ? `side-bend ${formatSignedAngle(metrics.lateralTrunkFlexionDeg)}`
    : `spine ${formatSignedAngle(metrics.lateralTrunkFlexionDeg)}`;
  chip(ctx, `${GOLF_PHASE_LABEL[metrics.phase]} · ${spineLabel}`, 16, height - 36);

  const club = metrics.clubHead;
  if (club) {
    solid(ctx, club.gripX, club.gripY, club.x, club.y, '#f8fafc', 2);
    ctx.save();
    ctx.fillStyle = club.method === 'image' ? '#facc15' : 'rgba(250, 204, 21, 0.45)';
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(club.x, club.y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    chip(ctx, club.method === 'image' ? 'Club head' : 'Club (prior)', club.x + 12, club.y - 4);
  }
}
