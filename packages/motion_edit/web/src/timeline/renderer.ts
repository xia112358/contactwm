import type { Session } from '../core/types';

export type TimelineMetrics = {
  width: number;
  height: number;
  labelW: number;
  rulerH: number;
  rowH: number;
  plotW: number;
  rows: number;
};

type TimelineRenderState = {
  canvas: HTMLCanvasElement;
  session: Session | null;
  frame: number;
  stableCuts: number[];
  partColors: number[];
};

// Session masks are immutable until the next API payload. Cache interval scans
// so scrubbing a long rollout does not rescan every frame of every channel.
const intervalCache = new WeakMap<Session, [number, number][][]>();
function contactIntervals(session: Session): [number, number][][] {
  const cached = intervalCache.get(session);
  if (cached) return cached;
  const lanes = session.contact_force.part_order.map((_, lane) => {
    const intervals: [number, number][] = [];
    let start = -1;
    for (let frame = 0; frame <= session.qpos.length; frame++) {
      const active = frame < session.qpos.length && Boolean(session.contact_force.masks[frame]?.[lane]);
      if (active && start < 0) start = frame;
      if (!active && start >= 0) { intervals.push([start, frame]); start = -1; }
    }
    return intervals;
  });
  intervalCache.set(session, lanes); return lanes;
}

export function getTimelineMetrics(canvas: HTMLCanvasElement, rows = 8): TimelineMetrics {
  const rect = canvas.getBoundingClientRect();
  const laneCount = Math.max(1, rows);
  const labelW = rect.width < 560 ? 54 : 68;
  const rulerH = 28;
  return {
    width: rect.width,
    height: rect.height,
    labelW,
    rulerH,
    rowH: Math.max(1, (rect.height - rulerH) / laneCount),
    plotW: Math.max(1, rect.width - labelW),
    rows: laneCount,
  };
}

export function frameToX(value: number, metrics: TimelineMetrics, frameCount: number): number {
  return metrics.labelW + value / Math.max(1, frameCount - 1) * metrics.plotW;
}

export function xToFrame(value: number, metrics: TimelineMetrics, frameCount: number): number {
  const ratio = Math.max(0, Math.min(1, (value - metrics.labelW) / metrics.plotW));
  return Math.round(ratio * Math.max(0, frameCount - 1));
}

function niceStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-6)));
  const scaled = raw / power;
  return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;
}

export function formatTimelineTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return minutes
    ? `${minutes}:${rest.toFixed(rest < 10 ? 1 : 0).padStart(rest < 10 ? 4 : 2, '0')}`
    : `${rest.toFixed(rest < 10 ? 1 : 0)}s`;
}

export function renderTimeline(state: TimelineRenderState): void {
  const { canvas, session, frame, stableCuts, partColors } = state;
  const metrics = getTimelineMetrics(canvas, session?.contact_force.part_order.length ?? 8);
  const context = canvas.getContext('2d');
  if (!context) return;

  const dpr = Math.min(window.devicePixelRatio, 2);
  const width = Math.max(1, Math.round(metrics.width * dpr)), height = Math.max(1, Math.round(metrics.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, metrics.width, metrics.height);
  context.fillStyle = '#0b1117';
  context.fillRect(0, 0, metrics.width, metrics.height);
  if (!session) return;

  const frameCount = session.qpos.length;
  context.fillStyle = '#0d141b';
  context.fillRect(0, 0, metrics.labelW, metrics.height);
  context.fillStyle = '#111923';
  context.fillRect(metrics.labelW, 0, metrics.plotW, metrics.rulerH);

  for (let lane = 0; lane < metrics.rows; lane++) {
    const y = metrics.rulerH + lane * metrics.rowH;
    context.fillStyle = lane % 2 === 0 ? '#121a22' : '#0f171f';
    context.fillRect(metrics.labelW, y, metrics.plotW, metrics.rowH);
    context.strokeStyle = '#22303c';
    context.beginPath();
    context.moveTo(0, Math.round(y) + 0.5);
    context.lineTo(metrics.width, Math.round(y) + 0.5);
    context.stroke();
    context.fillStyle = `#${partColors[lane % partColors.length].toString(16).padStart(6, '0')}`;
    context.fillRect(8, y + metrics.rowH / 2 - 3, 6, 6);
    context.fillStyle = '#aebbc7';
    context.font = '600 10px ui-monospace,monospace';
    context.textBaseline = 'middle';
    context.fillText(session.contact_force.part_order[lane] || '', 20, y + metrics.rowH / 2);
  }

  const duration = Math.max(1, frameCount - 1) / session.fps;
  const majorSeconds = niceStep(duration / Math.max(2, Math.floor(metrics.plotW / 92)));
  const majorFrames = Math.max(1, Math.round(majorSeconds * session.fps));
  const minorFrames = Math.max(1, Math.round(majorFrames / 5));
  const ticks = new Set<number>();
  for (let tick = 0; tick < frameCount; tick += minorFrames) ticks.add(tick);
  for (let tick = 0; tick < frameCount; tick += majorFrames) ticks.add(tick);
  for (const tick of [...ticks].sort((a, b) => a - b)) {
    const x = Math.round(frameToX(tick, metrics, frameCount)) + 0.5;
    const major = tick % majorFrames === 0;
    context.strokeStyle = major ? '#344451' : '#1d2a34';
    context.globalAlpha = major ? 0.9 : 0.65;
    context.beginPath();
    context.moveTo(x, major ? metrics.rulerH - 7 : metrics.rulerH - 4);
    context.lineTo(x, metrics.height);
    context.stroke();
    context.globalAlpha = 1;
    if (major) {
      context.fillStyle = '#aeb5bd';
      context.font = '10px ui-monospace,monospace';
      context.fillText(formatTimelineTime(tick / session.fps), Math.min(x + 4, metrics.width - 38), 10);
      context.fillStyle = '#6f7780';
      context.font = '9px ui-monospace,monospace';
      context.fillText(String(tick), Math.min(x + 4, metrics.width - 32), 21);
    }
  }
  context.fillStyle = '#7e8790';
  context.font = '9px ui-monospace,monospace';
  context.fillText('TIME', 8, 9);
  context.fillText('FRAME', 8, 21);

  const intervals = contactIntervals(session);
  for (let lane = 0; lane < intervals.length; lane++) {
    for (const [start, tick] of intervals[lane]) {
        const x0 = frameToX(start, metrics, frameCount);
        const x1 = frameToX(Math.max(start, tick - 1), metrics, frameCount) + Math.max(1, metrics.plotW / frameCount);
        context.fillStyle = `#${partColors[lane % partColors.length].toString(16).padStart(6, '0')}`;
        context.globalAlpha = 0.78;
        context.beginPath();
        context.roundRect(x0, metrics.rulerH + lane * metrics.rowH + 3, Math.max(2, x1 - x0), Math.max(2, metrics.rowH - 6), 2);
        context.fill();
        context.globalAlpha = 1;
    }
  }

  if (!session.contact_force.part_order.length || (session.contact_force.mask_status && session.contact_force.mask_status !== 'verified_newton_primary_faces')) {
    context.fillStyle = '#8c9aaa'; context.font = '12px sans-serif';
    context.fillText('接触状态未知 · 时间轴用于回放定位', metrics.labelW + 12, metrics.rulerH + 24);
    for (const segment of session.review?.segments ?? []) {
      const x0 = frameToX(segment.start, metrics, frameCount), x1 = frameToX(segment.end, metrics, frameCount);
      context.fillStyle = '#27505e';
      context.fillRect(x0, metrics.rulerH + 42, Math.max(2, x1 - x0), 18);
      if (x1 - x0 > 36) { context.fillStyle = '#dce7ed'; context.fillText(segment.label, x0 + 3, metrics.rulerH + 55); }
    }
  }

  context.setLineDash([3, 3]);
  for (const cut of stableCuts) {
    const x = Math.round(frameToX(cut, metrics, frameCount)) + 0.5;
    context.strokeStyle = '#d9b96e';
    context.globalAlpha = 0.38;
    context.beginPath();
    context.moveTo(x, metrics.rulerH);
    context.lineTo(x, metrics.height);
    context.stroke();
  }
  context.setLineDash([]);
  context.globalAlpha = 1;

  const cursorX = Math.round(frameToX(frame, metrics, frameCount)) + 0.5;
  context.strokeStyle = '#72ead3';
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(cursorX, 0);
  context.lineTo(cursorX, metrics.height);
  context.stroke();
  context.fillStyle = '#72ead3';
  context.beginPath();
  context.moveTo(cursorX - 5, 0);
  context.lineTo(cursorX + 5, 0);
  context.lineTo(cursorX, 7);
  context.closePath();
  context.fill();
  context.lineWidth = 1;
}
