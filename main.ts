// GTA VI trophy earn-rate forecaster (make.fun: gta6-trophy-rarity-forecast-01).
//
// Reads every case file <id>.in in the input directory and writes one
// <id>.out forecast per case. Each forecast gives five quantiles of every
// trophy's Exophase earn rate (percent) at the case's target instant.
//
// Approach, per trophy, all in logit space (logit(p) = ln(p / (100 - p))):
//   1. Collapse the read history into distinct snapshots: Exophase refreshes
//      its counts on its own schedule, so long runs of identical reads carry
//      no new information, while each jump between snapshots is one refresh.
//   2. Fit a weighted straight line of logit(rate) against
//      ln(1 + hours since release), weighting recent snapshots more, and
//      read the line at the target. Trophy earn rates rise fast just after
//      release and then flatten, which is what the log-time axis captures.
//      The fitted median is blended with the last observed rate so a weak
//      fit cannot drag the forecast far from what was last seen.
//   3. Spread: a random-walk term that grows with the horizon, plus a
//      refresh term built from the typical size of the logit jumps between
//      snapshots (scaled by how many refreshes the horizon should hold),
//      plus the fit's residual scatter. A floor keeps the band honest on
//      short horizons where the history has seen no refresh yet -- the
//      launch-night practice cases are exactly that trap: every read is the
//      same stale page, and the answer is the first refreshed page.
// Deterministic: pure arithmetic, no randomness, no clock reads, no I/O
// beyond the case files. Node standard library only.

import * as fs from "node:fs";
import * as path from "node:path";

const LEVELS: number[] = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z: number[] = [-1.645, -0.674, 0, 0.674, 1.645];
const RATE_FLOOR = 0.01;
const RATE_CEIL = 99.99;
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const HOUR_MS = 3_600_000;
const DAY_MINUTES = 1440;

// Tuning constants, fixed in advance (never fitted to the practice answers,
// which are not in the kit).
const WALK_SIGMA = 0.5; // logit units per sqrt(day), the random-walk term
const SPREAD_FLOOR = 0.3; // minimum logit spread, even on a 90-minute horizon
const SPREAD_CAP = 2.0; // maximum logit spread
const MAX_MEDIAN_MOVE = 2.5; // logit units the median may move from the last rate
const SLOPE_CLAMP = 2.0; // per unit of log-time
const FIT_WINDOW_HOURS = 48; // only snapshots this recent feed the trend fit
const RECENCY_HALF_LIFE_HOURS = 24;
const MAX_REFRESHES_COUNTED = 24; // cap on refreshes feeding the spread
const DEFAULT_JUMP = 0.35; // typical refresh jump when history shows none
const DEFAULT_REFRESH_HOURS = 4; // assumed refresh cadence when none seen

interface TrophyDef {
  id: number;
}

interface CaseDoc {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: TrophyDef[];
  history: Array<{
    at: string;
    players_tracked: number;
    rates: Array<number | null>;
  }>;
}

function clampRate(x: number): number {
  if (!Number.isFinite(x)) return RATE_FLOOR;
  return Math.min(RATE_CEIL, Math.max(RATE_FLOOR, x));
}

function logitOf(ratePercent: number): number {
  const c = clampRate(ratePercent);
  return Math.log(c / (100 - c));
}

function rateFromLogit(l: number): number {
  const v = 100 / (1 + Math.exp(-l));
  return Math.min(RATE_CEIL, Math.max(RATE_FLOOR, v));
}

function round4(x: number): number {
  const r = Math.round(x * 10000) / 10000;
  // Keep the value inside the scorer's clamp and the 0..100 output range.
  return Math.min(99.99, Math.max(0.01, r));
}

// Log-time coordinate: hours since release, on a log scale that starts at 0.
function logTime(ms: number): number {
  const hours = Math.max(0, (ms - RELEASE_MS) / HOUR_MS);
  return Math.log(1 + hours);
}

interface Snapshot {
  ms: number;
  logit: number;
}

// Collapse consecutive equal rates into one snapshot each, keeping the time
// the new value first appeared.
function snapshotsFor(series: Array<{ ms: number; rate: number | null }>): Snapshot[] {
  const out: Snapshot[] = [];
  for (const point of series) {
    if (point.rate === null || !Number.isFinite(point.rate)) continue;
    const l = logitOf(point.rate);
    if (out.length > 0 && Math.abs(l - out[out.length - 1].logit) < 1e-12) continue;
    out.push({ ms: point.ms, logit: l });
  }
  return out;
}

// Weighted least squares of y on x. Returns null when the fit is degenerate.
function weightedLine(
  xs: number[],
  ys: number[],
  ws: number[],
): { intercept: number; slope: number; residStd: number } | null {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw <= 0) return null;
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ss += ws[i] * r * r;
  }
  return { intercept, slope, residStd: Math.sqrt(ss / sw) };
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

interface ForecastRow {
  id: number;
  q: number[];
}

function forecastCase(doc: CaseDoc): ForecastRow[] {
  const checkpointMs = Date.parse(doc.at);
  const targetMs = Date.parse(doc.target);
  const horizonDays = Math.max(1 / 1440, doc.horizon_minutes / DAY_MINUTES);
  const targetX = logTime(targetMs);

  // Pool refresh behaviour across all trophies: typical jump size between
  // snapshots and typical gap between snapshots.
  const allJumps: number[] = [];
  const allGapsHours: number[] = [];
  const perTrophy: Snapshot[][] = [];
  for (let j = 0; j < doc.trophies.length; j++) {
    const series = doc.history.map((r) => ({
      ms: Date.parse(r.at),
      rate: r.rates[j] ?? null,
    }));
    const snaps = snapshotsFor(series);
    perTrophy.push(snaps);
    for (let i = 1; i < snaps.length; i++) {
      allJumps.push(Math.abs(snaps[i].logit - snaps[i - 1].logit));
      allGapsHours.push((snaps[i].ms - snaps[i - 1].ms) / HOUR_MS);
    }
  }
  const typicalJump = allJumps.length > 0 ? medianOf(allJumps) : DEFAULT_JUMP;
  const refreshHours =
    allGapsHours.length > 0 ? Math.max(0.25, medianOf(allGapsHours)) : DEFAULT_REFRESH_HOURS;
  const expectedRefreshes = (doc.horizon_minutes / 60) / refreshHours;
  const refreshSigma =
    typicalJump * Math.sqrt(Math.min(expectedRefreshes, MAX_REFRESHES_COUNTED));
  const walkSigma = WALK_SIGMA * Math.sqrt(horizonDays);

  const rows: ForecastRow[] = [];
  for (let j = 0; j < doc.trophies.length; j++) {
    const snaps = perTrophy[j];
    const lastLogit = snaps.length > 0 ? snaps[snaps.length - 1].logit : logitOf(0);

    // Trend fit on recent distinct snapshots.
    let medianLogit = lastLogit;
    let residStd = 0;
    const windowStart = checkpointMs - FIT_WINDOW_HOURS * HOUR_MS;
    const fitSnaps = snaps.filter((s) => s.ms >= windowStart);
    if (fitSnaps.length >= 3) {
      const xs = fitSnaps.map((s) => logTime(s.ms));
      const ys = fitSnaps.map((s) => s.logit);
      const ws = fitSnaps.map(
        (s) => Math.pow(0.5, (checkpointMs - s.ms) / (RECENCY_HALF_LIFE_HOURS * HOUR_MS)),
      );
      const fit = weightedLine(xs, ys, ws);
      if (fit) {
        const slope = Math.min(SLOPE_CLAMP, Math.max(-SLOPE_CLAMP, fit.slope));
        const predicted = fit.intercept + slope * targetX;
        // Trust the fit more the more distinct snapshots support it.
        const fitWeight = Math.min(0.8, 0.35 + 0.09 * fitSnaps.length);
        medianLogit = fitWeight * predicted + (1 - fitWeight) * lastLogit;
        residStd = Math.min(0.6, fit.residStd);
      }
    }
    // Never let the median run away from the last observed rate.
    medianLogit = Math.min(
      lastLogit + MAX_MEDIAN_MOVE,
      Math.max(lastLogit - MAX_MEDIAN_MOVE, medianLogit),
    );

    const sigma = Math.min(
      SPREAD_CAP,
      Math.max(SPREAD_FLOOR, Math.sqrt(walkSigma * walkSigma + refreshSigma * refreshSigma + residStd * residStd)),
    );

    const q = Z.map((z) => round4(rateFromLogit(medianLogit + z * sigma)));
    // Rounding can only tie neighbours; enforce non-decreasing explicitly.
    for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
    rows.push({ id: doc.trophies[j].id, q });
  }
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const doc = JSON.parse(
      fs.readFileSync(path.join(inputDir, file), "utf8"),
    ) as CaseDoc;
    const rows = forecastCase(doc);
    const outPath = path.join(outputDir, `${doc.id}.out`);
    fs.writeFileSync(outPath, JSON.stringify({ trophies: rows }), "utf8");
  }
}

main();
