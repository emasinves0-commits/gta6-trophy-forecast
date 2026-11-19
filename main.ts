// GTA VI PS5 trophy earn-rate forecaster: dilution/progression model on Exophase reads (Node 22, stdlib only).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

// ---- tunables (priors; everything data-dependent is fitted at runtime) ----
const K_DISTINCT = 6;        // distinct reads used in a fit
const RECENCY = 0.7;         // per-read weight decay (newest = 1)
const N_DAMP = 0.55;         // damping of players-growth slope (launch growth decelerates)
const E_DAMP = 0.7;          // damping of earners-growth slope
const SLOPE_TAU = 0.3;       // prior sd of a trophy's earners slope around the cross-trophy centre
const N_SLOPE_MAX = 1.5;
const NEVER_REFRESHED_SHIFT = -1.45; // logit shift when history is one unchanged page
const NEVER_REFRESHED_SIGMA = 0.65;
const BASE_SIGMA = 0.2;      // model-uncertainty floor (logit)
const N_SLOPE_SD = 0.32;     // uncertainty of players slope (per unit log-time)
const PLAT_SLOPE_CAP = 0.25; // platinum's earners grow very slowly

interface Read { at: string; players_tracked: number; rates: Array<number | null>; }
interface Case {
  levels: number[]; horizon_minutes: number; at: string; target: string;
  trophies: Array<{ id: number; tier: string }>; history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const expit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const hours = (iso: string, t0: number): number => (Date.parse(iso) - t0) / 3.6e6;

function wmedian(v: number[]): number {
  if (v.length === 0) return 0;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Slope through the anchor (last point): minimise sum w (dy - s dx)^2. Returns slope and its standard error. */
function anchoredSlope(dx: number[], dy: number[], w: number[], relErr: number[]): { s: number; se: number } {
  let num = 0, den = 0;
  for (let i = 0; i < dx.length; i++) { num += w[i] * dx[i] * dy[i]; den += w[i] * dx[i] * dx[i]; }
  if (den < 1e-9) return { s: 0, se: 1 };
  // error propagation: slope = sum(w dx dy)/den, dy errors = relErr_i (+ anchor error, folded in relErr)
  let v = 0;
  for (let i = 0; i < dx.length; i++) v += (w[i] * dx[i]) ** 2 * relErr[i] ** 2;
  return { s: num / den, se: Math.sqrt(v) / den };
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  const hist = c.history;
  const nT = c.trophies.length;
  const last = hist[hist.length - 1];
  const t0 = Date.parse(hist[0].at);
  const xOf = (iso: string): number => Math.log(1 + Math.max(0, hours(iso, t0)));
  const xT = xOf(c.target);

  // Distinct states (a repeated page is a stale read): keep the first read of each run.
  const key = (r: Read): string => `${r.players_tracked}|${r.rates.join(',')}`;
  const distinct: Read[] = [];
  let lastChangeAt = hist[0].at;
  for (const r of hist) {
    if (distinct.length === 0 || key(r) !== key(distinct[distinct.length - 1])) { distinct.push(r); lastChangeAt = r.at; }
  }
  const staleHours = Math.max(0, hours(last.at, 0) - hours(lastChangeAt, 0));
  const D = distinct.slice(-K_DISTINCT);
  const L = D[D.length - 1];
  const xL = xOf(L.at);
  const u = Math.max(0, xT - xL); // extrapolation in log-time

  const persist = (extra: (i: number) => { shift: number; sigma: number }) =>
    c.trophies.map((t, i) => {
      const r0 = last.rates[i] ?? 0;
      const { shift, sigma } = extra(i);
      return { id: t.id, r0, shift, sigma };
    });

  type Spec = { id: number; r0: number; shift: number; sigma: number; hi?: boolean };
  let specs: Spec[];

  if (D.length < 2) {
    // Never-refreshed page: the next refresh will show many more players; rates fall sharply.
    specs = persist(() => ({ shift: NEVER_REFRESHED_SHIFT, sigma: NEVER_REFRESHED_SIGMA }));
  } else {
    const dx = D.slice(0, -1).map((r) => xOf(r.at) - xL);
    const w = D.slice(0, -1).map((_, k, a) => RECENCY ** (a.length - 1 - k));
    // players growth
    const lnN = (r: Read): number => Math.log(r.players_tracked);
    const dyN = D.slice(0, -1).map((r) => lnN(r) - lnN(L));
    const nFit = anchoredSlope(dx, dyN, w, dx.map(() => 0.02));
    const sN = Math.min(N_SLOPE_MAX, Math.max(0, nFit.s)) * N_DAMP;
    const lnNgrowth = sN * u;

    // earners growth per trophy
    const raw: Array<{ s: number; se: number; ok: boolean }> = [];
    for (let i = 0; i < nT; i++) {
      const rL = L.rates[i];
      if (rL == null || rL <= 0) { raw.push({ s: 0, se: 1, ok: false }); continue; }
      const eL = (rL * L.players_tracked) / 100;
      const pts: number[] = [], ys: number[] = [], ws: number[] = [], es: number[] = [];
      for (let k = 0; k < D.length - 1; k++) {
        const r = D[k].rates[i];
        if (r == null || r <= 0) continue;
        const e = (r * D[k].players_tracked) / 100;
        pts.push(dx[k]); ys.push(Math.log(e / eL)); ws.push(w[k]);
        const rel = Math.hypot(Math.min(1, 0.005 / r + 1 / Math.sqrt(Math.max(e, 1))), Math.min(1, 0.005 / rL + 1 / Math.sqrt(Math.max(eL, 1))));
        es.push(rel);
      }
      if (pts.length === 0) { raw.push({ s: 0, se: 1, ok: false }); continue; }
      const f = anchoredSlope(pts, ys, ws, es);
      raw.push({ s: f.s, se: f.se, ok: true });
    }
    const centre = wmedian(raw.filter((r) => r.ok && r.se < 0.5).map((r) => r.s));
    const centreUse = raw.some((r) => r.ok && r.se < 0.5) ? centre : sN;
    const staleF = Math.min(1, staleHours / 12);

    specs = c.trophies.map((t, i) => {
      const r0 = L.rates[i] ?? last.rates[i] ?? 0;
      const rr = raw[i];
      let s: number, postVar: number;
      if (!rr.ok) { s = centreUse; postVar = SLOPE_TAU ** 2; }
      else {
        const wt = SLOPE_TAU ** 2 / (SLOPE_TAU ** 2 + rr.se ** 2);
        s = wt * rr.s + (1 - wt) * centreUse;
        postVar = wt * rr.se ** 2;
      }
      s = Math.max(0, s) * E_DAMP;
      if (t.tier === 'platinum') s = Math.min(s, PLAT_SLOPE_CAP);
      s = Math.min(s, sN + 0.05); // earners cannot outgrow players for long
      const noRate = !(r0 > 0);
      const shiftLn = (s * u - lnNgrowth); // ln(E ratio) - ln(N ratio)
      // convert to logit shift: for small rates logit shift == ln rate shift
      const thin = D.length <= 2 ? 1 : 0; // a two-read fit (one of them stale) is weak evidence: lean toward more dilution
      let shift = shiftLn - 0.25 * staleF - 0.12 * thin * Math.min(1, u);
      let sigma2 = BASE_SIGMA ** 2 + (N_SLOPE_SD * u) ** 2 + (u ** 2) * postVar + (0.3 * staleF) ** 2;
      if (noRate) { sigma2 += 0.5 ** 2; }
      return { id: t.id, r0, shift, sigma: Math.sqrt(sigma2) };
    });
  }

  return specs.map((sp) => {
    const r0 = sp.r0;
    let q: number[];
    if (r0 >= 99) {
      // essentially everyone: stays near 100
      q = [99.9, 99.97, 99.99, 99.99, 100];
      if (r0 < 99.99) q = Z.map((z) => round4(expit(logit(r0) + Math.min(0, sp.shift) * 0.3 + z * 0.4)));
    } else if (!(r0 > 0)) {
      // zero earners in every read: small prior rate, wide spread
      const base = Math.log(0.015 / 99.985) + Math.min(0, sp.shift) * 0.5;
      q = Z.map((z) => expit(base + z * Math.max(sp.sigma, 0.9)));
    } else {
      const l0 = logit(r0);
      // rounding of the printed target rate adds noise, esp. near the 0.01 floor
      const lm = l0 + sp.shift;
      const rt = expit(lm);
      const rnd = Math.min(0.8, 0.0029 / Math.max(rt, 0.003));
      const sig = Math.hypot(sp.sigma, rnd);
      q = Z.map((z) => expit(lm + z * sig));
    }
    q = q.map((v) => round4(Math.min(100, Math.max(0.01, v))));
    for (let j = 1; j < 5; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: sp.id, q };
  });
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const finiteRate = (v: unknown): number => (isNum(v) ? v : 0);

/** Persistence forecast: all five quantiles equal the trophy's last rate, clamped to 0.01..99.99. */
function persistence(c: Case, i: number): number[] {
  const hist = Array.isArray(c.history) ? c.history : [];
  const v = round4(clamp(finiteRate(hist[hist.length - 1]?.rates?.[i])));
  return [v, v, v, v, v];
}

const quantilesOk = (q: unknown): q is number[] =>
  Array.isArray(q) && q.length === 5 && q.every((v) => isNum(v) && v >= 0 && v <= 100);

/** Final per-row validation: finite, within 0..100, non-decreasing; repair what is repairable, else persistence. */
function validated(c: Case, i: number, q: unknown): number[] {
  if (!Array.isArray(q) || q.length !== 5 || !q.every(isNum)) return persistence(c, i);
  const fixed = q.map((v) => round4(Math.min(100, Math.max(0.01, v))));
  for (let j = 1; j < 5; j++) if (fixed[j] < fixed[j - 1]) fixed[j] = fixed[j - 1];
  return quantilesOk(fixed) ? fixed : persistence(c, i);
}

function solve(c: Case): Array<{ id: number; q: number[] }> {
  let raw: Array<{ id: number; q: number[] }> | null = null;
  try { raw = forecast(c); } catch { raw = null; }
  return c.trophies.map((t, i) => {
    const row = raw?.[i];
    // non-finite anywhere in a trophy's quantiles -> that trophy falls back to persistence
    const q = row && row.id === t.id && Array.isArray(row.q) && row.q.length === 5 && row.q.every(isNum)
      ? row.q : persistence(c, i);
    return { id: t.id, q: validated(c, i, q) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
  const id = file.slice(0, -3);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: solve(c) })}\n`);
}
