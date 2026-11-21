// GTA VI PS5 trophy earn-rate forecaster: dilution/progression model on Exophase reads (Node 22, stdlib only).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

// ---- tunables (priors; everything data-dependent is fitted at runtime) ----
const K_DISTINCT = 40;       // distinct reads kept for a fit
const WIN_H = 36;            // fit window (hours before the last distinct read)
const REC_H = 18;            // recency e-folding (hours) of fit weights
const TAU_N0 = 24;           // prior e-folding (h) of players-growth deceleration; refined from the data when the window shows it
const TAU_E = 72;            // e-folding (h) of earners-growth decay (progression persists; saturation does the capping)
const A_PRIOR_SD = 0.04;     // sd (per hour) of a trophy's earners log-growth around the cross-trophy level curve
const PLAT_A_CAP = 0.03;     // platinum's earners grow very slowly
const L_CAP = 0.847;         // logit of the soft ceiling for rising rates (70%)
const REL_SD = 0.3;         // relative uncertainty of the predicted move (logit)
const N_SD = 0.3;            // extra relative uncertainty of players growth
const NEVER_REFRESHED_SHIFT = -1.45; // logit shift at ~90 min when history is one unchanged page
const NEVER_REFRESHED_SIGMA = 0.65;
const NR_H0 = 2;           // hours (last read to target) at which the never-refreshed values apply (the practice horizon)
const NR_SHIFT_K = 0.6;      // extra downward shift per e-fold of horizon
const NR_SIGMA_K = 0.25;     // extra spread per e-fold of horizon
const BASE_SIGMA = 0.3;      // model-uncertainty floor (logit)
const HZ_SD = 0.15;          // spread that grows with the horizon: HZ_SD^2 * days

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
      return { id: t.id, r0, shift, sigma, cap: false };
    });

  type Spec = { id: number; r0: number; shift: number; sigma: number; raw: number; cap: boolean };
  let specs: Spec[];

  if (D.length < 2) {
    // Never-refreshed page: the next refresh will show many more players; rates fall sharply (more so the longer the horizon).
    const hz = Math.max(0, hours(c.target, 0) - hours(last.at, 0));
    const lg = Math.log(Math.max(1, hz / NR_H0));
    specs = persist(() => ({ shift: NEVER_REFRESHED_SHIFT - NR_SHIFT_K * lg, sigma: NEVER_REFRESHED_SIGMA + NR_SIGMA_K * lg }));
  } else {
    // ---- dynamics branch: per-trophy log-earners growth (per hour, slowly decaying) vs players growth (decelerating) ----
    const tL = hours(L.at, 0);
    const uH = Math.max(0, hours(c.target, 0) - tL);          // extrapolation in hours
    const win = D.filter((r) => tL - hours(r.at, 0) <= WIN_H);
    const W = win.length >= 2 ? win : D.slice(-2);
    const tt = W.map((r) => hours(r.at, 0) - tL);              // <= 0
    const wt = tt.map((t) => Math.exp(t / REC_H));

    // players growth: slope per hour, deceleration from the slope's own trend when the window shows it
    const lnN = W.map((r) => Math.log(Math.max(1, r.players_tracked)));
    const wls = (xs: number[], ys: number[], ws: number[]) => {
      let sw = 0, sx = 0, sy = 0;
      for (let k = 0; k < xs.length; k++) { sw += ws[k]; sx += ws[k] * xs[k]; sy += ws[k] * ys[k]; }
      const mx = sx / sw, my = sy / sw;
      let sxx = 0, sxy = 0;
      for (let k = 0; k < xs.length; k++) { sxx += ws[k] * (xs[k] - mx) ** 2; sxy += ws[k] * (xs[k] - mx) * (ys[k] - my); }
      return { slope: sxx > 1e-9 ? sxy / sxx : 0, mx, my, sxx, sw };
    };
    const nf = wls(tt, lnN, tt.map(() => 1));
    const bAvg = Math.max(0, nf.slope);
    const cOff = -nf.mx;                                       // hours between window centre and anchor
    let tauN = TAU_N0;
    if (W.length >= 3) {
      const ks: number[] = [], ms: number[] = [], wsd: number[] = [];
      for (let k = 1; k < W.length; k++) {
        const dt = tt[k] - tt[k - 1]; const g = (lnN[k] - lnN[k - 1]) / dt;
        if (dt > 0.5 && g > 1e-4) { ks.push(Math.log(g)); ms.push((tt[k] + tt[k - 1]) / 2); wsd.push(dt); }
      }
      if (ks.length >= 2) {
        const f = wls(ms, ks, wsd);
        const span = ms[ms.length - 1] - ms[0];
        const invFit = Math.min(1 / 10, Math.max(1 / 150, -f.slope));
        const wFit = Math.min(1, span / 24);
        tauN = 1 / (wFit * invFit + (1 - wFit) / TAU_N0);
      }
    }
    const bL = bAvg * Math.exp(-cOff / tauN);
    const growN = bL * tauN * (1 - Math.exp(-uH / tauN));

    // earners growth per trophy
    const fits = c.trophies.map((_, i) => {
      const xs: number[] = [], ys: number[] = [], ws: number[] = [];
      for (let k = 0; k < W.length; k++) {
        const r = W[k].rates[i];
        if (r == null || r <= 0 || r >= 99.99) continue;
        const e = (r * W[k].players_tracked) / 100;
        const s2 = (0.005 / r) ** 2 + 1 / Math.max(e, 1);
        xs.push(tt[k]); ys.push(Math.log(e)); ws.push(wt[k] / s2);
      }
      if (xs.length < 2) return null;
      const f = wls(xs, ys, ws);
      if (f.sxx < 1e-6) return null;
      return { a: f.slope, se: Math.sqrt(1 / f.sxx), lnE: ys[ys.length - 1] };
    });
    // cross-trophy prior: kernel regression of slope on earners level
    const good = fits.map((f, i) => ({ f, i })).filter((o) => o.f && o.f.se < 0.5);
    const prior = (lvl: number): number => {
      let sw = 0, sa = 0;
      for (const o of good) { const w = Math.exp(-0.5 * ((o.f!.lnE - lvl) / 1.0) ** 2) / (o.f!.se ** 2 + 0.02 ** 2); sw += w; sa += w * o.f!.a; }
      return sw > 0 ? sa / sw : bL;
    };
    const aDecay = (cOff2: number) => Math.exp(-cOff2 / TAU_E);

    specs = c.trophies.map((t, i) => {
      const r0 = L.rates[i] ?? last.rates[i] ?? 0;
      const f = fits[i];
      const lvl = r0 > 0 ? Math.log((r0 * L.players_tracked) / 100) : 0;
      const pr = prior(lvl);
      let a: number, sa: number;
      if (!f) { a = pr; sa = A_PRIOR_SD; }
      else {
        const wFit = A_PRIOR_SD ** 2 / (A_PRIOR_SD ** 2 + f.se ** 2);
        a = wFit * f.a + (1 - wFit) * pr; sa = Math.sqrt(wFit) * f.se;
        sa = Math.hypot(sa, (1 - wFit) * A_PRIOR_SD);
      }
      a = Math.max(0, a) * aDecay(cOff);
      if (t.tier === 'platinum') a = Math.min(a, PLAT_A_CAP);
      const growE = a * TAU_E * (1 - Math.exp(-uH / TAU_E));
      const raw = growE - growN;
      const sig2 = BASE_SIGMA ** 2 + HZ_SD ** 2 * (uH / 24) + (REL_SD * (growE + growN)) ** 2 + (sa * uH) ** 2 + (N_SD * growN) ** 2;
      return { id: t.id, r0, shift: raw, sigma: Math.sqrt(sig2), cap: true };
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
      const place = (d: number): number => {
        if (!sp.cap || d <= 0 || l0 >= L_CAP) return l0 + d;
        return L_CAP - (L_CAP - l0) * Math.exp(-d / (L_CAP - l0));
      };
      const rt = expit(place(sp.shift));
      const rnd = Math.min(0.8, 0.0029 / Math.max(rt, 0.003)); // rounding of the printed target rate, esp. near the 0.01 floor
      const sig = Math.hypot(sp.sigma, rnd);
      q = Z.map((z) => expit(place(sp.shift + z * sig)));
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
