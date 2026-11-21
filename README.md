# GTA VI trophy rarity forecast

[![competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emasinves0-commits/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emasinves0-commits/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty: <https://make.fun/bounty/gta6-trophy-rarity-forecast-01>.
`main.ts` (Node 22, standard library only, offline, deterministic) forecasts each GTA VI PS5 trophy's Exophase earn rate.

## Approach

A rate is earners / players_tracked. Two forces move it: dilution (new players have earned nothing) and progression (existing players earn more).

- Stale reads are collapsed: consecutive identical pages are one observation. The fit window is the last 36 hours of distinct reads.
- Earners are reconstructed per read (rate x players / 100). Each trophy's ln-earners growth per hour is fitted by recency- and precision-weighted regression (Poisson plus print-rounding noise), shrunk toward a cross-trophy kernel curve over earners level. It decays slowly (e-fold 72 h), so early/progressing trophies keep rising; there is no cap tying earners growth to players growth.
- Players growth per hour is fitted on the same window; its deceleration is estimated from the trend of the interval slopes when the window has 3+ distinct reads, otherwise a prior e-fold of 24 h applies.
- Rate change = earners growth - players growth, applied on the logit scale with a soft ceiling (70%) for rises; falls taper at the 0.01 print floor.
- Spread = floor + horizon term + a share of the predicted move + the trophy's slope standard error x horizon + players-growth uncertainty, plus print rounding near the floor. It is applied before the soft ceiling, so upside compresses near the ceiling.
- A page that has never refreshed gets a downward shift and spread that grow with the log of the horizon (practice's 2 h horizon keeps the original values).
- Edge cases: near-100% trophies stay near 100; zero-earner trophies get a small prior rate with a wide spread; nulls are tolerated; any failure falls back to persistence.

Backtests on real Exophase snapshots (launch-night history, 19 Nov 22:41 and 21 Nov 01:02 pages): B0 0.358, B1 0.460 with 91% coverage (the previous model scored 0.874 and 0.747).

