# GTA VI trophy rarity forecast

[![competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emasinves0-commits/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emasinves0-commits/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty: <https://make.fun/bounty/gta6-trophy-rarity-forecast-01>.
`main.ts` (Node 22, standard library only, offline, deterministic) forecasts each GTA VI PS5 trophy's Exophase earn rate.

## Approach

A rate is earners / players_tracked, and two forces move it: dilution (new players have earned nothing) and progression (existing players earn more).

- Stale reads are collapsed: consecutive identical pages are one observation, so a flat history is not mistaken for no growth.
- Earners are reconstructed per distinct read (rate x players / 100).
- Players growth is fitted as ln(players) against ln(1+hours) over the last six distinct reads, anchored on the latest read, recency-weighted, and damped (launch growth decelerates).
- Each trophy's earners slope is fitted the same way. Its standard error comes from the rounding of the printed rate and from Poisson noise. It is then shrunk toward the cross-trophy median slope, so noisy trophies borrow strength. It is capped by the players slope, and the platinum's is capped low.
- At the target, rate = earners(target) / players(target), done on the logit scale.
- The spread is a floor plus players-slope uncertainty plus per-trophy slope uncertainty, each scaled by the extrapolation distance in log-time. Rounding noise of the printed target rate is added, which matters near the 0.01 floor. A stale tail widens the spread and shifts the median down.
- A page that has never refreshed (one distinct read) gets a strong downward shift and a wide spread, since the next refresh brings many more players.
- Edge cases: trophies near 100% stay near 100%. Trophies with zero earners everywhere get a small prior rate with a wide spread. Null rates are tolerated. Any failure falls back to persistence.

Local scenario checks (synthetic, not the true answers) score well below the house baselines.
