# AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts

> **Building the SIH presentation?** Start with [docs/presentation/SIH_PPT_REFERENCE.md](docs/presentation/SIH_PPT_REFERENCE.md): slide-by-slide content, the full tech stack, verified numbers, judge Q&A, and the fixes the current deck ([docs/presentation/SIH2026-IDEA-Presentation-Format.pptx](docs/presentation/SIH2026-IDEA-Presentation-Format.pptx)) needs.

Smart India Hackathon prototype. A two-stage hybrid AI pipeline that:

1. **Stage 1 (tracking core):** screens a 50-member global ensemble forecast (0 to 10 days)
   for extremes by computing the **Extreme Forecast Index (EFI)** against a **30-year ERA5
   climatology**. A **message-passing GNN on an icosahedral mesh** turns those signals into
   calibrated probabilities of an observed extreme, and a tracker draws **4D bounding boxes**
   (a lat/lon box at every 6-hourly lead time) around each moving anomaly.
2. **Stage 2 (downscaling core):** crops the most severe anomaly and runs a **conditional
   denoising diffusion model** with a **physics-informed loss** (water conservation and
   moisture-convergence consistency) to generate high-resolution rainfall scenarios that
   keep the extreme peaks, which an MSE-trained CNN blurs away.
3. **Delivery:** a **FastAPI alert API** that drops a pinpoint coordinate on the core of the
   anomaly and returns **low / moderate / severe alerts within a 5 km radius**, plus a
   **Leaflet map dashboard**.

The demo case is **Cyclone Amphan (May 2020)**, forecast from the real ECMWF 50-member
ensemble initialised on 15 May 2020, five days before landfall. Amphan is held out: the
models never see it during training.

![architecture](docs/architecture.png)

---

## Quick start

```bash
cd sih-prototype
# Linux only, optional, much smaller download: CPU-only PyTorch first
python3 -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt     # exact tested versions
python3 -m pytest -q tests                     # 22 unit tests, a few seconds

python3 run_demo.py            # full pipeline on real data (bundled Amphan sample, no download)
python3 run_demo.py --serve    # same, then serves the API + dashboard on http://localhost:8000
```

### Windows setup (D:\sih-prototype)

1. Use **Python 3.11 or 3.12** (64-bit, from python.org); 3.10 also works. The
   `requirements.txt` pins have no wheels for pre-release Pythons such as a 3.15 beta, so if
   that is your default `python`, keep it and make the project's venv with 3.11 or 3.12
   explicitly, as below. `py -0p` lists the versions the Windows launcher can see and where
   they live; an entry whose folder no longer exists will fail, so pick one that does.
2. Right-click `sih-prototype.zip` → *Extract All…* → choose `D:\`. You should now have
   `D:\sih-prototype\run_demo.py`. Windows' *Extract All* sometimes suggests a folder named
   after the zip, which gives `D:\sih-prototype\sih-prototype\run_demo.py`. That works too:
   just `cd` into whichever folder holds `run_demo.py`.
   Or clone it from GitHub straight into that folder:
   `git clone https://github.com/omgawande1523/SIH-NEW.git D:\sih-prototype`
3. Open **PowerShell** and run:

```powershell
cd D:\sih-prototype
py -3.11 -m venv .venv                  # not plain `python`, which may be the 3.15 beta
# no 3.11? use 3.12 instead:  py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python --version                        # must say 3.11.x or 3.12.x inside the venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # the Windows torch wheel is already CPU-only
python -m pytest -q tests               # optional: 22 unit tests

python run_demo.py --serve              # then open http://localhost:8000/dashboard
```

The zip already holds the trained weights and the Amphan sample data (about 30 MB in
`data\cache\era5`), so this first run needs no data download and takes about 3 minutes on a 4-core CPU.
`--retrain` downloads the full training set (about 200 MB) and retrains. Stop the server
with Ctrl+C. Next time, just `cd D:\sih-prototype`, run `.\.venv\Scripts\Activate.ps1`,
then `python run_demo.py --serve`.

Then open:

* `outputs/dashboard.html`: the map (open it straight in a browser, or at `/dashboard` when serving;
  the base map and map library load from the internet)
* `http://localhost:8000/docs`: interactive API docs
* `outputs/stage1_tracking.png`, `outputs/stage2_downscaling_comparison.png`,
  `outputs/stage2_forecast_alerts.png`, `docs/architecture.png`: figures for the slides

Other options:

| Command | What it does |
|---|---|
| `python3 run_demo.py --source synthetic` | Fully offline: synthetic ensembles, climatology and 0.25° fields with the same format. Use this if the venue has no internet. |
| `python3 run_demo.py --retrain` | Retrain both models instead of reusing the weights in `models/`. |
| `python3 run_demo.py --fast` | Tiny training runs (a few minutes) saved to `models/<source>_fast/`, just to check everything is wired up. The real weights are left alone. |
| `uvicorn sih.api.app:app --port 8000` | Serve the API from the last run's outputs. |

Trained weights for both sources ship in `models/` and the downloaded data is cached in
`data/cache/`, so a normal run reuses them and finishes in about 3 minutes on a laptop CPU (most of it downscaling 24 members at 6 leads).
Retraining from scratch on CPU takes roughly 5 minutes for Stage 1 and 30 to 60 minutes
for Stage 2. No GPU is needed.

### Example API calls

```bash
curl localhost:8000/health
curl localhost:8000/tracks
curl localhost:8000/alerts/core                              # headline core pinpoint + 5 km impact circle
curl localhost:8000/alerts/first-warning                     # earliest lead that reaches an alert level
curl "localhost:8000/alerts/core?lead_hours=126"             # the core on landfall day
curl "localhost:8000/alerts/point?lat=22.57&lon=88.36&lead_hours=126"   # Kolkata: at risk within 5 km?
curl "localhost:8000/alerts/zones?p_min=0.5"                 # GeoJSON low/moderate/severe polygons
```

---

## Architecture

```mermaid
flowchart LR
  A[Ensemble forecast<br/>NEPS-G / ECMWF ENS<br/>member x lead x lat x lon] --> B[Climate CDF<br/>30-yr ERA5 / IMDAA<br/>+-15 days, same hour]
  B --> C[EFI, SOT, P&gt;q99,<br/>mean & spread<br/>rain / wind / heat]
  C --> D[Stage 1 GNN<br/>grid -> icosahedral multi-mesh<br/>6 message-passing layers<br/>mesh -> grid]
  D --> E[Tracker<br/>connected regions, linked<br/>across leads = 4D boxes<br/>+ ensemble cyclone plume]
  E --> F[Crop 12x12 deg window<br/>around the anomaly core]
  F --> G[Stage 2 diffusion<br/>conditional U-Net, DDIM<br/>physics-informed loss]
  G --> H[5 km alert lattice<br/>P(low/moderate/severe<br/>within 5 km)]
  H --> I[FastAPI alerts]
  H --> J[Leaflet dashboard]
```

### Stage 1: EFI + icosahedral-mesh GNN (`sih/stage1/`)

* **Climate distribution** (`efi.py`): for every grid point and valid time, the 30-year
  ERA5 sample (1990 to 2019, ±15 days, same hour of day) gives 101 climate quantiles.
* **EFI** = (2/π) ∫ (p − F_f(p)) / √(p(1−p)) dp, where F_f(p) is the share of ensemble
  members below the climate p-quantile (+1: all members beyond the climate maximum; about 0.91 on the 101-quantile grid used here).
  We also compute the **Shift of Tails** (how far the ensemble's 90th percentile goes past
  the climate 99th), the standardised ensemble mean and spread, and the fraction of members
  above q99 / below q1. These are computed for 24 h rain, 10 m wind and 2 m temperature.
* **Mesh** (`mesh.py`): an icosahedron refined recursively (GraphCast-style multi-mesh:
  edges from every refinement level are merged, so messages travel both locally and
  thousands of km in one hop). Working on the mesh avoids lat-lon pixel distortion.
* **GNN** (`gnn.py`): encoder (grid → mesh bipartite message passing), processor
  (6 interaction-network layers on the multi-mesh), decoder (mesh → grid). Inputs include
  ±24 h temporal context. Output: per-hazard probability that ERA5 will actually exceed
  its climate 99th percentile. It is trained on six other cases (Fani, Vayu, Tauktae, Yaas,
  the 2019 north-India heatwave and a quiet period) and tested on Amphan.
* **Tracker** (`tracker.py`): thresholds the probability, labels connected regions, links
  them across 6-hourly leads, and outputs each track's per-step box (the 4D box: one
  lat/lon box per valid time) and its space-time envelope (everything the system sweeps
  over its lifetime). Rain and wind regions that touch a tropical low (ensemble-mean MSLP
  at least 3 hPa below its surroundings, with 10 m wind of 9 m/s or more) are split around
  it: cells within 600 km form the cyclone object and the rest become separate objects.
  Without this, Amphan and the monsoon westerlies feeding it merged into one wind region
  spanning 0 to 22.5N and 61.5 to 96E. Tracks that follow a low are listed first. For those
  tracks it also finds every ensemble member's MSLP minimum, which gives a track plume and
  the strike fraction.

### Stage 2: physics-informed conditional diffusion (`sih/stage2/`)

* **Model:** conditional U-Net denoiser (2.2 M parameters), v-prediction, cosine noise
  schedule, 30-step DDIM sampling. Condition channels: coarse rain and 10 m wind
  (interpolated to the fine grid), fine-scale orography and land-sea mask.
* **Physics-informed loss** on the model's clean-field estimate:
  1. *Water conservation:* block-averaging the generated field must return the coarse
     rainfall, so downscaling cannot invent or lose water.
  2. *Moisture-convergence consistency:* heavy rain generated where the synoptic-scale
     moisture flux ∇·(TCWV·V) is divergent is penalised, which rules out downpours with no
     converging moisture to feed them. In ERA5, 77% of heavy-rain area lies in convergent air
     against about 54% of all area, so this constraint carries real information. It is a
     proxy built from column water vapour and 10 m wind: ERA5's own vertically integrated
     moisture divergence would be better, but it is empty in the public WeatherBench 2 copy.
* **Baseline:** the same U-Net trained with plain MSE. It shows the spectral smoothing
  problem: the peaks come out weaker and the small-scale power spectrum collapses.
* **Probabilistic alerts:** downscaling all 50 members is slow on a CPU, so 24 are chosen by
  importance sampling: the 12 wettest members in the anomaly box (the tail matters most)
  plus 12 evenly spaced through the rest. Each member carries a weight (1/50 for the wet
  twelve, 38/50 shared by the other twelve), so the weighted statistics estimate the whole
  ensemble instead of over-counting the wettest members. Each member gets 2 diffusion
  samples, which are resampled to a 5 km lattice. Each lattice point gets P(max rain within
  5 km ≥ threshold) for three levels, scaled from IMD daily classes to 6 hours: low ≥ 16,
  moderate ≥ 29, severe ≥ 51 mm/6 h (IMD heavy 64.5, very heavy 115.6, extremely heavy
  204.5 mm/day). A level is issued when its probability reaches 50%. The alert core at each
  lead is the lattice point with the highest P(severe), and every probability comes with a
  Monte Carlo standard error (`prob_se`, from the effective sample size of the weights,
  about 38 here).
* **Headline alert rule** (fixed in advance, `sih/products.py:_headline`): the headline is
  the downscaled lead whose core has the highest P(severe). Leads within one standard error
  of that maximum count as tied, and the earliest of them wins, because it is the more
  skilful forecast and gives the most warning. The level is whatever that core reads; the
  rule never picks a lead to get a higher level.

### Data (`sih/data/`)

* `loader.py` is the entry point for real NCMRWF files: `open_forecast()` reads NetCDF,
  GRIB2 (via cfgrib) or Zarr, lazily with Dask, and maps NEPS-G / NCUM / ECMWF / IMDAA
  variable and dimension names onto one standard layout `(member, lead, lat, lon)`.
* `fetch_era5.py` pulls the demo data anonymously from the public WeatherBench 2 mirror on
  Google Cloud: ECMWF IFS 50-member ensembles (1.5°), ERA5 at 1.5° (climatology and
  verification) and at 0.25° (downscaling training data and truth).
* `synthetic.py` builds an offline stand-in with identical files: Holland-profile cyclones
  on an Amphan-like track with growing ensemble spread, heat domes, a 30-year synthetic
  climate, and 0.25° convective fields with consistent convergence and orography.

---

## Results on the held-out case (Cyclone Amphan)

All numbers below come from `python3 run_demo.py` (real-data mode) and are rewritten to
`outputs/stage1_metrics.json` and `outputs/stage2_metrics.json` on every run.

**Stage 1: does the GNN detect the extremes that actually happened?** The target is
whether ERA5 exceeded its own 30-year 99th percentile at each grid point and lead time
(0 to 10 days). AUC: 1.0 is perfect, 0.5 is chance. With only seven events, a single
held-out case says little, so `scripts/stage1_crossval.py` trains seven times, each time
holding one event out, and scores it (`outputs/stage1_crossval.json`). When the held-out
event falls inside the 1990-2019 climate sample (Fani, Vayu and the 2019 heatwave, and the
quiet 2018 case), that year is dropped from the climatology for every event in the fold, so
the held-out event's EFI and its "exceeded q99" labels never use its own observations.
Doing so changed the mean AUCs by 0.005 at most:

| Hazard | GNN mean AUC | Raw EFI mean AUC | Events where GNN wins | Day 6-10 only: GNN vs EFI |
|---|---|---|---|---|
| Heavy rain (24 h) | **0.92** | 0.88 | 4 of 7 | **0.85** vs 0.81 |
| Damaging wind (10 m) | **0.91** | 0.86 | 7 of 7 | **0.84** vs 0.77 |
| Extreme heat (2 m T) | 0.85 | 0.84 | 5 of 7 | 0.79 vs 0.78 |

The GNN adds most at long lead times, where the raw EFI gets noisy, and it helps
consistently for wind. For rain it wins on the cyclones but not on the heatwave, Vayu or
quiet cases, and for heat it is roughly level with the EFI. On Amphan alone the rain AUC
is 0.955 against 0.896. Seven events is still a small sample, and the three 2019 events come from
one season, so a fold can still train on a sister event from the same year: treat these
as a proof of concept, not a verified skill score.

**Tracking.** From the 15 May 00 UTC ensemble, the tracker picks up Amphan as a
damaging-wind track (T01, +24 to +156 h) and a heavy-rain track (T02, +24 to +150 h), each
with per-step boxes about 6 to 9 degrees across that move north with the storm. The
monsoon wind surge south of it is a separate track (T03, 0 to 16.5N), not part of the
cyclone's box. The envelopes are still long (about 6N to 27N) because the storm itself
travels about 2,000 km in six days; the per-step boxes are the tight part. The ensemble-mean cyclone centre (MSLP minimum)
moves 12.4N 85.8E (+48 h) → 16.7N 86.0E (+96 h) → 19.4N 87.2E (+120 h) → 22.7N 88.1E
(+144 h), with 96 to 100% of members holding a closed low over days 2 to 6. Amphan made
landfall near Sagar Island (about 21.6N 88.3E) on 20 May.

**Does the cyclone split work beyond Amphan?** `scripts/tracker_check.py` runs the tracker on
all seven events with the constants unchanged (600 km radius, 3 hPa depth, 9 m/s wind) and
writes `outputs/tracker_check.json` (the script needs the full training data, which
`python -m sih.data.fetch_era5` downloads, about 200 MB). "Cut out" counts the steps where the cyclone object was
separated from a larger rain or wind region that would otherwise have been its box.

| Event (ensemble init) | Tracks that follow a low | Ensemble-mean low: start → middle → end | Steps cut out of a larger region (largest: region vs cyclone cells) | What happened (IMD, approximate) |
|---|---|---|---|---|
| Fani (27 Apr 2019) | wind +0–180 h, rain +24–198 h | 4.5N 90.0E → 13.5N 85.5E (+90 h) → 21.0N 88.5E (+180 h) | wind 10 of 30, rain 3 of 29 (13 vs 7) | Formed near 5N 89E, recurved, landfall at Puri (19.8N 85.8E) about +146 h, then into Bangladesh |
| Vayu (9 Jun 2019) | wind +6–126 h, rain +24–120 h | 10.5N 72.0E → 16.5N 70.5E (+66 h) → 22.5N 66.0E (+126 h) | wind 10 of 20, rain 6 of 15 (28 vs 12) | Formed off Kerala near 11N 72E, moved north, turned west off Gujarat, no landfall |
| Tauktae (12 May 2021) | rain +66–180 h, wind +66–162 h | 12.0N 73.5E (+66 h) → 18.0N 69.0E → 21.0N 67.5E (+180 h) | rain 20 of 20, wind 17 of 17 (74 vs 27) | Depression near 11.5N 72.5E about +50 h, up the west coast, landfall near Diu (20.8N 71.1E) about +135 h |
| Yaas (21 May 2021) | wind +48–174 h, rain +42–156 h | 16.5N 90.0E (+42 h) → 18.0N 88.5E → 24.0N 87.0E (+156 h) | wind 17 of 19, rain 8 of 20 (131 vs 33) | Depression near 15.5N 90E about +48 h, landfall near Balasore (21.4N 87.0E) about +124 h |
| Amphan (15 May 2020, test) | wind +24–156 h, rain +24–150 h | 10.5N 87.0E → 16.5N 85.5E (+90 h) → 24.0N 88.5E (+150 h) | wind 19 of 22, rain 7 of 22 (142 vs 27) | Landfall near Sagar Island (21.6N 88.3E) about +130 h |
| 2019 heatwave (28 May 2019) | none | 5 lows found, none attached to a rain or wind region | – | No cyclone |
| Quiet case (2 May 2018) | none | 3 lows found, none attached | – | No cyclone |

All four training cyclones are found and followed from genesis to landfall or decay, along
the right paths, and there are no false storms on the heatwave or quiet cases. The split
matters most when the monsoon is active (Tauktae, Yaas, Amphan), where the storm would
otherwise sit inside a region four to five times its size. Limits: Stage 1 probabilities for
the six training events come from the shipped GNN, which was trained on them, so they are
in-sample (the split itself only uses the ensemble pressure and wind). Tauktae's low runs about 250 to 400 km west of the real track from day 5 on; the
ensemble forecast itself put the storm there, so that is forecast error, not tracker error. Some lows belong to no track (13 in Vayu, 19 in Tauktae, mostly heat lows
over Pakistan and weak lows away from any rain or wind), and Vayu, Yaas and the heatwave each
have a few minima on the domain edge; none of these became a track. Seven events is still a
small test.

**Alert cores** (the 5 km point with the highest chance of severe rain at each downscaled
lead, from 24 importance-weighted members; `GET /alerts/core?lead_hours=`):

| Lead | Valid (UTC) | Core | P(low / moderate / severe within 5 km) | Level |
|---|---|---|---|---|
| +24 h | 16 May 00:00 | 9.1N 84.1E | 33% / 7% / 3% | none |
| +48 h (earliest warning) | 17 May 00:00 | 12.5N 84.9E | 91% / 52% / 7% | moderate |
| **+72 h (headline)** | 18 May 00:00 | 14.1N 85.2E | 70% / 51% / 27% (±7 / ±8 / ±7) | **moderate** |
| +102 h | 19 May 06:00 | 16.7N 85.2E | 47% / 30% / 21% | none |
| +126 h | 20 May 06:00 | 19.8N 86.0E | 36% / 25% / 18% | none |
| +150 h | 21 May 06:00 | 24.3N 86.6E | 14% / 13% / 13% | none |

The downscaled leads are six steps spread evenly along the cyclone rain track (+24 to
+150 h). The headline is +72 h under the fixed rule above: its P(severe) of 27% is the
highest, +102 h (21%) is within one standard error, and the earlier of the two wins. It
reads **moderate** because P(moderate) is 51%, which is right on the 50% line and within
its ±8-point sampling error, so present it as "moderate, borderline". In the previous
version the headline read "none" at +96 h; the rule did not change. What changed is the
tracker: the rain track is now the cyclone object rather than a region merged with the
monsoon band, so the six downscaled leads (and the box each is centred on) moved. At 4 to 6
days' lead no 5 km point reaches 50% at any level, which is what an honest probabilistic
system should say that far out. The cores now sit within about 50 km of the ensemble-mean
cyclone centre (for example 14.1N 85.2E against 14.4N 85.4E at +72 h).

**Earliest actionable warning** (`GET /alerts/first-warning`, and a line in the demo output
and dashboard legend): the first downscaled lead whose core reaches any level. For Amphan it
is +48 h, 17 May 00 UTC, level **moderate** (P(low) 91%, P(moderate) 52%). This sits beside
the headline and does not replace it: the headline answers "where is the worst risk", the
first warning answers "when could we first say something".

**Stage 2: are the extremes preserved?** This is a perfect-coarse test on 17 Amphan
time steps (18 to 22 May 2020, never seen in training). The ERA5 0.25° rain is
block-averaged to 1.5°, downscaled back, and compared with the original. Each diffusion
score averages 4 samples per case, and CRPS scores the 4 samples together as an ensemble
(for a single field CRPS equals the mean absolute error).

This is a perfect-prognosis setup: Stage 2 is trained and scored on coarse fields made by
block-averaging ERA5, then applied to ECMWF forecast fields in the demo. The forecast
model's own biases (too-smooth or misplaced rain, a displaced cyclone) are never seen in
training, so these scores are an upper bound on what it does with real forecasts. Training
on pairs of (coarse forecast, fine analysis) would fix that and needs a forecast archive
matched to high-resolution truth.

| Method | Peak (per case, mean) | 99th pct | RMSE | CRPS | Water-mass error | Heavy rain in divergent air |
|---|---|---|---|---|---|---|
| Bilinear interpolation | 44% | 64% | 6.2 | 2.39 | 22% | 14% |
| U-Net trained with MSE | 84% | 91% | **4.4** | 1.70 | 11% | 14% |
| Diffusion, no physics loss | 110% | 94% | 6.0 | **1.40** | 8% | 20% |
| **Physics-informed diffusion** | **101%** | **97%** | 5.7 | **1.40** | **4%** | 20% |
| ERA5 truth | 100% | 100% | 0 | 0 | 0% | 18% |

RMSE and CRPS are in mm/6 h. The MSE U-Net has the lowest RMSE because it hedges toward
the mean, which is exactly the spectral smoothing the proposal describes. On CRPS, the
proper score for a probabilistic forecast, the diffusion ensemble is best. It also keeps
the peaks and the small-scale power spectrum (see
`outputs/stage2_downscaling_comparison.png`).

**What the physics loss adds** (`scripts/stage2_ablation.py` retrains the same model with
the physics terms switched off and scores both on identical noise). The water-conservation
term roughly halves the water-mass error (8% → 4%) and stops the model overshooting peaks
(110% → 101%), which also lowers RMSE. The moisture-convergence term makes no measurable
difference to the convergence check (20% either way), so for now it is a constraint we
enforce rather than one we can show pays off. CRPS is the same with or without it.

"Heavy rain in divergent air" is the share of >16 mm/6 h area where the synoptic moisture
flux is diverging. Lower is not automatically better: the real atmosphere scores 18%, and
the smoother models land lower (14%). On this check the diffusion model (20%) is close to ERA5, but none of the models is clearly
more physical than the others.

![stage 2](outputs/stage2_downscaling_comparison.png)
![stage 1](outputs/stage1_tracking.png)

### Talking points for the pitch

* One command runs the whole pipeline on a real historical forecast in about 3 minutes on
  a laptop CPU, so the claim that inference is cheap once trained holds even without a GPU.
* The GNN beats the raw EFI at spotting real extremes, most clearly at day 6 to 10 and for
  wind (7 of 7 events in hold-one-out testing).
* The tracker follows Amphan 5.5 days ahead from the 15 May ensemble, and the alert
  probabilities stay below 50% at 4 to 6 days, as they should that far out.
* Spectral smoothing, measured: the MSE U-Net keeps 84% of the peak and the diffusion model
  keeps it all. The diffusion model also has the best probabilistic score (CRPS).
* Physics in the loss is measured, not just claimed: switching off the water-conservation
  term doubles the water-mass error (4% → 8%) and makes peaks overshoot.
* The alerts are probabilities over the ensemble, weighted so the tail sampling doesn't
  inflate them.
* Honest limits: see the next section.

---

## What is real, what is scaled down, what is synthetic

| Part of the proposal | In this prototype |
|---|---|
| NEPS-G 12 km ensemble | **Substituted:** ECMWF IFS 50-member ensemble (the closest public equivalent) at 1.5°. NEPS-G files plug in through `sih/data/loader.py`. |
| 30-year ERA5 / IMDAA baseline | **Real:** ERA5 1990 to 2019 at 1.5°. IMDAA would load the same way. |
| Icosahedral mesh GNN | **Real, scaled down:** refinement level 5 (~220 km node spacing) over the Indian region, because the input grid is 1.5°. 12 km input would use level 9. Message passing is written in plain PyTorch scatter ops, which is what DGL's `update_all` does, so there is no DGL dependency. |
| 12 km → 5 km downscaling | **Scaled:** the demo downscales 1.5° → 0.25° (×6), because public 5 km truth for India isn't openly available. The same code trains on any coarse/fine pair, e.g. NEPS-G 12 km with IMDAA-R / NCUM-R 4 km. The alert lattice is 5 km, bilinearly resampled from the 0.25° output. |
| Physics-informed loss | **Real:** water conservation plus a moisture-flux convergence penalty. ERA5's vertically integrated moisture divergence is empty in the public copy, so the penalty uses column water vapour × 10 m wind as a proxy. The ablation shows the conservation term helps (water-mass error 8% → 4%); the convergence term has no measurable effect yet. |
| Hugging Face Diffusers | **Replaced** by a ~150-line DDPM/DDIM implementation (`sih/stage2/diffusion.py`) so the physics loss can act on the clean-field estimate inside the training loop. |
| Cartopy maps | **Replaced** by Leaflet (Folium) for the dashboard and a land-sea-mask coastline in the figures, which avoids a heavy GIS install. |
| Stage 2 train/apply | **Perfect prognosis:** trained on block-averaged ERA5, applied to ECMWF forecasts, so forecast biases are not learned (see Results). |
| Training set size | **Small:** 6 ensemble cases for Stage 1 and 240 ERA5 fields for Stage 2, so it trains on a laptop CPU. The numbers are a proof of concept, not an operational verification. |

## Repository layout

```
run_demo.py               one-command pipeline
sih/config.py             domain, events, thresholds, model sizes
sih/data/                 loader (NetCDF/GRIB2/Zarr), ERA5+ECMWF fetcher, synthetic generator
sih/stage1/               efi.py, mesh.py, gnn.py, pipeline.py (train/infer), tracker.py
sih/stage2/               unet.py, diffusion.py (DDPM + physics loss), pipeline.py
sih/products.py           tracks -> downscaling -> 5 km alert lattice
sih/api/app.py            FastAPI alert service
sih/dashboard/build.py    Leaflet dashboard + figures
scripts/                  cross-validation, physics ablation, tracker check, API smoke test, architecture diagram
tests/                    unit tests (python -m pytest -q tests)
data/cache/<source>/      cached inputs    models/<source>/  trained weights    outputs/  products
```
