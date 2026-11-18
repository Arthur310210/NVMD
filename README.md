# NVMD
## Layout

```
nvmd/vmd.py          batched VMD (ADMM); matches vmdpy to 1e-8 (tests/test_vmd.py)
nvmd/baselines.py    EMD/EEMD/VMD × DWT/NLM/DWT-NLM (9 baselines)
nvmd/models.py       Fig. 3 encoder-decoder, NVMD(A) columns, NVMD(P) PNN groups, DDAE (FCN-DAE [33])
nvmd/train.py        Eq. (19) L1 IMF loss, Adam 1e-3, column-by-column training, timing
nvmd/data.py         MIT-BIH / EEGdenoiseNet / Ninapro DB2 loaders, splits, SNR-exact noise
nvmd/targets.py      clean IMF targets u_k (VMD of the clean signal, K = 1..7), disk-cached
nvmd/metrics.py      SNR_imp, RMSE, correlation (Eqs. 25–31)
nvmd/experiment.py   whole pipeline -> results.json, report.md, tables/*.csv, figs/*.png
nvmd/paper_values.py the paper's Tables I–V, used for the side-by-side comparison
scripts/run_experiment.py  command-line entry point
```

## Running

```bash
pip install -r requirements.txt

# ECG (MIT-BIH, downloaded from PhysioNet with wfdb)
python scripts/run_experiment.py --dataset ecg --data-root data/mitdb --download --out results/ecg
# EEG (EEGdenoiseNet, downloaded from GitHub)
python scripts/run_experiment.py --dataset eeg --data-root data/eeg --download --out results/eeg
# sEMG (Ninapro DB2 needs registration: put the *.mat files under data/ninapro_db2)
python scripts/run_experiment.py --dataset semg --data-root data/ninapro_db2 --out results/semg

# smoke test (tiny subsets, 2 epochs, K <= 4): a few minutes on CPU
python scripts/run_experiment.py --dataset ecg --data-root data/mitdb --quick --out results/quick_ecg
```

Useful flags: `--epochs`, `--batch-size`, `--K-max`, `--baseline-max-test N`,
`--eemd-trials`, `--skip-baselines`, `--device cuda`.
Everything is cached (`cache/` splits and VMD targets, `ckpt/` one checkpoint per column),
so an interrupted run resumes where it stopped.

**Compute.** The full protocol trains 28 NVMD(A) branches + 28 NVMD(P) branches (≈30M / 40M
parameters) for 100 epochs on 54k (ECG), 34.5k (EEG) and 157.7k (sEMG) fragments; use a GPU.
The clean-target VMD for sEMG (≈200k × K=1..7 × 2000 samples) needs ≈40 GB of disk.
Baselines run on the first `baseline_max_test` (default 200) test fragments per SNR because
EEMD at 0.7–3 s per signal × 7 K × 7 SNRs on the whole test set would take days; raise it if
you have the time.

## How the paper was implemented

| Paper item | Implementation |
|---|---|
| Eq. (2)/(4) clean IMF targets | VMD of the **clean** segment for each K = 1..7 |
| Fig. 3 encoder | 5 × [Conv1d k3 s1 → BN → LeakyReLU → MaxPool 2], filters 32-64-128-256-512 |
| Fig. 3 decoder | 4 × [ConvTranspose1d k3 s2 p1 op1 → BN → LeakyReLU], filters 256-128-64-32, then ConvTranspose1d → 1 channel, no activation |
| NVMD(A), Fig. 2a | column c = c parallel encoder-decoder branches, one IMF each; columns independent |
| NVMD(P), Fig. 2b, Eq. (15) | column c = c branches + 1×1-conv lateral adapters W from the frozen earlier columns at every matching depth; PNN groups of 3 columns (1–3, 4–6, 7) |
| Eq. (19) | Σ_k ‖u_k − û_k‖₁ (mean over batch/time) |
| Training | Adam, lr 1e-3, 100 epochs |
| Eq. (23) K* | argmax over c of test SNR_imp (pooled over the test SNRs) |
| Eq. (29) T_IMFs | mean over k = 1..7 of the test-stage time per signal |
| Data | MIT-BIH lead 1, 1500 × 256-sample segments per record = 72,000 (54k/7.2k/10.8k); EEGdenoiseNet 34,500/7,500/11,050; Ninapro DB2 2000-sample segments, 157,700/19,800/100,050 |
| Noise | WGN scaled so every fragment has exactly the target SNR; train/val fragments get a random SNR from the paper's training list; every test fragment is evaluated at every test SNR |

## Unspecified details (our choices — tune these first)

- **Batch size** 64; best epoch kept by validation loss; seed 0.
- **VMD**: α = 2000, τ = 0, DC off, uniform init, tol = 1e-7, 500 iterations.
- **IMF order**: sorted by centre frequency, IMF1 = lowest. Section II-C says IMF1 is the
  *highest*-frequency (noise) mode, but Section IV-C says IMF1 is the smooth low-frequency mode
  that was easiest to learn, and Table V's IMF1 values only make sense for the dominant
  low-frequency mode. Switch with `VMD["order"] = "descending"`.
- **Normalisation**: inputs/targets divided by the std of the clean training set; RMSE is
  reported in these normalised units (the paper's RMSE units are not stated).
- **Lateral connections**: branch k of column c receives from branch min(k, c_j) of every
  earlier column of its group (`lateral_mode="all"` connects every branch).
- **Baselines**: sym8 DWT, level 4, soft universal threshold (MAD/0.6745); NLM patch half-width
  10, search half-width 64, λ = 0.6σ; EEMD 50 trials, noise width 0.2; every component is
  denoised and the components are summed. EMD/EEMD outputs are merged/padded to exactly K
  components.
- **EEG noise**: each fragment = a clean EEG epoch + (an EOG or EMG artifact epoch, chosen 50/50)
  + WGN, artifact:WGN power 1:1, scaled to the target SNR. Clean and artifact epochs are split
  75/10/15 before mixing so test epochs never appear in training.
- **Fixed lengths**: 2000-sample sEMG is reflect-padded to 2016 inside the network and cropped.
- **DDAE**: FCN-DAE of ref. [33] (plus DDAE-F3, one Fig. 3 branch, as an extra), both trained
  noisy → clean with the same L1 loss and optimiser as NVMD.
- **Fig. 3 inconsistency**: it shows 5 poolings but only 4 upsampling layers; we made the output
  deconvolution stride 2 so the output length equals the input.

## Inconsistencies in the paper itself

- sEMG Tables II/III columns (−14, −10, −6, −2, 2 dB) are the **training** SNRs, not the stated
  test SNRs (−12, −8, −4, −2, 0, 2 dB); Table IV/V sEMG uses −6 dB, also not a test SNR.
  Fig. 5 uses 1 dB (ECG) and −5 dB (EEG), not in the test lists either. We evaluate every
  requested SNR by re-corrupting the whole test set.
- ECG Table II omits the 10 dB test level; EEG omits 0 dB.
- Table III RMSE **increases** with input SNR for every method, which is the opposite of what
  should happen; it may be a column-order or units issue.
- Tables IV/V don't say whether they are NVMD(A) or NVMD(P); their overall SNR_imp (ECG −1 dB,
  K=3: 11.30) matches neither Table II row exactly (10.86 / 11.73). We report both.
- sEMG split percentages sum to 103% (57/10/36); the absolute counts are used.
- The acknowledgement mentions an "MOA dataset" that is not used anywhere.
- ECG noise sources BW/PLI/MA are described in Section II-A, but Section IV only uses WGN; we
  follow Section IV.

## Reproduced runs (CPU profile)

`--profile cpu` keeps everything the paper specifies (data, split proportions, train/test
SNRs, architecture, L1 IMF loss, Adam 1e-3, K* rule) but trains on random subsets of the
paper's splits with fewer epochs, so the full pipeline runs on a 2-core CPU:

| | ECG (MIT-BIH, 48 records) | EEG (EEGdenoiseNet) | sEMG |
|---|---|---|---|
| data | real, full 72,000-fragment pool | real | **synthetic stand-in** (Ninapro DB2 not reachable) |
| train / val / test | 6,000 / 600 / 1,500 | 6,000 / 600 / 1,500 | 3,000 / 400 / 600 |
| epochs, K range | 20, K = 1..6 | 20, K = 1..6 | 15, K = 1..5 |
| baselines | 100 test fragments / SNR | 60 | 30 |

```bash
python scripts/run_experiment.py --dataset ecg  --data-root data/mitdb --profile cpu --out runs/cpu_ecg
python scripts/run_experiment.py --dataset eeg  --data-root data/eeg   --profile cpu --out runs/cpu_eeg
python scripts/run_experiment.py --dataset semg --data-root none --synthetic-semg --profile cpu --out runs/cpu_semg
```

Each run writes `report.md` (Tables I–V next to the paper's values plus a "paper claims
checked" table), `tables/*.csv`, `figs/fig5_snrimp_vs_K.png`, `figs/fig6_time_vs_K.png`,
`figs/fig_waveforms.png` (Figs. 7–9) and `figs/fig_imfs_nvmdp.png`. The waveform fragment is the
test fragment with the **median** NVMD(P) SNR_imp, i.e. a typical example, not a hand-picked one.

### Changes made after reading the paper's references
- **DDAE** now follows the paper's ref. [33] (Chiang et al. 2019, FCN-DAE: Conv1d k16 with
  40/20/20/20/40/1 filters, ELU + BN, mirrored transposed convolutions; layer list from the public
  DeepFilter reimplementation). The previous DDAE (one Fig. 3 encoder-decoder, ~1 M parameters)
  is kept as an extra row, **DDAE-F3**, and is a stronger model than the paper's DDAE.
- **VMD targets** use the standard α = 2000, τ = 0. The paper's own Table V (IMF1 SNR_imp 13.32 dB
  vs overall 9.12 dB at K = 1) shows its clean IMFs do not sum exactly to the clean signal, which is
  what τ = 0 gives; the reports log the reconstruction SNR of the IMF sum for each K.
- bf16 autocast (speed only) and an inference batch size that scales with the signal length.

### Baseline caveat
With K = 1, EMD/EEMD "decomposition" returns the signal itself, so EMD-x and EEMD-x at K* = 1
are simply DWT / NLM on the raw signal; Eq. (23) picks K = 1 for most baselines here. This makes
our EMD/EEMD baselines stronger than the paper's on ECG and sEMG.

## What has been verified

- `tests/test_vmd.py`: the batched VMD equals the vmdpy reference to < 1e-8 for K = 1, 3, 5.
- MIT-BIH files: all 704 match PhysioNet's SHA256SUMS.
- Full CPU-profile runs for ECG, EEG and the sEMG stand-in (see `runs/*/report.md`).
