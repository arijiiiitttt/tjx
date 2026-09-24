<p align="center">
<a>
<img alt="tjx logo" src="/public/logo/tjx.png" width="132">
</a>
</p>

<p align="center">
<a href="https://discord.com/users/thearijiiiitttt_"><img alt="Discord" src="https://img.shields.io/badge/discord-community-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
<a href="#-installation"><img alt="Node" src="https://img.shields.io/badge/node-18%2B-339933?style=flat-square&logo=node.js&logoColor=white" /></a>
</p>

<p align="center">
<b>tjx</b> train a real image classifier <em>entirely in your browser</em> 📦
</p>

> **New to this project?** This guide walks through everything step by step, with no skipped steps.


## Table of Contents

1. [What is tjx?](#-what-is-ml-studio)
2. [Before You Start](#-before-you-start)
3. [Installation](#-installation)
4. [Running tjx](#-running-ml-studio)
5. [The Guided Workflow](#-the-guided-workflow)
6. [Training Modes](#-training-modes)
7. [What's Real (Not Simulated)](#-whats-real-not-simulated)
8. [Project Layout](#-project-layout)
9. [Privacy Model](#-privacy-model)
10. [Scripts](#-scripts)
11. [Testing](#-testing)
12. [Troubleshooting](#-troubleshooting)

<br/>

## What is tjx?

**tjx** is a local-first image-classification workstation that runs completely in the browser. You add images, train a neural network, evaluate it, and run inference — all without a backend. Decode, tensors, forward pass, backpropagation, weight updates, evaluation, storage, and webcam inference stay on your device.

Your images **never leave the browser**.

**tjx can:**
- Ingest images from files or the webcam, with quality checks and duplicate detection
- Train with **transfer learning** (frozen MobileNetV2 + trainable head) or **from scratch** (small CNN)
- Import a custom TF.js backbone when TF Hub is unreachable
- Evaluate with per-class precision/recall/F1 and a full confusion matrix
- Export models as a zip or as separate `model.json` / `weights.bin` / metadata files
- Persist projects, samples, experiments, and models in IndexedDB

<br/>

## Before You Start

You need:
- **Node.js 18+**
- A modern browser with **WebGPU** or **WebGL** (Chrome / Edge recommended)
- No API key and no cloud account

The camera requires `https://` or `http://localhost`.

<br/>

## Installation

```bash
# 1. Go to the project folder
cd mlstudio

# 2. Install dependencies
npm install

# 3. (Optional) type-check
npm run typecheck
```

<br/>

## Running tjx

```bash
npm run dev
# → http://localhost:5173
```

Build for production:

```bash
npm run build      # type-checks, then builds to dist/
npm run preview    # serve the production build
```

<br/>

## The Guided Workflow

| Step | What you do |
| --- | --- |
| **1. Dataset** | Create classes, add images (files or webcam) |
| **2. Training** | Choose mode, backend, augmentation; run training |
| **3. Evaluation** | Inspect accuracy, confusion matrix, confident mistakes |
| **4. Inference** | Test with a still image or live webcam |
| **5. Models** | Load, export, import, or delete saved models |

A vertical step rail on wide screens echoes progress; the dashboard summarizes images, experiments, models, and the active backend.

<br/>

## Training Modes

| Mode | What trains | Best when |
| --- | --- | --- |
| **Transfer learning** | Small head on frozen MobileNetV2 | ≥ ~30 images per class; default |
| **From scratch** | Small CNN, full backprop + Adam | Learning / demos; no network needed |
| **Custom backbone** | Your TF.js layers model as features | Offline / air-gapped / no TF Hub |

- Transfer: backbone is downloaded once from TF Hub and cached in IndexedDB; later runs work offline.
- From scratch: includes a built-in **gradient check** (autodiff vs numerical central differences) on the active backend.
- Custom backbone: import `model.json` + `weights.bin` on the Training page; validated against a real third-party model during development.

<br/>

## What's Real (Not Simulated)

- **Backend probe** — WebGPU → WebGL → WASM → CPU is verified with an actual gradient step, not just “does it load”
- **Dataset health** — blank/dark/bright/tiny images, SHA-256 exact duplicates, perceptual near-duplicates, class imbalance
- **Split** — seeded, stratified per class, burst-aware (camera bursts stay together; boundary frames purged to avoid leakage)
- **Evaluation** — precision / recall / F1 per class, confusion matrix, most confident mistakes
- **Import safety** — untrusted models face a layer allow-list, size limits, and weight/metadata consistency checks
- **Memory** — tensors are scoped with `tf.tidy` or disposed; tests assert tensor counts return to baseline

<br/>

## Project Layout

```
src/
  app/          shell, Zustand store, defaults
  features/     React pages: dataset, training, evaluation, inference, models, settings, dashboard
  ml/           pure ML logic (no React): runtime, preprocessing, augmentation, data, models, training, metrics, inference, export
  workers/      Web Worker + RPC host (training off the UI thread)
  storage/      IndexedDB repository (in-memory fallback)
  components/   shared UI
  types/        domain & ML types
tests/          Vitest unit + integration tests
docs/           ARCHITECTURE.md (design, limits, decisions)
```

<br/>

## Privacy Model

- Images, tensors, gradients, and weights live **only in this browser**
- IndexedDB stores projects locally; you can request persistent storage in Settings
- No analytics, no upload, no remote training
- Status bar reminder: *Images stay on device · training runs locally*

<br/>

## Scripts

```bash
npm run dev         # Vite dev server (copies WASM assets first)
npm run build       # tsc --noEmit + production build
npm run typecheck   # TypeScript only
npm run test        # Vitest suite
npm run preview     # Preview dist/
```

<br/>

## Testing

```bash
npm test
```

- `tests/pure.test.ts` / `tests/ml.test.ts` — real training (forward → backprop → weight update) on CPU, memory baseline, gradient check, export/import round-trip including rejection of tampered files
- `tests/app.test.tsx` — Zustand store + every page via Testing Library with a fake ML host

See `docs/ARCHITECTURE.md` for known limits from headless Chromium smoke tests.

<br/>

## Troubleshooting

| Problem | Solution |
| --- | --- |
| Backend stuck on “starting…” | Wait for the probe; try Settings → Restart runtime |
| WebGPU unavailable | Falls back to WebGL → WASM → CPU automatically |
| Camera permission denied | Use `http://localhost` or HTTPS; allow camera in the browser |
| TF Hub backbone fails to download | Training page → import a custom TF.js backbone (`model.json` + weights) |
| Storage warning / data not saved | Settings → Request persistent storage, or free browser quota |
| Training needs at least 2 classes | Dataset page: add ≥ 2 classes with samples |
| Tensor / GPU memory climbing | Expected during training; should drop after dispose / page change |
| `npm install` / lockfile issues | Use Node 18+; delete `node_modules` and reinstall |

<br/>
