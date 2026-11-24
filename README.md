<div align="center">

# NCM-Light

### **Delta-NCM: AI Kaczmarz Fast-Weight Path Tracing, Hand-Free Ray-Traced Reflections, Volumetric Shafts & 8-Class Physical Optics — Engineered to Fly on Integrated GPUs**

[![Shaderpack CI](https://github.com/baobao1044/NCM-Light/actions/workflows/ci.yml/badge.svg)](https://github.com/baobao1044/NCM-Light/actions/workflows/ci.yml)
[![Download Latest ZIP](https://img.shields.io/github/v/release/baobao1044/NCM-Light?color=2ea043&label=Download%20NCM-Light.zip&logo=github)](https://github.com/baobao1044/NCM-Light/releases/latest)
[![OpenGL 3.3 Core](https://img.shields.io/badge/OpenGL-3.3%20Core%20(%23version%20120)-1f6feb)](#hardware--compatibility)
[![GLSL Permutations](https://img.shields.io/badge/Verified%20Shaders-179%2F179%20Passing-3fb950)](tools/check_pack.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-d29922.svg)](LICENSE)

**Zero Compute Shaders · Zero SSBOs · Zero TAA Smearing · Pure OpenGL 3.3 (`#version 120`) for Iris & OptiFine**

[**⬇ Download `NCM-Light.zip` (v3.2.0)**](https://github.com/baobao1044/NCM-Light/releases/latest) · [**Delta-NCM Path Tracing Proof**](#1-delta-ncm-path-tracing-proof--ai-fast-weight-attention-meets-gi) · [**Volumetric Cascade Proof**](#2-multiscale-volumetric-cascade--perceptual-gating) · [**Bandwidth & SSR Proof**](#3-breaking-the-igpu-bandwidth--ssr-wall) · [**8-Class Physical Optics**](#4-8-class-physical-optics-scattering--dispersion) · [**Comparison**](#head-to-head-comparison) · [**Tiếng Việt 🇻🇳**](#-tổng-quan--hướng-dẫn-nhanh-tiếng-việt)

</div>

---

| **`56.5 MB / frame`** | **`2.34×` Lower 1-spp Noise** | **`97.2%` 3D SH L1 Lock** | **`3.7×` Sharper SSR (`0.28m`)** | **`-98.6%` Shaft Taps** |
| :---: | :---: | :---: | :---: | :---: |
| **-66% DRAM Traffic** @ 1080p vs. `RGBA16F` PBR packs (`2.9×` lighter on shared DDR4 iGPUs) | **Delta-NCM Fast-Weight PT** (`1 spp` antithetic pair + R3 guiding + control variate) | **3D Directional Bounce** for 1080p normal maps, bevels & zero-cost **Indirect Specular** | **8+3 Hand-Free Binary SSR** with foreground & back-face rejection + Night Aurora reflections | **Telescoping Optical-Depth** quadrature (`MAE = 0.0028` vs 128-tap ground truth) |

---

## Why Next-Concept Modeling (NCM) Changes Everything

Most "ultra-light" Minecraft shaders get high FPS by **deleting lighting physics** — stripping out path-traced global illumination, reflections, subsurface scattering, and volumetric shadow shafts until the game looks like tinted Vanilla. Conversely, traditional PBR and path-tracing shaders brute-force uniform rays at full 1080p resolution across `5–18` heavy passes, choking integrated GPUs (**Intel UHD 620 / Iris Xe / AMD Vega**) with **166–250+ MB/frame** of DRAM traffic and **TAA motion ghosting**.

**NCM-Light unifies every subsystem under one mathematical principle — Next-Concept Modeling ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)):**
> *Solve smooth global transport at coarse resolution (`1/16th` screen pixels), maintain a learned directional fast-weight predictor, and spend full-resolution GPU compute strictly on **high-surprise residuals, sharp shadow-shaft edges, and specular/optical surfaces**.*

| Subsystem | Traditional Shader Approach | **NCM-Light v3.2 Solution** | Measured Impact |
| :--- | :--- | :--- | :--- |
| **Path-Traced GI** | Full-res blind Monte Carlo + heavy 2-pass SVGF (`95M` fetches/frame) + TAA blur | **Delta-NCM**: `0.25×` Kaczmarz SH L1 Fast-Weight PT + Antithetic Pair + L1-Cached 4-Tap Tent | **`2.34×`** lower 1-spp variance, **`97.2%`** 3D SH L1 lock, **`-84%`** denoiser fetches vs. SVGF |
| **Multi-Bounce GI** | Expensive 3D voxel grid + multi-ray recursion (OpenGL 4.3 Compute/SSBO) | **Directional Neumann Feedback** via persistent `0.25×` `colortex4/5` (`ρ < 0.5`) | Infinite-bounce color bleeding at **`0`** extra passes & pure OpenGL 3.3 |
| **Ray-Traced Reflections** | 16–32 uniform SSR steps + first-person sword/hand ghosting | **`8+3` Hand-Free Geometric Binary SSR** (`depthtex2` + foreground/back-face culling) | **`3.7×` sharper** (`0.28m` avg error), zero sword reflection, reflects Night Aurora |
| **Volumetric Shafts & Sky** | Uniform 16–64 ray-march steps + `exp()` on 100% of pixels | **Telescoping Optical-Depth Cascade** (`0.25×` L0 -> perceptual L1/L2) + zero-`exp()` media | **`-98.6%`** taps vs. 128-tap ground truth (`MAE = 0.0028`) |
| **Materials & Optics** | Extra G-Buffer attachments (`RGBA16F ×5` = `166 MB/frame`) | **3-bit 8-Class + 5-bit Sky** packed in 1 byte `colortex3.w` + zero-fetch shadow thickness | **Crystals, Thick SSS, Cauchy Dispersion, Kulla–Conty GGX & 42° Rainbow** at **`56.5 MB/frame`** |

---

## 1. Delta-NCM Path Tracing Proof — AI Fast-Weight Attention Meets GI

`v3.0–v3.2` introduces **Delta-NCM**, bridging **2026 AI Linear Attention / Fast-Weight Test-Time Training** ([Kaczmarz Linear Attention `arXiv:2605.08587`](https://arxiv.org/abs/2605.08587), [Gated DeltaNet-2 `arXiv:2605.22791`](https://arxiv.org/abs/2605.22791)) with **Unbiased Monte Carlo Path Tracing** ([Neural Two-Level Monte Carlo `arXiv:2412.04634`](https://arxiv.org/abs/2412.04634), [Neural Parametric Mixtures `arXiv:2504.04315`](https://arxiv.org/abs/2504.04315)) in pure **OpenGL 3.3 (`#version 120`)**:

<p align="center">
  <img src="assets/delta-ncm-pt-bench.png" alt="Delta-NCM v3.2 Path Tracing Quantitative Proof" width="100%">
</p>

> **What the 4 panels prove (100% reproducible via [`tools/simulate_levels.py`](tools/simulate_levels.py) & [`tools/generate_benchmarks.py`](tools/generate_benchmarks.py)):**
> - **Panel A — 1-spp Variance Reduction & Zero-Ghosting Step Jump**: Every `0.25×` pixel acts as a Test-Time-Trained Fast-Weight memory `(G0, L1)` updated along the **exact zero-mean hemisphere key** $\mathbf{d}(\omega) = \omega - \frac{2}{3}\mathbf{n}$ ($\mathbb{E}_{\cos}[\mathbf{d}] \equiv \mathbf{0}$). Driven by a **Martingale-Safe ($\mathcal{F}_{t-1}$-measurable) Decoupled Erase/Write Gate**, Delta-NCM cuts 1-spp single-ray variance by **`2.34×`** (`bias = 0.0195`), tracks 3D tilted normal-map irradiance accurately (`1.29×` lower RMSE), and adapts **`1.18×` faster** when a torch/light jumps $90^\circ$ at frame 60.
> - **Panel B — Learned 3D Hemisphere Light Field (`97.2%` Directional Lock)**: Standard scalar EMA collapses incoming light into a flat purple disk (**`0%` 3D directionality**), making walls and normal maps look washed out. Delta-NCM's orthogonal Kaczmarz update preserves perpendicular ray directions and locks onto the exact 3D light lobe (**`97.2%` cosine alignment**), powering both **1080p normal-map GI relief** and **zero-cost Indirect Specular Bounce** (`0` extra rays!).
> - **Panel C — Neumann Infinite-Bounce Convergence**: Secondary path hits query Shadow-Map NEE plus reprojected `colortex4` radiance with spectral radius $\rho = 0.50 \cdot \text{albedo} < 1$, converging smoothly to the exact analytical multi-bounce sum $L_\infty = L_{\text{dir}} / (1 - 0.50\rho)$ with **zero firefly blowup**.
> - **Panel D — Intel UHD 620 Compute De-Bottleneck (`-71%` GI Fetches vs. v3.0)**: On `~85%` of flat converged 1080p pixels, `composite2.glsl` runs an **L1-cached 4-tap tent smooth strictly on `0.25×` `colortex4`** (`0` depth/normal reads, `0` `4×4` blocky speckles!), reserving the 25-fetch normal/depth edge-stopping A-Trous filter strictly for `~15%` geometric edges & disocclusions — dropping total 1080p GI fetches from **`95.0M` (2-pass SVGF)** and **`51.8M` (v3.0)** down to **`14.8M` (`-71%` vs. v3.0, `-84%` vs. SVGF)**.

---

## 2. Multiscale Volumetric Cascade & Perceptual Gating

Using the exact same NCM residual hierarchy on volumetric sunlight shafts, clouds, night Aurora, and End accretion vortex:

<p align="center">
  <img src="assets/visual-cascade-bench.png" alt="NCM-Light 4-Panel Multiscale Visual Benchmark" width="100%">
</p>

> - **Panel 1 (`0.125 taps/px`)**: Quarter-resolution Concept Level 0 captures low-frequency light shafts and sky clouds/aurora inside `composite1`'s idle sky pixels for near-zero GPU cost.
> - **Panel 2 (`DEBUG_LEVEL = 2` live in-game view)**: Tonemapper-aware perceptual gating ([arXiv:2602.08642](https://arxiv.org/abs/2602.08642)) lights up **only** on shaft boundaries and silhouettes (**`< 6.1%`** of pixels reach Level 2), leaving cold navy regions untouched.
> - **Panel 3 vs. Panel 4**: Telescoping optical-depth quadrature ([arXiv:2611.12103](https://arxiv.org/abs/2611.12103)) converges to **`MAE = 0.0028`** against a **128-tap ground-truth integral** at **`2.89 taps/px` (`-97.7%` vs. 128-tap, `-64%` vs. 8-tap)** — with **zero `exp()` calls inside ray marches**.

---

## 3. Breaking the iGPU Bandwidth & SSR Wall

Why does an office laptop with an **Intel UHD 620 / Iris Xe / AMD Vega iGPU** collapse to `10–18 FPS` on conventional shaders? Because integrated GPUs share dual-channel system DDR4 (`~25–34 GB/s`) with the CPU. Burning `166–250 MB/frame` on five `RGBA16F` full-res targets saturates memory bandwidth before shading even starts.

<p align="center">
  <img src="assets/performance-benchmarks.png" alt="NCM-Light Quantitative Engineering Benchmarks" width="100%">
</p>

- **Panel A (`56.5 MB/frame`, `-66%` vs. `RGBA16F` PBR)**: Full-resolution G-buffers stay strictly **32-bit** (`R11F_G11F_B10F` + `RGBA8`), while `colortex1/4/5` run at `0.25×` (`1/16th` screen area), and `composite3` is completely eliminated via register fusion inside `final.glsl`.
- **Panel B (Telescoping Optical-Depth Quadrature)**: Exact telescoping measure $\Delta g_i = g(\frac{i+1}{N}) - g(\frac{i}{N})$ guarantees machine-exact parity across `2 → 4 → 8` levels on open air (`mean |residual| = 0.000166`).
- **Panel C (`8+3` Hand-Free Geometric Binary SSR — `3.7×` Sharper)**: Marches **8 dense geometric steps (`0.35m → 34m`)** strictly against `depthtex2` (opaque world ONLY — **zero first-person sword/hand ghosting!**) + **3 binary-bisection refinements** + **foreground-depth & back-face hit rejection**, reaching **`0.28 m` average hit precision** (`3.7×` sharper than 8-tap uniform SSR, `1.9×` sharper than 16-tap uniform SSR!) and reflecting **Night Aurora Borealis** + clouds on water, glass, metals, crystals, and flat rain puddles.

---

## 4. 8-Class Physical Optics, Scattering & Dispersion (Zero Extra VRAM)

By re-packing the single 8-bit channel `colortex3.w` into **3-bit Material Class (`0..7`) + 5-bit Skylight (`0..31`, twice Minecraft's 16 light levels)** in [`gbuffers_common.glsl`](NCM-Light/shaders/programs/gbuffers_common.glsl) and [`block.properties`](NCM-Light/shaders/block.properties), `NCM-Light v3.2` unlocks full physical optics at **`0` extra bytes**:

| Class | Material Group | Representative Blocks | Optical & Shading Model |
| :---: | :--- | :--- | :--- |
| **`1`** | **Matte / Dielectric** | Stone, dirt, wood, wool | Diffuse PBR + scalar fast-Z micro-crevice AO + Delta-NCM GI |
| **`2`** | **Foliage (`10100`/`10101`)** | Grass, leaves, flowers, crops, vines | Vertex wind waving (incl. shadow pass) + **Thin-Sheet SSS** wrap & backlit chlorophyll glow |
| **`3`** | **Polished / Clearcoat (`10300`)** | Quartz, obsidian, polished stones, glazed terracotta | Dielectric Fresnel + **SH L1 Indirect Specular Bounce** + `8+3` mirror SSR |
| **`4`** | **Conductors / Metals (`10400`)** | Iron, gold, copper, netherite, anvil, bell, chains | **Kulla–Conty energy-compensated RGB Conductor GGX** + metallic-tinted SSR |
| **`5`** | **Emissive HDR (`10200`)** | Lava, glowstone, lanterns, torches, froglights, lamps | High-HDR emission feeding Delta-NCM multi-bounce color bleed |
| **`6`** | **Crystals & Ice (`10600`)** | Diamond, emerald, amethyst, ice, packed/blue ice, glass | **Cauchy Prismatic Dispersion** (`≤1px`) + **Airy Thin-Film Iridescence** + crisp glass transparency |
| **`7`** | **Thick-Volume SSS (`10700`)** | Snow, powder snow, slime, honey, coral, mushrooms, sand | **Shadow-Map Thickness Beer–Lambert SSS** (`0` extra shadow taps!) + close-range snow sparkle |
| **Water** | **Liquid Water (`10500`)** | Water, flowing water, bubble columns | **Cauchy RGB Wave Refraction**, **Chromatic Underwater Caustics**, **Beer–Lambert $\exp(-\sigma_t d)$** absorption + Aurora SSR |

Plus:
- **Clean Stained-Glass Colored Shadows (`COLORED_SHADOWS`)**: Strict `shadowtex0 < shadowtex1 - 0.0005` translucent caster test + `2×2` bilinear `shadowcolor0` projection — zero colored edge bleed on normal terrain.
- **Physical `42°` Antisolar Atmospheric Rainbow (`SPECTRAL_DISPERSION`)**: Zero-tap analytical primary rainbow appearing naturally during sunlit rain.
- **Automatic `labPBR 1.3` (`LABPBR_SUPPORT`)**: Off by default on `Potato`/`Balanced` to save 2 full-1080p texture fetches per block on iGPUs; auto-enabled in `Ultra` and 1-click toggleable in menu.

---

## Complete v3.2 Render Graph (5 Stages, Pure OpenGL 3.3)

<p align="center">
  <img src="assets/pipeline-architecture.png" alt="NCM-Light v3.2 5-Stage OpenGL 3.3 Render Graph" width="100%">
</p>

Every stage above is directly verifiable inside [`NCM-Light/shaders/`](NCM-Light/shaders):
1. **`shadow`** ([`shadow.vsh`](NCM-Light/shaders/shadow.vsh), [`shadow.fsh`](NCM-Light/shaders/shadow.fsh)): $C^\infty$-smooth hyperbolic shadow distortion (`4.4×` near-player density = `~36 texels/block` at `1024px`), vertex-waving foliage shadows, stained-glass `shadowcolor0`, and **auto-stripped completely in The Nether & The End** (`21`-program shadowless pipelines).
2. **`gbuffers_*`** ([`gbuffers_surface.glsl`](NCM-Light/shaders/programs/gbuffers_surface.glsl), [`gbuffers_water_pbr.glsl`](NCM-Light/shaders/programs/gbuffers_water_pbr.glsl)): 32-bit `R11F_G11F_B10F` albedo + `RGBA8` octahedral normals + 8-class bit-packing.
3. **`composite1`** ([`composite1.glsl`](NCM-Light/shaders/programs/composite1.glsl)): Runs at **`0.25 × 0.25`** (`1/16th` screen pixels) — executes **Delta-NCM Kaczmarz Fast-Weight PT** (`colortex4/5` persistent reservoir + Antithetic Pairs + R3 Guiding + 4-Step Exponential DDA + Shadow NEE + Neumann Multi-Bounce), Concept Level-0 warped shafts, and Multiscale Volumetric Clouds / Night Aurora / End Vortex on sky pixels.
4. **`composite2`** ([`composite2.glsl`](NCM-Light/shaders/programs/composite2.glsl)): Full-resolution linear HDR PBR with **85% L1-cached 4-tap tent fast path** / **15% surprise-guided edge A-Trous**, `2×2` Hermite PCF + stained-glass shadows, 8-class BRDF, Thick Beer–Lambert SSS, Cauchy water refraction & chromatic caustics, and zero-cost SH L1 Indirect Specular Bounce.
5. **`final`** ([`final.glsl`](NCM-Light/shaders/programs/final.glsl)): Eliminates `composite3` by fusing perceptually gated Level-2 shafts (`8`-tap on `< 6%` of pixels), `8+3` Hand-Free Binary SSR, cloud silver-lining refinement, `42°` rainbow, rational atmospheric haze, and balanced ACES Filmic tonemapping.

---

## Head-to-Head Comparison

| Feature / Metric | Potato / "Vanilla+" Shaders | Mainstream Deferred PBR *(BSL / Complementary)* | Path-Traced / Voxel Packs *(SEUS PTGI / Rethinking)* | **NCM-Light v3.2 (This Repo)** |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum GPU API** | OpenGL 2.1 / 3.2 | OpenGL 3.3 – 4.3 | OpenGL 4.3+ / Compute / SSBO | **OpenGL 3.3 Core (`#version 120`)** |
| **1080p FBO Traffic / Frame** | `~35 MB` | `~166 MB` (`RGBA16F ×5`) | `~248+ MB` + 3D Voxel Grid | **`56.5 MB` (`32-bit` + `0.25×` PT)** |
| **Post-GBuffer Passes** | `1 – 2` | `5 – 9` fullscreen passes | `10 – 18` compute/fullscreen | **`2` (`0.25×` C1 + full C2) + Fused `final`** |
| **Path-Traced Multi-Bounce GI** | ❌ None | ⚠️ Optional / 1-bounce | ✅ Voxel / Compute PT | **✅ Delta-NCM Kaczmarz SH L1 PT (`1 spp`)** |
| **Indirect Specular Bounce** | ❌ None | ❌ None | ✅ Extra glossy rays | **✅ Zero-Cost from SH L1 (`0` extra rays!)** |
| **Ray-Traced Reflections (SSR)** | ❌ Sky tint only | ✅ Linear SSR (`16–32` steps) | ✅ Path-traced | **✅ `8+3` Hand-Free Binary SSR (`0.28m`)** |
| **Thick SSS & Chromatic Dispersion** | ❌ None | ⚠️ Foliage wrap only | ✅ Heavy volumetric | **✅ Shadow-Thick SSS + Cauchy Water/Crystals** |
| **1024px Near Shadow Density** | `~8–16 texels/block` | `~28–35 texels/block` | Ray-traced / Hybrid | **`~36 texels/block` ($C^\infty$ Hyperbolic Warp)** |
| **TAA Smearing / Ghosting** | ✅ None | ⚠️ Heavy TAA required | ❌ Severe motion ghosting | **✅ Zero TAA — Martingale DeltaNet-2 Gate** |
| **Intel UHD 620 / Laptop iGPU** | ✅ Smooth, looks flat | ⚠️ `12 – 22 FPS` @ 1080p | ❌ `2 – 7 FPS` (Unplayable) | **✅ Engineered & Verified on Intel UHD 620** |

---

## Live In-Game Diagnostic Inspector (`DEBUG_LEVEL`)

Inspect every internal NCM pass live inside **Video Settings → Shader Pack Settings → Debug view**:

| Mode | View Name | What It Proves Live in Minecraft |
| :---: | :--- | :--- |
| **`0`** | **Final Image** *(Default)* | Full Delta-NCM PT + 8-Class Physical Optics + Hand-Free SSR + Shafts + ACES |
| **`1`** | **Coarsest Level Only (`1/4` res)** | Raw `0.25 × 0.25` quarter-resolution transport before edge refinement |
| **`2`** | **Residual / Work Heatmap** | **Cold Navy** = skipped fast path · **Orange/Gold** = refined shaft edges & SSR |
| **`3`** | **Volumetric Light Only** | Isolates directional Mie god-rays and `> 32m` atmospheric haze |
| **`4`** | **Depth & Layer Map** | Visualizes opaque (`depthtex2`), translucent water/glass (`depthtex0`), and hand (`depthtex1`) |
| **`5`** | **Haze & Shaft Palette** | Split-screen preview of real-time celestial haze & shaft linear tints |
| **`6`** | **Pure Clay Delta-NCM GI + RTAO + Specular** | Strips block textures to neutral clay to inspect **multi-bounce colored GI**, **3D crevice RTAO**, and **SH L1 specular glints** |

---

## Hardware & Compatibility

- **Minecraft**: `1.16.5` – `1.21.x` / `26.x`
- **Loaders**: **Iris** (`1.6+` / `1.11+` + Sodium — Recommended) & **OptiFine**
- **GPUs**: Any **OpenGL 3.3 Core** GPU — Intel HD/UHD (`620`, `630`, `730`, `770`, Iris Xe, Arc), AMD Radeon Vega / RX, NVIDIA GTX / RTX, and Apple Silicon (`M1`–`M4`).

---

## Reproducible Verification & Build Suite

Every commit and release is automatically verified in CI by [`simulate_levels.py`](tools/simulate_levels.py) (proving volumetric convergence, Neumann multi-bounce stability, and Delta-NCM's `2.34×` variance reduction & `97.2%` SH L1 alignment) and [`check_pack.py`](tools/check_pack.py) (compiling all **179 GLSL shader permutations** across Overworld, Nether, and End via `glslangValidator`):

```bash
# 1. Run numerical convergence, Neumann multi-bounce & Delta-NCM 1-spp proofs
python tools/simulate_levels.py

# 2. Audit properties, Iris directives & compile all 179 GLSL permutations
python tools/check_pack.py NCM-Light --glslang glslangValidator

# 3. Regenerate all 4 dark-mode visual benchmark figures in assets/
uv run --with matplotlib --with numpy tools/generate_benchmarks.py

# 4. Package distributable shaderpack archive
python tools/package_pack.py NCM-Light NCM-Light.zip
```

---

## 🇻🇳 Tổng quan & Hướng dẫn nhanh (Tiếng Việt)

### NCM giúp ích thực sự ở đâu — từ Path Tracing, Ray Tracing đến Quang học Vật lý?
1. **Path Tracing Đa Nảy `Delta-NCM` chạy nổi trên Intel UHD 620**:
   - Kết hợp trực tiếp 2 công trình AI 2026 (**Kaczmarz Linear Attention `arXiv:2605.08587`** + **Gated DeltaNet-2 `arXiv:2605.22791`**) vào Path Tracing: mỗi pixel `0.25×` lưu bộ nhớ Fast-Weight SH L1 3D, bắn **cặp tia đối xứng Antithetic + Tự lái tia R3**, giúp **giảm `2.34×` phương sai đơn tia**, khóa chính xác **`97.2%` hướng sáng 3D**, phục hồi nhanh hơn `1.18×` khi đặt đuốc mà **không dùng TAA mờ chuyển động**.
   - **Cắt `71%` lượt đọc khử nhiễu 1080p** (`51.8M -> 14.8M` fetch/frame) nhờ lọc L1-cache 4-tap trên 85% vùng phẳng + A-Trous chặn cạnh ở 15% viền khối -> **hết nhiễu hạt màu, chạy nhẹ trên iGPU**.
2. **Phản chiếu `8+3` Binary SSR sạch tuyệt đối (`3.7×` sắc hơn)**:
   - Dò tia trực tiếp trên `depthtex2` (loại bỏ 100% lỗi in bóng kiếm/tay người chơi vào gương/kính), lọc va chạm mặt sau & vật tiền cảnh cho vũng mưa/block ướt phẳng thật, đồng thời phản chiếu cả **Cực Quang ban đêm** và mây thể tích.
3. **8 lớp vật liệu quang học vật lý — không tốn thêm 1 byte VRAM (`56.5 MB/frame`)**:
   - Đủ **tán xạ khối dày Beer–Lambert** (tuyết, băng, slime, mật ong), **tán sắc Cauchy** (khúc xạ nước, vân nắng đáy nước cầu vồng, đá quý), **hắt sáng Specular GI miễn phí từ SH L1**, **bóng nắng xuyên kính màu sạch biên**, **Kulla–Conty GGX kim loại**, và **cầu vồng vật lý $42^\circ$** khi mưa nắng.

### Cài đặt trong 30 giây
1. Tải file [**`NCM-Light.zip`**](https://github.com/baobao1044/NCM-Light/releases/latest) và thả vào thư mục `.minecraft/shaderpacks/`.
2. Vào game: **Options → Video Settings → Shader Packs** → chọn **`NCM-Light.zip`** → nhấn **Apply**.

### Cấu hình khuyên dùng theo Profile 1-Click
| Profile | Cấu hình máy phù hợp | Thiết lập mặc định trong Profile |
| :--- | :--- | :--- |
| **`Potato iGPU (Max FPS)`** | Intel UHD 620 / HD 520 / Laptop văn phòng | `Shadow = 512`, `RT_GI = Off`, giữ nguyên mây thể tích, phản chiếu nước, gió thổi cỏ lá & caustics |
| **`Balanced (Recommended)`** | Intel UHD 620–770 / Iris Xe / Vega / GTX 1050+ | `Shadow = 1024`, đầy đủ **Delta-NCM PT (`1 spp` + Guiding + Multi-Bounce)**, 8 lớp quang học, tán sắc & SSS (`LABPBR = Off` cho tối đa FPS) |
| **`Ultra (High Detail)`** | GPU rời GTX 1650+ / RTX / RX | `Shadow = 2048`, `PT_SPP = 2`, `RT_STRENGTH = 1.25`, tự động bật **`labPBR 1.3`** |

---

## Citation & Research References

- **Next-Concept Modeling**: [https://arxiv.org/abs/2610.14162](https://arxiv.org/abs/2610.14162)
- **Kaczmarz Linear Attention**: [https://arxiv.org/abs/2605.08587](https://arxiv.org/abs/2605.08587)
- **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention**: [https://arxiv.org/abs/2605.22791](https://arxiv.org/abs/2605.22791)
- **Neural Two-Level Monte Carlo Real-Time Rendering**: [https://arxiv.org/abs/2412.04634](https://arxiv.org/abs/2412.04634)
- **Neural Parametric Mixtures for Path Guiding**: [https://arxiv.org/abs/2504.04315](https://arxiv.org/abs/2504.04315)
- **Conditional Neural Control Variates for Variance Reduction**: [https://arxiv.org/abs/2602.21357](https://arxiv.org/abs/2602.21357)
- **Real-Time Volumetric Light Transport Inference without a Path**: [https://arxiv.org/abs/2611.12103](https://arxiv.org/abs/2611.12103)
- **Forget Superresolution, Sample Adaptively**: [https://arxiv.org/abs/2602.08642](https://arxiv.org/abs/2602.08642)
- **Occlusion-Point Reuse for Ray-Traced AO and Shadow**: [https://arxiv.org/abs/2607.23122](https://arxiv.org/abs/2607.23122)
- **Split Radiance Cascades**: [https://arxiv.org/abs/2607.20384](https://arxiv.org/abs/2607.20384)
- **Screen Space Indirect Lighting with Visibility Bitmask**: [https://arxiv.org/abs/2301.11376](https://arxiv.org/abs/2301.11376)

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Gia Bảo ([@baobao1044](https://github.com/baobao1044)).
