# NCM-Light

[![Shaderpack CI](https://github.com/baobao1044/NCM-Light/actions/workflows/ci.yml/badge.svg)](https://github.com/baobao1044/NCM-Light/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/baobao1044/NCM-Light?label=Download%20NCM-Light.zip)](https://github.com/baobao1044/NCM-Light/releases/latest)
[![OpenGL 3.3 Core](https://img.shields.io/badge/OpenGL-3.3%20Core%20(%23version%20120)-blue)](#hardware--compatibility)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NCM-Light** is an ultra-lightweight, iGPU-friendly **Linear HDR PBR + Multiscale Ray-Tracing & Volumetric** shaderpack for **Minecraft (Iris & OptiFine)** engineered strictly within **OpenGL 3.3 Core (`#version 120`)** — zero compute shaders, zero SSBOs, and zero temporal ghosting.

By mapping recent multiscale residual modeling and real-time light-transport formulations from arXiv onto a **32-bit bandwidth-compressed G-Buffer** and a **2-pass + `final` render graph**, `NCM-Light` delivers crisp shadows, 3D crevice RTAO, warm global-illumination color bleeding, coherent mirror reflections, and directional sun shafts even on integrated GPUs (such as Intel UHD Graphics 620).

---

## Key Highlights

- **Ultra-Lean 2-Pass + `final` Architecture**:
  ```text
  shadow (1024px, 4.4x hyperbolic distortion)
    └─► gbuffers_* (32-bit packed G-Buffer)
          └─► composite1 (0.25 × 0.25 quarter-res = 1/16th screen pixels!)
                └─► composite2 (full-res Linear HDR PBR + GGX + edge-aware gather)
                      └─► final (Level-2 shafts + Binary RTR + Haze + ACES Filmic)
  ```
- **~65% Lower VRAM/DRAM Bandwidth**: Replaces heavy `64-bit RGBA16F` full-screen attachments with `32-bit` hardware formats (`R11F_G11F_B10F` for Linear HDR radiance and bit-exact `RGBA8` octahedral normal + lightmap + IntegratedPBR material packing).
- **100% Grain-Free Without TAA**: Deterministic golden-quadrant hemisphere GI/AO, sub-texel Hermite-bilinear PCF on a $C^\infty$-distorted shadow map (`~36 texels/block` near the player at `1024px`), centered-midpoint volumetric quadrature, and coherent `5+3` binary-bisection reflections eliminate stochastic salt-and-pepper noise without temporal blur.
- **Dimension-Specialized Pipelines**: Automatically strips shadow passes and celestial ray marches in **The Nether (`world-1`)** and **The End (`world1`)** while preserving emissive RTGI, specular reflections, and volumetric atmosphere.

---

## Theoretical Foundations & Algorithmic Mapping

`NCM-Light` synthesizes ideas from recent graphics and multiscale modeling literature into closed-form fragment shaders:

| Reference | Core Idea Adapted in `NCM-Light` | Implementation |
|---|---|---|
| **Next-Concept Modeling** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162)) | **Telescoping Multiscale Residual Hierarchy**: Evaluates coarse transport at `0.25 × 0.25` resolution (`1/16` pixel count) and selectively refines higher dyadic levels via `est + w * (fine - est)` only where residual predictors fire. | `composite1.glsl`, `composite2.glsl`, `final.glsl` |
| **Volumetric Transport without a Path** ([arXiv:2611.12103](https://arxiv.org/abs/2611.12103)) | **Conditioned Optical-Depth Importance Warp & Rational Extinction**: Concentrates `2 → 4 → 8` quadrature nodes toward expected first-interaction optical depth with exact telescoping step measure $\sum \Delta g_i \equiv 1$, and replaces `1 - exp(-tau)` with $\hat{\tau} = \frac{\tau}{1 + c\tau}$. | `common.glsl` (`marchShafts`, `analyticHaze`) |
| **Forget Superresolution, Sample Adaptively** ([arXiv:2602.08642](https://arxiv.org/abs/2602.08642)) | **Tonemapper-Aware Perceptual Gating & $2\times 2$ Gather Upsampling**: Modulates refinement thresholds by De Vries–Rose dark sensitivity and ACES highlight compression (`perceptualImportance`), paired with a $C^0$-continuous `2×2` normal/depth edge-stopping gather on discontinuities (`~85%` of pixels take a 1-tap hardware bilinear fast path). | `common.glsl`, `composite2.glsl` (`gatherCoarseEdge2x2`) |
| **Visibility-Bitmask SSIL & Split Radiance Cascades** ([arXiv:2301.11376](https://arxiv.org/abs/2301.11376), [arXiv:2607.20384](https://arxiv.org/abs/2607.20384)) | **Finite-Thickness Dual-Radius Hemisphere Probing**: Probes disjoint near (`~0.55m` crevice) and mid (`~2.6m–5.2m` bounce) intervals across 4 deterministic golden quadrants (`0.50` depth taps per screen pixel) with finite thickness slabs so thin foliage/fences do not cast dark halos. | `common.glsl` (`traceOmniHemisphereRT`) |
| **Occlusion-Point Reuse** ([arXiv:2607.23122](https://arxiv.org/abs/2607.23122)) | **Geometry-Consistent Occluder Weighting**: Rational depth-normal kernels and hyperbolic $C^\infty$ shadow-map distortion (`SHADOW_DISTORT = 0.78`) with L1-cache-aligned `2×2` Hermite sub-texel PCF. | `shadow.vsh`, `common.glsl` (`surfaceShadow`) |

---

## G-Buffer Layout (32-Bit Bandwidth-Tuned)

| Buffer | Resolution | Format | Bytes/px | Contents |
|---|---|---|---|---|
| `colortex0` | `1.00 × 1.00` | `R11F_G11F_B10F` | `4 B` | Albedo (`gbuffers_*`) $\to$ Linear HDR lit scene (`composite2`) |
| `colortex2` | `1.00 × 1.00` | `RGBA8` | `4 B` | Translucent water/glass tag $\to$ `(estRays1, wMain1, packedTrans, volGate1)` |
| `colortex3` | `1.00 × 1.00` | `RGBA8` | `4 B` | Octahedral world normal `.xy`, blocklight `.z`, `matTag * 42 + skyQ` `.w` |
| `colortex1` | `0.25 × 0.25` | `RGBA16F` | `0.5 B` eff. | Quarter-res Level-0 `(rays0, ao0, volGate0, edgeGate0)` |
| `colortex4` | `0.25 × 0.25` | `R11F_G11F_B10F` | `0.25 B` eff. | Quarter-res Level-0 Linear HDR RTGI bounce `gi0.rgb` |

**IntegratedPBR Material Classes (`shaders/block.properties`)**:
- `10100`: Foliage, leaves, crops & flowers (subsurface sun-transmission + cross-mesh self-shadow guard)
- `10200`: Emissive blocks — torches, lanterns, lava, glowstone, froglights (`6.8×` HDR GI bounce)
- `10300`: Polished stone, quartz, obsidian, ice & stained glass (glossy dielectric binary RTR)
- `10400`: Iron, gold, diamond, emerald, netherite & copper blocks (metallic tinted mirror RTR)
- `10500`: Liquid water (procedural wave normals + Beer-like teal absorption + Fresnel mirror RTR)

---

## Hardware & Compatibility

- **Minecraft**: `1.16.5` – `1.21.x` / `26.x`
- **Shader Loaders**: **Iris** (`1.6+` / `1.11+` with Sodium) and **OptiFine**
- **GPU API**: Any GPU supporting **OpenGL 3.3 Core Profile** (Intel UHD 620+, AMD Vega/RDNA, NVIDIA GTX/RTX, Apple M-series via translation layers)

---

## Development & Automated Verification

`NCM-Light` ships with a static analyzer and numerical convergence simulator that validate all `152` GLSL shader permutations (including Iris `CommentDirectiveParser` `lastIndexOf` invariants and OpenGL 3.3 blend constraints) before packaging:

```bash
# 1. Verify multiscale optical-depth quadrature convergence & tap savings
python tools/simulate_levels.py

# 2. Validate shader structure, lang files, Iris directives & compile all 152 GLSL permutations
python tools/check_pack.py NCM-Light --glslang glslangValidator

# 3. Build distributable shaderpack archive
python tools/package_pack.py NCM-Light NCM-Light.zip
```

---

## Cài đặt nhanh (Tiếng Việt)

1. Tải file **`NCM-Light.zip`** mới nhất tại mục [**Releases**](https://github.com/baobao1044/NCM-Light/releases/latest) và chép vào thư mục `shaderpacks/` của Minecraft.
2. Vào **Options → Video Settings → Shader Packs**, chọn **`NCM-Light.zip`** → nhấn **Apply**.
3. **Mẹo tối ưu FPS cho iGPU (Intel UHD / Laptop văn phòng)**:
   - Mặc định `Shadow resolution = 1024` đã được nén biến dạng `4.4×` rất nét quanh người chơi. Nếu chơi ở Render Distance `12–16 chunks` trên iGPU yếu, bạn có thể hạ **Shadow resolution** xuống **`512`** trong *Shader Pack Settings* để tăng mạnh FPS mà bóng đổ vẫn mịn đẹp.
   - Chuyển **Debug view (`DEBUG_LEVEL`)** sang **`2`** để xem bản đồ nhiệt vùng tinh chỉnh NCM, hoặc **`6`** để soi trực tiếp ánh sáng gián tiếp RTGI + RTAO trên mô hình đất sét trắng.

---

## License

Released under the [MIT License](LICENSE).
