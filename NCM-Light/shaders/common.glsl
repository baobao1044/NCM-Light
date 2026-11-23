/*
    NCM-Light  --  High-FPS Cinematic Linear HDR PBR + arXiv Multiscale RT & Volumetrics
    Tuned for crisp 4.4x distorted shadows, deep 3D crevice RTAO, warm RTGI color bounce,
    GGX micro-specular relief, and high 1% low FPS on integrated & discrete GPUs (OpenGL 3.3).

    IMPORTANT: Never write the directive keyword for drawbuffers inside comments
    in any included .glsl header -- Iris scans expanded sources with lastIndexOf()!

    Loader-consumed 32-bit fast buffer directives:
    const int colortex0Format = R11F_G11F_B10F;
    const int colortex1Format = RGBA16F;
    const int colortex2Format = RGBA8;
    const int colortex3Format = RGBA8;
    const int colortex4Format = R11F_G11F_B10F;
    const int shadowcolor0Format = RGBA8;
    const float shadowIntervalSize = 2.0;
    const float shadowDistanceRenderMul = 1.0;
*/

#ifndef NCM_COMMON_GLSL
#define NCM_COMMON_GLSL

#include "/options.glsl"

/* ============================= Constants ============================= */

const float PI = 3.14159265;
const float REF_BLOCK = 62.0;
const float HEIGHT_SCALE = 30.0;
const float RAY_START = 0.04;
const float RAY_END = 1.0;
const float SKY_DIST = 192.0;
const float MIN_AIR_RUN = 16.0;

const float CLEAR_AIR_RADIUS = 32.0;
const float SKY_HAZE_WEIGHT = 0.0;
const float SKY_RAYS_WEIGHT = 0.45;

const float HAZE_COEFF = 0.0034;
const float RAYS_COEFF = 0.0120;
const float FOG_CAP = 0.42;
const float RAYS_CAP = 0.75;
const float SHADOW_BIAS = 0.00045;

const int TAPS_CONCEPT0 = 2;
const int TAPS_CONCEPT1 = 4;
const int TAPS_CONCEPT2 = 8;
const int MAX_TAPS = TAPS_CONCEPT2;

/* ============================== Uniforms ============================= */

uniform sampler2D colortex0;    /* sRGB albedo -> Linear HDR lit scene (R11G11B10F) */
uniform sampler2D colortex1;    /* 1/4-res level 0: (rays0, ao0, volGate0, edgeGate0) */
uniform sampler2D colortex2;    /* full-res RGBA8:  (rays,  wL1, packedTrans, volGate1) */
uniform sampler2D colortex3;    /* full-res RGBA8:  (oct01.xy, blockLight, matSkyByte) */
uniform sampler2D colortex4;    /* 1/4-res RTGI 0:  gi0.rgb (R11G11B10F) */

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D depthtex2;
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowModelView;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 moonPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;
uniform vec3 fogColor;
uniform vec3 skyColor;
uniform ivec2 eyeBrightnessSmooth;

uniform float frameTimeCounter;
uniform float rainStrength;
uniform float blindness;
uniform float near;
uniform float far;
uniform float viewWidth;
uniform float viewHeight;
uniform int isEyeInWater;

/* =========== Natural ACES Filmic Grading & Perceptual Gating ========= */

vec3 srgbToLinear(vec3 c) {
    vec3 x = max(c, vec3(0.0));
    return x * x * (0.715 + 0.285 * x);
}

vec3 linearToSrgb(vec3 lin) {
    vec3 x = clamp(lin, 0.0, 1.0);
    vec3 s1 = sqrt(x);
    vec3 s2 = sqrt(s1);
    return 1.128 * s1 - 0.128 * s2;
}

/* Balanced ACES Filmic curve with highlight desaturation & lime-chroma taming */
vec3 acesTonemapAndGrade(vec3 hdr) {
    vec3 x = max(hdr, vec3(0.0)) * 0.88;
    vec3 mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    mapped = clamp(mapped, 0.0, 1.0);

    /* Subtle filmic highlight rolloff & natural saturation balance (no radioactive lime!) */
    float luma = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
    mapped = mix(vec3(luma), mapped, 1.04 - 0.10 * luma);
    return linearToSrgb(clamp(mapped, 0.0, 1.0));
}

/* Tonemapper-Aware Perceptual Sample Importance (Balint et al., arXiv:2602.08642v1 Sec. 5) */
float perceptualImportance(float linLuma) {
    float x = max(linLuma, 0.0);
    float deVriesRose = x / (x + 0.035);
    float shoulderGain = 1.0 / (1.0 + 0.55 * x * x);
    return clamp(0.32 + 1.38 * deVriesRose * shoulderGain, 0.25, 1.15);
}

/* =================== 32-Bit RGBA8 G-Buffer Decoding ================== */

vec3 octDecodeNormal(vec2 oct01) {
    vec2 e = oct01 * 2.0 - 1.0;
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        vec2 signNotZero = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * signNotZero;
    }
    float len2 = dot(v, v);
    return (len2 > 1e-6) ? (v * inversesqrt(len2)) : vec3(0.0, 1.0, 0.0);
}

void unpackSurfaceData(vec4 surf, out bool valid, out vec3 worldNormal,
                       out float blockLight, out float skyLight, out float matTag)
{
    float rawByte = floor(surf.w * 255.0 + 0.5);
    if (rawByte < 21.0) {
        valid = false;
        worldNormal = vec3(0.0, 1.0, 0.0);
        blockLight = 0.0;
        skyLight = 1.0;
        matTag = 0.0;
        return;
    }
    valid = true;
    matTag = floor((rawByte + 2.0) / 42.0);
    skyLight = clamp((rawByte - matTag * 42.0) / 36.0, 0.0, 1.0);
    worldNormal = octDecodeNormal(surf.xy);
    blockLight = surf.z;
}

/* =================== Fast Noise, Basis & Projection ================== */

float ignNoise(vec2 pixelCoord) {
    return fract(52.9829189 * fract(dot(pixelCoord, vec2(0.06711056, 0.00583715))));
}

void buildOrthonormalBasis(vec3 n, out vec3 b1, out vec3 b2) {
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    b2 = vec3(b, s + n.y * n.y * a, -n.y);
}

float fastViewZ(float d) {
    float cz = d * 2.0 - 1.0;
    float vz = gbufferProjectionInverse[2].z * cz + gbufferProjectionInverse[3].z;
    float vw = gbufferProjectionInverse[2].w * cz + gbufferProjectionInverse[3].w;
    return vz / max(abs(vw), 1e-6) * (vw < 0.0 ? -1.0 : 1.0);
}

vec3 viewFromDepth(vec2 uv, float d) {
    vec3 clip = vec3(uv * 2.0 - 1.0, d * 2.0 - 1.0);
    float vw = gbufferProjectionInverse[2].w * clip.z + gbufferProjectionInverse[3].w;
    float invW = (vw < 0.0 ? -1.0 : 1.0) / max(abs(vw), 1e-6);
    vec3 v;
    v.x = gbufferProjectionInverse[0].x * clip.x + gbufferProjectionInverse[3].x;
    v.y = gbufferProjectionInverse[1].y * clip.y + gbufferProjectionInverse[3].y;
    v.z = gbufferProjectionInverse[2].z * clip.z + gbufferProjectionInverse[3].z;
    return v * invW;
}

vec3 fastProjectToScreen(vec3 vPos) {
    float clipW = gbufferProjection[2].w * vPos.z + gbufferProjection[3].w;
    if (clipW <= 1e-4) return vec3(-1.0);
    float invW = 1.0 / clipW;
    vec2 ndcXY = (vec2(gbufferProjection[0].x, gbufferProjection[1].y) * vPos.xy
                + gbufferProjection[2].xy * vPos.z
                + gbufferProjection[3].xy) * invW;
    float ndcZ = (gbufferProjection[2].z * vPos.z + gbufferProjection[3].z) * invW;
    return vec3(ndcXY * 0.5 + 0.5, ndcZ * 0.5 + 0.5);
}

vec3 reconstructGeomWorldNormal(vec2 uv, vec3 centerViewPos) {
    vec2 px = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float dR = texture2D(depthtex0, clamp(uv + vec2(px.x, 0.0), 0.001, 0.999)).r;
    float dU = texture2D(depthtex0, clamp(uv + vec2(0.0, px.y), 0.001, 0.999)).r;
    vec3 pR = viewFromDepth(uv + vec2(px.x, 0.0), dR) - centerViewPos;
    vec3 pU = viewFromDepth(uv + vec2(0.0, px.y), dU) - centerViewPos;
    vec3 vn = cross(pR, pU);
    float l2 = dot(vn, vn);
    if (l2 < 1e-8) return vec3(0.0, 1.0, 0.0);
    vn *= inversesqrt(l2);
    return normalize(mat3(gbufferModelViewInverse) * vn);
}

/* =========== Warm-Sun / Cool-Azure-Sky Cinematic Palette ============= */

float sunElevation() {
    float sunLen2 = dot(sunPosition, sunPosition);
    float upLen2 = dot(upPosition, upPosition);
    if (sunLen2 > 1e-4 && upLen2 > 1e-4) {
        return dot(sunPosition * inversesqrt(sunLen2), upPosition * inversesqrt(upLen2));
    }
    if (sunLen2 > 1e-4) {
        return (mat3(gbufferModelViewInverse) * (sunPosition * inversesqrt(sunLen2))).y;
    }
    return 0.6;
}

vec3 activeLightDirView(out float valid) {
    float len2 = dot(shadowLightPosition, shadowLightPosition);
    if (len2 > 1e-4) {
        valid = 1.0;
        return shadowLightPosition * inversesqrt(len2);
    }
    float sunLen2 = dot(sunPosition, sunPosition);
    if (sunLen2 > 1e-4) {
        valid = 1.0;
        vec3 sDir = sunPosition * inversesqrt(sunLen2);
        return (sunElevation() >= 0.0) ? sDir : -sDir;
    }
    valid = 0.0;
    return vec3(0.0, 1.0, 0.0);
}

vec3 worldShadowLightDir() {
    float valid;
    vec3 vDir = activeLightDirView(valid);
    if (valid < 0.5) return vec3(0.0, 1.0, 0.0);
    vec3 wDir = mat3(gbufferModelViewInverse) * vDir;
    float wLen2 = dot(wDir, wDir);
    return (wLen2 > 1e-6) ? (wDir * inversesqrt(wLen2)) : vec3(0.0, 1.0, 0.0);
}

float directLightWeatherFade() {
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return 0.0;
#else
    float horizonFade = smoothstep(0.015, 0.12, abs(sunElevation()));
    float rainFade = 1.0 - 0.82 * clamp(rainStrength, 0.0, 1.0);
    return horizonFade * rainFade;
#endif
}

vec3 directCelestialIrradiance() {
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(0.0);
#else
    float up = sunElevation();
    float day = smoothstep(-0.02, 0.22, up);
    float uRatio = up / 0.14;
    float twilight = exp(-(uRatio * uRatio));
    /* Warm golden noon sun so grass/wood/stone look rich and warm, never chalky lime! */
    vec3 noonSun = vec3(2.42, 2.14, 1.76);
    vec3 duskSun = vec3(2.45, 1.12, 0.42);
    vec3 moonCol = vec3(0.15, 0.23, 0.40);
    vec3 col = mix(moonCol, noonSun, day);
    col = mix(col, duskSun, twilight * 0.78);
    return col * directLightWeatherFade();
#endif
}

vec3 skyAmbientIrradiance() {
#if defined(DIMENSION_NETHER)
    return max(srgbToLinear(fogColor) * 0.95, vec3(0.14, 0.05, 0.04));
#elif defined(DIMENSION_END)
    return vec3(0.16, 0.12, 0.24);
#else
    float up = sunElevation();
    float day = smoothstep(-0.10, 0.26, up);
    float uRatio = up / 0.16;
    float twilight = exp(-(uRatio * uRatio));
    /* Luminous cool sky-blue shadow fill so shadowed grass/dirt stays clean & readable,
       leaving deep 3D darkening specifically to RTAO crevices! */
    vec3 daySky = vec3(0.46, 0.64, 0.96);
    vec3 nightSky = vec3(0.032, 0.050, 0.095);
    vec3 duskSky = vec3(0.46, 0.30, 0.32);
    vec3 amb = mix(nightSky, daySky, day);
    amb = mix(amb, duskSky, twilight * 0.45);
    float rain = clamp(rainStrength, 0.0, 1.0);
    vec3 overcast = vec3(0.30, 0.34, 0.40) * mix(0.18, 1.0, day);
    return mix(amb, overcast, rain * 0.75);
#endif
}

vec3 hazeTintLinear() {
#if defined(DIMENSION_NETHER)
    float fLen = dot(fogColor, fogColor);
    return (fLen > 1e-4) ? srgbToLinear(fogColor) * 1.2 : vec3(0.14, 0.02, 0.015);
#elif defined(DIMENSION_END)
    return vec3(0.03, 0.018, 0.055);
#else
    float up = sunElevation();
    float day = smoothstep(-0.08, 0.26, up);
    float uRatio = up / 0.15;
    float twilight = exp(-(uRatio * uRatio));
    vec3 dayHaze = mix(vec3(0.42, 0.60, 0.92), srgbToLinear(clamp(fogColor, vec3(0.0), vec3(1.0))) * 1.25, 0.55);
    vec3 nightHaze = vec3(0.015, 0.025, 0.052);
    vec3 duskHaze = vec3(0.85, 0.34, 0.14);
    vec3 col = mix(nightHaze, dayHaze, day);
    col = mix(col, duskHaze, twilight * 0.55);
    float rain = clamp(rainStrength, 0.0, 1.0);
    vec3 rainHaze = vec3(0.20, 0.23, 0.28) * mix(0.15, 1.0, day);
    return mix(col, rainHaze, rain * 0.80);
#endif
}

vec3 shaftTintLinear() {
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(0.0);
#else
    float up = sunElevation();
    float day = smoothstep(-0.02, 0.24, up);
    float uRatio = up / 0.15;
    float twilight = exp(-(uRatio * uRatio));
    vec3 moonShaft = vec3(0.12, 0.20, 0.38);
    vec3 noonShaft = vec3(1.38, 1.12, 0.76);
    vec3 duskShaft = vec3(1.68, 0.62, 0.20);
    vec3 tint = mix(moonShaft, noonShaft, day);
    tint = mix(tint, duskShaft, twilight * 0.75);
    if (isEyeInWater == 1) tint = mix(tint, vec3(0.10, 0.42, 0.68), 0.80);
    return tint;
#endif
}

/* == Normalized Reduced-Extinction Haze (Leonard & Westermann, arXiv:2611.12103v1) == */

float adaptiveGroundLevel() {
    return clamp(cameraPosition.y - 2.0, -56.0, REF_BLOCK);
}

float analyticHaze(vec3 dir, float tHit, float skyHazeWeight) {
    if (skyHazeWeight <= 0.001) return 0.0;
    float clearRadius = mix(CLEAR_AIR_RADIUS, 8.0, clamp(rainStrength, 0.0, 1.0));
    float effHit = max(tHit - clearRadius, 0.0);
    if (effHit <= 0.01) return 0.0;

#if defined(DIMENSION_NETHER)
    float optHaze = 0.75 * effHit;
#elif defined(DIMENSION_END)
    float optHaze = 0.35 * effHit;
#else
    float groundY = adaptiveGroundLevel();
    float rainBoost = 1.0 + 1.10 * clamp(rainStrength, 0.0, 1.0);
    float hDelta = max(cameraPosition.y - groundY, 0.0) / HEIGHT_SCALE;
    float baseDens = clamp((1.0 / (1.0 + hDelta + 0.45 * hDelta * hDelta)) * rainBoost, 0.015, 1.10);
    float slopeAtten = 1.0 / (1.0 + max(dir.y, 0.0) * (0.5 * (tHit + clearRadius)) / HEIGHT_SCALE);
    float optHaze = max(baseDens * slopeAtten * effHit, 0.012 * effHit);
#endif
    float tau = optHaze * HAZE_COEFF * DENSITY_SCALE;
    float tauHat = tau / (1.0 + 0.82 * tau);
    float onset = smoothstep(0.0, 28.0, effHit);
    return min(tauHat * onset, FOG_CAP) * skyHazeWeight;
}

/* ======= 4.4x Sharp Distorted Shadow Map + Sub-Texel Hermite PCF ====== */

vec3 distortShadowNDC(vec3 ndc) {
    float r = sqrt(dot(ndc.xy, ndc.xy) + 0.0016);
    float factor = mix(1.0, r, SHADOW_DISTORT);
    return vec3(ndc.xy / factor, ndc.z * 0.5);
}

vec3 surfaceShadowColored(vec3 playerPos, vec3 worldNormal, bool isFoliage) {
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(1.0);
#else
    float dist2 = dot(playerPos, playerPos);
    if (dist2 > shadowDistance * shadowDistance) return vec3(1.0);

    float dist = sqrt(dist2);
    vec3 lightDir = worldShadowLightDir();
    vec3 effNormal = isFoliage ? vec3(0.0, 1.0, 0.0) : worldNormal;
    float ndotl = dot(effNormal, lightDir);
    float distScale = 1.0 + dist * 0.035;
    float normalScale = (ndotl < 0.0) ? 0.032 : (0.030 + 0.045 * (1.0 - ndotl));
    float sunOffset = isFoliage ? 0.20 : (0.025 * distScale);
    vec3 biasedPos = playerPos
                   + effNormal * (normalScale * distScale)
                   + lightDir * sunOffset;

    vec4 shadowClip = shadowProjection * (shadowModelView * vec4(biasedPos, 1.0));
    vec3 preNDC = shadowClip.xyz;
    float rawEdge = max(abs(preNDC.x), abs(preNDC.y));
    if (rawEdge > 0.98 || preNDC.z > 1.0 || preNDC.z < -1.0) return vec3(1.0);

    vec3 dNDC = distortShadowNDC(preNDC);
    vec2 suv = dNDC.xy * 0.5 + 0.5;
    float sampleZ = dNDC.z * 0.5 + 0.5 - SHADOW_BIAS * distScale;

    float resF = float(shadowMapResolution);
    float invRes = 1.0 / resF;
    vec2 texelCoord = suv * resF - 0.5;
    vec2 baseTex = floor(texelCoord);
    vec2 fracTex = clamp(texelCoord - baseTex, 0.0, 1.0);
    vec2 uv00 = (baseTex + 0.5) * invRes;

    float z00 = texture2D(shadowtex1, uv00).r;
    float z10 = texture2D(shadowtex1, uv00 + vec2(invRes, 0.0)).r;
    float z01 = texture2D(shadowtex1, uv00 + vec2(0.0, invRes)).r;
    float z11 = texture2D(shadowtex1, uv00 + vec2(invRes, invRes)).r;

    float s00 = step(sampleZ, z00);
    float s10 = step(sampleZ, z10);
    float s01 = step(sampleZ, z01);
    float s11 = step(sampleZ, z11);

    vec2 w = fracTex * fracTex * (3.0 - 2.0 * fracTex);
    float opaqueLit = mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);
    float edgeFade = smoothstep(0.82, 0.98, rawEdge);
    opaqueLit = mix(opaqueLit, 1.0, edgeFade);

    if (opaqueLit <= 0.004) return vec3(0.0);

#if COLORED_SHADOWS
    float zAll = texture2D(shadowtex0, suv).r;
    if (zAll < sampleZ) {
        vec4 sCol = texture2D(shadowcolor0, suv);
        if (sCol.a > 0.96) return vec3(0.0);
        vec3 glassLin = srgbToLinear(sCol.rgb);
        float gLum = dot(glassLin, vec3(0.2126, 0.7152, 0.0722));
        vec3 vivid = clamp(mix(vec3(gLum), glassLin, 1.35) * 1.45, vec3(0.0), vec3(1.2));
        vec3 transTint = mix(vec3(1.0), vivid * (1.0 - 0.45 * sCol.a), clamp(sCol.a * 1.35, 0.0, 0.96));
        return mix(transTint, vec3(1.0), edgeFade) * opaqueLit;
    }
#endif

    return vec3(opaqueLit);
#endif
}

/* ====== Grain-Free Distorted Optical-Depth Volumetric Shafts ====== */

float screenSunVisibility(vec3 sampleViewPos, vec3 lightDirView, float t) {
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return 0.0;
#else
    vec3 probeView = sampleViewPos + lightDirView * max(1.5, 0.22 * t);
    vec3 proj = fastProjectToScreen(probeView);
    if (proj.z < 0.0 || proj.x <= 0.0 || proj.x >= 1.0 || proj.y <= 0.0 || proj.y >= 1.0) return 1.0;
    float probeDepth = texture2D(depthtex1, proj.xy).r;
    if (probeDepth >= 0.9999) return 1.0;
    return clamp((-fastViewZ(probeDepth) - (-probeView.z)) * 0.25, 0.0, 1.0);
#endif
}

float marchShafts(vec3 dir, float tHit, int taps, out float contrast, out float subResidual) {
    contrast = 0.0;
    subResidual = 0.0;
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return 0.0;
#else
    if (RAYS_STRENGTH <= 0.001) return 0.0;
    float weatherFade = directLightWeatherFade();
    float lightValid;
    vec3 lightDirView = activeLightDirView(lightValid);
    if (weatherFade <= 0.001 || lightValid < 0.5) return 0.0;

    vec3 rayView = mat3(gbufferModelView) * dir;
    float rLen2 = dot(rayView, rayView);
    if (rLen2 > 1e-6) rayView *= inversesqrt(rLen2);

    float cosTheta = dot(rayView, lightDirView);
    float cone = clamp((cosTheta - 0.05) / 0.95, 0.0, 1.0);
    if (cone <= 0.001) return 0.0;

    float cone2 = cone * cone;
    float cone4 = cone2 * cone2;
    float miePhase = (cone4 * cone * 0.82 + cone2 * 0.22) * (0.35 + 0.85 * SUN_SCATTER);
    if (miePhase <= 0.003) return 0.0;

    float marchDist = min(tHit, shadowDistance);
    float span = RAY_END - RAY_START;
    float invTaps = 1.0 / float(taps);

    float warpAlpha = clamp(0.48 * marchDist / HEIGHT_SCALE, 0.0, 0.85);
    float groundY = adaptiveGroundLevel();
    float hCam = max(cameraPosition.y - groundY, 0.0) / HEIGHT_SCALE;
    float camDens = clamp(1.0 / (1.0 + hCam + 0.45 * hCam * hCam), 0.015, 1.05);
    float upSlope = max(dir.y, 0.0) / HEIGHT_SCALE;

#if SHADOWS
    vec4 sOrig4 = shadowProjection * (shadowModelView * vec4(0.0, 0.0, 0.0, 1.0));
    vec3 sOrigNDC = sOrig4.xyz;
    vec3 sDirNDC  = mat3(shadowProjection) * (mat3(shadowModelView) * dir);
#endif

    float optRays = 0.0;
    float optRaysSub = 0.0;
    float vMin = 1.0;
    float vMax = 0.0;
    float gPrev = 0.0;
    bool subToggle = true;

    for (int i = 0; i < MAX_TAPS; i++) {
        if (i >= taps) break;

        float uHi = float(i + 1) * invTaps;
        float gNext = uHi / (1.0 + warpAlpha * (1.0 - uHi));
        float gMid = 0.5 * (gPrev + gNext);
        float df = gNext - gPrev;
        gPrev = gNext;

        float f = RAY_START + span * gMid;
        float t = marchDist * f;
        float dt = (marchDist * span) * df;
        float dens = (camDens / (1.0 + upSlope * t)) * dt;

        float sunVis = 1.0;
#if SHADOWS
        vec3 preNDC = sOrigNDC + sDirNDC * t;
        float rawEdge = max(abs(preNDC.x), abs(preNDC.y));
        if (rawEdge < 0.99 && preNDC.z > -1.0 && preNDC.z < 1.0) {
            vec3 dNDC = distortShadowNDC(preNDC);
            vec3 sUVZ = dNDC * 0.5 + 0.5 - vec3(0.0, 0.0, SHADOW_BIAS);
            float lit = step(sUVZ.z, texture2D(shadowtex1, sUVZ.xy).r);
            sunVis = mix(lit, 1.0, smoothstep(0.86, 0.99, rawEdge));
        }
#else
        sunVis = screenSunVisibility(rayView * t, lightDirView, t);
#endif

        vMin = min(vMin, sunVis);
        vMax = max(vMax, sunVis);

        optRays += sunVis * dens;
        if (subToggle) {
            optRaysSub += sunVis * dens * 2.0;
        }
        if ((i - 2 * (i / 2)) == 0) subToggle = !subToggle;
    }

    contrast = vMax - vMin;
    float raysScale = miePhase * weatherFade;
    float tauAll = optRays * RAYS_COEFF * DENSITY_SCALE;
    float tauSub = optRaysSub * RAYS_COEFF * DENSITY_SCALE;
    float raysAll = min((tauAll / (1.0 + 0.65 * tauAll)) * raysScale, RAYS_CAP);
    float raysSub = min((tauSub / (1.0 + 0.65 * tauSub)) * raysScale, RAYS_CAP);

    subResidual = (contrast > 0.02) ? abs(raysAll - raysSub) : 0.0;
    return raysAll;
#endif
}

/* == Deterministic 4-Quadrant Omnidirectional RTGI & Deep Crevice RTAO ==
   Runs ONLY at 1/4 resolution (0.50 depth taps per screen pixel, ZERO noise!):
   Covers all 4 quadrants around the 3D surface normal at near (0.55m) + mid (2.6m)
   radii with finite-thickness pass-through (arXiv:2301.11376v2 & 2607.20384v2). */

vec3 evaluateHitBounceRadiance(vec2 hitUV) {
    vec3 hitAlbedo = srgbToLinear(texture2D(colortex0, hitUV).rgb);
    vec4 hitSurf = texture2D(colortex3, hitUV);
    float rawByte = floor(hitSurf.w * 255.0 + 0.5);
    if (rawByte < 21.0) return hitAlbedo * 0.8;

    float hitMat = floor((rawByte + 2.0) / 42.0);
    if (abs(hitMat - 5.0) < 0.5) {
        return hitAlbedo * 6.8;
    }

    float hitSky = clamp((rawByte - hitMat * 42.0) / 36.0, 0.0, 1.0);
    float hitBlock = hitSurf.z;
    float day = smoothstep(-0.05, 0.25, sunElevation());
    vec3 sunBounce = hitAlbedo * vec3(1.08, 1.00, 0.88) * (2.35 * day * hitSky * hitSky);
    vec3 torchBounce = hitAlbedo * vec3(1.0, 0.66, 0.34) * (hitBlock * hitBlock * 3.8);
    return min(sunBounce + torchBounce + hitAlbedo * 0.25 * hitSky, vec3(6.5));
}

vec4 traceOmniHemisphereRT(vec3 viewPos, vec3 viewNormal) {
#if !RT_GI
    return vec4(0.0, 0.0, 0.0, 1.0);
#else
    float viewDist = -viewPos.z;
    if (viewDist > 40.0) return vec4(0.0, 0.0, 0.0, 1.0);

    float distAtten = 1.0 - smoothstep(26.0, 40.0, viewDist);
    vec3 b1, b2;
    buildOrthonormalBasis(viewNormal, b1, b2);

    vec3 startPos = viewPos + viewNormal * max(0.05, viewDist * 0.005);
    float rNear = 0.52 + viewDist * 0.015;
    float rFar  = clamp(2.2 + viewDist * 0.07, 2.2, 5.2);

    /* 4 deterministic Golden-Quadrant unit vectors in tangent space (cosPhi, sinPhi):
       100% spatially invariant -> ZERO checkerboard, ZERO grain, 360-deg corner coverage! */
    vec2 dirs[4];
    dirs[0] = vec2( 0.92388,  0.38268);
    dirs[1] = vec2(-0.38268,  0.92388);
    dirs[2] = vec2(-0.92388, -0.38268);
    dirs[3] = vec2( 0.38268, -0.92388);

    float occSum = 0.0;
    vec3 giSum = vec3(0.0);

    for (int k = 0; k < 4; k++) {
        vec3 rayDir = (b1 * dirs[k].x + b2 * dirs[k].y) * 0.78 + viewNormal * 0.6258;
        float dirOcc = 0.0;
        vec2 hitUV = vec2(-1.0);
        float hitWeight = 0.0;

        /* Near radius: deep 90-deg block crevice AO + close bounce */
        vec3 pNear = startPos + rayDir * rNear;
        vec3 prjN = fastProjectToScreen(pNear);
        if (prjN.z > 0.0 && prjN.x > 0.003 && prjN.x < 0.997 && prjN.y > 0.003 && prjN.y < 0.997) {
            float sdN = texture2D(depthtex1, prjN.xy).r;
            if (sdN < 0.9999) {
                float penN = fastViewZ(sdN) - pNear.z;
                if (penN > 0.02 && penN < 0.95) {
                    dirOcc = 1.0 - smoothstep(0.55, 0.95, penN);
                    hitUV = prjN.xy;
                    hitWeight = 1.0;
                }
            }
        }

        /* Far radius: room-scale sunlit wall & torch GI bounce + canopy AO */
        vec3 pFar = startPos + rayDir * rFar;
        vec3 prjF = fastProjectToScreen(pFar);
        if (prjF.z > 0.0 && prjF.x > 0.003 && prjF.x < 0.997 && prjF.y > 0.003 && prjF.y < 0.997) {
            float sdF = texture2D(depthtex1, prjF.xy).r;
            if (sdF < 0.9999) {
                float penF = fastViewZ(sdF) - pFar.z;
                if (penF > 0.04 && penF < 2.2) {
                    float fOcc = (1.0 - smoothstep(1.3, 2.2, penF)) * 0.58;
                    dirOcc = max(dirOcc, fOcc);
                    if (hitWeight < 0.5) {
                        hitUV = prjF.xy;
                        hitWeight = 0.78;
                    }
                }
            }
        }

        occSum += dirOcc;
        if (hitWeight > 0.0) {
            giSum += evaluateHitBounceRadiance(hitUV) * hitWeight;
        }
    }

    /* Deep, sculpted crevice AO (down to ~0.34 in tight 3-wall corners!) */
    float aoVis = 1.0 - clamp((occSum * 0.25) * distAtten * 0.88, 0.0, 0.68);
    vec3 giAvg = (giSum * 0.25) * distAtten;
    return vec4(giAvg, aoVis);
#endif
}

/* Ultra-Fast 2-Tap Full-Res Micro-Crevice AO (ONLY 2 depth reads, 0 loops, 0 grain!):
   Carves crisp 1080p contact bevels where dirt meets grass, stairs, slabs & foliage bases! */
float microCreviceAO(vec2 uv, vec3 viewPos, vec3 viewNormal) {
    float viewDist = -viewPos.z;
    if (viewDist > 24.0) return 1.0;
    float fade = 1.0 - smoothstep(16.0, 24.0, viewDist);
    vec2 stepUV = vec2(3.0 / max(viewWidth, 1.0), 3.0 / max(viewHeight, 1.0));

    float dA = texture2D(depthtex1, clamp(uv + stepUV, 0.002, 0.998)).r;
    float dB = texture2D(depthtex1, clamp(uv - stepUV, 0.002, 0.998)).r;

    float occ = 0.0;
    if (dA < 0.9999) {
        vec3 deltaA = viewFromDepth(uv + stepUV, dA) - viewPos;
        float dist2A = dot(deltaA, deltaA);
        if (dist2A > 1e-5 && dist2A < 0.85) {
            occ += clamp(dot(viewNormal, deltaA * inversesqrt(dist2A)) - 0.12, 0.0, 0.85) * (1.0 - dist2A);
        }
    }
    if (dB < 0.9999) {
        vec3 deltaB = viewFromDepth(uv - stepUV, dB) - viewPos;
        float dist2B = dot(deltaB, deltaB);
        if (dist2B > 1e-5 && dist2B < 0.85) {
            occ += clamp(dot(viewNormal, deltaB * inversesqrt(dist2B)) - 0.12, 0.0, 0.85) * (1.0 - dist2B);
        }
    }
    return 1.0 - clamp(occ * 0.62 * fade, 0.0, 0.42);
}

/* ====== GGX Micro-Facet Specular Highlight for All Outdoor Surfaces ====== */

vec3 evaluateGGXSpecular(vec3 worldNormal, vec3 viewDirToSurface, vec3 lightDir,
                         float roughness, float f0)
{
    vec3 V = -viewDirToSurface;
    vec3 H = normalize(V + lightDir);
    float ndoth = max(dot(worldNormal, H), 0.0);
    float ndotl = max(dot(worldNormal, lightDir), 0.0);
    float vdoth = max(dot(V, H), 0.0);
    if (ndotl <= 0.001 || ndoth <= 0.001) return vec3(0.0);

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    float D = a2 / max(PI * denom * denom, 1e-4);

    float omc = 1.0 - vdoth;
    float omc2 = omc * omc;
    float F = f0 + (1.0 - f0) * (omc2 * omc2 * omc);
    return vec3(min(D * F * ndotl * 0.32, 1.6));
}

/* == Multiscale Volumetric Clouds, Night Aurora & End Accretion Vortex == */

float cloudHash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float cloudNoise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = cloudHash21(i);
    float b = cloudHash21(i + vec2(1.0, 0.0));
    float c = cloudHash21(i + vec2(0.0, 1.0));
    float d = cloudHash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float cloudFBM(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < 4; i++) {
        v += a * cloudNoise2D(p);
        p = rot * p * 2.05 + vec2(17.1, 31.7);
        a *= 0.5;
    }
    return v;
}

vec4 evaluateVolumetricSkyL0(vec3 dir) {
#if defined(DIMENSION_NETHER)
    return vec4(0.0, 0.0, 0.0, 1.0);
#elif defined(DIMENSION_END)
    float angle = atan(dir.z, dir.x);
    float elev = abs(dir.y);
    float swirl = sin(angle * 3.0 + elev * 8.5 - frameTimeCounter * 0.18) * 0.5 + 0.5;
    float fbm = cloudFBM(dir.xz / max(elev + 0.35, 0.35) * 1.8 + vec2(frameTimeCounter * 0.04, -frameTimeCounter * 0.03));
    float ring = exp(-pow((elev - 0.32) / 0.24, 2.0));
    vec3 vortexCol = mix(vec3(0.06, 0.02, 0.14), vec3(0.32, 0.10, 0.52), swirl * fbm);
    vec3 coreGlow = vec3(0.55, 0.24, 0.85) * (ring * fbm * 0.42);
    float starHash = cloudHash21(floor(dir.xz / max(abs(dir.y) + 0.25, 0.25) * 90.0));
    float stars = smoothstep(0.992, 0.999, starHash) * smoothstep(0.10, 0.45, elev);
    return vec4(vortexCol * 0.45 + coreGlow + vec3(0.75, 0.65, 1.0) * (stars * 0.85), 0.92);
#else
    vec3 skyAdd = vec3(0.0);
    float skyTrans = 1.0;

    float up = sunElevation();
    float nightMask = smoothstep(-0.04, -0.22, up) * (1.0 - smoothstep(0.25, 0.70, rainStrength));
    if (nightMask > 0.01 && dir.y > 0.05) {
        vec2 ap = dir.xz / (dir.y + 0.20);
        float curtain1 = sin(ap.x * 2.4 + sin(ap.y * 1.7 + frameTimeCounter * 0.08) * 1.5);
        float curtain2 = cos(ap.y * 2.1 - cos(ap.x * 1.4 - frameTimeCounter * 0.06) * 1.3);
        float wave = smoothstep(0.15, 0.95, 0.5 + 0.5 * curtain1 * curtain2);
        float elevFade = smoothstep(0.05, 0.26, dir.y) * (1.0 - smoothstep(0.45, 0.92, dir.y));
        vec3 auroraCol = mix(vec3(0.06, 0.65, 0.34), vec3(0.30, 0.12, 0.62), smoothstep(0.12, 0.55, dir.y));
        skyAdd += auroraCol * (wave * elevFade * nightMask * 0.48);
    }

#if VOLUMETRIC_CLOUDS
    if (dir.y > 0.025) {
        float baseH = max(192.0, cameraPosition.y + 42.0);
        float thickH = 48.0;
        float t0 = (baseH - cameraPosition.y) / dir.y;
        float t1 = (baseH + thickH - cameraPosition.y) / dir.y;
        if (t0 > 0.0 && t0 < 2400.0) {
            float coverage = mix(0.46, 0.78, clamp(rainStrength, 0.0, 1.0));
            vec2 wind = vec2(frameTimeCounter * 2.2, frameTimeCounter * 0.85);
            vec3 sunDir = worldShadowLightDir();
            float sunCos = max(dot(dir, sunDir), 0.0);
            float hgSilver = 0.35 + 1.35 * (sunCos * sunCos * sunCos * sunCos);

            float day = smoothstep(-0.08, 0.24, up);
            vec3 sunTint = directCelestialIrradiance();
            vec3 ambCloud = mix(vec3(0.04, 0.06, 0.12), vec3(0.44, 0.58, 0.84), day);
            vec3 cloudCol = vec3(0.0);

            for (int i = 0; i < 3; i++) {
                float frac = (float(i) + 0.5) / 3.0;
                float t = mix(t0, t1, frac);
                vec3 p = cameraPosition + dir * t;
                float densBase = cloudFBM((p.xz + wind) * 0.0038);
                float heightProfile = 1.0 - abs(frac * 2.0 - 1.0);
                float dens = smoothstep(1.0 - coverage, 1.0 - coverage + 0.28, densBase) * (0.45 + 0.55 * heightProfile);
                if (dens > 0.002) {
                    float sunDens = cloudFBM((p.xz + sunDir.xz * 22.0 + wind) * 0.0038);
                    float shadowMask = exp(-max(sunDens - (1.0 - coverage), 0.0) * 2.8);
                    vec3 stepLight = ambCloud * (0.48 + 0.35 * frac) + sunTint * (shadowMask * hgSilver * 0.55);
                    float stepAlpha = 1.0 - exp(-dens * 1.15);
                    cloudCol += skyTrans * stepLight * stepAlpha;
                    skyTrans *= (1.0 - stepAlpha);
                }
            }

            float horizonFade = smoothstep(0.025, 0.16, dir.y);
            skyAdd += cloudCol * horizonFade;
            skyTrans = mix(1.0, skyTrans, horizonFade);
        }
    }
#endif

    return vec4(skyAdd, skyTrans);
#endif
}

vec4 evaluateCloudEdgeRefine(vec3 dir) {
#if !VOLUMETRIC_CLOUDS || defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec4(0.0, 0.0, 0.0, 1.0);
#else
    if (dir.y <= 0.025) return vec4(0.0, 0.0, 0.0, 1.0);
    float baseH = max(192.0, cameraPosition.y + 42.0);
    float tMid = (baseH + 24.0 - cameraPosition.y) / dir.y;
    if (tMid <= 0.0 || tMid >= 2400.0) return vec4(0.0, 0.0, 0.0, 1.0);

    float coverage = mix(0.46, 0.78, clamp(rainStrength, 0.0, 1.0));
    vec2 wind = vec2(frameTimeCounter * 2.2, frameTimeCounter * 0.85);
    vec3 p = cameraPosition + dir * tMid;
    float dens = smoothstep(1.0 - coverage, 1.0 - coverage + 0.26, cloudFBM((p.xz + wind) * 0.0038));
    float alpha = (1.0 - exp(-dens * 1.65)) * smoothstep(0.025, 0.16, dir.y);

    vec3 sunDir = worldShadowLightDir();
    float sunCos = max(dot(dir, sunDir), 0.0);
    float hgSilver = 0.35 + 1.35 * (sunCos * sunCos * sunCos * sunCos);
    float day = smoothstep(-0.08, 0.24, sunElevation());
    vec3 ambCloud = mix(vec3(0.04, 0.06, 0.12), vec3(0.44, 0.58, 0.84), day);
    vec3 col = (ambCloud * 0.65 + directCelestialIrradiance() * (hgSilver * 0.42)) * alpha;
    return vec4(col, 1.0 - alpha);
#endif
}

/* ======= Procedural Water Caustics (Zero Texture Reads) ======= */

vec3 evaluateWaterCaustics(vec3 floorWorldPos, vec3 lightDir, float waterDepth, float sunLit) {
#if !WATER_CAUSTICS || defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(0.0);
#else
    if (sunLit <= 0.01 || waterDepth <= 0.04) return vec3(0.0);
    vec2 p = floorWorldPos.xz - floorWorldPos.y * (lightDir.xz / max(lightDir.y, 0.28));
    float t = frameTimeCounter * 1.45;
    float w1 = sin(p.x * 2.3 + p.y * 1.4 + t) * cos(p.y * 2.1 - p.x * 1.2 - t * 0.8);
    float w2 = sin(p.x * 1.6 - p.y * 2.5 - t * 1.1) * cos(p.y * 1.8 + p.x * 1.9 + t * 0.9);
    float c = clamp(1.0 - abs(w1 + w2), 0.0, 1.0);
    c = c * c * c;
    float depthMask = smoothstep(0.04, 0.55, waterDepth) / (1.0 + 0.10 * waterDepth);
    return directCelestialIrradiance() * vec3(0.55, 0.88, 1.05) * (c * depthMask * sunLit * 1.10);
#endif
}

/* ======= Coherent Speckle-Free 5+3 Binary Ray-Traced Reflections ======= */

vec3 waterWaveNormal(vec2 worldXZ) {
    float t = frameTimeCounter * 1.65;
    float w1 = sin(worldXZ.x * 1.9 + worldXZ.y * 1.1 + t) * 0.016;
    float w2 = cos(worldXZ.x * 1.2 - worldXZ.y * 2.3 - t * 1.3) * 0.012;
    return normalize(vec3(w1, 1.0, w2));
}

vec3 analyticSkyReflectionLinear(vec3 worldReflectDir) {
    float horizon = clamp(worldReflectDir.y * 2.0 + 0.12, 0.0, 1.0);
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return hazeTintLinear() * mix(0.7, 1.3, horizon);
#else
    float up = sunElevation();
    float day = smoothstep(-0.08, 0.26, up);
    vec3 zenith = mix(vec3(0.008, 0.016, 0.042), vec3(0.14, 0.34, 0.82), day);
    vec3 baseSky = mix(hazeTintLinear(), zenith, horizon);
#if VOLUMETRIC_CLOUDS
    if (worldReflectDir.y > 0.05) {
        float baseH = max(192.0, cameraPosition.y + 42.0);
        float tC = (baseH + 20.0 - cameraPosition.y) / worldReflectDir.y;
        if (tC > 0.0 && tC < 2200.0) {
            vec2 cXZ = (cameraPosition + worldReflectDir * tC).xz + vec2(frameTimeCounter * 2.2, frameTimeCounter * 0.85);
            float cov = mix(0.46, 0.78, clamp(rainStrength, 0.0, 1.0));
            float cDens = smoothstep(1.0 - cov, 1.0 - cov + 0.28, cloudNoise2D(cXZ * 0.0038)) * smoothstep(0.05, 0.22, worldReflectDir.y);
            vec3 cCol = mix(vec3(0.05, 0.08, 0.16), vec3(0.72, 0.78, 0.88), day);
            baseSky = mix(baseSky, cCol, cDens * 0.65);
        }
    }
#endif
    return baseSky;
#endif
}

vec4 traceReflectionRT(vec3 viewPos, vec3 worldNormal,
                       float f0, float smoothness, float skyAccess)
{
    vec3 viewNormal = normalize(mat3(gbufferModelView) * worldNormal);
    vec3 V = normalize(viewPos);
    vec3 R = reflect(V, viewNormal);

    float oneMinusCos = 1.0 - clamp(dot(-V, viewNormal), 0.001, 1.0);
    float omc2 = oneMinusCos * oneMinusCos;
    float fresnel = clamp(f0 + (1.0 - f0) * (omc2 * omc2 * oneMinusCos), 0.0, 1.0) * smoothness;

    vec3 worldR = normalize(mat3(gbufferModelViewInverse) * R);
    vec3 skyRefl = analyticSkyReflectionLinear(worldR) * skyAccess;

    float lightValid;
    vec3 lightDirView = activeLightDirView(lightValid);
    vec3 specGlint = vec3(0.0);
#if !defined(DIMENSION_NETHER) && !defined(DIMENSION_END)
    if (lightValid > 0.5 && skyAccess > 0.2) {
        float specDot = max(dot(R, lightDirView), 0.0);
        if (specDot > 0.85) {
            float s2 = specDot * specDot;
            float s8 = s2 * s2; s8 *= s8;
            float s64 = s8 * s8; s64 *= s64;
            specGlint = directCelestialIrradiance() * (s64 * s64 * 1.35 * smoothness * skyAccess);
        }
    }
#endif

    if (fresnel < 0.018) {
        return vec4(skyRefl + specGlint, fresnel);
    }

    float maxDist = min((far > 1.0) ? far : 64.0, 64.0);
    vec3 startPos = viewPos + viewNormal * max(0.07, -viewPos.z * 0.005);

    float tPrev = 0.0;
    float tHit = 0.0;
    vec2 hitUV = vec2(0.5);
    bool hit = false;

    for (int i = 0; i < 5; i++) {
        float f = float(i + 1) * 0.20;
        float tCurr = 0.30 + (maxDist - 0.30) * (f * f * f);
        vec3 probe = startPos + R * tCurr;
        if (probe.z >= -near) break;

        vec3 proj = fastProjectToScreen(probe);
        if (proj.z < 0.0 || proj.x <= 0.002 || proj.x >= 0.998 || proj.y <= 0.002 || proj.y >= 0.998) {
            break;
        }

        float sd = texture2D(depthtex1, proj.xy).r;
        if (sd < 0.9999) {
            float pen = fastViewZ(sd) - probe.z;
            if (pen > 0.0 && pen < 1.35 + 0.36 * tCurr) {
                tHit = tCurr;
                hitUV = proj.xy;
                hit = true;
                break;
            }
        }
        tPrev = tCurr;
    }

    if (!hit) {
        return vec4(skyRefl + specGlint, fresnel);
    }

    float tLo = tPrev;
    float tHi = tHit;
    for (int b = 0; b < 3; b++) {
        float tMid = 0.5 * (tLo + tHi);
        vec3 midPos = startPos + R * tMid;
        vec3 mproj = fastProjectToScreen(midPos);
        if (mproj.z >= 0.0 && mproj.x > 0.0 && mproj.x < 1.0 && mproj.y > 0.0 && mproj.y < 1.0) {
            float md = texture2D(depthtex1, mproj.xy).r;
            float mz = (md < 0.9999) ? fastViewZ(md) : -1e6;
            if (mz - midPos.z > 0.0) {
                tHi = tMid;
                hitUV = mproj.xy;
            } else {
                tLo = tMid;
            }
        }
    }

    float border = min(min(hitUV.x, 1.0 - hitUV.x), min(hitUV.y, 1.0 - hitUV.y));
    float edgeFade = smoothstep(0.0, 0.08, border);
    vec3 hitCol = texture2D(colortex0, hitUV).rgb;
    vec3 reflCol = mix(skyRefl, hitCol, edgeFade) + specGlint * (1.0 - 0.90 * edgeFade);
    return vec4(reflCol, fresnel);
}

/* ==================== Gating & Volumetric Application ================= */

float refineWeight(float residual) {
    return smoothstep(REFINE_THRESHOLD, REFINE_THRESHOLD + max(REFINE_SOFTNESS, 0.01), residual);
}

float relativeResidual(float residual, float levelMag) {
    float absR = abs(residual);
    float visFloor = smoothstep(0.005, 0.022, absR);
    return clamp((absR / (abs(levelMag) + 0.08)) * visFloor * 1.35, 0.0, 1.0);
}

void levelSetup(vec2 uv, out vec3 viewPos, out vec3 camRel, out vec3 dir,
                out float tHit, out vec2 skyWeights, out bool noGeometry)
{
    float d = texture2D(depthtex0, uv).r;
    viewPos = viewFromDepth(uv, d);
    camRel = mat3(gbufferModelViewInverse) * viewPos;
    float dist = max(length(camRel), 1e-4);
    dir = camRel / dist;

    float skyLen = (far > 1.0) ? min(far, SKY_DIST) : SKY_DIST;
    float groundY = adaptiveGroundLevel();
    float airRun = max(cameraPosition.y - groundY + 20.0, MIN_AIR_RUN);
    noGeometry = (d >= 0.9999);

    float skyHit = (dir.y < -0.05) ? min(skyLen, airRun) : skyLen;
    tHit = noGeometry ? skyHit : min(dist, (far > 1.0) ? far : skyLen);
    skyWeights = noGeometry ? vec2(SKY_HAZE_WEIGHT, SKY_RAYS_WEIGHT) : vec2(1.0, 1.0);
}

vec3 applyVolumetricsLinear(vec3 sceneHDR, float haze, float rays, float tHit, bool noGeometry) {
    vec3 color = sceneHDR;

    if (isEyeInWater == 2) {
        float lavaFog = noGeometry ? 0.98 : clamp((tHit * 0.45) / (1.0 + tHit * 0.45), 0.0, 0.98);
        return mix(color, vec3(2.2, 0.28, 0.015), lavaFog);
    }
    if (isEyeInWater == 3) {
        float snowFog = noGeometry ? 0.98 : clamp((tHit * 0.55) / (1.0 + tHit * 0.55), 0.0, 0.98);
        return mix(color, vec3(0.75, 0.88, 1.0), snowFog);
    }
    if (isEyeInWater == 1) {
        float wTau = tHit * 0.045 * DENSITY_SCALE;
        float waterFog = noGeometry ? 0.94 : clamp(wTau / (1.0 + 0.85 * wTau), 0.0, 0.94);
        vec3 deepCol = vec3(0.012, 0.085, 0.22) * mix(0.20, 1.0, smoothstep(-0.15, 0.25, sunElevation()));
        color = mix(color, deepCol, waterFog);
        color += rays * RAYS_STRENGTH * shaftTintLinear();
        return color;
    }

    if (!noGeometry) {
        float hazeFactor = clamp(haze * FOG_STRENGTH, 0.0, FOG_CAP);
        color = mix(color, hazeTintLinear(), hazeFactor);
    }

    color += rays * RAYS_STRENGTH * shaftTintLinear();

    if (blindness > 0.001) {
        float bTau = tHit * 0.35 * blindness;
        float bFog = clamp(bTau / (1.0 + bTau), 0.0, 1.0) * clamp(blindness, 0.0, 1.0);
        color = mix(color, vec3(0.0), bFog);
    }

    return color;
}

#endif
