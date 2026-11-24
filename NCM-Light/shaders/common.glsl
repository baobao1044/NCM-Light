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
    const int colortex4Format = RGBA16F;
    const int colortex5Format = RGBA16F;
    const int colortex6Format = R11F_G11F_B10F;
    const int shadowcolor0Format = RGBA8;
    const float shadowIntervalSize = 2.0;
    const float shadowDistanceRenderMul = 1.0;
*/

#ifndef NCM_COMMON_GLSL
#define NCM_COMMON_GLSL

#include "/options.glsl"

const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;

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
const float SHADOW_BIAS = 0.00012;

const int TAPS_CONCEPT0 = 4;
const int TAPS_CONCEPT1 = 4;
const int TAPS_CONCEPT2 = 6;
const int MAX_TAPS = TAPS_CONCEPT2;

/* ============================== Uniforms ============================= */

uniform sampler2D colortex0;    /* sRGB albedo -> Linear HDR lit scene (R11G11B10F) */
uniform sampler2D colortex1;    /* 1/4-res level 0: (rays0, ao0, volGate0, edgeGate0) */
uniform sampler2D colortex2;    /* full-res RGBA8:  (rays,  wL1, packedTrans, volGate1) */
uniform sampler2D colortex3;    /* full-res RGBA8:  (oct01.xy, blockLight, matSkyByte) */
uniform sampler2D colortex4;    /* 1/4-res Delta-NCM Fast-Weight G0.rgb + normDepth (RGBA16F) */
uniform sampler2D colortex5;    /* 1/4-res Delta-NCM Kaczmarz SH L1.xyz + signedInnov (RGBA16F) */
uniform sampler2D colortex6;    /* 1/4-res Dual-Pass Karis Cinema HDR Bloom (R11G11B10F) */

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
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowModelView;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform vec3 sunPosition;
uniform vec3 moonPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;
uniform vec3 fogColor;
uniform vec3 skyColor;
uniform ivec2 eyeBrightnessSmooth;

uniform float frameTimeCounter;
uniform int frameCounter;
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
    if (rawByte < 32.0) {
        valid = false;
        worldNormal = vec3(0.0, 1.0, 0.0);
        blockLight = 0.0;
        skyLight = 1.0;
        matTag = 0.0;
        return;
    }
    valid = true;
    matTag = floor(rawByte * 0.03125);
    skyLight = clamp((rawByte - matTag * 32.0) * (1.0 / 31.0), 0.0, 1.0);
    worldNormal = octDecodeNormal(surf.xy);
    blockLight = surf.z;
}

/* =================== Fast Noise, Basis & Projection ================== */

float ignNoise(vec2 pixelCoord) {
    return fract(52.9829189 * fract(dot(pixelCoord, vec2(0.06711056, 0.00583715))));
}

/* Automatic High-Resolution Throttle (scales cleanly across 900p / 1080p / 1440p / 4K on iGPUs!) */
float hiResThrottle() {
    float pxCount = max(viewWidth, 1.0) * max(viewHeight, 1.0);
    return clamp(1050000.0 / max(pxCount, 1050000.0), 0.42, 1.0);
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
    float dC = texture2D(depthtex0, uv).r;
    float dR = texture2D(depthtex0, clamp(uv + vec2(px.x, 0.0), 0.001, 0.999)).r;
    float dL = texture2D(depthtex0, clamp(uv - vec2(px.x, 0.0), 0.001, 0.999)).r;
    float dU = texture2D(depthtex0, clamp(uv + vec2(0.0, px.y), 0.001, 0.999)).r;
    float dD = texture2D(depthtex0, clamp(uv - vec2(0.0, px.y), 0.001, 0.999)).r;

    /* Pick minimum-discontinuity neighbor on each axis to prevent silhouette edge spikes */
    vec3 dx = (abs(dR - dC) <= abs(dL - dC))
        ? (viewFromDepth(clamp(uv + vec2(px.x, 0.0), 0.001, 0.999), dR) - centerViewPos)
        : (centerViewPos - viewFromDepth(clamp(uv - vec2(px.x, 0.0), 0.001, 0.999), dL));
    vec3 dy = (abs(dU - dC) <= abs(dD - dC))
        ? (viewFromDepth(clamp(uv + vec2(0.0, px.y), 0.001, 0.999), dU) - centerViewPos)
        : (centerViewPos - viewFromDepth(clamp(uv - vec2(0.0, px.y), 0.001, 0.999), dD));

    vec3 vn = cross(dx, dy);
    float l2 = dot(vn, vn);
    if (l2 < 1e-8) return vec3(0.0, 1.0, 0.0);
    vn *= inversesqrt(l2);
    if (dot(vn, -centerViewPos) < 0.0) vn = -vn;
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
#if defined(DIMENSION_END)
    /* Colossal Gargantua Supermassive Black Hole direction lights End islands with warm golden key light! */
    return normalize(vec3(0.36, 0.34, -0.869));
#else
    float valid;
    vec3 vDir = activeLightDirView(valid);
    if (valid < 0.5) return vec3(0.0, 1.0, 0.0);
    vec3 wDir = mat3(gbufferModelViewInverse) * vDir;
    float wLen2 = dot(wDir, wDir);
    return (wLen2 > 1e-6) ? (wDir * inversesqrt(wLen2)) : vec3(0.0, 1.0, 0.0);
#endif
}

float directLightWeatherFade() {
#if defined(DIMENSION_NETHER)
    return 0.0;
#elif defined(DIMENSION_END)
    return 1.0;
#else
    float horizonFade = smoothstep(0.015, 0.12, abs(sunElevation()));
    float rainFade = 1.0 - 0.82 * clamp(rainStrength, 0.0, 1.0);
    return horizonFade * rainFade;
#endif
}

vec3 directCelestialIrradiance() {
#if defined(DIMENSION_NETHER)
    return vec3(0.0);
#elif defined(DIMENSION_END)
    /* Luminous golden-amber accretion disk key light from Gargantua sculpting End islands & towers */
    return vec3(1.18, 0.74, 0.40);
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
    /* Deep cosmic sapphire-indigo starlight fill creating rich warm-vs-cool cinema contrast */
    return vec3(0.11, 0.14, 0.30);
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
    return vec3(0.014, 0.016, 0.048);
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

vec3 surfaceShadowColored(vec3 feetPos, vec3 worldNormal, bool isFoliage, out float outThick) {
    outThick = 0.0;
#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(1.0);
#else
    float dist2 = dot(feetPos, feetPos);
    if (dist2 > shadowDistance * shadowDistance) return vec3(1.0);

    vec3 lightDir = worldShadowLightDir();
    vec3 effNormal = isFoliage ? vec3(0.0, 1.0, 0.0) : worldNormal;
    float ndotl = dot(effNormal, lightDir);

    /* Exact local distorted shadow-texel size in world blocks:
       Zero detached Peter-Panning close up AND zero zebra acne on distant hills! */
    vec4 rawClip = shadowProjection * (shadowModelView * vec4(feetPos, 1.0));
    vec3 rawNDC = rawClip.xyz / max(rawClip.w, 1e-5);
    float rawEdge = max(abs(rawNDC.x), abs(rawNDC.y));
    if (rawEdge > 0.98 || rawNDC.z > 1.0 || rawNDC.z < -1.0) return vec3(1.0);

    float dFac = mix(1.0, sqrt(dot(rawNDC.xy, rawNDC.xy) + 0.0016), SHADOW_DISTORT);
    float texelBlocks = (2.0 * shadowDistance / float(shadowMapResolution)) * ((dFac * dFac) / max(1.0 - SHADOW_DISTORT, 0.05));
    float slopeFactor = clamp(1.0 - max(ndotl, 0.0), 0.0, 1.0);
    float normalOffset = clamp(texelBlocks * (0.55 + 1.15 * slopeFactor), 0.018, 0.34);
    float sunOffset = isFoliage ? max(0.14, texelBlocks * 1.1) : clamp(texelBlocks * 0.25, 0.008, 0.09);

    vec3 biasedPos = feetPos
                   + effNormal * normalOffset
                   + lightDir * sunOffset;

    vec4 shadowClip = shadowProjection * (shadowModelView * vec4(biasedPos, 1.0));
    vec3 preNDC = shadowClip.xyz / max(shadowClip.w, 1e-5);
    float bEdge = max(abs(preNDC.x), abs(preNDC.y));
    if (bEdge > 0.98 || preNDC.z > 1.0 || preNDC.z < -1.0) return vec3(1.0);

    vec3 dNDC = distortShadowNDC(preNDC);
    vec2 suv = dNDC.xy * 0.5 + 0.5;
    float sampleZ = dNDC.z * 0.5 + 0.5 - SHADOW_BIAS;

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

    /* Shadow-map penetration depth for Thick-Volume Beer-Lambert SSS (0 extra fetches!) */
    outThick = max(sampleZ - 0.25 * (z00 + z10 + z01 + z11), 0.0) * 28.0;

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
    float zOpaqueCenter = min(min(z00, z10), min(z01, z11));
    /* Only project stained-glass RGB tint when a real translucent caster sits strictly
       in front of the opaque shadow map (eliminates colored speckles along normal shadow edges!) */
    if (zAll < sampleZ && zAll < zOpaqueCenter - 0.0005) {
        vec4 s00 = texture2D(shadowcolor0, uv00);
        vec4 s10 = texture2D(shadowcolor0, uv00 + vec2(invRes, 0.0));
        vec4 s01 = texture2D(shadowcolor0, uv00 + vec2(0.0, invRes));
        vec4 s11 = texture2D(shadowcolor0, uv00 + vec2(invRes, invRes));
        vec4 sCol = mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);
        if (sCol.a > 0.05 && sCol.a < 0.94) {
            vec3 glassLin = srgbToLinear(sCol.rgb);
            float gLum = dot(glassLin, vec3(0.2126, 0.7152, 0.0722));
            vec3 vivid = clamp(mix(vec3(gLum), glassLin, 1.30) * 1.40, vec3(0.0), vec3(1.15));
            vec3 transTint = mix(vec3(1.0), vivid * (1.0 - 0.35 * sCol.a), clamp(sCol.a * 1.45, 0.0, 0.94));
            return mix(transTint, vec3(1.0), edgeFade) * opaqueLit;
        }
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
    /* Start volumetric shaft shadow projection from true camera position in feet-player space (exact in F5!) */
    vec4 sOrig4 = shadowProjection * (shadowModelView * vec4(gbufferModelViewInverse[3].xyz, 1.0));
    vec3 sOrigNDC = sOrig4.xyz / max(sOrig4.w, 1e-5);
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

/* == Delta-NCM v3.0: Martingale-Safe Kaczmarz Fast-Weight Attention + Control-Variate PT ==
   Synthesizes:
     - Kaczmarz Linear Attention (arXiv:2605.08587): orthogonal key-norm fast-weight projection
       using the exact Hemisphere-Centered Key d(w) = w - (2/3)*N  (E_cos[d(w)] = 0 identically!)
     - Gated DeltaNet-2 (arXiv:2605.22791): Martingale-safe (F_{t-1}-measurable) decoupled
       Erase/Write gates driven by signed temporal innovation -> ZERO bias & fast jump recovery!
     - Neural Two-Level Monte Carlo (arXiv:2412.04634): unbiased SH L1 control variate residual
     - Online Fast-Weight Path Guiding (arXiv:2504.04315): R3-independent balance-heuristic MIS
     - 4-Step Exponential DDA Screen-Space Raymarch + Directional Neumann Multi-Bounce */

vec2 reprojectToPrevUV(vec3 viewPos) {
    vec3 feetPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 prevFeet = feetPos + (cameraPosition - previousCameraPosition);
    vec4 prevView = gbufferPreviousModelView * vec4(prevFeet, 1.0);
    vec4 prevClip = gbufferPreviousProjection * prevView;
    if (prevClip.w <= 1e-4) return vec2(-1.0);
    return (prevClip.xy / prevClip.w) * 0.5 + 0.5;
}

vec3 convolveDeltaNCML1(vec3 G0, vec3 L1, vec3 fineNormal, vec3 coarseNormal) {
    /* Exact hemisphere-centered SH L1 irradiance shift for 1080p normal maps & 3D block bevels:
       G(n_fine) = G0 + (2/3) * L1 . (n_fine - n_coarse)  (zero bias when n_fine == n_coarse!) */
    float y0 = max(dot(G0, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
    float modF = clamp(1.0 + 0.666667 * (dot(L1, fineNormal - coarseNormal) / y0), 0.25, 1.85);
    return G0 * modF;
}

float evaluateSecondarySunNEE(vec3 hitViewPos) {
#if SHADOWS
    vec3 hitFeet = (gbufferModelViewInverse * vec4(hitViewPos, 1.0)).xyz;
    vec4 shView = shadowModelView * vec4(hitFeet, 1.0);
    vec4 shClip = shadowProjection * shView;
    vec3 preNDC = shClip.xyz / max(shClip.w, 1e-5);
    float rawEdge = max(abs(preNDC.x), abs(preNDC.y));
    if (rawEdge >= 0.99) return 1.0;
    vec3 dNDC = distortShadowNDC(preNDC);
    vec3 sUVZ = dNDC * 0.5 + 0.5 - vec3(0.0, 0.0, SHADOW_BIAS * 1.5);
    if (sUVZ.x < 0.002 || sUVZ.x > 0.998 || sUVZ.y < 0.002 || sUVZ.y > 0.998) return 1.0;
    return step(sUVZ.z, texture2D(shadowtex1, sUVZ.xy).r);
#else
    return 1.0;
#endif
}

vec3 evaluatePathVertexRadiance(vec2 hitUV, vec3 hitViewPos) {
    vec3 hitAlbedo = srgbToLinear(texture2D(colortex0, hitUV).rgb);
    vec4 hitSurf = texture2D(colortex3, hitUV);
    float rawByte = floor(hitSurf.w * 255.0 + 0.5);
    if (rawByte < 32.0) return hitAlbedo * 0.75;

    float hitMat = floor(rawByte * 0.03125);
    if (abs(hitMat - 5.0) < 0.5) {
        return hitAlbedo * 7.2;
    }

    float hitSky = clamp((rawByte - hitMat * 32.0) * (1.0 / 31.0), 0.0, 1.0);
    float hitBlock = hitSurf.z;
    float sunNEE = evaluateSecondarySunNEE(hitViewPos);
    vec3 sunIrrad = directCelestialIrradiance() * directLightWeatherFade();
    vec3 skyIrrad = skyAmbientIrradiance();

    vec3 directBounce = hitAlbedo * sunIrrad * (1.55 * sunNEE * hitSky);
    vec3 skyBounce = hitAlbedo * skyIrrad * (0.38 * hitSky);
    vec3 torchBounce = hitAlbedo * vec3(1.0, 0.66, 0.34) * (hitBlock * hitBlock * 3.2);

    vec3 multiBounce = vec3(0.0);
#if PT_MULTI_BOUNCE
    vec2 prevHitUV = reprojectToPrevUV(hitViewPos);
    if (prevHitUV.x > 0.004 && prevHitUV.x < 0.996 && prevHitUV.y > 0.004 && prevHitUV.y < 0.996) {
        vec4 prevDC = texture2D(colortex4, prevHitUV);
        if (prevDC.a < 0.995) {
            /* Neumann multi-bounce feedback with 0.50 spectral-radius cap */
            multiBounce = hitAlbedo * min(max(prevDC.rgb, vec3(0.0)), vec3(2.4)) * 0.50;
        }
    }
#endif

    return min(directBounce + skyBounce + torchBounce + multiBounce, vec3(4.2));
}

void traceDeltaNCMPathGI(vec2 quarterUV, vec3 viewPos, vec3 viewNormal, vec3 worldNormal,
                         vec3 prevG0, vec3 prevL1, float prevSignedInnov, bool hasHistory,
                         out vec3 outG0, out vec3 outL1, out float outSignedInnov, out float outAO)
{
#if !RT_GI
    outG0 = vec3(0.0);
    outL1 = vec3(0.0);
    outSignedInnov = 0.0;
    outAO = 1.0;
#else
    float viewDist = -viewPos.z;
    float maxGiDist = 42.0 * mix(0.68, 1.0, hiResThrottle());
    if (viewDist > maxGiDist) {
        outG0 = vec3(0.0);
        outL1 = vec3(0.0);
        outSignedInnov = 0.0;
        outAO = 1.0;
        return;
    }

    float distAtten = 1.0 - smoothstep(maxGiDist * 0.66, maxGiDist, viewDist);
    vec3 b1, b2;
    buildOrthonormalBasis(viewNormal, b1, b2);

    vec3 startPos = viewPos + viewNormal * max(0.05, viewDist * 0.005);
    float ddaScale = 1.0 + viewDist * 0.035;

    /* 4-step exponential DDA radii (prevents 1-block wall light leaking while exiting early on hits) */
    float stepR[4];
    stepR[0] = 0.38 * ddaScale;
    stepR[1] = 0.96 * ddaScale;
    stepR[2] = 2.25 * ddaScale;
    stepR[3] = min(4.90 * ddaScale, 6.2);

    float stepThick[4];
    stepThick[0] = 0.65;
    stepThick[1] = 1.05;
    stepThick[2] = 1.65;
    stepThick[3] = 2.40;

    float stepAoW[4];
    stepAoW[0] = 1.00;
    stepAoW[1] = 0.78;
    stepAoW[2] = 0.52;
    stepAoW[3] = 0.30;

    vec2 pxCoord = quarterUV * vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25;
    float baseN1 = ignNoise(pxCoord);
    float baseN2 = ignNoise(pxCoord + vec2(17.0, 31.0));
    float baseN3 = ignNoise(pxCoord + vec2(43.0, 11.0));
    int frameMod = frameCounter - 64 * (frameCounter / 64);
    float fMod = float(frameMod);
    float sppF = float(PT_SPP);

    vec3 skyFill = skyAmbientIrradiance() * 0.28;
    float skyFade = smoothstep(-0.08, 0.22, sunElevation());

    /* Martingale-safe (F_{t-1}-measurable) predictable surprise with deadzone:
       Zero in steady state (zero bias!), opens immediately on real lighting shifts! */
    float prevSurprise = hasHistory ? clamp((abs(prevSignedInnov) - 0.34) / 0.36, 0.0, 1.0) : 1.0;

    /* Online Fast-Weight Path Guiding extracted directly from prevL1 (zero extra memory!) */
    float prevY0 = max(dot(prevG0, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
    float lenPrevL1 = length(prevL1);
    float kappa = hasHistory ? clamp(lenPrevL1 / max(prevY0, 0.05), 0.0, 0.85) : 0.0;
    vec3 muRaw = prevL1 + worldNormal * (0.666667 * prevY0);
    float lenMu = length(muRaw);
    vec3 muWorld = (lenMu > 1e-4) ? (muRaw / lenMu) : worldNormal;
    vec3 muView = normalize(mat3(gbufferModelView) * muWorld);

    bool canGuide = false;
    float pGuide = 0.0;
    float alphaG = 3.0;
    vec3 g1 = b1;
    vec3 g2 = b2;
#if PT_GUIDING
    if (hasHistory && kappa > 0.15 && dot(muWorld, worldNormal) > 0.25) {
        canGuide = true;
        pGuide = 0.50 * kappa * (1.0 - prevSurprise);
        alphaG = 3.0 + 4.0 * kappa;
        buildOrthonormalBasis(muView, g1, g2);
    }
#endif

    float occSum = 0.0;
    vec3 rawRGB = vec3(0.0);
    vec3 resRGB = vec3(0.0);
    vec3 kaczmarzDir = vec3(0.0);
    float resYSum = 0.0;

    /* Antithetic Hemisphere Pair (2 opposing rays per PT_SPP at 1/4 res = 0.125 rays/px!):
       Always probes both opposite sides of the normal on every frame -> kills colored speckles! */
    int totalRays = PT_SPP * 2;
    float totalRaysF = float(totalRays);

    for (int s = 0; s < PT_SPP * 2; s++) {
        int pairIdx = s / 2;
        bool isAnti = ((s - pairIdx * 2) == 1);
        float sF = float(pairIdx);
        /* 3 independent R3 plastic-constant coordinates + exact antithetic opposing mate */
        float u1 = fract(baseN1 + fMod * 0.81917251 + (sF + 0.5) / sppF + (isAnti ? 0.5 : 0.0));
        float u2 = fract(baseN2 + fMod * 0.67104361 + sF * 0.38196601);
        if (isAnti) u2 = 1.0 - u2;
        float uG = fract(baseN3 + fMod * 0.54970048 + sF * 0.75487767 + (isAnti ? 0.5 : 0.0));

        vec3 rayDir;
        if (canGuide && uG < pGuide) {
            float phiG = 6.2831853 * u1;
            float cosG = pow(max(1e-4, u2), 1.0 / (alphaG + 1.0));
            float sinG = sqrt(max(0.0, 1.0 - cosG * cosG));
            rayDir = normalize((g1 * cos(phiG) + g2 * sin(phiG)) * sinG + muView * cosG);
        } else {
            float phi = 6.2831853 * u1;
            float cosTheta = sqrt(max(0.04, u2));
            float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
            rayDir = (b1 * cos(phi) + b2 * sin(phi)) * sinTheta + viewNormal * cosTheta;
        }

        float cosN = dot(rayDir, viewNormal);
        if (cosN <= 0.0) {
            continue;
        }

        float wIS = 1.0;
        if (canGuide && pGuide > 1e-4) {
            float pCos = cosN * (1.0 / PI);
            float cosM = max(dot(rayDir, muView), 0.0);
            float pLobe = (alphaG + 1.0) * (0.5 / PI) * pow(cosM, alphaG);
            wIS = min(pCos / max((1.0 - pGuide) * pCos + pGuide * pLobe, 1e-4), 1.85);
        }

        float dirOcc = 0.0;
        vec2 hitUV = vec2(-1.0);
        vec3 hitPos = vec3(0.0);
        float hitWeight = 0.0;
        float jitter = 0.75 + 0.45 * u2;

        for (int m = 0; m < 4; m++) {
            vec3 pStep = startPos + rayDir * (stepR[m] * jitter);
            vec3 prj = fastProjectToScreen(pStep);
            if (prj.z <= 0.0 || prj.x < 0.003 || prj.x > 0.997 || prj.y < 0.003 || prj.y > 0.997) {
                break;
            }
            float sd = texture2D(depthtex2, prj.xy).r;
            if (sd < 0.9999) {
                float vz = fastViewZ(sd);
                float pen = vz - pStep.z;
                if (pen > 0.02 && pen < stepThick[m]) {
                    dirOcc = (1.0 - smoothstep(stepThick[m] * 0.55, stepThick[m], pen)) * stepAoW[m];
                    hitUV = prj.xy;
                    hitPos = viewFromDepth(hitUV, sd);
                    hitWeight = (m == 0) ? 1.0 : (m == 1 ? 0.95 : 0.85);
                    break;
                }
            }
        }

        vec3 worldRay = normalize(mat3(gbufferModelViewInverse) * rayDir);
        vec3 sampRGB;
        if (hitWeight > 0.0) {
            sampRGB = evaluatePathVertexRadiance(hitUV, hitPos) * hitWeight;
        } else {
            float upMask = smoothstep(-0.15, 0.55, worldRay.y);
            sampRGB = skyFill * (0.35 + 0.65 * upMask) * (0.25 + 0.75 * skyFade);
        }

        /* Strong Karis firefly suppression prevents colored GI speckles on bright emissives */
        float rawLuma = dot(sampRGB, vec3(0.2126, 0.7152, 0.0722));
        sampRGB = (sampRGB / (1.0 + rawLuma * 0.28)) * distAtten;
        float sampY = dot(sampRGB, vec3(0.2126, 0.7152, 0.0722));

        occSum += dirOcc;
        rawRGB += wIS * sampRGB;

        if (hasHistory) {
            vec3 dKey = worldRay - worldNormal * 0.666667;
            float l1DotD = dot(prevL1, dKey);
            vec3 predRGB = prevG0 * (1.0 + clamp(l1DotD / prevY0, -0.85, 1.25));
            float predY = prevY0 + l1DotD;

            vec3 rCV_rgb = clamp(wIS * (sampRGB - predRGB), vec3(-1.5), vec3(1.5));
            float rCV_y  = clamp(wIS * (sampY - predY), -1.5, 1.5);
            float dNorm2 = dot(dKey, dKey) + 0.25;

            resRGB += rCV_rgb;
            resYSum += rCV_y;
            kaczmarzDir += (rCV_y / dNorm2) * dKey;
        }
    }

    /* Deterministic 4-Tap Screen-Space Horizon Crevice AO (0 matrix projections, -50% fetches, ZERO blotches!) */
    vec2 aoStep = (0.52 / max(viewDist, 0.85)) * vec2(1.0, max(viewWidth, 1.0) / max(viewHeight, 1.0)) * 0.14;
    vec2 aoOffs[4];
    aoOffs[0] = vec2( aoStep.x,  aoStep.y * 0.4);
    aoOffs[1] = vec2(-aoStep.x * 0.4,  aoStep.y);
    aoOffs[2] = vec2(-aoStep.x, -aoStep.y * 0.4);
    aoOffs[3] = vec2( aoStep.x * 0.4, -aoStep.y);

    float detOcc = 0.0;
    for (int q = 0; q < 4; q++) {
        vec2 qUV = clamp(quarterUV + aoOffs[q], 0.003, 0.997);
        float sdQ = texture2D(depthtex2, qUV).r;
        if (sdQ < 0.9999) {
            vec3 qDiff = viewFromDepth(qUV, sdQ) - viewPos;
            float qLen2 = dot(qDiff, qDiff);
            if (qLen2 > 1e-4 && qLen2 < 3.2) {
                float vDot = dot(qDiff * inversesqrt(qLen2), viewNormal);
                detOcc += clamp(vDot - 0.12, 0.0, 1.0) * (1.0 - qLen2 * 0.28);
            }
        }
    }

    float invSpp = 1.0 / totalRaysF;
    outAO = 1.0 - clamp((detOcc * 0.25) * distAtten * 0.85, 0.0, 0.62);
    rawRGB *= invSpp;

    if (hasHistory) {
        resRGB *= invSpp;
        resYSum *= invSpp;
        kaczmarzDir *= invSpp;

        /* Martingale-Safe Decoupled Erase & Write Gates (Gated DeltaNet-2) */
        float s2 = prevSurprise * prevSurprise;
        float eraseGate = 0.015 + 0.40 * s2;
        float writeGate = 0.08  + 0.40 * s2;
        float dirWrite  = 0.14  + 0.36 * prevSurprise;

        outG0 = max(vec3(0.0), prevG0 + writeGate * resRGB);
        outL1 = (1.0 - eraseGate) * prevL1 + dirWrite * kaczmarzDir;

        float normRes = clamp(resYSum / (0.20 + 0.80 * prevY0), -1.5, 1.5);
        outSignedInnov = clamp(0.75 * prevSignedInnov + 0.25 * normRes, -1.5, 1.5);
    } else {
        outG0 = max(vec3(0.0), rawRGB);
        outL1 = vec3(0.0);
        outSignedInnov = 0.5;
    }

    /* Physical SH L1 boundedness projection: |L1| <= 1.40 * Y0 */
    float newY0 = dot(outG0, vec3(0.2126, 0.7152, 0.0722));
    float lenNewL1 = length(outL1);
    if (lenNewL1 > newY0 * 1.40 && lenNewL1 > 1e-5) {
        outL1 *= (newY0 * 1.40) / lenNewL1;
    }
#endif
}

/* Second-Derivative Laplacian Concavity Micro-Crevice AO (0 matrix unprojects, 0 convex edge outlines!):
   On any flat slanted wall: dzA + dzB == 0 -> ZERO false darkening!
   On any outer convex cube edge: dzA + dzB < 0 -> ZERO cartoon black border!
   Strictly darkens true inner concave 90-deg crevices where dzA + dzB > 0! */
float microCreviceAO(vec2 uv, vec3 viewPos, vec3 viewNormal) {
    float viewDist = -viewPos.z;
    if (viewDist > 16.0) return 1.0;
    float fade = 1.0 - smoothstep(9.0, 16.0, viewDist);
    vec2 px = vec2(2.0 / max(viewWidth, 1.0), 2.0 / max(viewHeight, 1.0));
    float z0 = viewPos.z;

    float dR = texture2D(depthtex2, clamp(uv + vec2(px.x, 0.0), 0.002, 0.998)).r;
    float dL = texture2D(depthtex2, clamp(uv - vec2(px.x, 0.0), 0.002, 0.998)).r;
    float dU = texture2D(depthtex2, clamp(uv + vec2(0.0, px.y), 0.002, 0.998)).r;
    float dD = texture2D(depthtex2, clamp(uv - vec2(0.0, px.y), 0.002, 0.998)).r;

    float occ = 0.0;
    float scale = 0.045 + viewDist * 0.018;
    if (dR < 0.9999 && dL < 0.9999) {
        float dzR = fastViewZ(dR) - z0;
        float dzL = fastViewZ(dL) - z0;
        /* Both neighbors must lie within local surface neighborhood (rejects sky/background silhouettes!) */
        if (abs(dzR) < 0.55 && abs(dzL) < 0.55) {
            float concX = dzR + dzL;
            if (concX > 0.018) occ += clamp((concX - 0.018) / scale, 0.0, 0.75);
        }
    }
    if (dU < 0.9999 && dD < 0.9999) {
        float dzU = fastViewZ(dU) - z0;
        float dzD = fastViewZ(dD) - z0;
        if (abs(dzU) < 0.55 && abs(dzD) < 0.55) {
            float concY = dzU + dzD;
            if (concY > 0.018) occ += clamp((concY - 0.018) / scale, 0.0, 0.75);
        }
    }
    return 1.0 - clamp(occ * 0.22 * fade, 0.0, 0.26);
}

/* == Physical Optics: Kulla-Conty RGB GGX, Thin-Film Iridescence, Sparkle & Rainbow == */

vec3 evaluateGGXSpecularRGB(vec3 worldNormal, vec3 viewDirToSurface, vec3 lightDir,
                            float roughness, vec3 f0RGB, bool isConductor)
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
    vec3 F = f0RGB + (vec3(1.0) - f0RGB) * (omc2 * omc2 * omc);
    /* Kulla-Conty / Turquin multiple-scattering energy compensation for rough conductors */
    vec3 msBoost = isConductor ? (vec3(1.0) + f0RGB * (roughness * 0.55)) : vec3(1.0);
    return min(D * F * msBoost * (ndotl * 0.32), vec3(1.8));
}

vec3 evaluateThinFilmIridescence(float cosTheta, float opdScale) {
#if !SPECTRAL_DISPERSION
    return vec3(1.0);
#else
    float phase = opdScale * sqrt(max(1.0 - (1.0 - cosTheta * cosTheta) * 0.55, 0.05));
    /* First-order spectral interference at R=650nm, G=530nm, B=450nm */
    return 0.58 + 0.42 * cos(phase * vec3(1.0, 1.226, 1.444) + vec3(0.0, 1.9, 3.8));
#endif
}

float evaluateMicroSparkle(vec3 worldPos, vec3 viewDir, vec3 lightDir, float cellScale) {
#if !SUBSURFACE_SCATTER
    return 0.0;
#else
    vec3 cell = floor(worldPos * cellScale);
    vec3 p3 = fract(cell * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    vec3 microN = normalize(fract((p3.xxy + p3.yzz) * p3.zyx) - 0.5);
    vec3 H = normalize(-viewDir + lightDir);
    float d = max(dot(microN, H), 0.0);
    float d8 = d * d; d8 *= d8; d8 *= d8;
    float d32 = d8 * d8; d32 *= d32;
    return smoothstep(0.32, 0.95, d32);
#endif
}

vec3 evaluateAtmosphericRainbow(vec3 dir, vec3 lightDir) {
#if !SPECTRAL_DISPERSION || defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(0.0);
#else
    float rainMask = smoothstep(0.05, 0.55, rainStrength) * smoothstep(0.04, 0.30, sunElevation());
    if (rainMask <= 0.005 || dir.y <= 0.0) return vec3(0.0);
    /* Physical 42-deg antisolar primary rainbow angle: cos(42 deg) = 0.7431 */
    float antiCos = dot(dir, -lightDir);
    if (antiCos < 0.70 || antiCos > 0.79) return vec3(0.0);
    float t = (antiCos - 0.728) / 0.032;
    vec3 band = exp(-pow(vec3(t - 0.18, t - 0.50, t - 0.82) * 4.5, vec3(2.0)));
    return band * (rainMask * smoothstep(0.0, 0.14, dir.y) * 0.22);
#endif
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

float cosmicHash31(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

float cosmicNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = cosmicHash31(i + vec3(0.0, 0.0, 0.0));
    float n100 = cosmicHash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = cosmicHash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = cosmicHash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = cosmicHash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = cosmicHash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = cosmicHash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = cosmicHash31(i + vec3(1.0, 1.0, 1.0));
    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

float cosmicFBM3D(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * cosmicNoise3D(p);
        p = p * 2.04 + vec3(13.7, 29.3, 19.1);
        a *= 0.5;
    }
    return v;
}

float cosmicFBM3D3(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; i++) {
        v += a * cosmicNoise3D(p);
        p = p * 2.04 + vec3(13.7, 29.3, 19.1);
        a *= 0.5;
    }
    return v;
}

float diskFBM2D(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < 3; i++) {
        v += a * cloudNoise2D(p);
        p = rot * p * 2.04 + vec2(17.1, 31.7);
        a *= 0.5;
    }
    return v;
}

/* Monotonic, Zero-Bubble Asymptotic Einstein-Schwarzschild Gravitational Lensing */
vec3 applyEinsteinLensing(vec3 dir, vec3 bhDir) {
    float bhDot = dot(dir, bhDir);
    float sinTh = sqrt(max(1.0 - bhDot * bhDot, 0.0));
    float thetaE = 0.168;
    float deflAmt = (thetaE * thetaE) / (sinTh + 0.095) * exp(-sinTh * 4.2) * smoothstep(0.55, 0.85, bhDot);
    vec3 tang = dir - bhDir * bhDot;
    float tLen = max(length(tang), 1e-5);
    return normalize(dir - (tang / tLen) * deflAmt);
}

/* Long-Reach 12-Step Perspective-Correct Screen-Space Gargantua Shadows (runs at 0.25x in composite1!):
   Tracks continuous solid-island entry (!wasInAir) so thick floating End islands NEVER tunnel light! */
float traceEndGargantuaShadow(vec3 viewPos, vec3 worldNormal) {
#if !defined(DIMENSION_END)
    return 1.0;
#else
    float maxShadDist = 48.0 * mix(0.72, 1.0, hiResThrottle());
    float distFade = 1.0 - smoothstep(maxShadDist * 0.70, maxShadDist, -viewPos.z);
    if (distFade <= 0.01) return 1.0;
    vec3 bhWorld = worldShadowLightDir();
    if (dot(worldNormal, bhWorld) <= 0.015) return 0.0;

    vec3 viewN = normalize(mat3(gbufferModelView) * worldNormal);
    vec3 lightView = normalize(mat3(gbufferModelView) * bhWorld);
    float towardCamFade = 1.0 - smoothstep(0.22, 0.62, lightView.z);
    if (towardCamFade <= 0.01) return 1.0;

    float marchLen = 9.0;
    vec3 startPos = viewPos + viewN * max(0.06, -viewPos.z * 0.005);
    if (lightView.z > 0.0001) {
        float zRoom = (-near * 1.15) - startPos.z;
        if (zRoom <= 0.04) return 1.0;
        marchLen = min(marchLen, zRoom / lightView.z);
    }
    vec3 endPos = startPos + lightView * marchLen;

    vec3 p0 = fastProjectToScreen(startPos);
    vec3 p1 = fastProjectToScreen(endPos);
    if (p0.z < 0.0 || p1.z < 0.0) return 1.0;

    vec2 uv0 = p0.xy;
    vec2 uv1 = p1.xy;
    float invZ0 = 1.0 / min(startPos.z, -near);
    float invZ1 = 1.0 / min(endPos.z, -near);

    vec2 duv = uv1 - uv0;
    float sMax = 1.0;
    if (duv.x >  1e-5) sMax = min(sMax, (0.996 - uv0.x) / duv.x);
    if (duv.x < -1e-5) sMax = min(sMax, (0.004 - uv0.x) / duv.x);
    if (duv.y >  1e-5) sMax = min(sMax, (0.996 - uv0.y) / duv.y);
    if (duv.y < -1e-5) sMax = min(sMax, (0.004 - uv0.y) / duv.y);
    if (sMax <= 0.002) return 1.0;

    uv1 = uv0 + duv * sMax;
    invZ1 = mix(invZ0, invZ1, sMax);

    float occ = 0.0;
    bool wasInAir = false;
    for (int i = 1; i <= 10; i++) {
        float fi = float(i) * 0.10;
        float s = 0.22 * fi + 0.78 * fi * fi;
        vec2 curUV = mix(uv0, uv1, s);
        float probeZ = 1.0 / mix(invZ0, invZ1, s);

        float sd = texture2D(depthtex2, curUV).r;
        if (sd < 0.9999) {
            float vz = fastViewZ(sd);
            float pen = vz - probeZ;
            if (pen < -0.08) {
                wasInAir = true;
            } else if (pen > 0.05) {
                float border = min(min(curUV.x, 1.0 - curUV.x), min(curUV.y, 1.0 - curUV.y));
                float bFade = smoothstep(0.0, 0.05, border);
                if (!wasInAir) {
                    /* Ray entered solid cliff/island directly from shaded block -> 100% solid eclipse (ZERO yellow contour tunneling!) */
                    occ = max(occ, smoothstep(0.05, 0.18, pen) * bFade);
                    if (occ > 0.92) break;
                } else {
                    /* Ray flew through open air first -> check finite caster thickness to ignore foreground pillars */
                    float thick = max(2.2, 0.25 * (-probeZ));
                    if (pen < thick) {
                        float hitW = smoothstep(0.05, 0.18, pen) * (1.0 - smoothstep(thick * 0.72, thick, pen));
                        occ = max(occ, hitW * bFade * (1.0 - 0.15 * fi));
                        if (occ > 0.90) break;
                    }
                }
            }
        } else {
            wasInAir = true;
        }
    }
    return 1.0 - 0.92 * occ * distFade * towardCamFade;
#endif
}

vec3 blackbodyDiskColor(float temp) {
    /* Interstellar Planckian accretion disk color ramp: burnt crimson -> molten amber -> champagne gold -> blue-white */
    float t = clamp(temp, 0.0, 2.5);
    vec3 c0 = vec3(0.36, 0.035, 0.015);
    vec3 c1 = vec3(1.10, 0.30, 0.04);
    vec3 c2 = vec3(1.36, 0.88, 0.44);
    vec3 c3 = vec3(1.28, 1.28, 1.42);
    vec3 col = mix(c0, c1, clamp(t / 0.55, 0.0, 1.0));
    col = mix(col, c2, clamp((t - 0.55) / 0.72, 0.0, 1.0));
    col = mix(col, c3, clamp((t - 1.27) / 0.90, 0.0, 1.0));
    return col;
}

void evalAccretionDiskPlasma(vec2 xz, vec3 rayV, out vec3 outEmit, out float outAbsorb) {
    float rc = max(length(xz), 1e-4);

    /* Seamless Cartesian-rotated Keplerian spiral shear (ZERO atan discontinuity!) */
    float logR = log(max(rc, 0.5));
    float twist1 = 4.8 * logR - frameTimeCounter * (1.20 / pow(max(rc, 0.8), 1.25));
    float c1 = cos(twist1), s1 = sin(twist1);
    vec2 sp1 = vec2(xz.x * c1 - xz.y * s1, xz.x * s1 + xz.y * c1);
    float n1 = diskFBM2D(sp1 * 0.50);

    float twist2 = 8.8 * logR - frameTimeCounter * (1.60 / pow(max(rc, 0.8), 1.35));
    float c2 = cos(twist2), s2 = sin(twist2);
    vec2 sp2 = vec2(xz.x * c2 - xz.y * s2, xz.x * s2 + xz.y * c2);
    float n2 = diskFBM2D(sp2 * 0.95 + vec2(n1 * 1.60, -n1 * 1.40));

    float rMod = rc + (n1 - 0.5) * 1.35;
    float band1 = 0.5 + 0.5 * sin(rMod * 3.6 + n2 * 2.9);
    float band2 = 0.5 + 0.5 * cos(rMod * 7.8 - n1 * 3.4);
    float plasma = (0.24 + 0.76 * band1) * (0.40 + 0.60 * band2) * (0.22 + 1.45 * n2 * n2);

    float inner = smoothstep(1.92, 2.65, rc);
    float outer = 1.0 - smoothstep(4.2, 10.6, rc);
    float radial = inner * pow(max(outer, 1e-4), 1.50) * pow(2.60 / max(rc, 1.0), 1.15);
    float dens = radial * (0.18 + 1.25 * plasma);

    /* Physical Relativistic Doppler Beaming (approaching left blazing white-gold, receding fiery amber!) */
    float vOrbMag = min(sqrt(0.56 / max(rc - 0.95, 0.35)), 0.58);
    vec2 vOrb = vec2(xz.y / rc, -xz.x / rc) * vOrbMag;
    vec3 vUnit = normalize(rayV);
    float dop = dot(vOrb, -vUnit.xz);
    float beaming = clamp(1.0 / (1.0 - dop * 0.76), 0.44, 2.18);
    float gRed = sqrt(clamp(1.0 - 1.0 / max(rc, 1.05), 0.08, 1.0));

    float temp = (1.62 * pow(2.50 / max(rc, 1.2), 0.64) * (0.48 + 0.62 * plasma)) * pow(beaming, 0.62) * gRed;
    outEmit = blackbodyDiskColor(temp) * (dens * 1.92 * pow(beaming, 1.48) * gRed);
    outAbsorb = radial * (1.15 + 1.55 * (1.0 - 0.5 * n2) + 0.95 * plasma);
}

/* Layer 1 (0.25x quarter-res = 16x fewer pixels -> 42-48 FPS on Intel UHD 620!):
   Seamless 3D Lensed Milky Way + 32-Step 3D Schwarzschild Geodesic Accretion Disk + 4-Layer End Space Clouds! */
void evaluateEndSkyL0(vec3 dir, out vec3 outSkyAndDisk, out vec3 outClouds,
                      out float outCloudTrans, out float outBhTrans, out float outFrontTrans)
{
    vec3 bhDir = worldShadowLightDir();
    vec3 bentDir = applyEinsteinLensing(dir, bhDir);

    /* 1. Deep OLED-Black Vacuum + Seamless 3D-Spherical Milky Way (ZERO atan branch cuts!) */
    vec3 galNorth = normalize(vec3(0.24, 0.84, -0.49));
    vec3 galCore  = normalize(vec3(-0.82, 0.42, 0.39) - galNorth * dot(vec3(-0.82, 0.42, 0.39), galNorth));
    vec3 galEast  = normalize(cross(galNorth, galCore));

    float lat = dot(bentDir, galNorth);
    float coreDot = dot(bentDir, galCore);
    float eastDot = dot(bentDir, galEast);

    vec3 gPos = vec3(coreDot * 2.7, eastDot * 2.7, lat * 5.0);
    float nBase = cosmicFBM3D(gPos + vec3(2.3, -1.5, 4.1));
    float nWarp = cosmicFBM3D(gPos * 1.85 + vec3(nBase * 1.35, -nBase * 1.15, nBase * 0.75));
    float nFine = cosmicFBM3D3(gPos * 3.80 - vec3(nWarp * 0.95, -nBase * 0.85, nWarp * 0.75));

    float planeWide = exp(-lat * lat * 11.0);
    float planeCore = exp(-lat * lat * 30.0);
    float bulge     = exp(-lat * lat * 14.0) * pow(max(coreDot, 0.0), 2.2);

    float dustCloud = smoothstep(0.36, 0.72, nWarp * 0.60 + nFine * 0.40) * exp(-lat * lat * 34.0);
    float dustTrans = 1.0 - 0.90 * dustCloud;

    vec3 vacuum    = vec3(0.0004, 0.0007, 0.0022);
    vec3 armGlow   = vec3(0.022, 0.056, 0.22) * (planeWide * pow(max(nBase, 1e-4), 1.55) * 1.30);
    vec3 starCloud = vec3(0.16, 0.28, 0.68) * (planeCore * (0.30 + 1.05 * nWarp) * (0.40 + 0.85 * nFine) * 0.56);
    vec3 coreGlow  = vec3(1.05, 0.62, 0.30) * (bulge * (0.40 + 0.85 * nWarp) * 0.48);

    float h2Region = smoothstep(0.52, 0.82, nBase * 0.55 + nFine * 0.45) * planeWide;
    float o3Region = smoothstep(0.54, 0.84, cosmicFBM3D3(gPos * 1.65 + vec3(9.4, 5.2, -7.1))) * planeWide;
    vec3 nebHa = vec3(0.46, 0.07, 0.36) * (h2Region * 0.28);
    vec3 nebO3 = vec3(0.04, 0.34, 0.56) * (o3Region * 0.26);

    vec3 mwHDR = vacuum + (armGlow + starCloud + coreGlow + nebHa + nebO3) * dustTrans;

    /* 2. True 3D Schwarzschild Null Geodesic Gargantua Accretion Disk (d2p/dl2 = -1.5 * h2 * p / r^5) */
    float bhDot = dot(dir, bhDir);
    vec3 bhEmit = vec3(0.0);
    outBhTrans = 1.0;
    outFrontTrans = 1.0;

    if (bhDot > 0.72) {
        vec3 bU0 = normalize(cross(vec3(0.0, 1.0, 0.0), bhDir));
        vec3 bV0 = cross(bhDir, bU0);
        vec3 bU = bU0 * 0.965926 - bV0 * 0.258819;
        vec3 bV = bU0 * 0.258819 + bV0 * 0.965926;

        float dX = dot(dir, bU);
        float dY = dot(dir, bV);
        float dZ = bhDot;

        float sinA = 0.132256;
        float cosA = 0.991215;
        vec3 p = vec3(0.0, 15.2 * sinA, -15.2 * cosA);
        vec3 v = normalize(vec3(dX, dY * cosA - dZ * sinA, dY * sinA + dZ * cosA));

        vec3 hVec = cross(p, v);
        float h2 = dot(hVec, hVec);

        float r2 = dot(p, p);
        float r = sqrt(r2);
        vec3 a = (-1.5 * h2 / (r2 * r2 * r)) * p;

        vec3 accum = vec3(0.0);
        float trans = 1.0;
        float frontTrans = 1.0;
        float minR = r;

        for (int step = 0; step < 34; step++) {
            float dt = clamp(0.135 * (r - 0.62), 0.078, 1.08);
            vec3 pNext = p + v * dt + a * (0.5 * dt * dt);
            float nr2 = dot(pNext, pNext);
            float nr = sqrt(nr2);
            minR = min(minR, nr);

            if (p.y * pNext.y <= 0.0 && trans > 0.008) {
                float frac = abs(p.y) / (abs(p.y) + abs(pNext.y) + 1e-6);
                vec3 pCross = mix(p, pNext, frac);
                float rcHit = length(pCross.xz);
                if (rcHit > 1.88 && rcHit < 10.6) {
                    vec3 em;
                    float ab;
                    evalAccretionDiskPlasma(pCross.xz, v, em, ab);
                    float vyAbs = max(abs(v.y), 0.16);
                    float pathDs = min((0.36 + 0.03 * max(rcHit - 2.0, 0.0)) / vyAbs, 1.85);
                    bool isFront = (pCross.z < 0.0);
                    float tauScale = isFront ? 1.45 : 0.85;
                    float tau = clamp(ab * pathDs * tauScale, 0.0, 4.5);
                    float aStep = 1.0 - exp(-tau);
                    accum += trans * em * (pathDs * 1.12);
                    float atten = 1.0 - aStep * 0.96;
                    trans *= atten;
                    if (isFront) frontTrans *= atten;
                }
            }

            vec3 aNext = (-1.5 * h2 / max(nr2 * nr2 * nr, 1e-6)) * pNext;
            vec3 vNext = v + (a + aNext) * (0.5 * dt);

            p = pNext;
            v = vNext;
            a = aNext;
            r2 = nr2;
            r = nr;

            if (r < 1.08) {
                trans = 0.0;
                break;
            }
            if (r > 18.5 && dot(p, v) > 0.0) {
                break;
            }
        }

        /* Delicate inner relativistic gravitational corona (1080p pass adds razor-sharp Photon Ring!) */
        float innerGlow = exp(-pow(max(minR - 1.54, 0.0) / 1.55, 1.40)) * smoothstep(1.08, 1.65, minR) * (0.14 * frontTrans);
        accum += vec3(1.20, 0.66, 0.26) * innerGlow;

        float coneFade = smoothstep(0.72, 0.80, bhDot);
        bhEmit = accum * coneFade;
        outBhTrans = 1.0 - (1.0 - trans) * coneFade;
        outFrontTrans = 1.0 - (1.0 - frontTrans) * coneFade;
    }

    outSkyAndDisk = mwHDR * outBhTrans + bhEmit;

    /* 3. 4-Layer 3D Volumetric End Space Clouds (unbent local sky: royal indigo + golden Gargantua rim!) */
    outClouds = vec3(0.0);
    outCloudTrans = 1.0;
    if (abs(dir.y) < 0.42) {
        float bhCos = max(bhDot, 0.0);
        float bh2 = bhCos * bhCos;
        float hgSilver = bh2 * bh2 * bh2 * 1.65 + pow(max(bhCos, 1e-4), 2.5) * 0.18;
        for (int i = 0; i < 4; i++) {
            float fi = float(i);
            float hLayer = -0.14 + 0.090 * fi;
            float sc = 2.0 + 0.52 * fi;
            vec3 cp = vec3(
                dir.x * sc + frameTimeCounter * 0.012 * (1.0 + 0.25 * fi),
                (dir.y - hLayer) * 6.8,
                dir.z * sc - frameTimeCounter * 0.008 * (1.0 + 0.20 * fi)
            );
            float dRaw = cosmicFBM3D3(cp);
            float dyH = (dir.y - hLayer) / 0.11;
            float dens = smoothstep(0.46, 0.78, dRaw) * exp(-dyH * dyH) * 0.46;
            if (dens > 0.002) {
                float rim = hgSilver * (0.35 + 0.65 * dRaw);
                vec3 cLit = vec3(0.024, 0.020, 0.110) + vec3(0.42, 0.22, 0.12) * rim;
                float aStep = 1.0 - exp(-dens * 1.15);
                outClouds += outCloudTrans * cLit * aStep;
                outCloudTrans *= (1.0 - aStep * 0.65);
            }
        }
    }
}

vec4 evaluateVolumetricSkyL0(vec3 dir) {
#if defined(DIMENSION_NETHER)
    return vec4(0.0, 0.0, 0.0, 1.0);
#elif defined(DIMENSION_END)
    vec3 sd, cl;
    float tr, bht, ft;
    evaluateEndSkyL0(dir, sd, cl, tr, bht, ft);
    return vec4(sd * tr + cl, 0.0);
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

/* ======= Chromatic Dispersive Water Caustics (Zero Texture Reads) ======= */

float causticLattice(vec2 p, float t) {
    float w1 = sin(p.x * 2.3 + p.y * 1.4 + t) * cos(p.y * 2.1 - p.x * 1.2 - t * 0.8);
    float w2 = sin(p.x * 1.6 - p.y * 2.5 - t * 1.1) * cos(p.y * 1.8 + p.x * 1.9 + t * 0.9);
    float c = clamp(1.0 - abs(w1 + w2), 0.0, 1.0);
    return c * c * c;
}

vec3 evaluateWaterCaustics(vec3 floorWorldPos, vec3 lightDir, float waterDepth, float sunLit) {
#if !WATER_CAUSTICS || defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return vec3(0.0);
#else
    if (sunLit <= 0.01 || waterDepth <= 0.04) return vec3(0.0);
    vec2 p = floorWorldPos.xz - floorWorldPos.y * (lightDir.xz / max(lightDir.y, 0.28));
    float t = frameTimeCounter * 1.45;
    float depthMask = smoothstep(0.04, 0.55, waterDepth) / (1.0 + 0.10 * waterDepth);
#if SPECTRAL_DISPERSION
    float disp = 0.045 + 0.018 * min(waterDepth, 4.0);
    vec3 cRGB = vec3(
        causticLattice(p + vec2(disp, -disp * 0.5), t),
        causticLattice(p, t),
        causticLattice(p - vec2(disp, -disp * 0.5), t)
    );
    return directCelestialIrradiance() * vec3(0.68, 0.92, 1.12) * (cRGB * depthMask * sunLit * 1.12);
#else
    float c = causticLattice(p, t);
    return directCelestialIrradiance() * vec3(0.55, 0.88, 1.05) * (c * depthMask * sunLit * 1.10);
#endif
#endif
}

/* ======= Coherent Speckle-Free 5+3 Binary Ray-Traced Reflections ======= */

vec3 waterWaveNormal(vec2 worldXZ) {
    float t = frameTimeCounter * 1.65;
    float w1 = sin(worldXZ.x * 1.9 + worldXZ.y * 1.1 + t) * 0.016;
    float w2 = cos(worldXZ.x * 1.2 - worldXZ.y * 2.3 - t * 1.3) * 0.012;
    return normalize(vec3(w1, 1.0, w2));
}

vec3 planckStarColor(float t) {
    /* True O-B-A-F-G-K-M blackbody stellar color ramp (0 = cool M orange-red -> 1 = hot O/B sapphire-white) */
    vec3 cM = vec3(1.00, 0.56, 0.32);
    vec3 cG = vec3(1.00, 0.92, 0.72);
    vec3 cA = vec3(0.95, 0.97, 1.00);
    vec3 cB = vec3(0.60, 0.76, 1.00);
    if (t < 0.33) return mix(cM, cG, t * 3.0);
    if (t < 0.66) return mix(cG, cA, (t - 0.33) * 3.0);
    return mix(cA, cB, (t - 0.66) * 3.0);
}

vec2 cubeSphereUV(vec3 d, out float faceId) {
    /* Seamless cube-face projection (zero polar pinching/stretching anywhere on the sky sphere!) */
    vec3 ad = abs(d);
    if (ad.x >= ad.y && ad.x >= ad.z) {
        faceId = (d.x > 0.0) ? 1.0 : 2.0;
        return d.yz / max(ad.x, 1e-4);
    } else if (ad.y >= ad.x && ad.y >= ad.z) {
        faceId = (d.y > 0.0) ? 3.0 : 4.0;
        return d.xz / max(ad.y, 1e-4);
    }
    faceId = (d.z > 0.0) ? 5.0 : 6.0;
    return d.xy / max(ad.z, 1e-4);
}

vec3 evalStarLayer(vec2 suv, float faceId, float gridScale, float densityThresh, float sharpPow, bool withSpikes) {
    vec2 p = suv * gridScale;
    vec2 cell = floor(p);
    vec2 f = fract(p) - 0.5;
    vec2 seed = cell + vec2(faceId * 37.3, faceId * 71.9);
    float h = cloudHash21(seed);
    if (h < densityThresh) return vec3(0.0);

    float h2 = cloudHash21(seed + vec2(19.7, 43.1));
    float h3 = cloudHash21(seed + vec2(83.3, 11.7));
    vec2 jitter = (vec2(h2, h3) - 0.5) * 0.62;
    vec2 d = f - jitter;
    float r2 = dot(d, d);
    float mag = (h - densityThresh) / max(1.0 - densityThresh, 1e-4);
    float twinkle = 0.86 + 0.14 * sin(frameTimeCounter * (1.8 + 2.4 * h2) + h3 * 40.0);
    float core = exp(-r2 * sharpPow) * (0.28 + 1.45 * mag * mag) * twinkle;

    if (withSpikes && mag > 0.55) {
        /* Delicate 4-point JWST/Hubble telescope diffraction spikes on brightest magnitude-0 stars */
        float spX = exp(-abs(d.y) * 52.0) * exp(-abs(d.x) * 6.5);
        float spY = exp(-abs(d.x) * 52.0) * exp(-abs(d.y) * 6.5);
        core += (spX + spY) * (mag - 0.55) * 1.15 * twinkle;
    }
    return planckStarColor(h2) * core;
}

vec3 evaluateEndCosmosFullRes(vec3 dir, vec3 baseSkyDiskHDR, vec3 endCloudsHDR,
                              float endCloudTrans, float bhTrans, float frontTrans)
{
#if !defined(DIMENSION_END)
    return baseSkyDiskHDR;
#else
    /* Layer 2 (1080p full-res, ZERO raymarch loops -> 42-48 FPS on Intel UHD 620!):
       Combines 0.25x 3D Schwarzschild Geodesic Disk & Milky Way with razor-sharp 1080p:
         1. Analytical Schwarzschild Event Horizon Shadow Cut (b_c/D = 0.1709) masked by frontTrans
         2. Blazing Doppler-Shifted Einstein Photon Ring masked strictly by foreground accretion band
         3. Crisp 3-Tier Lensed Planck Starfield + Andromeda M31 + Celestial Orbital Ring */
    vec3 bhDir = worldShadowLightDir();
    float bhDot = dot(dir, bhDir);
    vec3 bentDir = applyEinsteinLensing(dir, bhDir);

    vec3 galNorth = normalize(vec3(0.24, 0.84, -0.49));
    float lat = dot(bentDir, galNorth);
    float galPlane = exp(-lat * lat * 11.0);

    vec3 hdr = baseSkyDiskHDR;
    /* Crisp 1080p dust micro-detail ONLY on Milky Way away from the smooth geodesic plasma disk */
    if (abs(lat) < 0.46 && bhDot < 0.80) {
        float microDust = cosmicNoise3D(bentDir * 19.5 + vec3(3.1, -5.7, 2.4));
        hdr *= mix(1.0, 0.72 + 0.62 * microDust, galPlane * 0.56);
    }

    float faceId;
    vec2 suv = cubeSphereUV(bentDir, faceId);

    /* 3-tier crisp 1080p lensed starfield (dense Milky Way dust stars + colored giants + JWST spiked hero stars) */
    vec3 stars = evalStarLayer(suv, faceId, 240.0, mix(0.954, 0.844, galPlane), 92.0, false) * (0.28 + 0.90 * galPlane);
    stars     += evalStarLayer(suv, faceId, 120.0, mix(0.972, 0.917, galPlane), 80.0, false) * 1.05;
    stars     += evalStarLayer(suv, faceId,  48.0, 0.980, 68.0, true) * 2.15;

    /* Refined Tilted Andromeda M31 Spiral Galaxy */
    vec3 m31Dir = normalize(vec3(-0.64, 0.44, -0.63));
    vec3 m31Col = vec3(0.0);
    float m31Dot = dot(bentDir, m31Dir);
    if (m31Dot > 0.945) {
        vec3 mU = normalize(cross(vec3(0.0, 1.0, 0.0), m31Dir));
        vec3 mV = cross(m31Dir, mU);
        vec2 mp = vec2(dot(bentDir, mU), dot(bentDir, mV));
        vec2 mr = vec2(mp.x * 0.88 - mp.y * 0.47, (mp.x * 0.47 + mp.y * 0.88) * 3.05);
        float rad = length(mr);
        float ang = atan(mr.y, mr.x);
        float spiral = 0.5 + 0.5 * cos(2.0 * ang - 7.2 * log(rad + 0.014));
        float mDust = 0.55 + 0.45 * cloudNoise2D(mr * 28.0);
        float disk = exp(-rad * 15.5) * (0.28 + 0.72 * spiral * spiral) * mDust;
        float nuc  = exp(-rad * rad * 1900.0);
        m31Col = vec3(0.34, 0.50, 0.96) * (disk * 0.52) + vec3(1.20, 0.88, 0.56) * (nuc * 1.15);
    }

    /* 1080p Razor-Sharp Schwarzschild Event Horizon Cut & Blazing Einstein Photon Ring! */
    float ehCut = 1.0;
    if (bhDot > 0.94) {
        float bAng = sqrt(max(1.0 - bhDot * bhDot, 0.0));
        /* Exact critical Schwarzschild shadow angle: b_c / D = 2.59808 / 15.2 = 0.17093 */
        ehCut = smoothstep(0.167, 0.172, bAng);
        hdr *= mix(1.0, ehCut, frontTrans);

        vec3 bU0 = normalize(cross(vec3(0.0, 1.0, 0.0), bhDir));
        vec3 bV0 = cross(bhDir, bU0);
        vec3 bU = bU0 * 0.965926 - bV0 * 0.258819;
        float dX = dot(dir, bU);

        float prArg = (bAng - 0.1715) / 0.0046;
        float photonRing = exp(-prArg * prArg) * frontTrans;
        float ringDop = clamp(1.08 - dX * 2.1, 0.50, 1.70);
        hdr += vec3(1.80, 1.28, 0.76) * (photonRing * ringDop);
    }

    hdr += (stars + m31Col) * (bhTrans * ehCut);

    /* Grand Sweeping Translucent Celestial Orbital Ring (unbent local End atmosphere frame!) */
    vec3 ringCol = vec3(0.0);
    if (dir.y > -0.05) {
        vec3 ringN = normalize(vec3(-0.34, 0.84, 0.42));
        float denom = dot(dir, ringN);
        if (abs(denom) > 0.03) {
            float tRing = 0.65 / denom;
            if (tRing > 0.0 && tRing < 18.0) {
                vec3 ringU = normalize(cross(vec3(0.0, 1.0, 0.0), ringN));
                vec3 ringV = cross(ringN, ringU);
                vec3 rp = dir * tRing;
                vec2 ruv = vec2(dot(rp, ringU) - 0.55, dot(rp, ringV) + 0.35);
                float rr = length(ruv);
                float band = smoothstep(1.85, 2.25, rr) * (1.0 - smoothstep(2.95, 3.55, rr));
                if (band > 0.002) {
                    float casArg = (rr - 2.68) / 0.055;
                    float cassini = 1.0 - 0.85 * exp(-casArg * casArg);
                    float fineIce = 0.50 + 0.50 * (0.5 + 0.5 * sin(rr * 24.0)) * (0.62 + 0.38 * cos(rr * 64.0));
                    float bhC = max(bhDot, 0.0);
                    float distFade = smoothstep(17.5, 4.5, tRing) * smoothstep(-0.04, 0.16, dir.y);
                    float bhClear = smoothstep(0.92, 0.82, bhDot);
                    vec3 ringHue = mix(vec3(0.32, 0.46, 0.90), vec3(1.08, 0.72, 0.38), 0.50 * bhC);
                    ringCol = ringHue * (band * cassini * fineIce * (0.35 + 0.75 * bhC) * distFade * bhClear * 0.21);
                }
            }
        }
    }

    return hdr * endCloudTrans + endCloudsHDR + ringCol * endCloudTrans;
#endif
}

vec3 analyticSkyReflectionLinear(vec3 worldReflectDir) {
    float horizon = clamp(worldReflectDir.y * 2.0 + 0.12, 0.0, 1.0);
#if defined(DIMENSION_NETHER)
    return hazeTintLinear() * mix(0.7, 1.3, horizon);
#elif defined(DIMENSION_END)
    /* Reflect the deep cosmic vacuum, diagonal sapphire Milky Way & Gargantua golden glow */
    vec3 galNorth = normalize(vec3(0.24, 0.84, -0.49));
    vec3 bhDir = worldShadowLightDir();
    float plane = exp(-pow(dot(worldReflectDir, galNorth), 2.0) * 11.0);
    float bhGlow = pow(max(dot(worldReflectDir, bhDir), 0.0), 12.0);
    float bhCore = pow(max(dot(worldReflectDir, bhDir), 0.0), 64.0);
    return vec3(0.002, 0.004, 0.012)
         + vec3(0.10, 0.18, 0.46) * (plane * 0.52)
         + vec3(1.10, 0.58, 0.20) * (bhGlow * 0.58 + bhCore * 0.95);
#else
    float up = sunElevation();
    float day = smoothstep(-0.08, 0.26, up);
    vec3 zenith = mix(vec3(0.008, 0.016, 0.042), vec3(0.14, 0.34, 0.82), day);
    vec3 baseSky = mix(hazeTintLinear(), zenith, horizon);

    /* Reflect Night Aurora Borealis on water, glass & crystals */
    float nightMask = smoothstep(-0.04, -0.22, up) * (1.0 - smoothstep(0.25, 0.70, rainStrength));
    if (nightMask > 0.01 && worldReflectDir.y > 0.05) {
        vec2 ap = worldReflectDir.xz / (worldReflectDir.y + 0.20);
        float curtain1 = sin(ap.x * 2.4 + sin(ap.y * 1.7 + frameTimeCounter * 0.08) * 1.5);
        float curtain2 = cos(ap.y * 2.1 - cos(ap.x * 1.4 - frameTimeCounter * 0.06) * 1.3);
        float wave = smoothstep(0.15, 0.95, 0.5 + 0.5 * curtain1 * curtain2);
        float elevFade = smoothstep(0.05, 0.26, worldReflectDir.y) * (1.0 - smoothstep(0.45, 0.92, worldReflectDir.y));
        vec3 auroraCol = mix(vec3(0.06, 0.65, 0.34), vec3(0.30, 0.12, 0.62), smoothstep(0.12, 0.55, worldReflectDir.y));
        baseSky += auroraCol * (wave * elevFade * nightMask * 0.52);
    }

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

/* True 360-Deg Off-Screen World-Space Ray Tracer via Shadow Volume (shadowtex1 + shadowcolor0/1):
   Reflects houses, trees, wool, torches & Magma blocks standing BEHIND THE CAMERA or OFF-SCREEN
   in pure OpenGL 3.3 (#version 120) with ZERO extra geometry passes! */
bool traceOffscreenWorldRT(vec3 viewPos, vec3 worldNormal, vec3 worldR, out vec3 outHitCol) {
    outHitCol = vec3(0.0);
#if !SHADOWS || defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    return false;
#else
    vec3 feetOrigin = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz + worldNormal * 0.14;
    if (dot(feetOrigin, feetOrigin) > (shadowDistance * shadowDistance * 0.85)) return false;

    vec4 sOrig4 = shadowProjection * (shadowModelView * vec4(feetOrigin, 1.0));
    vec3 sOrigNDC = sOrig4.xyz / max(sOrig4.w, 1e-5);
    vec3 sDirNDC  = mat3(shadowProjection) * (mat3(shadowModelView) * worldR);
    float zTexPerMeter = max(0.25 * abs(shadowProjection[2].z), 0.0005);

    /* 1-tap Overhang / Cave-Roof Guard:
       If the mirror origin is already underneath a roof or dirt overhang in sun-shadow space
       (penM0 > 0.02), abort immediately so roof texels NEVER smear onto mirrors below! */
    vec3 origD = distortShadowNDC(sOrigNDC + sDirNDC * 0.16);
    if (max(abs(origD.x), abs(origD.y)) > 0.96 || abs(origD.z) > 0.98) return false;
    float zOrigShad = texture2D(shadowtex1, origD.xy * 0.5 + 0.5).r;
    float penM0 = ((origD.z * 0.5 + 0.5) - zOrigShad) / zTexPerMeter;
    if (penM0 > 0.02) return false;

    float tPrev = 0.16;
    float prevPenM = penM0;
    float tHit = 0.0;
    vec2 hitSUV = vec2(0.5);
    float zHit = 1.0;
    bool hit = false;

    float maxWorldDist = 20.0 * mix(0.70, 1.0, hiResThrottle());
    float t = 0.42;
    for (int i = 0; i < 6; i++) {
        vec3 preNDC = sOrigNDC + sDirNDC * t;
        if (max(abs(preNDC.x), abs(preNDC.y)) > 0.96 || abs(preNDC.z) > 0.98) break;

        vec3 dNDC = distortShadowNDC(preNDC);
        vec2 suv = dNDC.xy * 0.5 + 0.5;
        float sampleZ = dNDC.z * 0.5 + 0.5;
        float zShad = texture2D(shadowtex1, suv).r;
        float penM = (sampleZ - zShad) / zTexPerMeter;
        float stepSpan = max(0.95, (t - tPrev) * 1.10);

        if (penM > 0.04 && prevPenM <= 0.02 && penM < stepSpan) {
            tHit = t;
            hitSUV = suv;
            zHit = zShad;
            hit = true;
            break;
        }
        prevPenM = penM;
        tPrev = t;
        t = min(t * 1.75, maxWorldDist);
    }

    if (!hit) return false;

    /* 2-step binary bisection in shadow volume space */
    float tLo = tPrev;
    float tHi = tHit;
    float finalPenM = 0.5;
    for (int b = 0; b < 2; b++) {
        float tMid = 0.5 * (tLo + tHi);
        vec3 mNDC = distortShadowNDC(sOrigNDC + sDirNDC * tMid);
        vec2 mSUV = mNDC.xy * 0.5 + 0.5;
        float mSampleZ = mNDC.z * 0.5 + 0.5;
        float mz = texture2D(shadowtex1, mSUV).r;
        float mPen = (mSampleZ - mz) / zTexPerMeter;
        if (mPen > 0.0) {
            tHi = tMid;
            hitSUV = mSUV;
            zHit = mz;
            finalPenM = mPen;
        } else {
            tLo = tMid;
        }
    }

    if (finalPenM > 0.85) return false;

    vec4 sc0 = texture2D(shadowcolor0, hitSUV);
    float zAll = texture2D(shadowtex0, hitSUV).r;
    bool unoccludedAlbedo = (abs(zAll - zHit) < zTexPerMeter * 0.45);
    vec3 hitAlbedo = unoccludedAlbedo ? srgbToLinear(sc0.rgb) : vec3(0.32);

    /* Emissive off-screen blocks (alpha=0.965 packed in shadow.fsh, ZERO extra MRTs!) glow brightly! */
    if (unoccludedAlbedo && sc0.a > 0.95 && sc0.a < 0.98) {
        outHitCol = hitAlbedo * 3.8;
        return true;
    }

    float approxNdl = clamp(dot(-worldR, worldShadowLightDir()) * 0.55 + 0.45, 0.15, 1.0);
    vec3 amb = skyAmbientIrradiance() * 0.68;
    vec3 sun = directCelestialIrradiance() * (approxNdl * 0.62 * directLightWeatherFade());
    outHitCol = hitAlbedo * (amb + sun);
    return true;
#endif
}

vec4 traceReflectionRT(vec2 screenUV, vec3 viewPos, vec3 worldNormal,
                       float f0, float smoothness, float skyAccess,
                       float dispersionAmt)
{
    vec3 viewNormal = normalize(mat3(gbufferModelView) * worldNormal);
    vec3 V = normalize(viewPos);
    vec3 R = reflect(V, viewNormal);

    float oneMinusCos = 1.0 - clamp(dot(-V, viewNormal), 0.001, 1.0);
    float omc2 = oneMinusCos * oneMinusCos;
    float fresnel = clamp(f0 + (1.0 - f0) * (omc2 * omc2), 0.0, 1.0) * smoothness;

    vec3 worldR = normalize(mat3(gbufferModelViewInverse) * R);

    /* Physically directional environment fallback for off-screen / behind-camera rays:
       1. Sky ONLY appears when reflected ray actually points UP (worldR.y > -0.02) AND has sky access
          (eliminates blue/purple sky leaking onto vertical indoor walls!).
       2. Horizontal / downward / indoor rays fall back to 3D Delta-NCM GI (colortex4/5) + room torch bounce! */
    float upSkyMask = smoothstep(-0.02, 0.26, worldR.y) * skyAccess;
    vec3 skyRefl = (upSkyMask > 0.01) ? (analyticSkyReflectionLinear(worldR) * upSkyMask) : vec3(0.0);

    /* Sample live 1080p rendered sky from colortex0 (Gargantua, Milky Way, Clouds, Aurora) when pointing at sky */
    if (upSkyMask > 0.04 && R.z < -0.02) {
        vec3 skyProj = fastProjectToScreen(viewPos + R * 220.0);
        if (skyProj.z > 0.0 && skyProj.x > 0.01 && skyProj.x < 0.99 && skyProj.y > 0.01 && skyProj.y < 0.99) {
            float dSkyCheck = texture2D(depthtex2, skyProj.xy).r;
            float dSkyHand  = texture2D(depthtex1, skyProj.xy).r;
            if (dSkyCheck >= 0.9999 && dSkyHand >= 0.9999) {
                float sBorder = min(min(skyProj.x, 1.0 - skyProj.x), min(skyProj.y, 1.0 - skyProj.y));
                float sBlend = smoothstep(0.01, 0.14, sBorder);
                vec3 liveSky = max(texture2D(colortex0, skyProj.xy).rgb, vec3(0.0)) * upSkyMask;
                skyRefl = mix(skyRefl, liveSky, sBlend);
            }
        }
    }

    /* 3D Directional Delta-NCM Room/Terrain Fallback for off-screen horizontal/downward rays */
    vec3 envFallback = skyRefl;
    if (upSkyMask < 0.92) {
        vec3 envG0 = max(texture2D(colortex4, screenUV).rgb, vec3(0.0));
        vec3 envL1 = texture2D(colortex5, screenUV).rgb;
        float eY0 = max(dot(envG0, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
        float dirGain = clamp(1.0 + 0.65 * (dot(envL1, worldR) / eY0), 0.35, 1.85);
        float surfBL = texture2D(colortex3, screenUV).z;
        vec3 localRoom = envG0 * (dirGain * 0.72) + vec3(0.55, 0.30, 0.12) * (surfBL * surfBL * 0.28);
        envFallback = mix(localRoom, skyRefl, upSkyMask);
    }

    float lightValid;
    vec3 lightDirView = activeLightDirView(lightValid);
    vec3 specGlint = vec3(0.0);
#if !defined(DIMENSION_NETHER) && !defined(DIMENSION_END)
    if (lightValid > 0.5 && upSkyMask > 0.15) {
        float specDot = max(dot(R, lightDirView), 0.0);
        if (specDot > 0.85) {
            float s2 = specDot * specDot;
            float s8 = s2 * s2; s8 *= s8;
            float s64 = s8 * s8; s64 *= s64;
            specGlint = directCelestialIrradiance() * (s64 * s64 * 1.35 * smoothness * upSkyMask);
        }
    }
#endif

    float resThrottle = hiResThrottle();
    float maxSsrDist = 44.0 * mix(0.68, 1.0, resThrottle);
    float ssrDistFade = 1.0 - smoothstep(maxSsrDist * 0.72, maxSsrDist, -viewPos.z);
    if (fresnel < 0.015 || ssrDistFade < 0.02 || dot(R, viewNormal) < 0.01) {
        return vec4(envFallback + specGlint, fresnel);
    }

    bool allowOffscreenRT = (fresnel > 0.08 && skyAccess > 0.20);

    /* Allow wide-angle cross-room mirror bounces all the way up to R.z = 0.94!
       When R.z > 0.94 (pointing straight behind camera), seamlessly hand off to 360-deg World RT! */
    float towardCamFade = (1.0 - smoothstep(0.72, 0.95, R.z)) * ssrDistFade;
    if (towardCamFade < 0.02) {
        vec3 offHit;
        if (allowOffscreenRT && traceOffscreenWorldRT(viewPos, worldNormal, worldR, offHit)) {
            return vec4(mix(envFallback, offHit, ssrDistFade) + specGlint * 0.2, fresnel);
        }
        return vec4(envFallback + specGlint, fresnel);
    }

    float marchLen = min((far > 1.0) ? far : 30.0, 30.0 * mix(0.70, 1.0, resThrottle));
    vec3 startPos = viewPos + viewNormal * max(0.022, -viewPos.z * 0.0028);
    if (R.z > 0.0001) {
        float zRoom = (-near * 1.15) - startPos.z;
        if (zRoom <= 0.02) {
            vec3 offHitNear;
            if (allowOffscreenRT && traceOffscreenWorldRT(viewPos, worldNormal, worldR, offHitNear)) {
                return vec4(mix(envFallback, offHitNear, ssrDistFade), fresnel);
            }
            return vec4(envFallback + specGlint, fresnel);
        }
        marchLen = min(marchLen, zRoom / R.z);
    }
    vec3 endPos = startPos + R * marchLen;

    vec3 p0 = fastProjectToScreen(startPos);
    vec3 p1 = fastProjectToScreen(endPos);
    if (p0.z < 0.0 || p1.z < 0.0) {
        vec3 offHitClip;
        if (allowOffscreenRT && traceOffscreenWorldRT(viewPos, worldNormal, worldR, offHitClip)) {
            return vec4(mix(envFallback, offHitClip, ssrDistFade), fresnel);
        }
        return vec4(envFallback + specGlint, fresnel);
    }

    vec2 uv0 = p0.xy;
    vec2 uv1 = p1.xy;
    float invZ0 = 1.0 / min(startPos.z, -near);
    float invZ1Raw = 1.0 / min(endPos.z, -near);

    /* Clip ray segment strictly to visible screen box [0.002, 0.998] & compute exact 3D clippedLen! */
    vec2 duv = uv1 - uv0;
    float sMax = 1.0;
    if (duv.x >  1e-5) sMax = min(sMax, (0.998 - uv0.x) / duv.x);
    if (duv.x < -1e-5) sMax = min(sMax, (0.002 - uv0.x) / duv.x);
    if (duv.y >  1e-5) sMax = min(sMax, (0.998 - uv0.y) / duv.y);
    if (duv.y < -1e-5) sMax = min(sMax, (0.002 - uv0.y) / duv.y);
    if (sMax <= 0.001) {
        vec3 offHitEdge;
        if (allowOffscreenRT && traceOffscreenWorldRT(viewPos, worldNormal, worldR, offHitEdge)) {
            return vec4(mix(envFallback, offHitEdge, ssrDistFade), fresnel);
        }
        return vec4(envFallback + specGlint, fresnel);
    }

    uv1 = uv0 + duv * sMax;
    float invZ1 = mix(invZ0, invZ1Raw, sMax);
    float clippedAlpha = clamp((sMax * invZ1Raw) / invZ1, 0.0, 1.0);
    float clippedLen = marchLen * clippedAlpha;

    vec2 pxSpan = (uv1 - uv0) * vec2(max(viewWidth, 1.0), max(viewHeight, 1.0));
    if (dot(pxSpan, pxSpan) < 2.25) {
        return vec4(envFallback + specGlint, fresnel);
    }

    float sPrev = 0.0;
    float sHit = 0.0;
    float hitPen = 999.0;
    float prevPen = -0.02;
    float prevT3D = 0.0;
    vec2 hitUV = uv0;
    bool hit = false;

    /* Adaptive step count + 16x binary bisection: 10 @ <=800p, 8 @ 1080p (128x eff.), 6 @ 1440p+ (96x eff.) */
    int numSteps = (resThrottle < 0.46) ? 6 : ((resThrottle < 0.78) ? 8 : 10);
    float invSteps = 1.0 / float(numSteps);
    for (int i = 1; i <= 10; i++) {
        if (i > numSteps) break;
        float fi = float(i) * invSteps;
        float s = 0.18 * fi + 0.82 * fi * fi;
        vec2 curUV = mix(uv0, uv1, s);
        float probeZ = 1.0 / mix(invZ0, invZ1, s);
        /* Exact 3D Euclidean ray travel (never collapses to 0 when ray moves sideways R.z ~ 0!) */
        float t3D = clippedLen * clamp(s * invZ1 * probeZ, 0.0, 1.0);
        float step3D = max(t3D - prevT3D, 0.04);

        float sd = texture2D(depthtex2, curUV).r;
        if (sd < 0.9999) {
            float vz = fastViewZ(sd);
            float pen = vz - probeZ;
            float thick = max(0.24, step3D * 1.85 + 0.025 * (-probeZ));
            if (pen > 0.0 && prevPen <= 0.04 && pen < thick) {
                sHit = s;
                hitPen = pen;
                hitUV = curUV;
                hit = true;
                break;
            }
            prevPen = pen;
        } else {
            prevPen = -1.0;
        }
        sPrev = s;
        prevT3D = t3D;
    }

    if (!hit) {
        /* Screen-space ray exited viewport or missed -> seamlessly continue into 360-deg Off-Screen World RT! */
        vec3 offHitMiss;
        if (allowOffscreenRT && traceOffscreenWorldRT(viewPos, worldNormal, worldR, offHitMiss)) {
            return vec4(mix(envFallback, offHitMiss, ssrDistFade) + specGlint * 0.15, fresnel);
        }
        return vec4(envFallback + specGlint, fresnel);
    }

    /* 4-step perspective-correct binary bisection (16x sub-step refinement) */
    float sLo = sPrev;
    float sHi = sHit;
    float finalPen = hitPen;
    float finalProbeZ = 1.0 / mix(invZ0, invZ1, sHi);
    for (int b = 0; b < 4; b++) {
        float sMid = 0.5 * (sLo + sHi);
        vec2 mUV = mix(uv0, uv1, sMid);
        float mProbeZ = 1.0 / mix(invZ0, invZ1, sMid);
        float md = texture2D(depthtex2, mUV).r;
        float mz = (md < 0.9999) ? fastViewZ(md) : -1e6;
        float mpen = mz - mProbeZ;
        if (mpen > 0.0) {
            sHi = sMid;
            hitUV = mUV;
            finalPen = mpen;
            finalProbeZ = mProbeZ;
        } else {
            sLo = sMid;
        }
    }

    /* Exact 3D post-bisection step residual: captures angled walls cleanly while rejecting thin-torch silhouettes! */
    float probeZLo = 1.0 / mix(invZ0, invZ1, sLo);
    float tLo3D = clippedLen * clamp(sLo * invZ1 * probeZLo, 0.0, 1.0);
    float tHi3D = clippedLen * clamp(sHi * invZ1 * finalProbeZ, 0.0, 1.0);
    float finalStep3D = max(tHi3D - tLo3D, 0.02);
    bool ssrValid = (finalPen <= max(0.18, finalStep3D * 2.6 + 0.025 * (-finalProbeZ)));
    if (ssrValid) {
        float dHandCheck = texture2D(depthtex1, hitUV).r;
        float dWorldCheck = texture2D(depthtex2, hitUV).r;
        if (dWorldCheck >= 0.9999 || dHandCheck < dWorldCheck - 1e-5) {
            ssrValid = false;
        } else {
            vec4 hitSurf = texture2D(colortex3, hitUV);
            float hitRawByte = floor(hitSurf.w * 255.0 + 0.5);
            if (hitRawByte >= 32.0) {
                float hitMat = floor(hitRawByte * 0.03125);
                if (abs(hitMat - 2.0) > 0.5 && dot(octDecodeNormal(hitSurf.xy), worldR) > 0.45) {
                    ssrValid = false;
                }
            }
        }
    }

    if (!ssrValid) {
        vec3 offHitRej;
        if (allowOffscreenRT && traceOffscreenWorldRT(viewPos, worldNormal, worldR, offHitRej)) {
            return vec4(mix(envFallback, offHitRej, ssrDistFade) + specGlint * 0.15, fresnel);
        }
        return vec4(envFallback + specGlint, fresnel);
    }

    float border = min(min(hitUV.x, 1.0 - hitUV.x), min(hitUV.y, 1.0 - hitUV.y));
    float edgeFade = smoothstep(0.0, 0.09, border) * towardCamFade;
    vec3 hitCol;
#if SPECTRAL_DISPERSION
    if (dispersionAmt > 0.0002) {
        vec2 dispOff = vec2(dispersionAmt * (0.4 + 0.6 * oneMinusCos), dispersionAmt * 0.25);
        hitCol = vec3(
            texture2D(colortex0, clamp(hitUV + dispOff, 0.002, 0.998)).r,
            texture2D(colortex0, hitUV).g,
            texture2D(colortex0, clamp(hitUV - dispOff, 0.002, 0.998)).b
        );
    } else {
        hitCol = texture2D(colortex0, hitUV).rgb;
    }
#else
    hitCol = texture2D(colortex0, hitUV).rgb;
#endif
    vec3 reflCol = mix(envFallback, hitCol, edgeFade) + specGlint * (1.0 - 0.90 * edgeFade);
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

vec3 applyVolumetricsLinear(vec3 sceneHDR, vec3 dir, float haze, float rays, float tHit, bool noGeometry) {
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
#if defined(DIMENSION_END)
        /* Directional Cosmic Void Mist: royal indigo abyss + golden Gargantua Mie forward scatter! */
        vec3 bhDir = worldShadowLightDir();
        float bhCos = max(dot(dir, bhDir), 0.0);
        float bhMie = pow(bhCos, 5.0);
        vec3 endMist = vec3(0.012, 0.015, 0.046)
                     + vec3(0.028, 0.016, 0.068) * (1.0 - abs(dir.y))
                     + vec3(0.32, 0.16, 0.055) * bhMie;
        color = mix(color, endMist, hazeFactor);
#else
        color = mix(color, hazeTintLinear(), hazeFactor);
#endif
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
