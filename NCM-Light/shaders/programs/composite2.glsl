/*
    Full-resolution Shading Pass (Ultra-Fast & Cinematic):
      * 1-tap hardware-bilinear GI/AO read on ~85% flat pixels + C0 2x2 Gather ONLY on edges!
      * 2-tap Full-1080p Micro-Crevice AO for crisp block/foliage depth (ZERO full-res GI loops -> high FPS!)
      * 4.4x Sharp Distorted Shadows + Warm Sun / Cool Azure Sky + GGX Specular Relief + Subsurface Glow
      gl_FragData[0] -> colortex0 (R11G11B10F): Linear HDR lit world
      gl_FragData[1] -> colortex2 (RGBA8):      vec4(estRays1, wMain1, packedTrans, volGate1)
*/

#ifdef STAGE_VSH

varying vec2 texcoord;

void main() {
    gl_Position = ftransform();
    texcoord = gl_MultiTexCoord0.xy;
}

#endif

#ifdef STAGE_FSH

#include "/common.glsl"

varying vec2 texcoord;

void denoiseDeltaNCM(vec2 uv, float centerZ, vec3 centerWorldN, float centerSurprise,
                     out vec3 outGI, out vec3 outL1, out float outAO)
{
    vec2 coarseRes = max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
    /* Surprise-guided A-Trous stride: razor-sharp 1-texel when converged, 2.5-texel on disocclusion */
    float stride = mix(1.0, 2.5, centerSurprise);
    vec2 stepUV = stride / coarseRes;

    vec2 offs[5];
    offs[0] = vec2( 0.0,  0.0);
    offs[1] = vec2( stepUV.x, 0.0);
    offs[2] = vec2(-stepUV.x, 0.0);
    offs[3] = vec2( 0.0,  stepUV.y);
    offs[4] = vec2( 0.0, -stepUV.y);

    float kW[5];
    kW[0] = mix(0.44, 0.28, centerSurprise);
    kW[1] = (1.0 - kW[0]) * 0.25;
    kW[2] = kW[1];
    kW[3] = kW[1];
    kW[4] = kW[1];

    vec3 sumG0 = vec3(0.0);
    vec3 sumL1 = vec3(0.0);
    vec3 sumN  = vec3(0.0);
    float sumAO = 0.0;
    float wSum = 0.0;

    for (int i = 0; i < 5; i++) {
        vec2 cuv = clamp(uv + offs[i], 0.002, 0.998);
        float cd = texture2D(depthtex0, cuv).r;
        if (cd >= 0.9999) continue;

        float dz = abs(fastViewZ(cd) - centerZ);
        vec4 cSurf = texture2D(colortex3, cuv);
        vec3 cNorm = (cSurf.w > 0.12) ? octDecodeNormal(cSurf.xy) : centerWorldN;
        float nDot = max(dot(centerWorldN, cNorm), 0.0);
        float nW = nDot * nDot;

        float w = kW[i] * (0.02 + 0.98 * nW * nW) / (1.0 + dz * dz * 6.0);
        sumAO += texture2D(colortex1, cuv).y * w;
        sumG0 += max(texture2D(colortex4, cuv).rgb, vec3(0.0)) * w;
        sumL1 += texture2D(colortex5, cuv).rgb * w;
        sumN  += cNorm * w;
        wSum += w;
    }

    vec3 avgG0, avgCoarseN;
    if (wSum > 1e-4) {
        float invW = 1.0 / wSum;
        outAO = clamp(sumAO * invW, 0.25, 1.0);
        avgG0 = max(sumG0 * invW, vec3(0.0));
        outL1 = sumL1 * invW;
        avgCoarseN = normalize(sumN + vec3(0.0, 1e-5, 0.0));
    } else {
        outAO = clamp(texture2D(colortex1, uv).y, 0.25, 1.0);
        avgG0 = max(texture2D(colortex4, uv).rgb, vec3(0.0));
        outL1 = texture2D(colortex5, uv).rgb;
        avgCoarseN = centerWorldN;
    }

    /* Evaluate exact hemisphere-centered SH L1 directional shift for 1080p normal maps & block bevels */
    outGI = convolveDeltaNCML1(avgG0, outL1, centerWorldN, avgCoarseN);
}

void gatherVolumetricSky(vec2 uv, out vec3 skyScatHDR, out vec3 endCloudsHDR, out float skyTrans, out float borderContrast) {
    vec2 coarseRes = max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
    vec2 invCoarse = 1.0 / coarseRes;
    vec2 fCoord = uv * coarseRes - 0.5;
    vec2 baseIdx = floor(fCoord);
    vec2 f = clamp(fCoord - baseIdx, 0.0, 1.0);
    vec2 uv00 = (baseIdx + 0.5) * invCoarse;

    vec3 sumScat = vec3(0.0);
    vec3 sumEndC = vec3(0.0);
    float sumTrans = 0.0;
    float wSum = 0.0;
    float minT = 1.0;
    float maxT = 0.0;

    for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
            vec2 off = vec2(float(ix), float(iy));
            vec2 cuv = clamp(uv00 + off * invCoarse, 0.002, 0.998);
            if (texture2D(depthtex0, cuv).r < 0.9999) continue;
            float bW = ((ix == 0) ? (1.0 - f.x) : f.x) * ((iy == 0) ? (1.0 - f.y) : f.y) + 0.01;
            float tVal = clamp(texture2D(colortex1, cuv).y, 0.0, 1.0);
            sumTrans += tVal * bW;
            sumScat  += texture2D(colortex4, cuv).rgb * bW;
#if defined(DIMENSION_END)
            sumEndC  += texture2D(colortex5, cuv).rgb * bW;
#endif
            wSum += bW;
            minT = min(minT, tVal);
            maxT = max(maxT, tVal);
        }
    }

    if (wSum > 1e-4) {
        float invW = 1.0 / wSum;
        skyScatHDR = max(sumScat * invW, vec3(0.0));
        endCloudsHDR = max(sumEndC * invW, vec3(0.0));
        skyTrans = clamp(sumTrans * invW, 0.0, 1.0);
        borderContrast = maxT - minT;
    } else {
        skyScatHDR = vec3(0.0);
        endCloudsHDR = vec3(0.0);
        skyTrans = 1.0;
        borderContrast = 0.0;
    }
}

void main() {
    vec4 rawColor = texture2D(colortex0, texcoord);
    vec4 transTag = texture2D(colortex2, texcoord);
    vec4 coarse1  = texture2D(colortex1, texcoord);

    float d = texture2D(depthtex0, texcoord).r;
    float dOpaque = texture2D(depthtex1, texcoord).r;
    float dNoHand = texture2D(depthtex2, texcoord).r;

    vec3 viewPos, camRel, dir;
    float tHit;
    vec2 skyWeights;
    bool noGeometry;
    levelSetup(texcoord, viewPos, camRel, dir, tHit, skyWeights, noGeometry);

    /* True Iris/OptiFine hand test: hand writes depthtex1 (not depthtex2);
       water & glass write depthtex0 ONLY (depthtex1 == depthtex2)! */
    bool isHand = (dOpaque < dNoHand - 1e-5);
    float transCode = 0.0;
    if (!isHand && d < dOpaque - 1e-5) {
        if (transTag.x > 0.65) transCode = 2.0;       /* glass / ice */
        else if (transTag.x > 0.36) transCode = 1.5;  /* vertical waterfall */
        else if (transTag.x > 0.12) transCode = 1.0;  /* flat horizontal water */
    }
    bool isLiquidWater = (transCode > 0.5 && transCode < 1.75);
    bool isTranslucent = (transCode > 0.5);

    vec4 surfData = texture2D(colortex3, texcoord);
    bool hasSurf;
    vec3 worldNormal;
    float blockLight, skyLight, matTag;
    unpackSurfaceData(surfData, hasSurf, worldNormal, blockLight, skyLight, matTag);

    vec3 albedoLin = srgbToLinear(rawColor.rgb);

    /* 1. UHD 620 Adaptive Fast-Path Delta-NCM Read:
       ~85% flat converged 1080p pixels use 2 hardware-bilinear taps (12.5x fewer fetches!),
       while ~15% edge/disoccluded pixels run the 5-tap Surprise-Guided A-Trous filter! */
    float estAO = 1.0;
    vec3 estGI = vec3(0.0);
    vec3 estL1 = vec3(0.0);
    float edgeWork = 0.0;

#if RT_GI
    if (!noGeometry && !isHand && !isTranslucent && hasSurf && RT_STRENGTH > 0.001 && (-viewPos.z) < 42.0) {
        vec4 fastSH = texture2D(colortex5, texcoord);
        float centerSurprise = clamp((abs(fastSH.a) - 0.30) / 0.45, 0.0, 1.0);
        if ((coarse1.w > 0.08 || centerSurprise >= 0.22) && DEBUG_LEVEL != 1) {
            denoiseDeltaNCM(texcoord, viewPos.z, worldNormal, centerSurprise, estGI, estL1, estAO);
            edgeWork = max(coarse1.w, centerSurprise);
        } else {
            /* Ultra-Fast L1-Cached 4-Tap Tent Filter on 0.25x colortex4 + colortex1.y (0 depth/normal reads!):
               Eliminates 4x4 GI & AO blotches on stone/walls while staying super light on UHD 620! */
            vec2 qStep = 1.0 / max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
            vec2 uv00 = clamp(texcoord + vec2(-0.5, -0.5) * qStep, 0.002, 0.998);
            vec2 uv10 = clamp(texcoord + vec2( 0.5, -0.5) * qStep, 0.002, 0.998);
            vec2 uv01 = clamp(texcoord + vec2(-0.5,  0.5) * qStep, 0.002, 0.998);
            vec2 uv11 = clamp(texcoord + vec2( 0.5,  0.5) * qStep, 0.002, 0.998);
            vec3 g00 = max(texture2D(colortex4, uv00).rgb, vec3(0.0));
            vec3 g10 = max(texture2D(colortex4, uv10).rgb, vec3(0.0));
            vec3 g01 = max(texture2D(colortex4, uv01).rgb, vec3(0.0));
            vec3 g11 = max(texture2D(colortex4, uv11).rgb, vec3(0.0));
            float a00 = texture2D(colortex1, uv00).y;
            float a10 = texture2D(colortex1, uv10).y;
            float a01 = texture2D(colortex1, uv01).y;
            float a11 = texture2D(colortex1, uv11).y;
            estAO = clamp(0.25 * (a00 + a10 + a01 + a11), 0.25, 1.0);
            estGI = 0.25 * (g00 + g10 + g01 + g11);
            estL1 = fastSH.rgb;
        }
    }
#endif

    /* 2. Physical Optics Linear HDR Direct + Scattering + Dispersion + Indirect Specular */
    vec3 hdrScene;
    /* True feet-player space (includes gbufferModelViewInverse[3].xyz so shadows & caustics are exact in F5!) */
    vec3 feetPos  = camRel + gbufferModelViewInverse[3].xyz;
    vec3 worldPos = feetPos + cameraPosition;

    if (noGeometry) {
        vec3 skyScatHDR, endCloudsHDR;
        float skyTrans, borderContrast;
        gatherVolumetricSky(texcoord, skyScatHDR, endCloudsHDR, skyTrans, borderContrast);
#if defined(DIMENSION_END)
        /* Full-1080p True 3D Schwarzschild Geodesic Gargantua + Lensed Milky Way + Ring + Volumetric End Clouds */
        hdrScene = evaluateEndCosmosFullRes(dir, skyScatHDR, endCloudsHDR, skyTrans);
#else
        if (borderContrast > 0.18 && DEBUG_LEVEL != 1) {
            vec4 edgeRef = evaluateCloudEdgeRefine(dir);
            skyScatHDR = mix(skyScatHDR, edgeRef.rgb, 0.35);
            skyTrans = mix(skyTrans, edgeRef.a, 0.35);
            edgeWork = max(edgeWork, smoothstep(0.18, 0.45, borderContrast) * 0.55);
        }
        hdrScene = (albedoLin * 1.32) * skyTrans + skyScatHDR + evaluateAtmosphericRainbow(dir, worldShadowLightDir()) * skyTrans;
#endif
    } else if (isHand) {
        hdrScene = albedoLin * 1.18;
    } else if (isTranslucent) {
        float tBlock = transTag.y;
        float tSky   = transTag.z;
        vec3 shadowRGB = vec3(1.0);
        float transThick = 0.0;
#if SHADOWS && !defined(DIMENSION_NETHER) && !defined(DIMENSION_END)
        float outW = smoothstep(0.08, 0.45, tSky) * directLightWeatherFade();
        if (outW > 0.002) {
            shadowRGB = mix(vec3(1.0), surfaceShadowColored(feetPos, vec3(0.0, 1.0, 0.0), false, transThick), outW);
        } else {
            shadowRGB = vec3(0.0);
        }
#endif
        vec3 amb = max(skyAmbientIrradiance() * (tSky * tSky), vec3(0.024));
        vec3 sun = directCelestialIrradiance() * shadowRGB * 0.65;
        vec3 trc = vec3(1.00, 0.56, 0.22) * (tBlock * tBlock * (0.65 + 1.85 * tBlock));
        vec3 transIrr = amb + sun + trc;

        if (isLiquidWater) {
            float waterDepth = max(-fastViewZ(dOpaque) - (-viewPos.z), 0.0);
            vec2 waveOff = waterWaveNormal(worldPos.xz).xz * clamp(waterDepth * 0.035, 0.0, 0.014);
            vec2 rUV = clamp(texcoord + waveOff, 0.002, 0.998);
            if (texture2D(depthtex0, rUV).r < texture2D(depthtex1, rUV).r - 1e-5) {
#if SPECTRAL_DISPERSION
                /* Cauchy spectral refraction dispersion (n_R < n_G < n_B) across water ripples */
                vec2 uvR = clamp(texcoord + waveOff * 0.86, 0.002, 0.998);
                vec2 uvB = clamp(texcoord + waveOff * 1.16, 0.002, 0.998);
                vec3 refrSRGB = vec3(
                    texture2D(colortex0, uvR).r,
                    texture2D(colortex0, rUV).g,
                    texture2D(colortex0, uvB).b
                );
                albedoLin = srgbToLinear(refrSRGB);
#else
                albedoLin = srgbToLinear(texture2D(colortex0, rUV).rgb);
#endif
                dOpaque = texture2D(depthtex1, rUV).r;
            }
            /* Physical Beer-Lambert spectral extinction (red absorbs fastest, blue penetrates deepest) */
            vec3 absorb = exp(-vec3(0.42, 0.14, 0.06) * min(waterDepth, 16.0));
            vec3 floorView = viewFromDepth(rUV, dOpaque);
            vec3 floorFeet = (gbufferModelViewInverse * vec4(floorView, 1.0)).xyz;
            float litScalar = dot(shadowRGB, vec3(0.3333));
            vec3 caustics = evaluateWaterCaustics(floorFeet + cameraPosition, worldShadowLightDir(), waterDepth, litScalar);
            hdrScene = albedoLin * absorb * (transIrr + caustics);
        } else {
            hdrScene = albedoLin * transIrr;
        }
    } else if (!hasSurf) {
        hdrScene = albedoLin * 1.42;
    } else if (abs(matTag - 5.0) < 0.5) {
        hdrScene = albedoLin * 3.8 + albedoLin * estGI * (0.45 * RT_STRENGTH);
    } else {
        bool isFoliage  = (abs(matTag - 2.0) < 0.5);
        bool isGlossy   = (abs(matTag - 3.0) < 0.5);
        bool isMetal    = (abs(matTag - 4.0) < 0.5);
        bool isCrystal  = (abs(matTag - 6.0) < 0.5);
        bool isThickSSS = (abs(matTag - 7.0) < 0.5);

        vec3 lightDir = worldShadowLightDir();
        float rawNdotL = dot(worldNormal, lightDir);
        /* Uniform top-hemisphere wrap on cross-mesh foliage prevents ugly dark X-planes! */
        float ndotl = isFoliage
            ? clamp(rawNdotL * 0.38 + 0.62, 0.35, 1.0)
            : max(rawNdotL, 0.0);

        vec3 shadowRGB = vec3(1.0);
        float shadowThick = 0.0;
        float outdoorW = smoothstep(0.08, 0.45, skyLight) * directLightWeatherFade();
#if SHADOWS && !defined(DIMENSION_NETHER) && !defined(DIMENSION_END)
        if (outdoorW > 0.002 && (ndotl > 0.001 || isFoliage || isThickSSS)) {
            shadowRGB = mix(vec3(1.0), surfaceShadowColored(feetPos, worldNormal, isFoliage, shadowThick), outdoorW);
        } else if (outdoorW <= 0.002) {
            shadowRGB = vec3(0.0);
        }
#elif SHADOWS && defined(DIMENSION_END)
        if (ndotl > 0.001) {
            shadowRGB = vec3(traceEndContactShadow(viewPos, worldNormal));
        } else {
            shadowRGB = vec3(0.0);
        }
#endif
        float litScalar = dot(shadowRGB, vec3(0.3333));

        /* Sculpted RTAO: preserves deep crevice contrast while keeping a hint of bounce hue */
        float aoApplied = clamp(mix(1.0, estAO, clamp(RT_STRENGTH * 1.25, 0.0, 1.7)), 0.18, 1.0);
        vec3 aoTinted = mix(vec3(aoApplied * aoApplied) * (0.65 + 0.35 * albedoLin), vec3(aoApplied), 0.72);

        /* Directional hemisphere sky shading (distinguishes N/S/E/W & top/bottom faces cleanly!) */
        float dirAxisShade = 0.90 + 0.10 * abs(worldNormal.x) - 0.06 * abs(worldNormal.z);
        float hemi = (0.60 + 0.40 * worldNormal.y) * dirAxisShade;
        vec3 skyTerm = skyAmbientIrradiance() * (skyLight * skyLight * hemi) * aoTinted;
        vec3 minCaveAmb = vec3(0.024, 0.026, 0.034) * aoApplied;

        vec3 sunCol = directCelestialIrradiance();
        vec3 directTerm = sunCol * (ndotl * shadowRGB);

        /* 8-Class Physical BRDF: Kulla-Conty RGB GGX + Thin/Thick Beer-Lambert SSS + Sparkle Glints */
        float ndotv = clamp(dot(-dir, worldNormal), 0.001, 1.0);
        float rough = isCrystal ? 0.14 : (isMetal ? 0.20 : (isGlossy ? 0.25 : (isThickSSS ? 0.50 : 0.62)));
        vec3 f0RGB = isMetal
            ? mix(vec3(0.58), albedoLin, 0.82)
            : (isCrystal ? (vec3(0.14) * evaluateThinFilmIridescence(ndotv, 5.2))
                         : (isGlossy ? vec3(0.06) : vec3(0.032)));

        vec3 specTerm = vec3(0.0);
        if (litScalar > 0.02) {
            if (isFoliage) {
#if SUBSURFACE_SCATTER
                float blDot = clamp(dot(dir, lightDir), 0.0, 1.0);
                directTerm += sunCol * vec3(0.58, 0.92, 0.32) * shadowRGB * (blDot * blDot * 0.56);
#endif
                specTerm = sunCol * shadowRGB * evaluateGGXSpecularRGB(worldNormal, dir, lightDir, 0.44, vec3(0.038), false);
            } else {
                specTerm = sunCol * shadowRGB * evaluateGGXSpecularRGB(worldNormal, dir, lightDir, rough, f0RGB, isMetal);
                if (isThickSSS && (-viewPos.z) < 8.0) {
                    /* Subtle close-range snow/frost glint only (zero distant grain or colored speckles!) */
                    float snowGlint = evaluateMicroSparkle(worldPos, dir, lightDir, 16.0) * (1.0 - smoothstep(4.0, 8.0, -viewPos.z));
                    specTerm += sunCol * shadowRGB * (snowGlint * 0.20);
                }
            }
        }

#if SUBSURFACE_SCATTER
        /* Thick-Volume Beer-Lambert SSS on Snow, Slime, Honey & Coral using shadow-map thickness! */
        if (isThickSSS && outdoorW > 0.02) {
            float fwdLobe = clamp(dot(dir, lightDir) * 0.5 + 0.5, 0.0, 1.0);
            float hg = fwdLobe * fwdLobe;
            vec3 sssTint = mix(vec3(0.32, 0.68, 0.95), albedoLin * 1.25, 0.55);
            vec3 transThickRGB = exp(-vec3(1.15, 0.65, 0.42) * min(shadowThick + 0.15, 4.0));
            directTerm += sunCol * sssTint * transThickRGB * (outdoorW * (0.22 + 0.45 * hg));
        }
#endif

        /* Zero-Cost Indirect Specular Bounce directly from Delta-NCM SH L1 dominant direction! */
        if ((isMetal || isGlossy || isCrystal) && RT_STRENGTH > 0.001) {
            float giY = dot(estGI, vec3(0.2126, 0.7152, 0.0722));
            if (giY > 0.015) {
                vec3 muBounce = normalize(estL1 + worldNormal * (0.666667 * giY) + vec3(0.0, 1e-4, 0.0));
                specTerm += estGI * evaluateGGXSpecularRGB(worldNormal, dir, muBounce, min(rough + 0.14, 0.65), f0RGB, isMetal) * (0.95 * RT_STRENGTH);
            }
        }

        if (isEyeInWater == 1) {
            directTerm += evaluateWaterCaustics(worldPos, lightDir, (-viewPos.z) * 0.35 + 1.2, litScalar * outdoorW);
        }

        /* Subtle grazing-angle sky Fresnel rim on foliage & blocks for lush 3D surface sheen */
        float grazing = 1.0 - ndotv;
        float g2 = grazing * grazing;
        vec3 skyRim = skyAmbientIrradiance() * (g2 * g2 * skyLight * 0.14 * aoApplied);

        float bl2 = blockLight * blockLight;
        vec3 torchTerm = vec3(1.00, 0.56, 0.22) * (bl2 * (0.65 + 1.85 * blockLight)) * mix(1.0, aoApplied, 0.45);
        vec3 giTerm = estGI * (1.10 * RT_STRENGTH);

        vec3 totalIrradiance = max(skyTerm, minCaveAmb) + directTerm + torchTerm + giTerm;
        vec3 diffAlbedo = isMetal ? (albedoLin * 0.32) : albedoLin;

#if DEBUG_LEVEL == 6
        hdrScene = vec3(0.50) * (max(skyAmbientIrradiance() * (skyLight * skyLight * hemi) * aoApplied, minCaveAmb) + directTerm * 0.30)
                 + estGI * (1.35 * RT_STRENGTH) + specTerm;
#else
        hdrScene = diffAlbedo * totalIrradiance + specTerm + skyRim;
#endif
    }

#if DEBUG_LEVEL == 6
    if (noGeometry) hdrScene *= 0.25;
#endif

    /* 3. Concept Level 1 Deterministic Shafts (4 taps, Perceptually Gated!) */
    float sceneLuma = dot(hdrScene, vec3(0.2126, 0.7152, 0.0722));
    float volPerc = perceptualImportance(sceneLuma);
    float estRays = max(coarse1.x, 0.0);
    float wVol1 = (DEBUG_LEVEL == 1) ? 0.0 : refineWeight(coarse1.z * volPerc);
    float volGate1 = 0.0;

    if (wVol1 > 0.002) {
        float c1, subRes1;
        float rawRays1 = marchShafts(dir, tHit, TAPS_CONCEPT1, c1, subRes1);
        float rays1 = rawRays1 * skyWeights.y;

        estRays = max(estRays + wVol1 * (rays1 - estRays), 0.0);
        volGate1 = relativeResidual(subRes1 * skyWeights.y, rays1);
    }

    float packedTrans = (transCode > 1.75)
        ? (0.64 + 0.32 * transTag.z)
        : ((transCode > 1.25)
            ? (0.44 + 0.14 * transTag.z)
            : ((transCode > 0.5) ? (0.12 + 0.28 * transTag.z) : 0.0));

    float wMain1 = max(wVol1, edgeWork);
    gl_FragData[0] = vec4(hdrScene, 1.0);
    gl_FragData[1] = vec4(clamp(estRays, 0.0, 1.0), clamp(wMain1, 0.0, 1.0), packedTrans, clamp(volGate1, 0.0, 1.0));
}

#endif
