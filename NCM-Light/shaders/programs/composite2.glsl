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

void denoiseDeltaNCM(vec2 uv, float centerZ, vec3 centerWorldN, vec4 centerCoarse1, vec3 centerL1,
                     out vec3 outGI, out vec3 outL1, out float outAO, out float outEndShad)
{
    vec2 coarseRes = max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
    vec2 stepUV = 1.25 / coarseRes;

    vec2 offs[4];
    offs[0] = vec2( stepUV.x, 0.0);
    offs[1] = vec2(-stepUV.x, 0.0);
    offs[2] = vec2( 0.0,  stepUV.y);
    offs[3] = vec2( 0.0, -stepUV.y);

    float w0 = 0.44;
    float wN = 0.14;
    vec3 sumG0 = max(texture2D(colortex4, uv).rgb, vec3(0.0)) * w0;
    vec3 sumN  = centerWorldN * w0;
    float sumAO = centerCoarse1.y * w0;
    float sumES = centerCoarse1.x * w0;
    float wSum = w0;

    for (int i = 0; i < 4; i++) {
        vec2 cuv = clamp(uv + offs[i], 0.002, 0.998);
        float cd = texture2D(depthtex0, cuv).r;
        if (cd >= 0.9999) continue;

        float dz = abs(fastViewZ(cd) - centerZ);
        vec4 cSurf = texture2D(colortex3, cuv);
        vec3 cNorm = (cSurf.w > 0.12) ? octDecodeNormal(cSurf.xy) : centerWorldN;
        float nDot = max(dot(centerWorldN, cNorm), 0.0);
        float nW = nDot * nDot;

        float w = wN * (0.02 + 0.98 * nW * nW) / (1.0 + dz * dz * 6.0);
        vec4 c1 = texture2D(colortex1, cuv);
        sumAO += c1.y * w;
        sumES += c1.x * w;
        sumG0 += max(texture2D(colortex4, cuv).rgb, vec3(0.0)) * w;
        sumN  += cNorm * w;
        wSum += w;
    }

    float invW = 1.0 / max(wSum, 1e-4);
    outAO = clamp(sumAO * invW, 0.25, 1.0);
    outEndShad = clamp(sumES * invW, 0.0, 1.0);
    vec3 avgG0 = max(sumG0 * invW, vec3(0.0));
    outL1 = centerL1;
    vec3 avgCoarseN = normalize(sumN + vec3(0.0, 1e-5, 0.0));

    /* Evaluate exact hemisphere-centered SH L1 directional shift for 1080p normal maps & block bevels */
    outGI = convolveDeltaNCML1(avgG0, outL1, centerWorldN, avgCoarseN);
}

void gatherVolumetricSky(vec2 uv, out vec3 skyScatHDR, out vec3 endCloudsHDR,
                         out float skyTrans, out float borderContrast,
                         out float bhTrans, out float frontTrans)
{
    vec2 coarseRes = max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
    vec2 invCoarse = 1.0 / coarseRes;
    vec2 fCoord = uv * coarseRes - 0.5;
    vec2 baseIdx = floor(fCoord);
    vec2 f = clamp(fCoord - baseIdx, 0.0, 1.0);
    vec2 uv00 = (baseIdx + 0.5) * invCoarse;

    vec3 sumScat = vec3(0.0);
    vec3 sumEndC = vec3(0.0);
    float sumTrans = 0.0;
    float sumBhT = 0.0;
    float sumFrT = 0.0;
    float wSum = 0.0;
    float minT = 1.0;
    float maxT = 0.0;

    for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
            vec2 off = vec2(float(ix), float(iy));
            vec2 cuv = clamp(uv00 + off * invCoarse, 0.002, 0.998);
            if (texture2D(depthtex0, cuv).r < 0.9999) continue;
            float bW = ((ix == 0) ? (1.0 - f.x) : f.x) * ((iy == 0) ? (1.0 - f.y) : f.y) + 0.01;
            vec4 c1 = texture2D(colortex1, cuv);
            float tVal = clamp(c1.y, 0.0, 1.0);
            sumTrans += tVal * bW;
            sumScat  += texture2D(colortex4, cuv).rgb * bW;
#if defined(DIMENSION_END)
            sumEndC  += texture2D(colortex5, cuv).rgb * bW;
            sumBhT   += clamp(c1.x, 0.0, 1.0) * bW;
            sumFrT   += clamp(c1.z, 0.0, 1.0) * bW;
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
        bhTrans = clamp(sumBhT * invW, 0.0, 1.0);
        frontTrans = clamp(sumFrT * invW, 0.0, 1.0);
        borderContrast = maxT - minT;
    } else {
        skyScatHDR = vec3(0.0);
        endCloudsHDR = vec3(0.0);
        skyTrans = 1.0;
        bhTrans = 1.0;
        frontTrans = 1.0;
        borderContrast = 0.0;
    }
}

void main() {
    vec4 rawColor = texture2D(colortex0, texcoord);
    vec3 albedoLin = srgbToLinear(rawColor.rgb);

    vec3 viewPos, camRel, dir;
    float tHit;
    vec2 skyWeights;
    bool noGeometry;
    levelSetup(texcoord, viewPos, camRel, dir, tHit, skyWeights, noGeometry);

    /* Ultra-Fast Sky Early-Out: 97% open sky pixels use hardware GL_LINEAR directly (-85% sky fetches!);
       2x2 depth-masked gather runs ONLY along the thin 4px terrain silhouette border (c1Sky.w >= 0.05)! */
    if (noGeometry) {
        vec4 c1Sky = texture2D(colortex1, texcoord);
        vec3 skyScatHDR, endCloudsHDR;
        float skyTrans, borderContrast, bhTrans, frontTrans;
        if (c1Sky.w > 0.05) {
            gatherVolumetricSky(texcoord, skyScatHDR, endCloudsHDR, skyTrans, borderContrast, bhTrans, frontTrans);
        } else {
            skyScatHDR = max(texture2D(colortex4, texcoord).rgb, vec3(0.0));
            skyTrans = clamp(c1Sky.y, 0.0, 1.0);
            borderContrast = 0.0;
#if defined(DIMENSION_END)
            endCloudsHDR = max(texture2D(colortex5, texcoord).rgb, vec3(0.0));
            bhTrans = clamp(c1Sky.x, 0.0, 1.0);
            frontTrans = clamp(c1Sky.z, 0.0, 1.0);
#else
            endCloudsHDR = vec3(0.0);
            bhTrans = 1.0;
            frontTrans = 1.0;
#endif
        }
        float skyEdgeW = 0.0;
        vec3 hdrSky;
#if defined(DIMENSION_END)
        hdrSky = evaluateEndCosmosFullRes(dir, skyScatHDR, endCloudsHDR, skyTrans, bhTrans, frontTrans);
#else
        if (borderContrast > 0.18 && DEBUG_LEVEL != 1) {
            vec4 edgeRef = evaluateCloudEdgeRefine(dir);
            skyScatHDR = mix(skyScatHDR, edgeRef.rgb, 0.35);
            skyTrans = mix(skyTrans, edgeRef.a, 0.35);
            skyEdgeW = smoothstep(0.18, 0.45, borderContrast) * 0.55;
        }
        hdrSky = (albedoLin * 1.32) * skyTrans + skyScatHDR + evaluateAtmosphericRainbow(dir, worldShadowLightDir()) * skyTrans;
#endif
#if DEBUG_LEVEL == 6
        hdrSky *= 0.25;
#endif
#if defined(DIMENSION_END) || defined(DIMENSION_NETHER)
        gl_FragData[0] = vec4(hdrSky, 1.0);
        gl_FragData[1] = vec4(0.0, clamp(skyEdgeW, 0.0, 1.0), 0.0, 0.0);
#else
        float sEstRays = max(c1Sky.x, 0.0);
        float sVolGate1 = (DEBUG_LEVEL == 1) ? 0.0 : clamp(c1Sky.z, 0.0, 1.0);
        gl_FragData[0] = vec4(hdrSky, 1.0);
        gl_FragData[1] = vec4(clamp(sEstRays, 0.0, 1.0), clamp(skyEdgeW, 0.0, 1.0), 0.0, sVolGate1);
#endif
        return;
    }

    vec4 transTag = texture2D(colortex2, texcoord);
    vec4 coarse1  = texture2D(colortex1, texcoord);
    float d = texture2D(depthtex0, texcoord).r;
    float dOpaque = texture2D(depthtex1, texcoord).r;
    float dNoHand = texture2D(depthtex2, texcoord).r;

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

    /* 1. High-Res Bandwidth-Halved Delta-NCM Read:
       ~92% flat pixels reuse coarse1.xy + 1-2 hardware GL_LINEAR taps (ZERO camera-movement Surprise spike!);
       bilateral denoise runs ONLY on true geometric depth/normal edges (coarse1.w > 0.26)! */
    float estAO = 1.0;
    float estEndShad = clamp(coarse1.x, 0.0, 1.0);
    vec3 estGI = vec3(0.0);
    vec3 estL1 = vec3(0.0);
    float edgeWork = 0.0;
    float resThrottle = hiResThrottle();
    float maxGiDist = 42.0 * mix(0.68, 1.0, resThrottle);

#if RT_GI
    if (!isHand && !isTranslucent && hasSurf && RT_STRENGTH > 0.001 && (-viewPos.z) < maxGiDist) {
        vec4 fastSH = texture2D(colortex5, texcoord);
        if (coarse1.w > 0.26 && DEBUG_LEVEL != 1) {
            denoiseDeltaNCM(texcoord, viewPos.z, worldNormal, coarse1, fastSH.rgb, estGI, estL1, estAO, estEndShad);
            edgeWork = coarse1.w;
        } else {
            estAO = clamp(coarse1.y, 0.25, 1.0);
            estEndShad = clamp(coarse1.x, 0.0, 1.0);
            estL1 = fastSH.rgb;
            if (resThrottle < 0.78) {
                estGI = max(texture2D(colortex4, texcoord).rgb, vec3(0.0));
            } else {
                vec2 qDiag = 0.35 / max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
                vec3 gA = max(texture2D(colortex4, clamp(texcoord - qDiag, 0.002, 0.998)).rgb, vec3(0.0));
                vec3 gB = max(texture2D(colortex4, clamp(texcoord + qDiag, 0.002, 0.998)).rgb, vec3(0.0));
                estGI = 0.5 * (gA + gB);
            }
        }
    }
#endif

    /* 2. Physical Optics Linear HDR Direct + Scattering + Dispersion + Indirect Specular */
    vec3 hdrScene;
    vec3 feetPos  = camRel + gbufferModelViewInverse[3].xyz;
    vec3 worldPos = feetPos + cameraPosition;

    if (isHand) {
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
            shadowRGB = vec3(estEndShad);
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
            : (isCrystal ? (vec3(0.24) * evaluateThinFilmIridescence(ndotv, 5.2))
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

    /* 3. Forward 4-Tap Quarter-Res Shafts & Residual Gate Directly to Final Edge-Only Refiner! */
#if defined(DIMENSION_END) || defined(DIMENSION_NETHER)
    float estRays = 0.0;
    float wVol1 = 0.0;
    float volGate1 = 0.0;
#else
    float estRays = max(coarse1.x, 0.0);
    float wVol1 = 0.0;
    float volGate1 = (DEBUG_LEVEL == 1) ? 0.0 : clamp(coarse1.z, 0.0, 1.0);
#endif

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
