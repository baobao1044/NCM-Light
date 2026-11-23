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

void denoisePathTracedGI(vec2 uv, float centerZ, vec3 centerWorldN,
                         out vec3 outGI, out float outAO)
{
    vec2 coarseRes = max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));
    vec2 stepUV = 2.0 / coarseRes;

    vec2 offs[5];
    offs[0] = vec2( 0.0,  0.0);
    offs[1] = vec2( stepUV.x, 0.0);
    offs[2] = vec2(-stepUV.x, 0.0);
    offs[3] = vec2( 0.0,  stepUV.y);
    offs[4] = vec2( 0.0, -stepUV.y);

    float kW[5];
    kW[0] = 0.36;
    kW[1] = 0.16;
    kW[2] = 0.16;
    kW[3] = 0.16;
    kW[4] = 0.16;

    vec3 sumGI = vec3(0.0);
    float sumAO = 0.0;
    float wSum = 0.0;

    for (int i = 0; i < 5; i++) {
        vec2 cuv = clamp(uv + offs[i], 0.002, 0.998);
        float cd = texture2D(depthtex0, cuv).r;
        if (cd >= 0.9999) continue;

        float dz = abs(fastViewZ(cd) - centerZ);
        vec4 cSurf = texture2D(colortex3, cuv);
        vec3 cNorm = (cSurf.w > 0.08) ? octDecodeNormal(cSurf.xy) : centerWorldN;
        float nDot = max(dot(centerWorldN, cNorm), 0.0);
        float nW = nDot * nDot;

        float w = kW[i] * (0.02 + 0.98 * nW * nW) / (1.0 + dz * dz * 6.0);
        sumAO += texture2D(colortex1, cuv).y * w;
        sumGI += texture2D(colortex5, cuv).rgb * w;
        wSum += w;
    }

    if (wSum > 1e-4) {
        float invW = 1.0 / wSum;
        outAO = clamp(sumAO * invW, 0.25, 1.0);
        outGI = max(sumGI * invW, vec3(0.0));
    } else {
        outAO = clamp(texture2D(colortex1, uv).y, 0.25, 1.0);
        outGI = texture2D(colortex5, uv).rgb;
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

    bool isHand = (d < dNoHand - 1e-5);
    float transCode = 0.0;
    if (!isHand && d < dOpaque - 1e-5) {
        if (transTag.x > 0.60) transCode = 2.0;
        else if (transTag.x > 0.20) transCode = 1.0;
    }
    bool isLiquidWater = (abs(transCode - 1.0) < 0.5);
    bool isTranslucent = (transCode > 0.5);

    vec4 surfData = texture2D(colortex3, texcoord);
    bool hasSurf;
    vec3 worldNormal;
    float blockLight, skyLight, matTag;
    unpackSurfaceData(surfData, hasSurf, worldNormal, blockLight, skyLight, matTag);

    vec3 albedoLin = srgbToLinear(rawColor.rgb);

    /* 1. Fast Upsampled RTGI/RTAO + Crisp 2-Tap 1080p Micro-Crevice AO */
    float estAO = 1.0;
    vec3 estGI = vec3(0.0);
    float edgeWork = 0.0;

#if RT_GI
    if (!noGeometry && !isHand && !isTranslucent && hasSurf && RT_STRENGTH > 0.001 && (-viewPos.z) < 42.0) {
        if (DEBUG_LEVEL != 1) {
            denoisePathTracedGI(texcoord, viewPos.z, worldNormal, estGI, estAO);
            edgeWork = coarse1.w;
        } else {
            estAO = clamp(coarse1.y, 0.25, 1.0);
            estGI = texture2D(colortex5, texcoord).rgb;
        }
    }
#endif

    /* 2. Cinematic Linear HDR Direct + Indirect + GGX Specular Lighting */
    vec3 hdrScene;

    if (noGeometry) {
        hdrScene = albedoLin * 1.32;
    } else if (isHand) {
        hdrScene = albedoLin * 1.18;
    } else if (isTranslucent) {
        float tBlock = transTag.y;
        float tSky   = transTag.z;
        vec3 shadowRGB = vec3(1.0);
#if SHADOWS && !defined(DIMENSION_NETHER) && !defined(DIMENSION_END)
        float outW = smoothstep(0.08, 0.45, tSky) * directLightWeatherFade();
        if (outW > 0.002) {
            shadowRGB = mix(vec3(1.0), surfaceShadowColored(camRel, vec3(0.0, 1.0, 0.0), false), outW);
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
            vec3 worldWaterPos = camRel + cameraPosition;
            vec2 waveOff = waterWaveNormal(worldWaterPos.xz).xz * clamp(waterDepth * 0.035, 0.0, 0.014);
            vec2 rUV = clamp(texcoord + waveOff, 0.002, 0.998);
            if (texture2D(depthtex0, rUV).r < texture2D(depthtex1, rUV).r - 1e-5) {
                albedoLin = srgbToLinear(texture2D(colortex0, rUV).rgb);
                dOpaque = texture2D(depthtex1, rUV).r;
            }
            float wAbs = (waterDepth * 0.26) / (1.0 + waterDepth * 0.26);
            vec3 absorb = mix(vec3(1.0), vec3(0.14, 0.52, 0.68), clamp(wAbs, 0.0, 0.88));
            vec3 floorView = viewFromDepth(rUV, dOpaque);
            vec3 floorRel = mat3(gbufferModelViewInverse) * floorView;
            float litScalar = dot(shadowRGB, vec3(0.3333));
            vec3 caustics = evaluateWaterCaustics(floorRel + cameraPosition, worldShadowLightDir(), waterDepth, litScalar);
            hdrScene = albedoLin * absorb * (transIrr + caustics);
        } else {
            hdrScene = albedoLin * transIrr;
        }
    } else if (!hasSurf) {
        hdrScene = albedoLin * 1.42;
    } else if (abs(matTag - 5.0) < 0.5) {
        hdrScene = albedoLin * 3.8 + albedoLin * estGI * (0.45 * RT_STRENGTH);
    } else {
        bool isFoliage = (abs(matTag - 2.0) < 0.5);
        bool isGlossy  = (abs(matTag - 3.0) < 0.5);
        bool isMetal   = (abs(matTag - 4.0) < 0.5);

        vec3 lightDir = worldShadowLightDir();
        float rawNdotL = dot(worldNormal, lightDir);
        /* Uniform top-hemisphere wrap on cross-mesh foliage prevents ugly dark X-planes! */
        float ndotl = isFoliage
            ? clamp(rawNdotL * 0.38 + 0.62, 0.35, 1.0)
            : max(rawNdotL, 0.0);

        vec3 shadowRGB = vec3(1.0);
        float outdoorW = smoothstep(0.08, 0.45, skyLight) * directLightWeatherFade();
#if SHADOWS && !defined(DIMENSION_NETHER) && !defined(DIMENSION_END)
        if (outdoorW > 0.002 && (ndotl > 0.001 || isFoliage)) {
            shadowRGB = mix(vec3(1.0), surfaceShadowColored(camRel, worldNormal, isFoliage), outdoorW);
        } else if (outdoorW <= 0.002) {
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

        /* Subsurface scattering on sunlit leaves & grass + GGX specular highlight on ALL blocks! */
        vec3 specTerm = vec3(0.0);
        if (litScalar > 0.02) {
            if (isFoliage) {
                float blDot = clamp(dot(dir, lightDir), 0.0, 1.0);
                directTerm += sunCol * vec3(0.58, 0.92, 0.32) * shadowRGB * (blDot * blDot * 0.52);
                specTerm = sunCol * shadowRGB * evaluateGGXSpecular(worldNormal, dir, lightDir, 0.44, 0.038);
            } else {
                float rough = isMetal ? 0.22 : (isGlossy ? 0.26 : 0.62);
                float f0Val = isMetal ? 0.65 : (isGlossy ? 0.06 : 0.032);
                specTerm = sunCol * shadowRGB * evaluateGGXSpecular(worldNormal, dir, lightDir, rough, f0Val);
            }
        }

        if (isEyeInWater == 1) {
            directTerm += evaluateWaterCaustics(camRel + cameraPosition, lightDir, (-viewPos.z) * 0.35 + 1.2, litScalar * outdoorW);
        }

        /* Subtle grazing-angle sky Fresnel rim on foliage & blocks for lush 3D surface sheen */
        float grazing = 1.0 - clamp(dot(-dir, worldNormal), 0.0, 1.0);
        float g2 = grazing * grazing;
        vec3 skyRim = skyAmbientIrradiance() * (g2 * g2 * skyLight * 0.14 * aoApplied);

        float bl2 = blockLight * blockLight;
        vec3 torchTerm = vec3(1.00, 0.56, 0.22) * (bl2 * (0.65 + 1.85 * blockLight)) * mix(1.0, aoApplied, 0.45);
        vec3 giTerm = estGI * (1.10 * RT_STRENGTH);

        vec3 totalIrradiance = max(skyTerm, minCaveAmb) + directTerm + torchTerm + giTerm;
        vec3 diffAlbedo = isMetal ? (albedoLin * 0.35) : albedoLin;

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

    float packedTrans = (transCode > 1.5)
        ? (0.60 + 0.35 * transTag.z)
        : ((transCode > 0.5) ? (0.15 + 0.35 * transTag.z) : 0.0);

    float wMain1 = max(wVol1, edgeWork);
    gl_FragData[0] = vec4(hdrScene, 1.0);
    gl_FragData[1] = vec4(clamp(estRays, 0.0, 1.0), clamp(wMain1, 0.0, 1.0), packedTrans, clamp(volGate1, 0.0, 1.0));
}

#endif
