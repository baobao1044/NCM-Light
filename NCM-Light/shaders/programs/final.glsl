/*
    Fused Final Pass (Single 1080p Screen Output):
      1. Concept Level 2 Deterministic Shafts (8 taps, Perceptually Gated).
      2. Coherent 5+3 Binary Ray-Traced Reflections (Water, Glass/Ice, Metals, Polished, Rain).
      3. Normalized Reduced-Extinction Distance Haze + Balanced ACES Filmic Tonemapping!
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

void main() {
    vec4 hdrColor = texture2D(colortex0, texcoord);
    vec4 state = texture2D(colortex2, texcoord);

    float d = texture2D(depthtex0, texcoord).r;
    float dNoHand = texture2D(depthtex2, texcoord).r;

    vec3 viewPos, camRel, dir;
    float tHit;
    vec2 skyWeights;
    bool noGeometry;
    levelSetup(texcoord, viewPos, camRel, dir, tHit, skyWeights, noGeometry);

    bool isHand = (d < dNoHand - 1e-5);
    vec3 sceneHDR = hdrColor.rgb;

    /* 1. Concept Level 2 Grain-Free Shafts (8 taps, Perceptually Gated!) */
    float estRays = state.x;
    float wMain1 = state.y;
    float packedTrans = state.z;
    float volGate1 = state.w;
    float perc2 = perceptualImportance(dot(sceneHDR, vec3(0.2126, 0.7152, 0.0722)));
    float wVol2 = (DEBUG_LEVEL == 1) ? 0.0 : refineWeight(volGate1 * perc2);

    if (wVol2 > 0.002) {
        float c2, subRes2;
        float rawRays2 = marchShafts(dir, tHit, TAPS_CONCEPT2, c2, subRes2);
        float rays2 = rawRays2 * skyWeights.y;
        estRays = max(estRays + wVol2 * (rays2 - estRays), 0.0);
    }

    float reflWork = 0.0;

    /* 2. Coherent Speckle-Free Ray-Traced Reflections (0 taps on matte dry terrain!) */
#if RT_REFLECTIONS
    if (!noGeometry && !isHand && RT_STRENGTH > 0.001 && (-viewPos.z) < 64.0) {
        vec3 worldPos = camRel + cameraPosition;
        float distFade = 1.0 - smoothstep(46.0, 64.0, -viewPos.z);

        if (packedTrans >= 0.10 && packedTrans < 0.55) {
            float transSky = clamp((packedTrans - 0.15) / 0.35, 0.0, 1.0);
            vec3 geomN = reconstructGeomWorldNormal(texcoord, viewPos);
            vec3 reflN = (geomN.y > 0.85) ? waterWaveNormal(worldPos.xz) : geomN;
            float smoothVal = clamp(0.92 * min(RT_STRENGTH, 1.25), 0.0, 0.98) * distFade;
            vec4 refl = traceReflectionRT(viewPos, reflN, 0.035, smoothVal, max(transSky, 0.25));
            sceneHDR = mix(sceneHDR, refl.rgb, clamp(refl.a, 0.0, 0.92));
            reflWork = 0.80;
        } else if (packedTrans >= 0.55) {
            float transSky = clamp((packedTrans - 0.60) / 0.35, 0.0, 0.98);
            vec3 geomN = reconstructGeomWorldNormal(texcoord, viewPos);
            float smoothVal = clamp(0.88 * min(RT_STRENGTH, 1.25), 0.0, 0.96) * distFade;
            vec4 refl = traceReflectionRT(viewPos, geomN, 0.06, smoothVal, max(transSky, 0.25));
            sceneHDR = mix(sceneHDR, refl.rgb, clamp(refl.a, 0.0, 0.86));
            reflWork = 0.75;
        } else {
            vec4 surfData = texture2D(colortex3, texcoord);
            bool hasSurf;
            vec3 worldNormal;
            float blockLight, skyLight, matTag;
            unpackSurfaceData(surfData, hasSurf, worldNormal, blockLight, skyLight, matTag);

            if (hasSurf) {
                bool isGlossy = (abs(matTag - 3.0) < 0.5);
                bool isMetal  = (abs(matTag - 4.0) < 0.5);

                if (isMetal) {
                    float smoothM = clamp(0.90 * min(RT_STRENGTH, 1.3), 0.0, 0.96) * distFade;
                    vec4 refl = traceReflectionRT(viewPos, worldNormal, 0.68, smoothM, skyLight);
                    float luma = max(dot(sceneHDR, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
                    vec3 metalTint = clamp(sceneHDR / luma, vec3(0.35), vec3(1.45));
                    vec3 tintedRefl = refl.rgb * mix(vec3(1.0), metalTint, 0.65);
                    sceneHDR = mix(sceneHDR, tintedRefl, clamp(refl.a, 0.0, 0.88));
                    reflWork = 0.85;
                } else if (isGlossy) {
                    float smoothG = clamp(0.84 * min(RT_STRENGTH, 1.3), 0.0, 0.95) * distFade;
                    vec4 refl = traceReflectionRT(viewPos, worldNormal, 0.065, smoothG, skyLight);
                    sceneHDR = mix(sceneHDR, refl.rgb, clamp(refl.a, 0.0, 0.85));
                    reflWork = 0.70;
                } else if (rainStrength > 0.04 && worldNormal.y > 0.80 && skyLight > 0.45) {
                    float puddleNoise = sin(worldPos.x * 0.42 + cos(worldPos.z * 0.37)) * cos(worldPos.z * 0.48 - sin(worldPos.x * 0.31));
                    float puddle = smoothstep(0.10, 0.65, puddleNoise * 0.5 + 0.5)
                                 * clamp(rainStrength, 0.0, 1.0)
                                 * smoothstep(0.45, 0.85, skyLight)
                                 * distFade;
                    if (puddle > 0.08) {
                        vec3 wetN = normalize(mix(worldNormal, waterWaveNormal(worldPos.xz * 1.4), 0.60));
                        vec4 wetRefl = traceReflectionRT(viewPos, wetN, 0.04, 0.65 * puddle * min(RT_STRENGTH, 1.4), skyLight);
                        sceneHDR = mix(sceneHDR * (1.0 - 0.18 * puddle), wetRefl.rgb, clamp(wetRefl.a, 0.0, 0.70));
                        reflWork = 0.55 * puddle;
                    }
                }
            }
        }
    }
#endif

    float haze = analyticHaze(dir, tHit, skyWeights.x);
#if DEBUG_LEVEL != 6
    sceneHDR = applyVolumetricsLinear(sceneHDR, haze, estRays, tHit, noGeometry);
#endif

    float workMetric = clamp(max(wMain1 * 0.48, max(wVol2, reflWork)), 0.0, 1.0);

#if DEBUG_LEVEL == 2
    vec3 cold = vec3(0.03, 0.05, 0.10);
    vec3 mid  = vec3(0.95, 0.45, 0.06);
    vec3 hot  = vec3(1.00, 0.92, 0.48);
    vec3 heat = (workMetric < 0.5)
        ? mix(cold, mid, workMetric * 2.0)
        : mix(mid, hot, (workMetric - 0.5) * 2.0);
    vec3 baseSrgb = acesTonemapAndGrade(sceneHDR);
    float luma = dot(baseSrgb, vec3(0.299, 0.587, 0.114));
    gl_FragColor = vec4(mix(vec3(luma) * 0.22, heat, 0.88), 1.0);
#elif DEBUG_LEVEL == 3
    vec3 volHDR = haze * FOG_STRENGTH * hazeTintLinear()
                + estRays * RAYS_STRENGTH * shaftTintLinear();
    gl_FragColor = vec4(acesTonemapAndGrade(volHDR * 1.6), 1.0);
#elif DEBUG_LEVEL == 4
    float dOpaque = texture2D(depthtex1, texcoord).r;
    if (noGeometry) {
        gl_FragColor = vec4(0.08, 0.32, 0.95, 1.0);
    } else {
        float g = clamp(1.0 - (-viewPos.z) / 128.0, 0.0, 1.0);
        vec3 dbg = vec3(g);
        if (isHand) dbg = mix(dbg, vec3(0.2, 0.95, 0.35), 0.5);
        else if (d < dOpaque - 1e-5) dbg = mix(dbg, vec3(0.15, 0.75, 0.95), 0.5);
        gl_FragColor = vec4(dbg, 1.0);
    }
#elif DEBUG_LEVEL == 5
    vec3 c = (texcoord.x < 0.5) ? acesTonemapAndGrade(hazeTintLinear())
                                : acesTonemapAndGrade(shaftTintLinear());
    gl_FragColor = vec4(c, 1.0);
#else
    gl_FragColor = vec4(acesTonemapAndGrade(sceneHDR), 1.0);
#endif
}

#endif
