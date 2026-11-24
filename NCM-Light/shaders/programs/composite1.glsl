/*
    Concept Level 0 (rendered at QUARTER resolution = 1/16th screen pixels!):
      * 2-tap Distorted Optical-Depth Volumetric Shafts
      * 4-Quadrant Omnidirectional Deterministic RTGI + Deep Crevice RTAO (0.50 taps/screen-px, ZERO noise!)
      gl_FragData[0] -> colortex1: vec4(rays0, ao0, volGate0, edgeGate0)
      gl_FragData[1] -> colortex4: vec4(gi0.rgb, 1.0)
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

vec2 detectEdges(vec2 uv, float centerHit, bool centerSky, vec3 centerWorldN, bool hasSurf) {
    vec2 stepUv = vec2(2.0 / max(viewWidth, 1.0), 2.0 / max(viewHeight, 1.0));
    vec2 uv0 = clamp(uv + vec2( stepUv.x,  stepUv.y), 0.001, 0.999);
    vec2 uv1 = clamp(uv + vec2(-stepUv.x,  stepUv.y), 0.001, 0.999);
    vec2 uv2 = clamp(uv + vec2( stepUv.x, -stepUv.y), 0.001, 0.999);
    vec2 uv3 = clamp(uv + vec2(-stepUv.x, -stepUv.y), 0.001, 0.999);

    float d0 = texture2D(depthtex0, uv0).r;
    float d1 = texture2D(depthtex0, uv1).r;
    float d2 = texture2D(depthtex0, uv2).r;
    float d3 = texture2D(depthtex0, uv3).r;

    bool anySky = (d0 >= 0.9999) || (d1 >= 0.9999) || (d2 >= 0.9999) || (d3 >= 0.9999);
    bool allSky = (d0 >= 0.9999) && (d1 >= 0.9999) && (d2 >= 0.9999) && (d3 >= 0.9999);

    if (centerSky != allSky || (anySky && !allSky)) {
        return vec2(1.0, 1.0);
    }
    if (centerSky) {
        return vec2(0.0, 0.0);
    }

    float z0 = -fastViewZ(d0);
    float z1 = -fastViewZ(d1);
    float z2 = -fastViewZ(d2);
    float z3 = -fastViewZ(d3);
    float zMin = min(min(z0, z1), min(z2, z3));
    float zMax = max(max(z0, z1), max(z2, z3));
    float silEdge = smoothstep(0.14, 0.42, (zMax - zMin) / (centerHit + 1.4));

    float normEdge = 0.0;
    if (hasSurf && centerHit < 40.0) {
        vec4 s0 = texture2D(colortex3, uv0);
        vec4 s1 = texture2D(colortex3, uv1);
        vec3 n0 = (s0.w > 0.08) ? octDecodeNormal(s0.xy) : centerWorldN;
        vec3 n1 = (s1.w > 0.08) ? octDecodeNormal(s1.xy) : centerWorldN;
        float minDot = min(dot(centerWorldN, n0), dot(centerWorldN, n1));
        normEdge = smoothstep(0.18, 0.65, 1.0 - minDot);
    }
    return vec2(silEdge, normEdge);
}

void main() {
    vec3 viewPos, camRel, dir;
    float tHit;
    vec2 skyWeights;
    bool noGeometry;
    levelSetup(texcoord, viewPos, camRel, dir, tHit, skyWeights, noGeometry);

    vec4 surfData = texture2D(colortex3, texcoord);
    bool hasSurf;
    vec3 worldN;
    float blockLight, skyLight, matTag;
    unpackSurfaceData(surfData, hasSurf, worldN, blockLight, skyLight, matTag);
    hasSurf = hasSurf && (!noGeometry);

    vec2 edges = detectEdges(texcoord, tHit, noGeometry, worldN, hasSurf);

    float contrast, subRes;
    float rawRays0 = marchShafts(dir, tHit, TAPS_CONCEPT0, contrast, subRes);
    float rays0 = rawRays0 * skyWeights.y;

    float bandPred = contrast * smoothstep(0.03, 0.18, rays0) * 0.35;
    float rayPred = max(bandPred, relativeResidual(subRes * skyWeights.y, rays0));
    float volGate0 = clamp(max(rayPred, edges.x * 0.55 * step(0.002, rays0)), 0.0, 1.0);

    vec3 outG0 = vec3(0.0);
    vec3 outL1 = vec3(0.0);
    float outSignedInnov = 0.0;
    float normDepth = 1.0;
    float ao0 = 1.0;
    float edgeGate0 = 0.0;

    if (noGeometry) {
#if defined(DIMENSION_END)
        evaluateEndSkyL0(dir, outG0, outL1, ao0, rays0, volGate0);
#else
        vec4 vSky = evaluateVolumetricSkyL0(dir);
        outG0 = vSky.rgb;
        ao0 = vSky.a;
#endif
        normDepth = 1.0;
        edgeGate0 = edges.x;
    } else {
        float currDist = -viewPos.z;
        normDepth = clamp(currDist / 64.0, 0.0, 0.99);
#if SHADOWS && defined(DIMENSION_END)
        /* Long-reach 12-step Gargantua shadow at 0.25x (16x cheaper than 1080p!) stored in unused rays0 slot */
        rays0 = hasSurf ? traceEndGargantuaShadow(viewPos, worldN) : 1.0;
#endif
#if RT_GI
        if (hasSurf && RT_STRENGTH > 0.001 && currDist < 42.0) {
            vec3 viewN = normalize(mat3(gbufferModelView) * worldN);
            vec3 prevG0 = vec3(0.0);
            vec3 prevL1 = vec3(0.0);
            float prevSignedInnov = 0.0;
            bool hasHistory = false;

            vec2 prevUV = reprojectToPrevUV(viewPos);
            if (prevUV.x > 0.004 && prevUV.x < 0.996 && prevUV.y > 0.004 && prevUV.y < 0.996) {
                vec4 histDC = texture2D(colortex4, prevUV);
                float histDist = histDC.a * 64.0;
                if (histDC.a < 0.995 && abs(histDist - currDist) < max(0.65, currDist * 0.14)) {
                    prevG0 = max(histDC.rgb, vec3(0.0));
                    vec4 histSH = texture2D(colortex5, prevUV);
                    prevL1 = histSH.rgb;
                    prevSignedInnov = histSH.a;
                    hasHistory = true;
                }
            }

            traceDeltaNCMPathGI(texcoord, viewPos, viewN, worldN, prevG0, prevL1, prevSignedInnov, hasHistory,
                                outG0, outL1, outSignedInnov, ao0);
            edgeGate0 = clamp(max(edges.x * 0.80, edges.y * 0.90), 0.0, 1.0);
        }
#endif
    }

    gl_FragData[0] = vec4(rays0, ao0, volGate0, edgeGate0);
    gl_FragData[1] = vec4(outG0, normDepth);
    gl_FragData[2] = vec4(outL1, outSignedInnov);
}

#endif
