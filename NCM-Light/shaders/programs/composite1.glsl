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

    vec3 gi0 = vec3(0.0);
    vec3 reservoirGI = vec3(0.0);
    float normDepth = 1.0;
    float ao0 = 1.0;
    float edgeGate0 = 0.0;

    if (noGeometry) {
        vec4 vSky = evaluateVolumetricSkyL0(dir);
        gi0 = vSky.rgb;
        ao0 = vSky.a;
    } else {
        normDepth = clamp((-viewPos.z) / 64.0, 0.0, 1.0);
#if RT_GI
        if (hasSurf && RT_STRENGTH > 0.001 && (-viewPos.z) < 42.0) {
            vec3 viewN = normalize(mat3(gbufferModelView) * worldN);
            vec4 rt0 = tracePathGI(texcoord, viewPos, viewN);
            float currLuma = dot(rt0.rgb, vec3(0.2126, 0.7152, 0.0722));
            gi0 = rt0.rgb / (1.0 + currLuma * 0.08);
            ao0 = rt0.a;
            edgeGate0 = clamp(max(edges.x * 0.80, edges.y * 0.90), 0.0, 1.0);

            reservoirGI = gi0;
            vec2 prevUV = reprojectToPrevUV(viewPos);
            if (prevUV.x > 0.004 && prevUV.x < 0.996 && prevUV.y > 0.004 && prevUV.y < 0.996) {
                vec4 hist = texture2D(colortex5, prevUV);
                float histDist = hist.a * 64.0;
                float currDist = -viewPos.z;
                if (abs(histDist - currDist) < max(0.35, currDist * 0.08)) {
                    reservoirGI = mix(gi0, max(hist.rgb, vec3(0.0)), 0.87);
                }
            }
        }
#endif
    }

    gl_FragData[0] = vec4(rays0, ao0, volGate0, edgeGate0);
    gl_FragData[1] = vec4(gi0, 1.0);
    gl_FragData[2] = vec4(reservoirGI, normDepth);
}

#endif
