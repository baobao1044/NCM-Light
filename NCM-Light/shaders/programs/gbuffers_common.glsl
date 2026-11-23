/*
    Shared helpers for gbuffers_* programs.
    Bit-exact 32-bit RGBA8 packing in colortex3 (50% less VRAM bandwidth than RGBA16F!):
      .xy = Octahedral World Normal mapped to [0, 1]
      .z  = Blocklight [0, 1]
      .w  = Exact Material Tag (1..5) * 42 + Quantised Skylight (0..36) / 255
*/

#ifndef NCM_GBUFFERS_COMMON_GLSL
#define NCM_GBUFFERS_COMMON_GLSL

uniform mat4 gbufferModelViewInverse;

vec3 safeWorldNormal(vec3 viewNormal) {
    float len2 = dot(viewNormal, viewNormal);
    if (len2 < 1e-6) return vec3(0.0, 1.0, 0.0);
    vec3 wn = mat3(gbufferModelViewInverse) * (viewNormal * inversesqrt(len2));
    float wLen2 = dot(wn, wn);
    return (wLen2 > 1e-6) ? (wn * inversesqrt(wLen2)) : vec3(0.0, 1.0, 0.0);
}

float entityDirectionalShade(vec3 worldNormal) {
    float top = clamp(worldNormal.y, -1.0, 1.0);
    float side = abs(worldNormal.x);
    return clamp(0.80 + 0.16 * top + 0.04 * side, 0.62, 1.0);
}

vec2 octEncodeNormal(vec3 n) {
    float l1 = max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    vec2 p = n.xy / l1;
    if (n.z < 0.0) {
        vec2 signNotZero = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        p = (1.0 - abs(p.yx)) * signNotZero;
    }
    return p * 0.5 + 0.5;
}

float classifyMaterialTag(float blockId) {
    if (abs(blockId - 10100.5) < 1.0) return 2.0;
    if (abs(blockId - 10200.0) < 0.5) return 5.0;
    if (abs(blockId - 10300.0) < 0.5) return 3.0;
    if (abs(blockId - 10400.0) < 0.5) return 4.0;
    return 1.0;
}

vec4 packSurfaceData(vec3 worldNormal, vec2 lmcoord, float matTag) {
    float nLen2 = dot(worldNormal, worldNormal);
    vec3 n = (nLen2 > 1e-6) ? (worldNormal * inversesqrt(nLen2)) : vec3(0.0, 1.0, 0.0);
    vec2 oct01 = octEncodeNormal(n);
    float blockLight = clamp((lmcoord.x - 0.03125) / 0.9375, 0.0, 1.0);
    float skyLight   = clamp((lmcoord.y - 0.03125) / 0.9375, 0.0, 1.0);
    float packedW = (matTag * 42.0 + floor(skyLight * 36.0 + 0.5)) / 255.0;
    return vec4(oct01, blockLight, packedW);
}

#endif
