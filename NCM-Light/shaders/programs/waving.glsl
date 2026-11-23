/*
    Zero-fragment-cost vertex waving for grounded plants (10100) and tree canopies (10101).
*/

#ifndef NCM_WAVING_GLSL
#define NCM_WAVING_GLSL

vec3 computeFoliageWave(vec3 worldPos, float blockId, float texY, float midTexY, float time, float rain) {
#if !WAVING_FOLIAGE
    return vec3(0.0);
#else
    bool isGroundedPlant = (abs(blockId - 10100.0) < 0.5);
    bool isLeafCanopy    = (abs(blockId - 10101.0) < 0.5);
    if (!isGroundedPlant && !isLeafCanopy) {
        return vec3(0.0);
    }

#if defined(DIMENSION_NETHER) || defined(DIMENSION_END)
    float dimScale = 0.25;
#else
    float dimScale = 1.0;
#endif
    float windAmp = (1.0 + 0.85 * clamp(rain, 0.0, 1.0)) * dimScale;
    float phase = time * 2.3 + worldPos.x * 0.85 + worldPos.z * 0.65 + worldPos.y * 0.35;

    if (isGroundedPlant) {
        float topMask = (texY < midTexY) ? 1.0 : 0.0;
        return vec3(sin(phase) * 0.055, 0.0, cos(phase * 0.83 + 1.1) * 0.042) * (topMask * windAmp);
    }

    return vec3(
        sin(phase * 0.82) * 0.032,
        cos(phase * 1.15) * 0.014,
        cos(phase * 0.74 + 0.7) * 0.028
    ) * windAmp;
#endif
}

#endif
