/*
    Quarter-Resolution Cinematic HDR Bloom Pass 1 (1/16th screen pixels):
      * 9-tap Karis-averaged soft-knee bright-pass downsample from colortex0 -> colortex6
      * Eliminates 1px firefly shimmer on stars, water glints & photon ring while preserving HDR glow!
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

vec3 extractBloomTap(vec2 uv, out float karisW) {
    vec3 c = min(max(texture2D(colortex0, clamp(uv, 0.002, 0.998)).rgb, vec3(0.0)), vec3(6.5));
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
#if defined(DIMENSION_END)
    float thresh = 0.56;
    float knee   = 0.45;
#else
    float thresh = 0.74;
    float knee   = 0.52;
#endif
    float soft = clamp(luma - (thresh - knee), 0.0, 2.0 * knee);
    float contrib = max(luma - thresh, (soft * soft) / (4.0 * knee + 1e-4));
    karisW = 1.0 / (1.0 + luma);
    return c * (contrib / max(luma, 1e-4));
}

void main() {
#if !BLOOM
    gl_FragData[0] = vec4(0.0, 0.0, 0.0, 1.0);
#else
    vec2 px = vec2(1.5 / max(viewWidth, 1.0), 1.5 / max(viewHeight, 1.0));

    vec2 offs[9];
    offs[0] = vec2( 0.0,  0.0);
    offs[1] = vec2( px.x, 0.0);
    offs[2] = vec2(-px.x, 0.0);
    offs[3] = vec2( 0.0,  px.y);
    offs[4] = vec2( 0.0, -px.y);
    offs[5] = vec2( px.x,  px.y);
    offs[6] = vec2(-px.x,  px.y);
    offs[7] = vec2( px.x, -px.y);
    offs[8] = vec2(-px.x, -px.y);

    float sW[9];
    sW[0] = 0.2500;
    sW[1] = 0.1250; sW[2] = 0.1250; sW[3] = 0.1250; sW[4] = 0.1250;
    sW[5] = 0.0625; sW[6] = 0.0625; sW[7] = 0.0625; sW[8] = 0.0625;

    vec3 sumCol = vec3(0.0);
    float sumW = 0.0;
    for (int i = 0; i < 9; i++) {
        float kw;
        vec3 b = extractBloomTap(texcoord + offs[i], kw);
        float w = sW[i] * kw;
        sumCol += b * w;
        sumW += w;
    }

    gl_FragData[0] = vec4(sumCol / max(sumW, 1e-4), 1.0);
#endif
}

#endif
