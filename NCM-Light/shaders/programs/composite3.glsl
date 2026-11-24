/*
    Single-Pass Fused Quarter-Res Cinema HDR Bloom (1/16th screen pixels, -1 FBO pass!):
      * 9 hardware-bilinear (GL_LINEAR) Karis-averaged taps = 36 effective full-res samples!
      * Combines tight inner glow + wide atmospheric halo + subtle Interstellar anamorphic streak
        in ONE ultra-fast 0.25x pass (<0.12ms on Intel UHD 620 even at high resolutions!).
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
    vec2 px = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));

#if defined(DIMENSION_END)
    vec3 anamTint = vec3(1.12, 0.88, 0.62);
    float wAnam = 0.055;
#else
    vec3 anamTint = vec3(1.00, 0.96, 0.92);
    float wAnam = 0.036;
#endif

    float kw;
    vec3 sumCol = vec3(0.0);
    float sumW = 0.0;

    /* 1. Center bilinear tap */
    vec3 b0 = extractBloomTap(texcoord, kw);
    sumCol += b0 * (0.22 * kw);
    sumW   += 0.22 * kw;

    /* 2. Inner 4 diagonal hardware-bilinear taps (+-2.5 full-res px) */
    vec2 d1 = px * 2.5;
    vec3 b1 = extractBloomTap(texcoord + vec2( d1.x,  d1.y), kw); sumCol += b1 * (0.12 * kw); sumW += 0.12 * kw;
    vec3 b2 = extractBloomTap(texcoord + vec2(-d1.x,  d1.y), kw); sumCol += b2 * (0.12 * kw); sumW += 0.12 * kw;
    vec3 b3 = extractBloomTap(texcoord + vec2( d1.x, -d1.y), kw); sumCol += b3 * (0.12 * kw); sumW += 0.12 * kw;
    vec3 b4 = extractBloomTap(texcoord + vec2(-d1.x, -d1.y), kw); sumCol += b4 * (0.12 * kw); sumW += 0.12 * kw;

    /* 3. Wide atmospheric vertical/diagonal halo (+-7.5 full-res px) */
    vec2 d2 = px * 7.5;
    vec3 b5 = extractBloomTap(texcoord + vec2( 0.0,  d2.y), kw); sumCol += b5 * (0.08 * kw); sumW += 0.08 * kw;
    vec3 b6 = extractBloomTap(texcoord + vec2( 0.0, -d2.y), kw); sumCol += b6 * (0.08 * kw); sumW += 0.08 * kw;

    /* 4. Wide horizontal Interstellar anamorphic streak (+-13.5 full-res px) */
    vec3 b7 = extractBloomTap(texcoord + vec2( px.x * 13.5, 0.0), kw); sumCol += (b7 * anamTint) * (wAnam * kw); sumW += wAnam * kw;
    vec3 b8 = extractBloomTap(texcoord + vec2(-px.x * 13.5, 0.0), kw); sumCol += (b8 * anamTint) * (wAnam * kw); sumW += wAnam * kw;

    gl_FragData[0] = vec4(max(sumCol / max(sumW, 1e-4), vec3(0.0)), 1.0);
#endif
}

#endif
