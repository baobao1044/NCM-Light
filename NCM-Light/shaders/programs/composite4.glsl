/*
    Quarter-Resolution Cinematic HDR Bloom Pass 2 (1/16th screen pixels):
      * 13-tap Dual-Scale Wide Kawase Blur + Subtle Interstellar Anamorphic Horizontal Streak!
      * Runs at 0.25x (only 0.81 taps/screen-px total!) for silky IterationT-class glow on iGPUs.
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
#if !BLOOM
    gl_FragData[0] = vec4(0.0, 0.0, 0.0, 1.0);
#else
    vec2 qStep = 1.0 / max(floor(vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)) * 0.25), vec2(1.0));

    vec3 acc = texture2D(colortex6, texcoord).rgb * 0.18;
    float wSum = 0.18;

    /* Tight inner halo (+-1.5 quarter-res texels = +-6 full-res px) */
    vec2 d1 = qStep * 1.50;
    acc += texture2D(colortex6, clamp(texcoord + vec2( d1.x,  d1.y), 0.002, 0.998)).rgb * 0.11;
    acc += texture2D(colortex6, clamp(texcoord + vec2(-d1.x,  d1.y), 0.002, 0.998)).rgb * 0.11;
    acc += texture2D(colortex6, clamp(texcoord + vec2( d1.x, -d1.y), 0.002, 0.998)).rgb * 0.11;
    acc += texture2D(colortex6, clamp(texcoord + vec2(-d1.x, -d1.y), 0.002, 0.998)).rgb * 0.11;
    wSum += 0.44;

    /* Wide photographic atmospheric halo (+-3.6 quarter-res texels = +-14.4 full-res px) */
    vec2 d2 = qStep * 3.60;
    acc += texture2D(colortex6, clamp(texcoord + vec2( d2.x,  0.0), 0.002, 0.998)).rgb * 0.065;
    acc += texture2D(colortex6, clamp(texcoord + vec2(-d2.x,  0.0), 0.002, 0.998)).rgb * 0.065;
    acc += texture2D(colortex6, clamp(texcoord + vec2( 0.0,  d2.y), 0.002, 0.998)).rgb * 0.065;
    acc += texture2D(colortex6, clamp(texcoord + vec2( 0.0, -d2.y), 0.002, 0.998)).rgb * 0.065;
    wSum += 0.26;

    /* Subtle Interstellar horizontal anamorphic optical streak (+-6.5 & +-10.5 quarter-res px) */
#if defined(DIMENSION_END)
    vec3 anamTint = vec3(1.12, 0.88, 0.62);
    float wA1 = 0.038;
    float wA2 = 0.022;
#else
    vec3 anamTint = vec3(1.00, 0.96, 0.92);
    float wA1 = 0.025;
    float wA2 = 0.015;
#endif
    vec3 streak = (texture2D(colortex6, clamp(texcoord + vec2( qStep.x * 6.5, 0.0), 0.002, 0.998)).rgb
                 + texture2D(colortex6, clamp(texcoord + vec2(-qStep.x * 6.5, 0.0), 0.002, 0.998)).rgb) * wA1
                + (texture2D(colortex6, clamp(texcoord + vec2( qStep.x * 10.5, 0.0), 0.002, 0.998)).rgb
                 + texture2D(colortex6, clamp(texcoord + vec2(-qStep.x * 10.5, 0.0), 0.002, 0.998)).rgb) * wA2;
    acc += streak * anamTint;
    wSum += 2.0 * (wA1 + wA2);

    gl_FragData[0] = vec4(max(acc / wSum, vec3(0.0)), 1.0);
#endif
}

#endif
