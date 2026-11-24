#version 120

#include "/options.glsl"
#include "/programs/waving.glsl"

attribute vec4 mc_Entity;
attribute vec2 mc_midTexCoord;

uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform vec3 cameraPosition;
uniform float frameTimeCounter;
uniform float rainStrength;

varying vec2 texcoord;
varying vec4 glcolor;
varying vec3 vWorldNormal;
varying vec2 vLightInfo;

void main() {
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    glcolor = gl_Color;
    vec2 lm = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    float isEmissive = (abs(mc_Entity.x - 10200.0) < 0.5) ? 1.0 : 0.0;
    float skyL = clamp((lm.y - 0.03125) / 0.9375, 0.0, 1.0);
    vLightInfo = vec2(isEmissive, skyL);

    vec3 sNorm = gl_NormalMatrix * gl_Normal;
    float snLen2 = dot(sNorm, sNorm);
    vec3 wNorm = (snLen2 > 1e-6) ? (mat3(shadowModelViewInverse) * (sNorm * inversesqrt(snLen2))) : vec3(0.0, 1.0, 0.0);
    float wnLen2 = dot(wNorm, wNorm);
    vWorldNormal = (wnLen2 > 1e-6) ? (wNorm * inversesqrt(wnLen2)) : vec3(0.0, 1.0, 0.0);

    vec4 sView = gl_ModelViewMatrix * gl_Vertex;
#if WAVING_FOLIAGE
    vec3 pPos = mat3(shadowModelViewInverse) * sView.xyz + shadowModelViewInverse[3].xyz;
    pPos += computeFoliageWave(pPos + cameraPosition, mc_Entity.x, texcoord.y, mc_midTexCoord.y, frameTimeCounter, rainStrength);
    sView.xyz = mat3(shadowModelView) * pPos + shadowModelView[3].xyz;
#endif
    gl_Position = gl_ProjectionMatrix * sView;

    /* Smooth C-infinity hyperbolic shadow distortion (4.4x texel density near player,
       zero origin kink on large Sodium greedy-meshed quads!) */
    float r = sqrt(dot(gl_Position.xy, gl_Position.xy) + 0.0016);
    float distortFactor = mix(1.0, r, SHADOW_DISTORT);
    gl_Position.xy /= distortFactor;
    gl_Position.z *= 0.5;
}
