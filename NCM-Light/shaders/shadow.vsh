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
varying float vIsEmissive;

void main() {
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    glcolor = gl_Color;
    vIsEmissive = (abs(mc_Entity.x - 10200.0) < 0.5) ? 1.0 : 0.0;

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
