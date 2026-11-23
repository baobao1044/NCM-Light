#version 120

varying vec2 texcoord;
varying vec4 glcolor;

void main() {
    gl_Position = ftransform();
    /* Smooth C-infinity hyperbolic shadow distortion (4.4x texel density near player,
       zero origin kink on large Sodium greedy-meshed quads!) */
    float r = sqrt(dot(gl_Position.xy, gl_Position.xy) + 0.0016);
    float distortFactor = mix(1.0, r, 0.78);
    gl_Position.xy /= distortFactor;
    gl_Position.z *= 0.5;

    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    glcolor = gl_Color;
}
