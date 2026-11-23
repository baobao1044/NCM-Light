/*
    Untextured vertex-colour gbuffer stage (gbuffers_basic, gbuffers_skybasic).
    Uses ftransform() so Iris's VanillaTransformer automatically injects thick
    line expansion when compiling for 3D block-outline / debug lines.
*/

#ifdef STAGE_VSH

varying vec4 glcolor;

void main() {
    gl_Position = ftransform();
    glcolor = gl_Color;
}

#endif

#ifdef STAGE_FSH

varying vec4 glcolor;

void main() {
#ifdef DISCARD_ALPHA
    if (glcolor.a < DISCARD_ALPHA) discard;
#endif
    gl_FragData[0] = glcolor;
}

#endif
