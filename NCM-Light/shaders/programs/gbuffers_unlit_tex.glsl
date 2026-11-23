/*
    Unlit textured gbuffer stage:
      gbuffers_textured, gbuffers_skytextured, gbuffers_armor_glint,
      gbuffers_spidereyes, gbuffers_beaconbeam, gbuffers_damagedblock.
*/

#ifdef STAGE_VSH

varying vec2 texcoord;
varying vec4 glcolor;

void main() {
    gl_Position = ftransform();
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    glcolor = gl_Color;
}

#endif

#ifdef STAGE_FSH

uniform sampler2D gtexture;

varying vec2 texcoord;
varying vec4 glcolor;

void main() {
    vec4 color = texture2D(gtexture, texcoord) * glcolor;
#ifdef DISCARD_ALPHA
    if (color.a < DISCARD_ALPHA) discard;
#endif
    gl_FragData[0] = color;
}

#endif
