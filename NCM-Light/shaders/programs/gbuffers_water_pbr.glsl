/*
    Translucent terrain stage (gbuffers_water: water, stained glass, ice).
    Writes clean albedo to colortex0 and 8-bit safe [0..1] tag to colortex2:
      .x = 0.40 for liquid water, 0.80 for glass/ice (0.0 when untouched)
      .y = blockLight [0..1]
      .z = skyLight   [0..1]
      .w = 1.0 (unblended overwrite under GL_SRC_ALPHA)
*/

#ifdef STAGE_VSH

attribute vec4 mc_Entity;

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying float transTagX;

void main() {
    gl_Position = ftransform();
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    glcolor = gl_Color;
    transTagX = (abs(mc_Entity.x - 10500.0) < 0.5) ? 0.40 : 0.80;
}

#endif

#ifdef STAGE_FSH

uniform sampler2D gtexture;

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying float transTagX;

void main() {
    vec4 color = texture2D(gtexture, texcoord) * glcolor;
    if (color.a < 0.01) discard;

    float blockLight = clamp((lmcoord.x - 0.03125) / 0.9375, 0.0, 1.0);
    float skyLight   = clamp((lmcoord.y - 0.03125) / 0.9375, 0.0, 1.0);

    gl_FragData[0] = color;
    gl_FragData[1] = vec4(transTagX, blockLight, skyLight, 1.0);
}

#endif
