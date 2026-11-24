/*
    Translucent terrain stage (gbuffers_water: water, stained glass, ice).
    Writes clean albedo to colortex0 and 8-bit safe [0..1] tag to colortex2:
      .x = 0.25 for flat horizontal water (exact Y-up normal!),
           0.48 for vertical waterfall sides, 0.82 for glass/ice (0.0 when untouched)
      .y = blockLight [0..1]
      .z = skyLight   [0..1]
      .w = 1.0 (unblended overwrite under GL_SRC_ALPHA)
*/

#ifdef STAGE_VSH

attribute vec4 mc_Entity;
uniform mat4 gbufferModelViewInverse;

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying float transTagX;

void main() {
    gl_Position = ftransform();
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    glcolor = gl_Color;
    vec3 vn = gl_NormalMatrix * gl_Normal;
    float vLen2 = dot(vn, vn);
    vec3 wn = (vLen2 > 1e-6) ? normalize(mat3(gbufferModelViewInverse) * (vn * inversesqrt(vLen2))) : vec3(0.0, 1.0, 0.0);
    bool isWater = (abs(mc_Entity.x - 10500.0) < 0.5);
    transTagX = isWater ? ((wn.y > 0.72) ? 0.25 : 0.48) : 0.82;
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
