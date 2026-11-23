/*
    Lightmap-lit single-target (colortex0 only) gbuffer stage:
      gbuffers_textured_lit (lit particles), gbuffers_water, gbuffers_weather,
      gbuffers_hand, gbuffers_hand_water, gbuffers_block_translucent,
      gbuffers_entities_translucent.

    Writes ONLY attachment 0 so fixed-function alpha blending never corrupts
    surface normals in colortex3.
*/

#include "/programs/gbuffers_common.glsl"

#ifdef STAGE_VSH

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying vec3 worldNormal;

void main() {
    gl_Position = ftransform();
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    glcolor = gl_Color;
    worldNormal = safeWorldNormal(gl_NormalMatrix * gl_Normal);
}

#endif

#ifdef STAGE_FSH

uniform sampler2D gtexture;
uniform sampler2D lightmap;
#ifdef USE_ENTITY_COLOR
uniform vec4 entityColor;
#endif

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying vec3 worldNormal;

void main() {
    vec4 color = texture2D(gtexture, texcoord) * glcolor;
#ifdef DISCARD_ALPHA
    if (color.a < DISCARD_ALPHA) discard;
#endif
#ifdef USE_ENTITY_COLOR
    color.rgb = mix(color.rgb, entityColor.rgb, entityColor.a);
#endif
#ifdef USE_DIRECTIONAL_SHADE
    color.rgb *= entityDirectionalShade(worldNormal);
#endif
    color.rgb *= texture2D(lightmap, lmcoord).rgb;
    gl_FragData[0] = color;
}

#endif
