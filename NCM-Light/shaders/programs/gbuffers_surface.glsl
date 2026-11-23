/*
    Opaque / cutout surface stage writing attachments 0 and 3:
      gbuffers_terrain, gbuffers_block, gbuffers_entities.

    Outputs clean un-tinted albedo * vertexAO into colortex0 (no muddy vanilla
    lightmap tint!), and full PBR normal + blockLight + skyLight + materialID
    into colortex3 (RGBA16F) for Linear HDR deferred PBR & NCM Ray Tracing.
*/

#include "/programs/gbuffers_common.glsl"

#ifdef STAGE_VSH

#ifdef USE_BLOCK_ID
attribute vec4 mc_Entity;
#endif

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying vec3 worldNormal;
varying float matTag;

void main() {
    gl_Position = ftransform();
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    glcolor = gl_Color;
    worldNormal = safeWorldNormal(gl_NormalMatrix * gl_Normal);
#ifdef USE_BLOCK_ID
    matTag = classifyMaterialTag(mc_Entity.x);
#else
    matTag = 1.0;
#endif
}

#endif

#ifdef STAGE_FSH

uniform sampler2D gtexture;
#ifdef USE_ENTITY_COLOR
uniform vec4 entityColor;
#endif

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying vec3 worldNormal;
varying float matTag;

void main() {
    vec4 color = texture2D(gtexture, texcoord) * glcolor;
    if (color.a < 0.1) discard;

#ifdef USE_ENTITY_COLOR
    color.rgb = mix(color.rgb, entityColor.rgb, entityColor.a);
#endif
#ifdef USE_DIRECTIONAL_SHADE
    color.rgb *= entityDirectionalShade(worldNormal);
#endif

    gl_FragData[0] = vec4(color.rgb, 1.0);
    gl_FragData[1] = packSurfaceData(worldNormal, lmcoord, floor(matTag + 0.5));
}

#endif
