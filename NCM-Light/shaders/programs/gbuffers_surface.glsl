/*
    Opaque / cutout surface stage writing attachments 0 and 3:
      gbuffers_terrain, gbuffers_block, gbuffers_entities.

    Outputs clean un-tinted albedo * vertexAO into colortex0, and packed
    world normal + blockLight + skyLight + materialID into colortex3 (RGBA8).
    Automatically detects labPBR 1.3 _n and _s textures when a PBR resource pack
    is enabled, while falling back seamlessly to IntegratedPBR otherwise.
*/

#include "/options.glsl"
#include "/programs/gbuffers_common.glsl"
#include "/programs/waving.glsl"

#ifdef STAGE_VSH

attribute vec4 at_tangent;
#ifdef USE_BLOCK_ID
attribute vec4 mc_Entity;
attribute vec2 mc_midTexCoord;
uniform mat4 gbufferModelView;
uniform vec3 cameraPosition;
uniform float frameTimeCounter;
uniform float rainStrength;
#endif

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying vec3 worldNormal;
varying vec3 worldTangent;
varying float tangentSign;
varying float matTag;

void main() {
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    glcolor = gl_Color;
    worldNormal = safeWorldNormal(gl_NormalMatrix * gl_Normal);
    worldTangent = safeWorldNormal(gl_NormalMatrix * at_tangent.xyz);
    tangentSign = (at_tangent.w < 0.0) ? -1.0 : 1.0;

#ifdef USE_BLOCK_ID
    matTag = classifyMaterialTag(mc_Entity.x);
    vec4 vView = gl_ModelViewMatrix * gl_Vertex;
#if WAVING_FOLIAGE
    vec3 pPos = mat3(gbufferModelViewInverse) * vView.xyz + gbufferModelViewInverse[3].xyz;
    pPos += computeFoliageWave(pPos + cameraPosition, mc_Entity.x, texcoord.y, mc_midTexCoord.y, frameTimeCounter, rainStrength);
    vView.xyz = mat3(gbufferModelView) * pPos + gbufferModelView[3].xyz;
#endif
    gl_Position = gl_ProjectionMatrix * vView;
#else
    matTag = 1.0;
    gl_Position = ftransform();
#endif
}

#endif

#ifdef STAGE_FSH

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;
#ifdef USE_ENTITY_COLOR
uniform vec4 entityColor;
#endif

varying vec2 texcoord;
varying vec2 lmcoord;
varying vec4 glcolor;
varying vec3 worldNormal;
varying vec3 worldTangent;
varying float tangentSign;
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

    vec3 finalWorldN = normalize(worldNormal);
    float finalMat = floor(matTag + 0.5);

#if LABPBR_SUPPORT
    vec4 normTex = texture2D(normals, texcoord);
    vec4 specTex = texture2D(specular, texcoord);

    vec2 nCentered = normTex.xy * 2.0 - 1.0;
    if (dot(normTex.xy, normTex.xy) > 0.01 && dot(nCentered, nCentered) > 0.0008 && dot(worldTangent, worldTangent) > 0.2) {
        float nz = sqrt(max(1.0 - dot(nCentered, nCentered), 0.0));
        vec3 T = normalize(worldTangent - finalWorldN * dot(finalWorldN, worldTangent));
        vec3 B = cross(finalWorldN, T) * tangentSign;
        finalWorldN = normalize(T * nCentered.x + B * nCentered.y + finalWorldN * nz);
        if (normTex.z > 0.02) {
            color.rgb *= mix(1.0, clamp(normTex.z, 0.25, 1.0), 0.75);
        }
    }

    if ((specTex.r + specTex.g) > 0.008 || (specTex.a > 0.02 && specTex.a < 0.996)) {
        if (specTex.a > 0.02 && specTex.a < 0.996) {
            finalMat = 5.0;
        } else if (specTex.g >= 0.898) {
            finalMat = 4.0;
        } else if (specTex.b >= 0.255 && finalMat < 1.5) {
            finalMat = 7.0;
        } else if (specTex.r >= 0.52 && finalMat < 5.5) {
            finalMat = 3.0;
        }
    }
#endif

    gl_FragData[0] = vec4(color.rgb, 1.0);
    gl_FragData[1] = packSurfaceData(finalWorldN, lmcoord, finalMat);
}

#endif
