#version 120

uniform sampler2D gtexture;

varying vec2 texcoord;
varying vec4 glcolor;
varying float vIsEmissive;

void main() {
    vec4 color = texture2D(gtexture, texcoord) * glcolor;
    if (color.a < 0.1) discard;
    /* Pack emissive tag into alpha=0.965 on opaque light blocks (0 extra MRT attachments!):
       stained glass keeps alpha < 0.92, opaque matte keeps alpha = 1.0! */
    float outAlpha = (vIsEmissive > 0.5 && color.a > 0.95) ? 0.965 : color.a;
    gl_FragData[0] = vec4(color.rgb, outAlpha);
}
