#version 120

uniform sampler2D gtexture;

varying vec2 texcoord;
varying vec4 glcolor;
varying vec3 vWorldNormal;
varying vec2 vLightInfo;

vec2 octEncodeShadowNormal(vec3 n) {
    float l1 = max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    vec2 p = n.xy / l1;
    if (n.z < 0.0) {
        vec2 s = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        p = (1.0 - abs(p.yx)) * s;
    }
    return p * 0.5 + 0.5;
}

void main() {
    vec4 color = texture2D(gtexture, texcoord) * glcolor;
    if (color.a < 0.1) discard;
    vec3 wn = normalize(vWorldNormal);
    gl_FragData[0] = color;
    gl_FragData[1] = vec4(octEncodeShadowNormal(wn), vLightInfo.x, vLightInfo.y);
}
