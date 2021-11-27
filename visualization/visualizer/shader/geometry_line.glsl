#version 410

layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 um4p;
uniform mat4 um4v;
uniform mat4 um4m;

void main(){
    float thickness = 0.04;
    float r = thickness / 2;
    mat4 mv = um4v * um4m;
    vec4 p1 = mv * gl_in[0].gl_Position;
    vec4 p2 = mv * gl_in[1].gl_Position;
    vec2 tmp = p2.xy - p1.xy;
    vec2 dir = normalize(p2.xy - p1.xy);
        if (tmp.x+tmp.y==0)
            dir = vec2(0, 1);
    vec2 normal = vec2(dir.y, -dir.x);
    vec4 offset1, offset2;
    offset1 = vec4(normal * r, 0, 0);
    offset2 = vec4(normal * r, 0, 0);

    vec4 coords[4];
    coords[0] = p1 + offset1;
    coords[1] = p1 - offset1;
    coords[2] = p2 + offset2;
    coords[3] = p2 - offset2;
    for (int i = 0; i < 4; ++i) {
        coords[i] = um4p * coords[i];
        gl_Position = coords[i];
        EmitVertex();
    }
    EndPrimitive();
}
