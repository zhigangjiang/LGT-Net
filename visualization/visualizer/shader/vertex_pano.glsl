#version 410
layout(location = 0) in vec3 iv3vertex;

uniform mat4 um4p;
uniform mat4 um4v;
uniform mat4 um4m;
out vec3 modelPosition;

void main(){
    gl_Position = um4p * um4v * um4m * vec4(iv3vertex, 1.0);
    //gl_Position = vec4(iv3vertex, 1.0);
    modelPosition = iv3vertex;
}
