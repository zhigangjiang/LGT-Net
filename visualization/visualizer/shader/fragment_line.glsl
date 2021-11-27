#version 410
#define pi 3.14159265359
layout(location = 0) out vec4 fragColor;
uniform int um4f;

void main(){
    if (um4f==0)
        fragColor = vec4(vec3(255, 250, 84)/255.0, 1.0);
    else if(um4f==1)
        fragColor = vec4(0, 0, 1, 1.0);
    else
        fragColor = vec4(vec3(154, 255, 154)/255.0, 1.0);

    fragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
