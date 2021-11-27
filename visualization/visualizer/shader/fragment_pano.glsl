#version 410
#define pi 3.14159265359
layout(location = 0) out vec4 fragColor;
in vec3 modelPosition;

uniform sampler2D pano;
uniform float alpha;
uniform int wallNum;
uniform vec2 wallPoints[100];


bool intersect1D(float a1, float a2, float b1, float b2) 
{
    if (a1 > a2) 
    {
        float tmp = a1; 
        a1 = a2; 
        a2 = tmp;
    }
    if (b1 > b2) 
    {
        float tmp = b1; 
        b1 =b2;
        b2 = tmp;
    }
    return max(a1, b1) <= min(a2, b2);
}
float cross(vec2 o, vec2 a, vec2 b)
{
    return (a.x-o.x) * (b.y-o.y) - (a.y-o.y) * (b.x-o.x);
}

bool intersect(vec2 a1, vec2 a2, vec2 b1, vec2 b2) 
{
    return intersect1D(a1.x, a2.x, b1.x, b2.x)
        && intersect1D(a1.y, a2.y, b1.y, b2.y)
        && cross(a1, a2, b1) * cross(a1, a2, b2) <= 0
        && cross(b1, b2, a1) * cross(b1, b2, a2) <= 0;
}

bool checkIntersectWalls(vec2 pts){
    vec2 a = pts * 0.99;
    vec2 b = vec2(0, 0);
    for (int i=0; i<wallNum; i++){
        vec2 c = wallPoints[i*2];
        vec2 d = wallPoints[i*2+1];
        //if(min(a.x,b.x)<=max(c.x,d.x) && min(c.y,d.y)<=max(a.y,b.y) && min(c.x,d.x)<=max(a.x,b.x) && min(a.y,b.y)<=max(c.y,d.y))
        //if(min(a.x,b.x)<=max(c.x,d.x) && max(a.x, b.x)>=min(c.x, d.x) && min(a.y,b.y)<=max(c.y,d.y) && max(a.y, b.y)>=min(c.y, d.y))
        if (intersect(a, b, c, d))
            return true;
        /*
        float u=(c.x-a.x)*(b.y-a.y)-(b.x-a.x)*(c.y-a.y);
        float v=(d.x-a.x)*(b.y-a.y)-(b.x-a.x)*(d.y-a.y);
        float w=(a.x-c.x)*(d.y-c.y)-(d.x-c.x)*(a.y-c.y);
        float z=(b.x-c.x)*(d.y-c.y)-(d.x-c.x)*(b.y-c.y);
        return (u*v<=1e-5 && w*z<=1e-5);
        */
    }

    return false;
}


void main(){
    float x = modelPosition.x;
    float y = modelPosition.y;
    float z = modelPosition.z;
    float normXYZ = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
    float normXZ = sqrt(pow(x, 2) + pow(z, 2));
    float lon = (atan(x, z) / pi + 1) * 0.5;
    float lat = (asin(y / normXYZ) / (0.5*pi) + 1) * 0.5;
    vec2 coord = vec2(lon, lat);
    if (!checkIntersectWalls(vec2(x, z)))
    //if (true)
        fragColor = vec4(texture(pano, coord).xyz, alpha);
    else{
        if (mod(y * 10, 10) < 5 ^^ mod(x * 10, 10) < 5 ^^ mod(z * 10, 10) < 5)
            fragColor = vec4(vec3(1.0, 1.0, 1.0), alpha);
        else
            fragColor = vec4(vec3(0.5, 0.5, 0.5), alpha);
    }
}
