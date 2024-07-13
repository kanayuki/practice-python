#version 330
# define pi 3.141592653589793238462643383279502

in vec2 v_vert;
in vec3 v_color;
uniform float time;
out vec3 f_color;

mat2 rotateMatrix(float theta)
{
    return mat2(cos(theta),-sin(theta),
                sin(theta),cos(theta));
}

float sdTorus(vec3 p, float r1, float r2)
{
    float d = length(p.xy)-r1;
    return length(vec2(d,p.z))-r2;
}

float sdCylinder(vec3 p, float r, float h)
{
    float d = length(p.xy)-r;
    vec2 w = vec2(d, abs(p.z)-h);
    return min(max(w.x,w.y), 0.0) + length(max(w,0.0));
}

float sdf(vec3 p)
{
    p.z-=0;
    p.xz *= rotateMatrix(time);
    p.xy *= rotateMatrix(time);

    // Symmetries
    p = abs(p);
    if(p.x>p.y) p.xy=p.yx;
    if(p.y>p.z) p.yz=p.zy;
    if(p.x>p.y) p.xy=p.yx;

    float cy = sdCylinder(p,0.3,3.0);
    float t = sdTorus(p.zyx,3.0,0.5);

    return min(cy,t);
 }

float rayMarching(vec3 raySource, vec3 direction)
{
    float d = 0.0;
    for (int i=0; i<100; i++)
    {
        float len = sdf(raySource+direction*d);
        if (len<0.001 || d>30.0) break;
        d += len;
    }
    return d;
}

vec3 getNormal(vec3 p)
{
    vec2 e = vec2(0.0001,0);
    float d = sdf(p);
    float pdx=sdf(p+e.xyy)-d;
    float pdy=sdf(p+e.yxy)-d;
    float pdz=sdf(p+e.yyx)-d;
    return normalize(vec3(pdx,pdy,pdz));

}



void main()
{
    // vec2 fragCoord = v_vert.xy;

    // Normalized pixel coordinates (from 0 to 1)
    float iTime = time;
    vec2 uv = v_vert*8;

    vec3 col = vec3(0.75);
    vec3 raySource = vec3(0,0,20);

    vec3 screen = vec3(uv,1.0);

    vec3 direction = normalize(screen-raySource);

    float d = rayMarching(raySource, direction );
    if(d<30.0)
    {
        vec3 p = raySource+direction*d;
        vec3 N = getNormal(p);
        vec3 light = vec3(5,0,5);
        // light.xz *= rotateMatrix(iTime);

        vec3 L = normalize(light-p);

        float diffuse = dot(N,L);
        // Lambert
        // diffuse = max(0,diffuse);
        // Half Lambert
        diffuse = diffuse * 0.5 + 0.5;


        // Phong
        vec3 R = normalize(2.0*N*dot(N,L)-L);
        vec3 V = -direction;
        // vec3 specular = 0.1*vec3(1)*pow(dot(R,V),8.0);

        // Blinn
        vec3 H = normalize(V+L);
        vec3 specular = 0.9*vec3(1)*pow(dot(H,N),5.0);


        vec3 ambient = vec3(0.1,0.1,0.3)*1.5;

        col = vec3(1,0,1);
        col = ambient + 0.5*col*diffuse + specular;
    }

    // Time varying pixel color
    // col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    // fragColor = vec4(col,1.0);
    // f_color = col;
    // f_color = v_color;
    f_color = col + v_color * 0.01;
    // gl_Position = vec4(in_vert * 1.0, 0.0, 1.0);
}