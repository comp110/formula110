"""Small GLSL studio shaders; no external rendering pipeline is needed."""

VERTEX = """#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;
varying vec3 world;
varying vec3 normal;
varying vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    world = (p3d_ModelMatrix * p3d_Vertex).xyz;
    mat3 m = mat3(p3d_ModelMatrix);
    vec3 a = cross(m[1],m[2]);
    vec3 b = cross(m[2],m[0]);
    vec3 c = cross(m[0],m[1]);
    normal = (a*p3d_Normal.x + b*p3d_Normal.y + c*p3d_Normal.z) / dot(m[0],a);
    uv = p3d_MultiTexCoord0;
}
"""

SURFACE = """#version 120
uniform vec4 p3d_ColorScale;
uniform vec3 eye;
uniform float phase;
uniform float reflection;
uniform float gloss;
varying vec3 world;
varying vec3 normal;
varying vec2 uv;
vec3 softbox(vec3 N, vec3 V, vec3 L, vec3 tint, vec3 paint, float power) {
    float diffuse = max(dot(N,L),0.0);
    float spec = pow(max(dot(N,normalize(L+V)),0.0),power);
    return tint * (paint * diffuse + spec * gloss * .62);
}
void main() {
    vec3 N = normalize(normal);
    vec3 V = normalize(eye-world);
    vec3 paint = p3d_ColorScale.rgb;
    vec3 col = paint * vec3(.11,.13,.17);
    col += softbox(N,V,normalize(vec3(-.45,.85,-.75)),vec3(.90,.89,.87),paint,52.0);
    col += softbox(N,V,normalize(vec3(.8,.35,.35)),vec3(.30,.44,.64),paint,70.0);
    col += softbox(N,V,normalize(vec3(.25,.55,-1.0)),vec3(.16,.18,.23),paint,32.0);
    vec3 moving = normalize(vec3(6.0*sin(phase),4.5,-8.0)-world);
    col += softbox(N,V,moving,vec3(.30,.34,.40),paint,48.0);
    // Long studio softboxes produce broad clear-coat reflections on curved parts.
    vec3 R = reflect(-V,N);
    float box = exp(-pow((R.x-.30*sin(phase))/.16,2.0)-pow((R.y-.2)/.65,4.0));
    box *= smoothstep(.15,.8,-R.z);
    col += vec3(.32,.36,.42)*box*gloss;
    float strip = exp(-pow((world.x-7.0*cos(phase)+.5*world.y)/.85,2.0));
    col += vec3(.22,.27,.33)*strip*pow(abs(N.z),8.0)*gloss;
    float fresnel = pow(1.0-max(dot(N,V),0.0),3.0);
    col += vec3(.16,.28,.42) * fresnel * gloss;
    if (reflection > .5) {
        col *= .32 * exp(min(world.y,0.0)*.65);
        col = mix(col,vec3(.005,.010,.019),clamp(-world.y*.12,0.0,.75));
    }
    // Gentle highlight roll-off retains the Carolina blue paint.
    col = col / (1.0 + col * .23);
    gl_FragColor = vec4(col,p3d_ColorScale.a);
}
"""

FLOOR = """#version 120
uniform sampler2D argyle_texture;
uniform float argyle_aspect;
uniform vec2 lane_bounds;
uniform vec3 eye;
uniform float car_x;
uniform float car_z;
uniform float car_heading;
uniform float phase;
varying vec3 world;
varying vec3 normal;
varying vec2 uv;
void main() {
    float pool = exp(-dot(world.xz*vec2(.12,.19),world.xz*vec2(.12,.19)));
    float lane = exp(-pow((world.z-.3)*.7,2.0));
    vec3 col = vec3(.006,.010,.018)+vec3(.014,.028,.044)*pool;
    col += vec3(.006,.013,.020)*lane;
    // Soft reflected pools move with the lighting rig, never with the camera.
    float left = exp(-pow((world.x+5.0+sin(phase))*.42,2.0)-pow((world.z-1.0)*.28,2.0));
    float right = exp(-pow((world.x-5.0+sin(phase))*.42,2.0)-pow((world.z-1.0)*.28,2.0));
    col += vec3(.028,.065,.12)*(left+right);
    // Inlaid argyle spans exactly the two rails. World-space UVs keep it
    // fixed to the floor, with the original artwork's aspect ratio intact.
    float lane_width = lane_bounds.y-lane_bounds.x;
    float inlay = smoothstep(lane_bounds.x,lane_bounds.x+.018,world.z);
    inlay *= 1.0-smoothstep(lane_bounds.y-.018,lane_bounds.y,world.z);
    vec2 inlay_uv = vec2(world.x/(lane_width*argyle_aspect)+.5,
                        (world.z-lane_bounds.x)/lane_width);
    vec4 argyle = texture2D(argyle_texture,inlay_uv);
    vec3 lacquer = mix(vec3(.10,.15,.21),argyle.rgb*.45,argyle.a);
    lacquer *= .65+.35*pool;
    lacquer += vec3(.016,.032,.055)*(left+right);
    // A smooth clear coat catches the studio softboxes above the printed ink.
    vec3 V = normalize(eye-world);
    vec3 R = reflect(-V,vec3(0,1,0));
    vec3 L1 = normalize(vec3(-5.5+2.0*sin(phase),4.0,14.0)-world);
    vec3 L2 = normalize(vec3(6.0+cos(phase),5.0,18.0)-world);
    float shine = pow(max(dot(R,L1),0.0),90.0);
    shine += .65*pow(max(dot(R,L2),0.0),120.0);
    lacquer += vec3(.20,.27,.35)*shine;
    col = mix(col,lacquer,inlay);
    vec2 delta = world.xz - vec2(car_x,car_z);
    float forward = dot(delta,vec2(sin(car_heading),cos(car_heading)));
    float lateral = dot(delta,vec2(cos(car_heading),-sin(car_heading)));
    float shadow = exp(-pow(forward/1.7,2.0)-pow(lateral/.55,2.0));
    col *= 1.0-.9*shadow;
    float coverage = 1.0-smoothstep(8.0,30.0,world.z);
    gl_FragColor = vec4(col,(.58 + .4*shadow)*coverage);
}
"""

GLOW = """#version 120
uniform vec4 p3d_ColorScale;
varying vec2 uv;
void main() {
    float r = length((uv-.5)*2.0);
    float falloff = exp(-r*r*5.0)*smoothstep(1.0,.55,r);
    vec3 col = p3d_ColorScale.rgb * p3d_ColorScale.a * falloff;
    float dither = fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453)-.5;
    gl_FragColor = vec4(max(vec3(0.0),col+dither/255.0),1.0);
}
"""

BEAM = """#version 120
uniform vec4 p3d_ColorScale;
varying vec2 uv;
void main() {
    float width = abs(uv.x-.5)*2.0;
    float across = (exp(-width*width*5.0)+.12*exp(-width*width*32.0));
    across *= 1.0-smoothstep(.6,1.0,width);
    float along = pow(1.0-uv.y,1.35)*smoothstep(0.0,.025,uv.y);
    float dither = fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453)-.5;
    vec3 col = p3d_ColorScale.rgb*p3d_ColorScale.a*across*along;
    gl_FragColor = vec4(max(vec3(0.0),col+dither/512.0),1.0);
}
"""

SCREEN_VERTEX = """#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
varying vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix*p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
"""

SMOKE = """#version 120
uniform vec4 p3d_ColorScale;
varying vec2 uv;
void main() {
    vec2 p = (uv-.5)*2.0;
    p += vec2(.09*sin(p.y*8.0),.08*sin(p.x*7.0));
    float r = length(p);
    float density = exp(-r*r*3.0)*(1.0-smoothstep(.65,1.0,r));
    gl_FragColor = vec4(p3d_ColorScale.rgb,p3d_ColorScale.a*density);
}
"""

FINISH = """#version 120
uniform sampler2D tex;
uniform vec2 pixel_size;
varying vec2 uv;
vec3 highlight(vec2 offset) {
    vec3 col = texture2D(tex,uv+pixel_size*offset).rgb;
    float lightness = dot(col,vec3(.2126,.7152,.0722));
    return col*smoothstep(.52,.95,lightness);
}
void main() {
    vec3 col = texture2D(tex,uv).rgb;
    vec3 bloom = highlight(vec2(3,0))+highlight(vec2(-3,0));
    bloom += highlight(vec2(0,3))+highlight(vec2(0,-3));
    bloom += highlight(vec2(7,7))+highlight(vec2(-7,7));
    bloom += highlight(vec2(7,-7))+highlight(vec2(-7,-7));
    col += bloom*.025;
    vec2 edge = (uv-.5)*2.0;
    col *= 1.0-.12*smoothstep(.35,1.35,dot(edge,edge));
    gl_FragColor = vec4(col,1.0);
}
"""
