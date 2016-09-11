#version 330 core
in vec2 vertexTexCoord;
in float vertexSegmentIndex;
in vec3 vertexNormal;

uniform float z_conv1;
uniform float z_conv2;

uniform sampler2D tex0;

layout(location=0) out vec4 color;
layout(location=1) out float depthColor;

void main()
{
    color = vec4(texture(tex0, vertexTexCoord));
    // calculating depth from camera from z buffer and saving in second image
    depthColor = z_conv1 / (gl_FragCoord.z + z_conv2);
}
