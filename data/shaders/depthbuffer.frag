#version 410 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoords;

out vec4 color;
uniform sampler2D screenTexture;

float LinearizeDepth(in vec2 uv)
{
    float zNear = 0.01;    // TODO: Replace by the zNear of your perspective projection
    float zFar  = 1000.0; // TODO: Replace by the zFar  of your perspective projection
    float depth = texture2D(screenTexture, uv).x;
    return (2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

void main()
{
    float c = LinearizeDepth(texCoords);
    color = vec4(c, c, c, 1.0);
}
