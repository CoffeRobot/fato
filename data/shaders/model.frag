#version 330 core
in vec2 vertexTexCoord;
in float vertexSegmentIndex;
in vec3 vertexNormal;



uniform sampler2D tex0;

//layout(location=0) out float out_grayscale;
layout(location=0) out vec4 color;
layout(location=1) out float out_z_buffer;
layout(location=2) out float out_segment_ind;
layout(location=3) out float out_normal_x;
layout(location=4) out float out_normal_y;
layout(location=5) out float out_normal_z;




void main()
{
    // no longer guaranteed unit length due to interpolation
    vec3 normal = normalize(vertexNormal);
    out_normal_x = normal.x;
    out_normal_y = normal.y;
    out_normal_z = normal.z;
    out_z_buffer = gl_FragCoord.z;
    out_segment_ind = vertexSegmentIndex;

    color = vec4(texture(tex0, vertexTexCoord));
    //out_grayscale = 0.299f*color.x + 0.587f*color.y + 0.114f*color.z;

    // add 1 to more easily identify rendering mask (will be equal to zero)
    //out_grayscale += 1.0f;
}
