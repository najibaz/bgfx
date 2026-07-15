$input a_position, i_data0, i_data1, i_data2, i_data3, i_data4

/*
 * Depth-only instanced vertex shader for the Hi-Z occluder pass.
 * (No fragment shader / no varyings: renders to a depth-only target.)
 */

#include "../common/common.sh"

void main()
{
	mat4 model = mtxFromCols(i_data0, i_data1, i_data2, i_data3);
	vec4 wpos = mul(model, vec4(a_position, 1.0));
	gl_Position = mul(u_viewProj, wpos);
}
