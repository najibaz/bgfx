$input a_position, i_data0, i_data1, i_data2, i_data3, i_data4
$output v_wpos, v_color

/*
 * IFC GPU-driven viewer - instanced vertex shader.
 * Per-instance model matrix is stored in i_data0..i_data3 (columns),
 * per-instance color in i_data4.xyz.
 */

#include "../common/common.sh"

void main()
{
	mat4 model = mtxFromCols(i_data0, i_data1, i_data2, i_data3);

	vec4 wpos = mul(model, vec4(a_position, 1.0));
	v_wpos = wpos.xyz;
	v_color = i_data4;

	gl_Position = mul(u_viewProj, wpos);
}
