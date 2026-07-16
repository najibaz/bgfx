$input a_position, i_data0, i_data1, i_data2, i_data3, i_data4

/*
 * Depth-only instanced vertex shader for the Hi-Z occluder pass.
 * (No fragment shader / no varyings: renders to a depth-only target.)
 */

#include "../common/common.sh"

void main()
{
	// i_data0.w holds the item index, not a matrix element (see vs_ifc.sc) --
	// clear it, or the transform is corrupted.
	//
	// No nudge here: this pass only feeds the Hi-Z occlusion pyramid, which
	// renders to its own target rather than the main depth buffer, so it never
	// has to match the colour pass bit-for-bit. A sub-mm offset would not change
	// the conservative occlusion test either way.
	mat4 model = mtxFromCols(vec4(i_data0.xyz, 0.0), i_data1, i_data2, i_data3);
	vec4 wpos = mul(model, vec4(a_position, 1.0));
	gl_Position = mul(u_viewProj, wpos);
}
