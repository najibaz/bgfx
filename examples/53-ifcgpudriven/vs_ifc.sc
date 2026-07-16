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
	// i_data0.w carries the draw-item index (set CPU-side), not a matrix element.
	// Force col0.w back to 0 so the transform is exactly the original affine.
	mat4 model = mtxFromCols(vec4(i_data0.xyz, 0.0), i_data1, i_data2, i_data3);

	vec4 wpos = mul(model, vec4(a_position, 1.0));
	v_wpos = wpos.xyz;
	v_color = i_data4;

	gl_Position = mul(u_viewProj, wpos);

	// Anti-z-fighting nudge (ported from the bim-viewer). Near-coplanar
	// duplicates -- the same wall in two federated models, overlapping segments
	// of one element -- either tie exactly or interpenetrate within the SNORM
	// quantization wobble (~1 mm on large elements), which stipples and shimmers
	// as the GPU-compacted draw order shuffles. A multiplicative offset keyed on
	// the stable item index makes the depth order strict and frame-stable.
	//
	// The multiplicative form from the bim-viewer, which works here because the
	// projection is now REVERSE-Z: the shift after the perspective divide is
	// ndc_z * eps, and reverse-Z puts the near plane at ndc_z = 1, so geometry
	// near the camera gets the full offset. (Under standard Z the near plane is
	// ndc_z = 0 and this same form decays to nothing exactly where the geometry
	// is densest -- which is why it did nothing before the reverse-Z change.)
	//
	// 256 levels rather than the bim-viewer's 32, with the step scaled down to
	// keep the same ~3.2e-4 maximum offset (so nothing becomes visible). Two
	// coplanar items only tie if their indices collide modulo the level count, so
	// 32 levels leaves ~3% of coplanar pairs still fighting; 256 leaves ~0.4%.
	// The step stays ~10x the f32 depth quantum at the near plane (ulp(1.0) is
	// 1.2e-7 and reverse-Z puts the near plane at 1.0), so it still separates.
	//
	// Faces that are coplanar *within a single draw item* share the item index,
	// and no per-item nudge can ever separate those.
	gl_Position.z = gl_Position.z * (1.0 + mod(i_data0.w, 256.0) * 1.25e-6);
}
