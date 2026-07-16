/*
 * Hi-Z pyramid downscale for the REVERSE-Z pipeline.
 *
 * Private copy of example 37's cs_gdr_downscale_hi_z, reducing with min instead
 * of max. Under reverse-Z the far plane is 0, so the *farthest* occluder in a
 * 2x2 block -- the conservative choice for an occlusion test -- is the smallest
 * value, not the largest.
 *
 * (37's shader is shared with this example's copy_z step and cannot simply be
 * flipped: example 37 is standard-Z and still needs max.)
 */

#include "bgfx_compute.sh"

IMAGE2D_RO(s_texOcclusionDepthIn, r32f, 0);
IMAGE2D_WO(s_texOcclusionDepthOut, r32f, 1);

uniform vec4 u_inputRTSize;

NUM_THREADS(16, 16, 1)
void main()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

	if (all(lessThan(coord.xy, u_inputRTSize.xy) ) )
	{
		vec4 depths = vec4(
			  imageLoad(s_texOcclusionDepthIn, ivec2(u_inputRTSize.zw * coord.xy                   ) ).x
			, imageLoad(s_texOcclusionDepthIn, ivec2(u_inputRTSize.zw * coord.xy + ivec2(1.0, 0.0) ) ).x
			, imageLoad(s_texOcclusionDepthIn, ivec2(u_inputRTSize.zw * coord.xy + ivec2(0.0, 1.0) ) ).x
			, imageLoad(s_texOcclusionDepthIn, ivec2(u_inputRTSize.zw * coord.xy + ivec2(1.0, 1.0) ) ).x
			);

		// reverse-Z: farthest occluder == min depth
		float minDepth = min(
			  min(depths.x, depths.y)
			, min(depths.z, depths.w)
			);

		imageStore(s_texOcclusionDepthOut, coord, vec4(minDepth, 0, 0, 1) );
	}
}
