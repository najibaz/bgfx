/*
 * GPU-driven culling + command compaction (frustum + Hi-Z occlusion).
 *
 * One thread per instance. Survivors grab a slot in the indirect command
 * buffer via an atomic counter (atomic-append compaction) and write their own
 * DrawIndexedIndirect command. The render pass then draws exactly the visible
 * count via a GPU-provided draw-count buffer.
 *
 * Occlusion uses a hierarchical-Z pyramid built from the previous frame's
 * visible set (temporal occluders). Adapted from example 37 (cs_gdr_occlude_props).
 *
 * u_cullParams.x = instance count
 * u_cullParams.y = frustum cull enable (0/1)
 * u_cullParams.z = occlusion cull enable (0/1)
 * u_cullParams.w = Hi-Z mip count
 * u_hiZSize.xy   = Hi-Z base size in texels; .zw = 1/size
 * u_viewProj     = provided by bgfx::setViewTransform on the dispatch view
 */

#include "bgfx_compute.sh"

SAMPLER2D(s_hiZ, 0);
BUFFER_RO(instanceDrawParams, uvec4, 1);  // indexCount, firstIndex, baseVertex, transparentFlag
BUFFER_RO(instanceBBox,       vec4, 2);   // [2i]=min.xyz, [2i+1]=max.xyz (world space)
BUFFER_RW(opaqueCount,        uint, 3);   // opaque atomic counter
BUFFER_WO(opaqueIndirect,     uvec4, 4);  // opaque DrawIndexedIndirect commands
BUFFER_RW(transpCount,        uint, 5);   // transparent atomic counter
BUFFER_WO(transpIndirect,     uvec4, 6);  // transparent DrawIndexedIndirect commands
BUFFER_RW(occCount,           uint, 7);   // occluder atomic counter
BUFFER_WO(occIndirect,        uvec4, 8);  // occluder commands (Hi-Z depth pass)

uniform vec4 u_cullParams;
uniform vec4 u_hiZSize;
uniform vec4 u_screenPixels;  // xy = screen size (px), z = min projected size (px) to keep

NUM_THREADS(64, 1, 1)
void main()
{
	uint idx = gl_GlobalInvocationID.x;
	if (idx >= uint(u_cullParams.x))
		return;

	vec3 bmin = instanceBBox[2u * idx].xyz;
	vec3 bmax = instanceBBox[2u * idx + 1u].xyz;

	vec3 corners[8];
	corners[0] = vec3(bmin.x, bmin.y, bmin.z);
	corners[1] = vec3(bmax.x, bmin.y, bmin.z);
	corners[2] = vec3(bmin.x, bmax.y, bmin.z);
	corners[3] = vec3(bmax.x, bmax.y, bmin.z);
	corners[4] = vec3(bmin.x, bmin.y, bmax.z);
	corners[5] = vec3(bmax.x, bmin.y, bmax.z);
	corners[6] = vec3(bmin.x, bmax.y, bmax.z);
	corners[7] = vec3(bmax.x, bmax.y, bmax.z);

	bool doFrustum   = u_cullParams.y > 0.5;
	bool doOcclusion = u_cullParams.z > 0.5;

	int outL = 0, outR = 0, outB = 0, outT = 0, outN = 0, outF = 0;
	// REVERSE-Z: near is 1 and far is 0, so the NEAREST corner is the one with
	// the LARGEST depth -- track a max, seeded at the far plane.
	float nearZ = 0.0;
	vec2 minXY = vec2(1.0, 1.0);
	vec2 maxXY = vec2(0.0, 0.0);

	for (int i = 0; i < 8; i++)
	{
		vec4 c = mul(u_viewProj, vec4(corners[i], 1.0));

		// frustum plane counters
		if (c.x < -c.w) outL++;
		if (c.x >  c.w) outR++;
		if (c.y < -c.w) outB++;
		if (c.y >  c.w) outT++;
		// REVERSE-Z inverts the depth planes: nearer than near is z > w, and
		// beyond far is z < 0 (the opposite of a standard 0..1 projection).
		if (c.z >  c.w) outN++;
		if (c.z <  0.0) outF++;

		// screen-space AABB + nearest depth (clamped)
		c.z = max(c.z, 0.0);
		vec3 ndc = c.xyz / c.w;
		ndc.xy = clamp(ndc.xy, -1.0, 1.0);
		vec2 uv = ndc.xy * vec2(0.5, -0.5) + vec2(0.5, 0.5);
		minXY = min(uv, minXY);
		maxXY = max(uv, maxXY);
		nearZ = saturate(max(nearZ, ndc.z));
	}

	bool visible = true;
	if (doFrustum && (outL == 8 || outR == 8 || outB == 8 || outT == 8 || outN == 8 || outF == 8))
		visible = false;

	// small-object rejection: skip items whose projected footprint is sub-threshold
	if (visible && u_screenPixels.z > 0.0 && outN < 8)
	{
		vec2 px = (maxXY - minXY) * u_screenPixels.xy;
		if (max(px.x, px.y) < u_screenPixels.z)
			visible = false;
	}

	if (visible && doOcclusion)
	{
		vec4 boxUVs = vec4(minXY, maxXY);
		ivec2 size = ivec2((maxXY - minXY) * u_hiZSize.xy);
		float mip = ceil(log2(float(max(size.x, size.y))));
		mip = clamp(mip, 0.0, u_cullParams.w);

		// Drop to the next finer mip only if the footprint still fits in 2x2
		// texels there (4 taps then cover it).
		//
		// NOTE: example 37 scales the box by exp2(-level) alone, but boxUVs are
		// UVs in [0,1] -- without multiplying by the Hi-Z size these are not
		// texel coords, floor/ceil snap them to 0/1, and the test is ~always
		// true, so the mip is *always* dropped and the 4 taps under-cover the
		// footprint. Under-covering means the taps can miss the gap an item is
		// actually visible through, and it gets culled. Scale by the mip's texel
		// dimensions so the 2x2 test means what it says.
		float levelLower = max(mip - 1.0, 0.0);
		vec2 scale = u_hiZSize.xy * exp2(-levelLower);
		vec2 a = floor(boxUVs.xy * scale);
		vec2 b = ceil(boxUVs.zw * scale);
		vec2 dims = b - a;
		if (dims.x <= 2.0 && dims.y <= 2.0)
			mip = levelLower;

#if BGFX_SHADER_LANGUAGE_GLSL
		boxUVs.y = 1.0 - boxUVs.y;
		boxUVs.w = 1.0 - boxUVs.w;
#endif
		vec4 depth = vec4(
			  texture2DLod(s_hiZ, boxUVs.xy, mip).x
			, texture2DLod(s_hiZ, boxUVs.zy, mip).x
			, texture2DLod(s_hiZ, boxUVs.xw, mip).x
			, texture2DLod(s_hiZ, boxUVs.zw, mip).x
			);
		// REVERSE-Z: the pyramid is min-reduced and stores the farthest occluder,
		// which is the SMALLEST value. Take the min of the four taps to stay
		// conservative, and cull only when the item's nearest point is still
		// behind that occluder -- which, with near = 1, means a smaller value.
		float occluderFar = min(min(depth.x, depth.y), min(depth.z, depth.w));

		if (nearZ < occluderFar)
			visible = false;
	}

	if (visible)
	{
		uvec4 dp = instanceDrawParams[idx];
		uint slot;
		bool transparent = (dp.w & 1u) != 0u;
		if (!transparent)
		{
			atomicFetchAndAdd(opaqueCount[0], 1u, slot);
			drawIndexedIndirect(opaqueIndirect, slot, dp.x, 1u, dp.y, dp.z, idx);

			// large structural elements also feed the Hi-Z occluder depth pass
			if ((dp.w & 2u) != 0u)
			{
				uint oslot;
				atomicFetchAndAdd(occCount[0], 1u, oslot);
				drawIndexedIndirect(occIndirect, oslot, dp.x, 1u, dp.y, dp.z, idx);
			}
		}
		else
		{
			atomicFetchAndAdd(transpCount[0], 1u, slot);
			drawIndexedIndirect(transpIndirect, slot, dp.x, 1u, dp.y, dp.z, idx);
		}
	}
}
