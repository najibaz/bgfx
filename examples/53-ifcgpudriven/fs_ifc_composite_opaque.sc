$input v_texcoord0

/*
 * Composite + Eye-Dome Lighting for scenes with NO transparent geometry.
 *
 * Identical to fs_ifc_composite minus the WBOIT resolve: when the scene has no
 * transparent material, the accum/reveal targets are never allocated and the
 * accumulation pass never runs, so there is nothing to resolve and
 * `final = opaque` (reveal would be a constant 1 everywhere).
 *
 * Selecting this variant on the CPU rather than branching on a uniform keeps
 * the two fullscreen samples and the fp16 accum target out of the frame
 * entirely. See the EDL notes in fs_ifc_composite.sc -- the reverse-Z reasoning
 * there applies here unchanged.
 */

#include "../common/common.sh"

SAMPLER2D(s_opaqueColor,   0);
SAMPLER2DMS(s_depth,       3);

uniform vec4 u_edlParams;  // x = strength, y = radius (px), z = max darken
uniform vec4 u_edlColor;   // xyz = edge colour

float sampleDepth(ivec2 coord, ivec2 dims)
{
	ivec2 c = ivec2(
		  clamp(coord.x, 0, dims.x - 1)
		, clamp(coord.y, 0, dims.y - 1)
		);
	return texelFetch(s_depth, c, 0).x;
}

void main()
{
	vec3 color = texture2D(s_opaqueColor, v_texcoord0).xyz;

	// ---- Eye-Dome Lighting ----
	ivec2 dims = ivec2(u_viewRect.zw);
	ivec2 p    = ivec2(gl_FragCoord.xy);

	float center = sampleDepth(p, dims);

	// Reverse-Z: 0 is the far plane, i.e. nothing was drawn here. Skip, so the
	// background never gets haloed.
	if (center > 1e-5)
	{
		int r = int(max(u_edlParams.y, 1.0) );
		float logCenter = log(center);

		float accumEdl = 0.0;
		float count    = 0.0;

		// 8 taps: 4 cardinal + 4 diagonal
		ivec2 offsets[8];
		offsets[0] = ivec2( r,  0);
		offsets[1] = ivec2(-r,  0);
		offsets[2] = ivec2( 0,  r);
		offsets[3] = ivec2( 0, -r);
		offsets[4] = ivec2( r,  r);
		offsets[5] = ivec2(-r,  r);
		offsets[6] = ivec2( r, -r);
		offsets[7] = ivec2(-r, -r);

		for (int i = 0; i < 8; i++)
		{
			float dn = sampleDepth(p + offsets[i], dims);
			if (dn > 1e-5)
			{
				// Positive only: darken where the neighbour is further away.
				accumEdl += max(0.0, logCenter - log(dn) );
				count    += 1.0;
			}
		}

		float meanLog = accumEdl / max(count, 1.0);
		// 300 is Potree's default sensitivity constant.
		float shade  = exp(-300.0 * meanLog * u_edlParams.x);
		float darken = clamp(1.0 - shade, 0.0, u_edlParams.z);

		// Blend toward a grey rather than multiplying to black: edges read as
		// separation lines instead of a wireframe over dense BIM geometry.
		color = mix(color, u_edlColor.xyz, darken);
	}

	gl_FragColor = vec4(color, 1.0);
}
