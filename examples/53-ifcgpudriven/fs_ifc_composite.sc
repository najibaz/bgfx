$input v_texcoord0

/*
 * WBOIT composite + Eye-Dome Lighting, resolved to the backbuffer.
 *
 * 1) Resolve the weighted-blended OIT targets over the opaque image:
 *      final = (accum.rgb / accum.a) * (1 - reveal) + opaque * reveal
 *    reveal is the product of transmittances, so reveal == 1 means nothing
 *    transparent covered the pixel and opaque passes through untouched.
 *
 * 2) Eye-Dome Lighting (ported from the bim-viewer's edl.wgsl, itself Potree
 *    style). For each pixel, sample neighbouring depths at radius R, take the
 *    mean positive log-depth difference, and darken proportionally. That paints
 *    a soft dark halo wherever a surface meets something further away -- i.e.
 *    the boundaries between elements -- which is what flat-shaded BIM geometry
 *    lacks. It keys off *depth discontinuity* rather than normals, so it also
 *    separates two abutting walls that shade almost identically.
 *
 * REVERSE-Z: depth 1 = near, 0 = far. So a *smaller* neighbour depth means the
 * neighbour is further away, and log(centre) - log(neighbour) > 0 marks a
 * silhouette. Under standard Z this comparison would be inverted -- the port is
 * only valid because this pipeline is now reverse-Z, same as the bim-viewer's.
 */

#include "../common/common.sh"

// Colour / accum / reveal are plain samplers: bgfx resolves those MSAA targets
// automatically on sample. Depth cannot be resolved, so it is bound as MSAA_SAMPLE
// and fetched per-sample here; sample 0 is plenty for edge detection.
SAMPLER2D(s_opaqueColor,   0);
SAMPLER2D(s_accum,         1);
SAMPLER2D(s_reveal,        2);
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
	// ---- WBOIT resolve ----
	vec3  opaque = texture2D(s_opaqueColor, v_texcoord0).xyz;
	vec4  accum  = texture2D(s_accum,       v_texcoord0);
	float reveal = texture2D(s_reveal,      v_texcoord0).x;

	vec3 avg   = accum.xyz / clamp(accum.w, 1e-4, 5e4);
	vec3 color = avg * (1.0 - reveal) + opaque * reveal;

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
