$input v_wpos, v_color

/*
 * Weighted-blended order-independent transparency (McGuire & Bavoil) —
 * accumulation pass for the transparent visible set.
 *
 * The GPU-driven cull compacts draws with an atomic append, so the transparent
 * draw order is arbitrary and re-races every frame. Alpha blending is
 * order-dependent, so that produced flicker. WBOIT sidesteps it entirely: both
 * outputs are accumulated with *commutative* blends, so the result does not
 * depend on draw order at all.
 *
 *   RT0 accum  (RGBA16F, cleared 0, blend ONE/ONE)
 *       sum of premultiplied colour * weight, and sum of alpha * weight in .a
 *   RT1 reveal (R16F, cleared 1, blend ZERO/INV_SRC_COLOR -> dst *= (1 - a))
 *       product of transmittances = prod(1 - alpha)
 *
 * The composite then resolves:
 *   final = (accum.rgb / accum.a) * (1 - reveal) + opaque * reveal
 *
 * Trade-off: this is a weighted *average*, not an exact back-to-front sort, so
 * deep stacks of glass read slightly differently than a true sort. In exchange
 * it needs no sort and cannot flicker.
 */

#include "../common/common.sh"

void main()
{
	// same shading as fs_ifc: face normal from screen-space derivatives
	vec3 n = normalize(cross(dFdx(v_wpos), dFdy(v_wpos)));
	vec3 L = normalize(vec3(0.4, 0.8, 0.5));
	float ndl = abs(dot(n, L));
	vec3 color = v_color.rgb * (0.30 + 0.70 * ndl);

	float a = v_color.a;

	// McGuire weight function tuned for NDC depth in [0,1]: near fragments
	// dominate, far ones fade, clamped to [1e-2, 3e3] to stay in fp16 range.
	float z = gl_FragCoord.z;
	float w = clamp(
		  pow(min(1.0, a * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - z * 0.9, 3.0)
		, 1e-2
		, 3e3
		);

	gl_FragData[0] = vec4(color * a, a) * w;  // accum  (blend ONE, ONE)
	gl_FragData[1] = vec4_splat(a);           // reveal (blend ZERO, INV_SRC_COLOR)
}
