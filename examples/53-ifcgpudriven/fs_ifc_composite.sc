$input v_texcoord0

/*
 * WBOIT composite: resolve the accumulation + revealage targets over the opaque
 * image and write the final frame to the backbuffer.
 *
 *   final = (accum.rgb / accum.a) * (1 - reveal) + opaque * reveal
 *
 * reveal is the product of transmittances, so reveal == 1 means nothing
 * transparent covered the pixel and the opaque colour passes through untouched.
 */

#include "../common/common.sh"

SAMPLER2D(s_opaqueColor, 0);
SAMPLER2D(s_accum,       1);
SAMPLER2D(s_reveal,      2);

void main()
{
	vec3  opaque = texture2D(s_opaqueColor, v_texcoord0).xyz;
	vec4  accum  = texture2D(s_accum,       v_texcoord0);
	float reveal = texture2D(s_reveal,      v_texcoord0).x;

	// Weighted average of the transparent fragments. Clamp guards the
	// no-coverage case (accum.a == 0) and fp16 overflow.
	vec3 avg = accum.xyz / clamp(accum.w, 1e-4, 5e4);

	gl_FragColor = vec4(avg * (1.0 - reveal) + opaque * reveal, 1.0);
}
