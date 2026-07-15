$input v_wpos, v_color

/*
 * IFC GPU-driven viewer - fragment shader.
 * The source data carries no vertex normals, so reconstruct a per-face normal
 * from screen-space derivatives of the world position and light it with a
 * fixed directional light plus ambient.
 */

#include "../common/common.sh"

void main()
{
	vec3 n = normalize(cross(dFdx(v_wpos), dFdy(v_wpos)));
	vec3 L = normalize(vec3(0.4, 0.8, 0.5));
	float ndl = abs(dot(n, L));
	vec3 color = v_color.rgb * (0.30 + 0.70 * ndl);
	gl_FragColor = vec4(color, v_color.a);
}
