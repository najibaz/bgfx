/*
 * Reset the GPU draw counter to zero before the cull pass.
 */

#include "bgfx_compute.sh"

BUFFER_RW(drawCount, uint, 0);

NUM_THREADS(1, 1, 1)
void main()
{
	drawCount[0] = 0u;
}
