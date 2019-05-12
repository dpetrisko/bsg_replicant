#include "test_barrier.h"

/*!
 * Runs the barrier kernel on a 2x2 tile group. 
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/barrier/ Manycore binary in the dev_cuda_tile_group_refactored branch of the BSG Manycore github repository.  
*/
int kernel_barrier () {
	fprintf(stderr, "Running the CUDA Barrier Kernel on a 2x2 tile group.\n\n");

	device_t device;
	uint8_t grid_dim_x = 4;
	uint8_t grid_dim_y = 4;
	uint8_t grid_origin_x = 0;
	uint8_t grid_origin_y = 1;
	eva_id_t eva_id = 0;
	char* elf = BSG_STRINGIFY(BSG_MANYCORE_DIR) "/software/spmd/bsg_cuda_lite_runtime" "/barrier/main.riscv";

	hb_mc_device_init(&device, eva_id, elf, grid_dim_x, grid_dim_y, grid_origin_x, grid_origin_y);


	uint8_t tg_dim_x = 2;
	uint8_t tg_dim_y = 2;

	int argv[1];

	hb_mc_tile_group_enqueue (&device, tg_dim_x, tg_dim_y, "kernel_barrier", 0, argv);

	hb_mc_device_tile_groups_execute(&device);
	
	hb_mc_device_finish(&device); /* freeze the tiles and memory manager cleanup */
	
	return HB_MC_SUCCESS;
}

#ifdef COSIM
void test_main(uint32_t *exit_code) {	
	bsg_pr_test_info("test_barrier Regression Test (COSIMULATION)\n");
	int rc = kernel_barrier();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main() {
	bsg_pr_test_info("test_barrier Regression Test (F1)\n");
	int rc = kernel_barrier();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif

