#include <bsg_manycore_platform.h>
#include <bsg_manycore_mmio.h>
#include <bsg_manycore_config.h>
#include <bsg_manycore_printing.h>

#include <cstring>

/**
 * Clean up the runtime platform
 * @param[in] mc    A manycore to clean up
 */
void hb_mc_platform_cleanup(hb_mc_manycore_t *mc)
{
    /* Do nothing for baremetal BlackParrot */
}

/**
 * Initialize the runtime platform
 * @param[in] mc    A manycore to initialize
 * @param[in] id    ID which selects the physical hardware from which this manycore is configured
 * @return HB_MC_FAIL if an error occured. HB_MC_SUCCESS otherwise.
 */
int hb_mc_platform_init(hb_mc_manycore_t *mc,
                        hb_mc_manycore_id_t id)
{
    /* Do nothing for baremetal BlackParrot */
    return HB_MC_SUCCESS;
}

/**
 * Transmit a request packet to manycore hardware
 * @param[in] mc      A manycore instance initialized with hb_mc_manycore_init()
 * @param[in] request A request packet to transmit to manycore hardware
 * @param[in] timeout A timeout counter. Unused - set to -1 to wait forever.
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_platform_transmit(hb_mc_manycore_t *mc,
                            hb_mc_packet_t *packet,
                            hb_mc_fifo_tx_t type,
                            long timeout)
{
    /* TODO: Implement BlackParrot transmit */
    return HB_MC_SUCCESS;
}

/**
 * Receive a packet from manycore hardware
 * @param[in] mc       A manycore instance initialized with hb_mc_manycore_init()
 * @param[in] response A packet into which data should be read
 * @param[in] timeout  A timeout counter. Unused - set to -1 to wait forever.
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_platform_receive(hb_mc_manycore_t *mc,
                           hb_mc_packet_t *packet,
                           hb_mc_fifo_rx_t type,
                           long timeout)
{
    /* TODO: Implement BlackParrot receive */
    return HB_MC_SUCCESS;
}

/**
 * Read the configuration register at an index
 * @param[in]  mc     A manycore instance initialized with hb_mc_manycore_init()
 * @param[in]  idx    Configuration register index to access
 * @param[out] config Configuration value at index
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_platform_get_config_at(hb_mc_manycore_t *mc, 
                                 unsigned int idx,
                                 hb_mc_config_raw_t *config)
{
    /* TODO: BlackParrot CSRs are unsupported */
    *config = 0;
    return HB_MC_SUCCESS;
}

/**
 * Stall until the all requests (and responses) have reached their destination.
 * @param[in]  mc     A manycore instance initialized with hb_mc_manycore_init()
 * @param[in] timeout A timeout counter. Unused - set to -1 to wait forever.
 * @return HB_MC_SUCCESS on success. Otherwise an error code defined in bsg_manycore_errno.h.
 */
int hb_mc_platform_fence(hb_mc_manycore_t *mc, 
                         long timeout)
{
    /* TODO: Implement BlackParrot fence */
    return 0;
}

