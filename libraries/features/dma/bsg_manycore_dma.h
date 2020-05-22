#ifndef __BSG_MANYCORE_DMA_HPP
#define __BSG_MANYCORE_DMA_HPP

#include <bsg_manycore.h>
#include <bsg_manycore_npa.h>
#include <bsg_manycore_config.h>

#ifdef __cplusplus
extern "C" {
#endif

        /**
         * Check if NPA is in DRAM.
         * @param[in]  mc     A manycore instance initialized with hb_mc_manycore_init()
         * @param[in]  npa    A valid hb_mc_npa_t
         * @return One if the NPA maps to DRAM - Zero otherwise.
         */
        static inline int hb_mc_manycore_npa_is_dram(hb_mc_manycore_t *mc,
                                                     const hb_mc_npa_t *npa)
        {
                const hb_mc_config_t *cfg = hb_mc_manycore_get_config(mc);
                return hb_mc_config_is_dram_y(cfg, hb_mc_npa_get_y(npa));
        }

        /**
         * Read memory from manycore DRAM via C++ backdoor
         * @param[in]  mc     A manycore instance initialized with hb_mc_manycore_init()
         * @param[in]  npa    A valid hb_mc_npa_t - must be an L2 cache coordinate
         * @param[in]  data   A host buffer to be read into from manycore hardware
         * @param[in]  sz     The number of bytes to read from manycore hardware
         * @return HB_MC_FAIL if an error occured. HB_MC_SUCCESS otherwise.
         */
        int hb_mc_manycore_dma_read_internal(hb_mc_manycore_t *mc,
                                             const hb_mc_npa_t *npa,
                                             void *data, size_t sz);

        /**
         * Write memory out to manycore DRAM via C++ backdoor
         * @param[in]  mc     A manycore instance initialized with hb_mc_manycore_init()
         * @param[in]  npa    A valid hb_mc_npa_t - must be an L2 cache coordinate
         * @param[in]  data   A buffer to be written out manycore hardware
         * @param[in]  sz     The number of bytes to write to manycore hardware
         * @return HB_MC_FAIL if an error occured. HB_MC_SUCCESS otherwise.
         */
        int hb_mc_manycore_dma_write_internal(hb_mc_manycore_t *mc,
                                              const hb_mc_npa_t *npa,
                                              const void *data, size_t sz);
#ifdef __cplusplus
}
#endif

#endif
