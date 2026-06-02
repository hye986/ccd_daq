/*
 * ccd_decode.c
 *
 * Fast CCD rolling-shutter UDP packet decoder, callable from Python via ctypes.
 * Compile with:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC \
 *       -o ccd_decode.so ccd_decode.c
 *
 * On macOS replace .so with .dylib.
 *
 * Provides two functions:
 *
 *   decode_strip_512(payload, payload_len, strip_out)
 *     Decode one 8256-byte UDP packet for a 512x512 single-HYB detector.
 *     strip_out must be uint16[512][8] (4096 bytes, C-contiguous).
 *     Returns 0 on success, -1 on bad payload length.
 *
 *   decode_strip_1024(payload, payload_len, strip_out)
 *     Identical interface, same shape — for the 1024x1024 4-HYB detector
 *     the HYB local geometry is identical; the global placement is done
 *     in Python using the GX/GY LUTs.
 *     Returns 0 on success, -1 on bad payload length.
 *
 *   decode_strip_2048(payload, payload_len, strip_out)
 *     Identical interface, same shape — for the 2048x1024 4-HYB detector
 *     the HYB local geometry is identical (1024x512); the global placement
 *     is done in Python using the GX/GY LUTs.
 *     Returns 0 on success, -1 on bad payload length.
 *
 * All three functions are thread-safe (no global mutable state).
 *
 * Memory layout of strip_out[local_y][x_step]:
 *   local_y = (7 - ADC_TO_ASIC[adc]) * 64 + mux
 *   x_step  = 0..7  (x = x0 - x_step)
 *
 * This is identical to the numpy _Y_IDX-based layout used in the Python decoder.
 */

#include <stdint.h>
#include <string.h>

/* ── constants ──────────────────────────────────────────────────────── */
#define N_X_PER_PACKET  8
#define N_MUX          64
#define N_ADC           8
#define HYB_SIZE       512
#define HEADER_LEN      64
#define ADC_DATA_LEN   (N_X_PER_PACKET * N_MUX * N_ADC * 2)   /* 8192 */
#define TOTAL_LEN      (HEADER_LEN + ADC_DATA_LEN)              /* 8256 */

/*
 * ADC_TO_ASIC wiring map: ADC_TO_ASIC[adc] = asic_index
 * Y base for each ADC channel: y_base[adc] = (7 - ADC_TO_ASIC[adc]) * 64
 */
static const int ADC_TO_ASIC[8] = {3, 1, 2, 0, 5, 4, 7, 6};
static const int Y_BASE[8]      = {
    (7-3)*64,   /* ADC0 -> ASIC3 -> y=256 */
    (7-1)*64,   /* ADC1 -> ASIC1 -> y=384 */
    (7-2)*64,   /* ADC2 -> ASIC2 -> y=320 */
    (7-0)*64,   /* ADC3 -> ASIC0 -> y=448 */
    (7-5)*64,   /* ADC4 -> ASIC5 -> y=128 */
    (7-4)*64,   /* ADC5 -> ASIC4 -> y=192 */
    (7-7)*64,   /* ADC6 -> ASIC7 -> y=0   */
    (7-6)*64,   /* ADC7 -> ASIC6 -> y=64  */
};

/*
 * decode_strip_512 / decode_strip_1024 / decode_strip_2048
 *
 * All have identical ADC geometry — they differ only in header parsing
 * (hyb_num field) which is handled in Python.  The strip decode is the
 * same for all HYBs.
 *
 * strip_out: uint16_t[HYB_SIZE][N_X_PER_PACKET]  i.e. uint16_t[512][8]
 */
static int
_decode_strip(const uint8_t *payload, int payload_len, uint16_t *strip_out)
{
    if (payload_len != TOTAL_LEN)
        return -1;

    const uint8_t *data = payload + HEADER_LEN;

    /*
     * Data layout in the packet (little-endian uint16):
     *   for x_step in 0..7:
     *     for mux in 0..63:
     *       ADC0, ADC1, ..., ADC7
     *
     * We want strip_out[local_y][x_step] where
     *   local_y = Y_BASE[adc] + mux
     *
     * Inner loop order chosen so the read pointer advances linearly
     * through the packet — cache friendly.
     */
    const uint16_t *src = (const uint16_t *)data;

    for (int x_step = 0; x_step < N_X_PER_PACKET; x_step++) {
        for (int mux = 0; mux < N_MUX; mux++) {
            for (int adc = 0; adc < N_ADC; adc++) {
                uint16_t val = *src++;   /* little-endian, host is LE */
                int local_y  = Y_BASE[adc] + mux;
                strip_out[local_y * N_X_PER_PACKET + x_step] = val;
            }
        }
    }
    return 0;
}

/* Public API — separate symbols so ctypes can find them by name */
int decode_strip_512(const uint8_t  *payload,
                     int             payload_len,
                     uint16_t       *strip_out)
{
    return _decode_strip(payload, payload_len, strip_out);
}

int decode_strip_1024(const uint8_t *payload,
                      int            payload_len,
                      uint16_t      *strip_out)
{
    return _decode_strip(payload, payload_len, strip_out);
}

int decode_strip_2048(const uint8_t *payload,
                      int            payload_len,
                      uint16_t      *strip_out)
{
    return _decode_strip(payload, payload_len, strip_out);
}
