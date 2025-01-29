/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#include <fstream>
#include <iostream>
#include <limits>

namespace vaip_vaiml_custom_op {
// minimum aligned IFM C dimension (ic) size
const uint32_t aligned_ic_size = 8;
// bytes size of layer parameters
const uint32_t LP_bytes = 64;
// bytes size of dynamic dispatch parameters
const uint32_t DD_bytes = 64;

union uint64_uint32 {
  uint64_t UI64;
  uint32_t UI32[2];
};

union uint64_int64 {
  uint64_t UI64;
  int64_t I64;
};

union uint64_uint8 {
  uint64_t UI64;
  uint8_t UI8[8];
};

// pad WTS if ic dim is less than 8, data format must be (oc, ic, kh, kw)
inline void pad_wts_ic_dim(uint8_t* wts_data, uint8_t* pad_wts_data,
                           uint32_t kh, uint32_t kw, uint32_t ic, uint32_t oc) {
  uint32_t idx_orig = 0;
  uint32_t idx_pad = 0;
  for (int i = 0; i < oc; i++) {
    for (int j = 0; j < ic; j++) {
      for (int k = 0; k < kh; k++) {
        for (int l = 0; l < kw; l++) {
          pad_wts_data[idx_pad++] = wts_data[idx_orig++];
        }
      }
    }
    for (int j = 0; j < (aligned_ic_size - ic); j++) {
      for (int k = 0; k < kh; k++) {
        for (int l = 0; l < kw; l++) {
          pad_wts_data[idx_pad++] = (uint8_t)0;
        }
      }
    }
  }
}

// 8-D transpose for WTS as AIE required.
// from data format (depth_iter_oc, ocg, oc_tile, depth_iter_ic, icg, ic_tile,
// kh, kw) to (depth_iter_oc, depth_iter_ic, ocg, icg, kh, kw, ic_tile,
// oc_tile), equivalent of above code is:
//     wts = np.transpose(wts, (0, 3, 1, 4, 6, 7, 5, 2))
inline void transpose_wts(uint8_t* wts_data, uint8_t* trans_wts_data,
                          uint32_t kh, uint32_t kw, uint32_t ic, uint32_t oc,
                          uint32_t ic_tile, uint32_t icg, uint32_t oc_tile,
                          uint32_t ocg) {

  uint32_t depth_iter_ic = ic / (ic_tile * icg);
  uint32_t depth_iter_oc = oc / (oc_tile * ocg);

  for (uint32_t i = 0; i < depth_iter_oc; i++) {
    for (uint32_t j = 0; j < ocg; j++) {
      for (uint32_t k = 0; k < oc_tile; k++) {
        for (uint32_t l = 0; l < depth_iter_ic; l++) {
          for (uint32_t m = 0; m < icg; m++) {
            for (uint32_t n = 0; n < ic_tile; n++) {
              for (uint32_t o = 0; o < kh; o++) {
                for (uint32_t p = 0; p < kw; p++) {
                  uint32_t new_i = i; // depth_iter_oc
                  uint32_t new_j = l; // depth_iter_ic
                  uint32_t new_k = j; // ocg
                  uint32_t new_l = m; // icg
                  uint32_t new_m = o; // kh
                  uint32_t new_n = p; // kw
                  uint32_t new_o = n; // ic_tile
                  uint32_t new_p = k; // oc_tile
                  uint32_t from_idx =
                      i * (ocg * oc_tile * depth_iter_ic * icg * ic_tile * kh *
                           kw) +
                      j * (oc_tile * depth_iter_ic * icg * ic_tile * kh * kw) +
                      k * (depth_iter_ic * icg * ic_tile * kh * kw) +
                      l * (icg * ic_tile * kh * kw) + m * (ic_tile * kh * kw) +
                      n * (kh * kw) + o * (kw) + p;
                  uint32_t to_idx =
                      new_i * (depth_iter_ic * ocg * icg * kh * kw * ic_tile *
                               oc_tile) +
                      new_j * (ocg * icg * kh * kw * ic_tile * oc_tile) +
                      new_k * (icg * kh * kw * ic_tile * oc_tile) +
                      new_l * (kh * kw * ic_tile * oc_tile) +
                      new_m * (kw * ic_tile * oc_tile) +
                      new_n * (ic_tile * oc_tile) + new_o * (oc_tile) + new_p;
                  trans_wts_data[to_idx] = wts_data[from_idx];
                }
              }
            }
          }
        }
      }
    }
  }
}

// append layer parameters for each sub-volume
inline void append_lp(std::vector<uint64_t>& res, uint32_t* lp_data) {
  uint64_uint32 cvt;
  for (int i = 0; i < (LP_bytes / sizeof(uint32_t)) /
                          (sizeof(uint64_t) / sizeof(uint32_t));
       i++) {
    for (int j = 0; j < sizeof(uint64_t) / sizeof(uint32_t); j++) {
      cvt.UI32[j] = lp_data[i * sizeof(uint64_t) / sizeof(uint32_t) + j];
    }
    res.push_back(cvt.UI64);
  }
}

// append sub-volume of c0 for each sub-volume
inline void append_c0(std::vector<uint64_t>& res, int64_t* c0_data,
                      uint32_t idx, uint32_t coeff_gran) {
  uint64_int64 cvt;
  for (int i = 0; i < coeff_gran; i++) {
    cvt.I64 = c0_data[idx * coeff_gran + i];
    res.push_back(cvt.UI64);
  }
}

// append sub-volume of WTS for each sub-volume
inline void append_wts(std::vector<uint64_t>& res, uint8_t* wts_data,
                       uint32_t idx, uint32_t wts_gran) {
  uint64_uint8 cvt;
  for (int i = 0; i < wts_gran / (sizeof(uint64_t) * sizeof(uint8_t)); i++) {
    for (int j = 0; j < sizeof(uint64_t) / sizeof(uint8_t); j++) {
      cvt.UI8[j] =
          wts_data[idx * wts_gran + i * sizeof(uint64_t) / sizeof(uint8_t) + j];
    }
    res.push_back(cvt.UI64);
  }
}

// pointer-to-pointer WTS generator for GT conv
inline std::vector<uint64_t> wts_gen_conv(uint32_t* lp_data, int64_t* c0_data,
                                          uint8_t* wts_data, uint32_t kh,
                                          uint32_t kw, uint32_t ic,
                                          uint32_t oc, // original wts shape
                                          uint32_t ic_tile, uint32_t icg,
                                          uint32_t oc_tile, uint32_t ocg) {

  int32_t padded_ic = ic < aligned_ic_size ? aligned_ic_size : ic;
  uint8_t* pad_wts_data = new uint8_t[kh * kw * padded_ic * oc];
  uint8_t* trans_wts_data = new uint8_t[kh * kw * padded_ic * oc];

  // pre-processing WTS
  if (ic < aligned_ic_size) {
    // pad WTS on dim 2 if necessary
    pad_wts_ic_dim(wts_data, pad_wts_data, kh, kw, ic, oc);
    // transpose WTS as AIE required
    transpose_wts(pad_wts_data, trans_wts_data, kh, kw, padded_ic, oc, ic_tile,
                  icg, oc_tile, ocg);
  } else {
    // transpose WTS as AIE required
    transpose_wts(wts_data, trans_wts_data, kh, kw, padded_ic, oc, ic_tile, icg,
                  oc_tile, ocg);
  }

  int32_t depth_iters_ic = padded_ic / (ic_tile * icg);
  int32_t depth_iters_oc = oc / (oc_tile * ocg);
  uint32_t coeff_gran = oc / depth_iters_oc;
  uint32_t wts_gran =
      (kh * kw * padded_ic * oc) / depth_iters_ic / depth_iters_oc;

  // union lp/c0/wts SV-by-SV
  std::vector<uint64_t> res;
  for (uint32_t i = 0; i < depth_iters_oc; i++) {
    for (uint32_t j = 0; j < depth_iters_ic; j++) {
      append_lp(res, lp_data);
      append_c0(res, c0_data, i, coeff_gran);
      append_wts(res, trans_wts_data, i * depth_iters_ic + j, wts_gran);
    }
  }

  delete[] pad_wts_data;
  delete[] trans_wts_data;
  // flatten union WTS before dumping out
  uint32_t ic_depths = padded_ic / (icg * ic_tile);
  uint32_t oc_depths = oc / (ocg * oc_tile);
  uint32_t coeffBytes = sizeof(int64_t);
  uint32_t wgtBytes = sizeof(uint8_t);
  uint32_t one_loop_union_bytes =
      (LP_bytes + ocg * oc_tile * coeffBytes +
       ocg * icg * kh * kw * ic_tile * oc_tile * wgtBytes) *
      ic_depths;
  std::vector<uint64_t> res_flatten;
  if (res.size() * sizeof(uint64_t) != one_loop_union_bytes * oc_depths) {
    one_loop_union_bytes =
        (DD_bytes + LP_bytes + ocg * oc_tile * coeffBytes +
         ocg * icg * kh * kw * ic_tile * oc_tile * wgtBytes) *
        ic_depths;
    for (uint32_t iter_oc = 0; iter_oc < oc_depths; iter_oc++) {
      // XXX: original python implementation is in conv2d_processArray.py line
      // 172 iter_h & iter_w is not used in the inner-most loop body, so we
      // simplify the nested loop into single iter_oc loop
      for (uint32_t i = 0;
           i < one_loop_union_bytes / (sizeof(uint64_t) / sizeof(uint8_t));
           i++) {
        res_flatten.push_back(res[i]);
      }
    }
    return res_flatten;
  }

  return res;
}
} // namespace vaip_vaiml_custom_op