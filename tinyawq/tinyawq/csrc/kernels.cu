/*
Inspired by :
@article{lin2023awq,
    title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
    author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
    journal={arXiv},
    year={2023}
}

And:
@misc{kim2024quick,
    title={QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference}, 
    author={Taesu Kim and Jongho Lee and Daehyun Ahn and Sarang Kim and Jiwoong Choi and Minkyu Kim and Hyungjun Kim},
    year={2024},
    eprint={2402.10076},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
*/

#include <torch/extension.h>


__device__ uint4 dequantize_s4_to_fp16x2_fused(uint32_t const& source)
{
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[2])
                    : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[3])
                    : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // TSK: add {960, 960}.
    static constexpr uint32_t FP16_MAGIC_NUM = 0x63806380;
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;

    // // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(FP16_MAGIC_NUM));
    // // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(FP16_MAGIC_NUM));

    return result;
}


__device__ void compute_gemm_x2(half const* A_shared, int const* B_ptr_local,
                              float* C_warp,
                              uint4 B_loaded_zero, uint4 B_loaded_zero_2,
                              uint4 B_loaded_scale, uint4 B_loaded_scale_2)
{
  half A_warp[16];

  // Load B
  uint4 B_loaded = *(uint4*)(B_ptr_local);

  // Copy A
  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[0])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[0]),
        "=r"(((unsigned *)(A_warp))[1]),
        "=r"(((unsigned *)(A_warp))[2]),
        "=r"(((unsigned *)(A_warp))[3])
      : "r"(addr)
    );

    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.x);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.y));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp))[0]),
        "=f"(((float *)(C_warp))[1]),
        "=f"(((float *)(C_warp))[2]),
        "=f"(((float *)(C_warp))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp))[0]),
        "f"(((float *)(C_warp))[1]),
        "f"(((float *)(C_warp))[2]),
        "f"(((float *)(C_warp))[3]));

    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[16 * (32 + 8)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[4]),
        "=r"(((unsigned *)(A_warp))[5]),
        "=r"(((unsigned *)(A_warp))[6]),
        "=r"(((unsigned *)(A_warp))[7])
      : "r"(addr)
    );

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4))[0]),
        "=f"(((float *)(C_warp + 4))[1]),
        "=f"(((float *)(C_warp + 4))[2]),
        "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4))[0]),
        "f"(((float *)(C_warp + 4))[1]),
        "f"(((float *)(C_warp + 4))[2]),
        "f"(((float *)(C_warp + 4))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 32))[0]),
        "=f"(((float *)(C_warp + 32))[1]),
        "=f"(((float *)(C_warp + 32))[2]),
        "=f"(((float *)(C_warp + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 32))[0]),
        "f"(((float *)(C_warp + 32))[1]),
        "f"(((float *)(C_warp + 32))[2]),
        "f"(((float *)(C_warp + 32))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4 + 32))[0]),
        "=f"(((float *)(C_warp + 4 + 32))[1]),
        "=f"(((float *)(C_warp + 4 + 32))[2]),
        "=f"(((float *)(C_warp + 4 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4 + 32))[0]),
        "f"(((float *)(C_warp + 4 + 32))[1]),
        "f"(((float *)(C_warp + 4 + 32))[2]),
        "f"(((float *)(C_warp + 4 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.y);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8))[0]),
        "=f"(((float *)(C_warp + 8))[1]),
        "=f"(((float *)(C_warp + 8))[2]),
        "=f"(((float *)(C_warp + 8))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8))[0]),
        "f"(((float *)(C_warp + 8))[1]),
        "f"(((float *)(C_warp + 8))[2]),
        "f"(((float *)(C_warp + 8))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12))[0]),
        "=f"(((float *)(C_warp + 12))[1]),
        "=f"(((float *)(C_warp + 12))[2]),
        "=f"(((float *)(C_warp + 12))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12))[0]),
        "f"(((float *)(C_warp + 12))[1]),
        "f"(((float *)(C_warp + 12))[2]),
        "f"(((float *)(C_warp + 12))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8 + 32))[0]),
        "=f"(((float *)(C_warp + 8 + 32))[1]),
        "=f"(((float *)(C_warp + 8 + 32))[2]),
        "=f"(((float *)(C_warp + 8 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8 + 32))[0]),
        "f"(((float *)(C_warp + 8 + 32))[1]),
        "f"(((float *)(C_warp + 8 + 32))[2]),
        "f"(((float *)(C_warp + 8 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12 + 32))[0]),
        "=f"(((float *)(C_warp + 12 + 32))[1]),
        "=f"(((float *)(C_warp + 12 + 32))[2]),
        "=f"(((float *)(C_warp + 12 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12 + 32))[0]),
        "f"(((float *)(C_warp + 12 + 32))[1]),
        "f"(((float *)(C_warp + 12 + 32))[2]),
        "f"(((float *)(C_warp + 12 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.z);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.y));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16))[0]),
        "=f"(((float *)(C_warp + 16))[1]),
        "=f"(((float *)(C_warp + 16))[2]),
        "=f"(((float *)(C_warp + 16))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16))[0]),
        "f"(((float *)(C_warp + 16))[1]),
        "f"(((float *)(C_warp + 16))[2]),
        "f"(((float *)(C_warp + 16))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20))[0]),
        "=f"(((float *)(C_warp + 20))[1]),
        "=f"(((float *)(C_warp + 20))[2]),
        "=f"(((float *)(C_warp + 20))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20))[0]),
        "f"(((float *)(C_warp + 20))[1]),
        "f"(((float *)(C_warp + 20))[2]),
        "f"(((float *)(C_warp + 20))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16 + 32))[0]),
        "=f"(((float *)(C_warp + 16 + 32))[1]),
        "=f"(((float *)(C_warp + 16 + 32))[2]),
        "=f"(((float *)(C_warp + 16 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16 + 32))[0]),
        "f"(((float *)(C_warp + 16 + 32))[1]),
        "f"(((float *)(C_warp + 16 + 32))[2]),
        "f"(((float *)(C_warp + 16 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20 + 32))[0]),
        "=f"(((float *)(C_warp + 20 + 32))[1]),
        "=f"(((float *)(C_warp + 20 + 32))[2]),
        "=f"(((float *)(C_warp + 20 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20 + 32))[0]),
        "f"(((float *)(C_warp + 20 + 32))[1]),
        "f"(((float *)(C_warp + 20 + 32))[2]),
        "f"(((float *)(C_warp + 20 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded.w);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.w));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24))[0]),
        "=f"(((float *)(C_warp + 24))[1]),
        "=f"(((float *)(C_warp + 24))[2]),
        "=f"(((float *)(C_warp + 24))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24))[0]),
        "f"(((float *)(C_warp + 24))[1]),
        "f"(((float *)(C_warp + 24))[2]),
        "f"(((float *)(C_warp + 24))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28))[0]),
        "=f"(((float *)(C_warp + 28))[1]),
        "=f"(((float *)(C_warp + 28))[2]),
        "=f"(((float *)(C_warp + 28))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28))[0]),
        "f"(((float *)(C_warp + 28))[1]),
        "f"(((float *)(C_warp + 28))[2]),
        "f"(((float *)(C_warp + 28))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24 + 32))[0]),
        "=f"(((float *)(C_warp + 24 + 32))[1]),
        "=f"(((float *)(C_warp + 24 + 32))[2]),
        "=f"(((float *)(C_warp + 24 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24 + 32))[0]),
        "f"(((float *)(C_warp + 24 + 32))[1]),
        "f"(((float *)(C_warp + 24 + 32))[2]),
        "f"(((float *)(C_warp + 24 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28 + 32))[0]),
        "=f"(((float *)(C_warp + 28 + 32))[1]),
        "=f"(((float *)(C_warp + 28 + 32))[2]),
        "=f"(((float *)(C_warp + 28 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28 + 32))[0]),
        "f"(((float *)(C_warp + 28 + 32))[1]),
        "f"(((float *)(C_warp + 28 + 32))[2]),
        "f"(((float *)(C_warp + 28 + 32))[3]));
  }

  // Load next B
  uint4 B_loaded_2 = *(uint4*)(B_ptr_local + 4);

  // Copy A
  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[16])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[0]),
        "=r"(((unsigned *)(A_warp))[1]),
        "=r"(((unsigned *)(A_warp))[2]),
        "=r"(((unsigned *)(A_warp))[3])
      : "r"(addr)
    );

    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.x);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.y));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp))[0]),
        "=f"(((float *)(C_warp))[1]),
        "=f"(((float *)(C_warp))[2]),
        "=f"(((float *)(C_warp))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp))[0]),
        "f"(((float *)(C_warp))[1]),
        "f"(((float *)(C_warp))[2]),
        "f"(((float *)(C_warp))[3]));

    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[16 + 16 * (32+8)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
    );

    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_warp))[4]),
        "=r"(((unsigned *)(A_warp))[5]),
        "=r"(((unsigned *)(A_warp))[6]),
        "=r"(((unsigned *)(A_warp))[7])
      : "r"(addr)
    );

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4))[0]),
        "=f"(((float *)(C_warp + 4))[1]),
        "=f"(((float *)(C_warp + 4))[2]),
        "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4))[0]),
        "f"(((float *)(C_warp + 4))[1]),
        "f"(((float *)(C_warp + 4))[2]),
        "f"(((float *)(C_warp + 4))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 32))[0]),
        "=f"(((float *)(C_warp + 32))[1]),
        "=f"(((float *)(C_warp + 32))[2]),
        "=f"(((float *)(C_warp + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 32))[0]),
        "f"(((float *)(C_warp + 32))[1]),
        "f"(((float *)(C_warp + 32))[2]),
        "f"(((float *)(C_warp + 32))[3]));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 4 + 32))[0]),
        "=f"(((float *)(C_warp + 4 + 32))[1]),
        "=f"(((float *)(C_warp + 4 + 32))[2]),
        "=f"(((float *)(C_warp + 4 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 4 + 32))[0]),
        "f"(((float *)(C_warp + 4 + 32))[1]),
        "f"(((float *)(C_warp + 4 + 32))[2]),
        "f"(((float *)(C_warp + 4 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.y);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w));

    // Compute
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8))[0]),
        "=f"(((float *)(C_warp + 8))[1]),
        "=f"(((float *)(C_warp + 8))[2]),
        "=f"(((float *)(C_warp + 8))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8))[0]),
        "f"(((float *)(C_warp + 8))[1]),
        "f"(((float *)(C_warp + 8))[2]),
        "f"(((float *)(C_warp + 8))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12))[0]),
        "=f"(((float *)(C_warp + 12))[1]),
        "=f"(((float *)(C_warp + 12))[2]),
        "=f"(((float *)(C_warp + 12))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12))[0]),
        "f"(((float *)(C_warp + 12))[1]),
        "f"(((float *)(C_warp + 12))[2]),
        "f"(((float *)(C_warp + 12))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 8 + 32))[0]),
        "=f"(((float *)(C_warp + 8 + 32))[1]),
        "=f"(((float *)(C_warp + 8 + 32))[2]),
        "=f"(((float *)(C_warp + 8 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 8 + 32))[0]),
        "f"(((float *)(C_warp + 8 + 32))[1]),
        "f"(((float *)(C_warp + 8 + 32))[2]),
        "f"(((float *)(C_warp + 8 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 12 + 32))[0]),
        "=f"(((float *)(C_warp + 12 + 32))[1]),
        "=f"(((float *)(C_warp + 12 + 32))[2]),
        "=f"(((float *)(C_warp + 12 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 12 + 32))[0]),
        "f"(((float *)(C_warp + 12 + 32))[1]),
        "f"(((float *)(C_warp + 12 + 32))[2]),
        "f"(((float *)(C_warp + 12 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.z);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.x));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.y));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.y));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16))[0]),
        "=f"(((float *)(C_warp + 16))[1]),
        "=f"(((float *)(C_warp + 16))[2]),
        "=f"(((float *)(C_warp + 16))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16))[0]),
        "f"(((float *)(C_warp + 16))[1]),
        "f"(((float *)(C_warp + 16))[2]),
        "f"(((float *)(C_warp + 16))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20))[0]),
        "=f"(((float *)(C_warp + 20))[1]),
        "=f"(((float *)(C_warp + 20))[2]),
        "=f"(((float *)(C_warp + 20))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20))[0]),
        "f"(((float *)(C_warp + 20))[1]),
        "f"(((float *)(C_warp + 20))[2]),
        "f"(((float *)(C_warp + 20))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 16 + 32))[0]),
        "=f"(((float *)(C_warp + 16 + 32))[1]),
        "=f"(((float *)(C_warp + 16 + 32))[2]),
        "=f"(((float *)(C_warp + 16 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 16 + 32))[0]),
        "f"(((float *)(C_warp + 16 + 32))[1]),
        "f"(((float *)(C_warp + 16 + 32))[2]),
        "f"(((float *)(C_warp + 16 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 20 + 32))[0]),
        "=f"(((float *)(C_warp + 20 + 32))[1]),
        "=f"(((float *)(C_warp + 20 + 32))[2]),
        "=f"(((float *)(C_warp + 20 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 20 + 32))[0]),
        "f"(((float *)(C_warp + 20 + 32))[1]),
        "f"(((float *)(C_warp + 20 + 32))[2]),
        "f"(((float *)(C_warp + 20 + 32))[3]));
  }

  {
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_fused(B_loaded_2.w);
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero_2.z));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale_2.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale_2.w));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero_2.w));
    asm volatile("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale_2.w));

    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24))[0]),
        "=f"(((float *)(C_warp + 24))[1]),
        "=f"(((float *)(C_warp + 24))[2]),
        "=f"(((float *)(C_warp + 24))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24))[0]),
        "f"(((float *)(C_warp + 24))[1]),
        "f"(((float *)(C_warp + 24))[2]),
        "f"(((float *)(C_warp + 24))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28))[0]),
        "=f"(((float *)(C_warp + 28))[1]),
        "=f"(((float *)(C_warp + 28))[2]),
        "=f"(((float *)(C_warp + 28))[3])
      : "r"(((unsigned *)(A_warp))[0]),
        "r"(((unsigned *)(A_warp))[1]),
        "r"(((unsigned *)(A_warp))[2]),
        "r"(((unsigned *)(A_warp))[3]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28))[0]),
        "f"(((float *)(C_warp + 28))[1]),
        "f"(((float *)(C_warp + 28))[2]),
        "f"(((float *)(C_warp + 28))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 24 + 32))[0]),
        "=f"(((float *)(C_warp + 24 + 32))[1]),
        "=f"(((float *)(C_warp + 24 + 32))[2]),
        "=f"(((float *)(C_warp + 24 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.x),
        "r"(B_loaded_fp16.y),
        "f"(((float *)(C_warp + 24 + 32))[0]),
        "f"(((float *)(C_warp + 24 + 32))[1]),
        "f"(((float *)(C_warp + 24 + 32))[2]),
        "f"(((float *)(C_warp + 24 + 32))[3]));
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(((float *)(C_warp + 28 + 32))[0]),
        "=f"(((float *)(C_warp + 28 + 32))[1]),
        "=f"(((float *)(C_warp + 28 + 32))[2]),
        "=f"(((float *)(C_warp + 28 + 32))[3])
      : "r"(((unsigned *)(A_warp))[4]),
        "r"(((unsigned *)(A_warp))[5]),
        "r"(((unsigned *)(A_warp))[6]),
        "r"(((unsigned *)(A_warp))[7]),
        "r"(B_loaded_fp16.z),
        "r"(B_loaded_fp16.w),
        "f"(((float *)(C_warp + 28 + 32))[0]),
        "f"(((float *)(C_warp + 28 + 32))[1]),
        "f"(((float *)(C_warp + 28 + 32))[2]),
        "f"(((float *)(C_warp + 28 + 32))[3]));
  }
}


__global__ void gemm_forward_4bit_cuda_quick_m32n128k32(int G, int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  float C_warp[32 * 2];
  __shared__ half A_shared[16 * (32 + 8) * 2];

  for (int i = 0; i < 32*2; ++i) C_warp[i] = 0;

  int oc_block_num = ((OC + 128 - 1) / 128);
  static constexpr int row_stride_warp = 32 * 8 / 32;
  bool ld_A_flag_1 = (threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;
  bool ld_A_flag_2 = (threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32 + 16) < M;

  half* A_ptr = A + (blockIdx.x / oc_block_num * 32 + (threadIdx.y * row_stride_warp) + threadIdx.x / (32 / 8)) * IC + (threadIdx.x % (32 / 8)) * 8;
  int* B_ptr = B + (threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8))) * 8;
  half* A_shared_ptr = A_shared + threadIdx.y * row_stride_warp * (32 + 8) + (threadIdx.x / (32 / 8)) * (32 + 8) + (threadIdx.x % (32 / 8)) * 8;
  int channel = threadIdx.y * (OC / 8) * 2 + (threadIdx.x / (128 / 8)) * (OC / 8) + (blockIdx.x % oc_block_num) * (128 / 8) + (threadIdx.x % (128 / 8));
  int* zeros_ptr = zeros + (channel / 4) * 2;
  half* scaling_factors_ptr = scaling_factors + (channel / 4) * 16;
  half* C_ptr = C + blockIdx.y * M * OC + (blockIdx.x % oc_block_num) * 128 + threadIdx.y * 64 + (threadIdx.x % 4) * 2;

  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx.y >= IC) k_bound -= 1;

  #pragma unroll
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx.y;
    if(ld_A_flag_1) *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    if(ld_A_flag_2) *(uint4*)(A_shared_ptr + 16 * (32 + 8)) = *(uint4*)(A_ptr + 16 * IC + (k_0_0 * 32));
    __syncthreads();
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);
    uint2 B_loaded_z = *(uint2*)(zeros_ptr + ((k_0_0 * 32) / G) * (OC / 8) * 2);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + ((k_0_0 * 32) / G) * OC * 2);
    uint4 B_loaded_scale_2 = *(uint4*)(scaling_factors_ptr + ((k_0_0 * 32) / G) * OC * 2 + 8);
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2_fused(B_loaded_z.x);
    uint4 B_loaded_zero_2 = dequantize_s4_to_fp16x2_fused(B_loaded_z.y);
    compute_gemm_x2(A_shared, B_ptr_local, C_warp, B_loaded_zero, B_loaded_zero_2, B_loaded_scale, B_loaded_scale_2);
    __syncthreads();
  }

  #pragma unroll
  for (int local_id = 0; local_id < 4; ++local_id) {
    #pragma unroll
    for (int chunk_id = 0; chunk_id < 4; ++chunk_id) {
      int row_offset = ((int)threadIdx.x) / 4 + local_id % 2 * 8;
      int row_offset_2 = ((int)threadIdx.x) / 4 + local_id % 2 * 8 + 16;
      if (row_offset < M) *(__half2*)(C_ptr + chunk_id * 16 + row_offset * OC + (local_id / 2) * 8) = __float22half2_rn(*(float2*)(C_warp + (chunk_id * 8) + local_id * 2));
      if (row_offset_2 < M) *(__half2*)(C_ptr + chunk_id * 16 + row_offset_2 * OC + (local_id / 2) * 8) = __float22half2_rn(*(float2*)(C_warp + 32 + (chunk_id * 8) + local_id * 2));
    }
  }
}


torch::Tensor gemm_forward_cuda_quick(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({split_k_iters, num_in_feats, _kernel.size(1) / 4 * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
    int group_size = num_in_channels / _scaling_factors.size(0);

    if (num_out_channels % 128 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 128");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    if (group_size % 32 != 0)
	      throw std::invalid_argument("Group size should be a multiple of 32");
    int oc_block_num = num_out_channels / 128 / 1;

    dim3 threads_per_block(32, 2);
    dim3 num_blocks_32(oc_block_num, split_k_iters);
    gemm_forward_4bit_cuda_quick_m32n128k32<<<num_blocks_32, threads_per_block>>>(
        group_size, 
        split_k_iters, 
        in_feats, 
        kernel, 
        scaling_factors, 
        zeros, 
        num_in_feats, 
        num_in_channels, 
        num_out_channels, 
        out_feats
    );
    if (split_k_iters > 1) return _out_feats.sum(0);

    return _out_feats;
}


#define HALF_FLT_MAX 65504.F
#define FINAL_MASK 0xffffffff


template<typename T>
inline __device__ T add(T a, T b) {
    return a + b;
}

template<>
inline __device__ half2 add(half2 a, half2 b) {
    return __hadd2(a, b);
}

template<>
inline __device__ half add(half a, half b) {
    return __hadd(a, b);
}

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));  //__shfl_sync bf16 return float when sm < 80
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;
    int                 wid  = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}


template<typename T>
__device__ __forceinline__ T clamp_inf_for_half(const float input)
{
    return input;
}

template<>
__device__ __forceinline__ half clamp_inf_for_half(const float input)
{
    // clamp inf values to enable fp16 training
    return input > 0.0f ? __float2half(min(input, HALF_FLT_MAX - 1000)) : __float2half(max(input, -HALF_FLT_MAX + 1000));
}


static inline __device__ float to_float(half src)
{
    return __half2float(src);
}

static inline __device__ float to_float(float src)
{
    return src;
}


template<typename T>
__global__ void generalT5LayerNorm(
    const T* __restrict input, const T* __restrict gamma, T* output, const float layernorm_eps, int m, int n)
{
    const int tid = threadIdx.x;
    __shared__ float s_variance;
    float variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((to_float(input[blockIdx.x * n + i]) * s_variance) * to_float(__ldg(&gamma[i])));
    }
}


template<typename T>
void invokeGeneralT5LayerNorm(T* out,
                              const T* input,
                              const T* gamma,
                              const float layernorm_eps,
                              const int m,
                              const int n)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));
    if (n % 32 != 0)
        block.x = 1024;
    block.x = block.x / (4 / sizeof(T));
    generalT5LayerNorm<T><<<grid, block>>>(input, gamma, out, layernorm_eps, m, n);
}


template void invokeGeneralT5LayerNorm(half* out,
                              const half* input,
                              const half* gamma,
                              const float layernorm_eps,
                              const int m,
                              const int n);


template void invokeGeneralT5LayerNorm(float* out,
                              const float* input,
                              const float* gamma,
                              const float layernorm_eps,
                              const int m,
                              const int n);


void layernorm_forward_cuda(
    torch::Tensor _input,
    torch::Tensor _gamma,
    torch::Tensor _out,
    float eps)
{
    int m = _input.size(0) * _input.size(1);
    int n = _input.size(2);

    auto input = reinterpret_cast<half*>(_input.data_ptr<at::Half>());
    auto gamma = reinterpret_cast<half*>(_gamma.data_ptr<at::Half>());
    auto out = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

    invokeGeneralT5LayerNorm(out, input, gamma, eps, m, n);
}
