// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

// Fused TurboQuant copy-to-KV-cache with Hadamard rotation and 4-bit quantization.
// Applies the Walsh-Hadamard transform to new K/V tokens, quantizes to 4-bit
// centroid indices packed into u32 words with fp32 L2 norm, then writes into
// the present KV cache (stored as u32).
// Each workgroup handles one (batch, head, seq) slice for either K or V.
class TurboQuantHadamardProgram final : public Program<TurboQuantHadamardProgram> {
 public:
  TurboQuantHadamardProgram(const std::string& kernel_name, bool has_past, bool kv_BNSH,
                            bool past_present_share_buffer, int head_size, int head_size_log2, int components,
                            int compressed_head_size_u32,
                            bool prepare_indirect_dispatch = false, bool use_seqlen_k = false)
      : Program{kernel_name}, has_past_(has_past), kv_BNSH_(kv_BNSH),
        past_present_share_buffer_(past_present_share_buffer),
        head_size_(head_size), head_size_log2_(head_size_log2), components_(components),
        compressed_head_size_u32_(compressed_head_size_u32),
        prepare_indirect_dispatch_(prepare_indirect_dispatch), use_seqlen_k_(use_seqlen_k) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"tile_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"num_slices_per_kv", ProgramUniformVariableDataType::Uint32},
                                          {"present_seq_length", ProgramUniformVariableDataType::Uint32},
                                          {"compressed_head_size_u32", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_;
  bool kv_BNSH_;
  bool past_present_share_buffer_;
  int head_size_;
  int head_size_log2_;
  int components_;
  int compressed_head_size_u32_;
  bool prepare_indirect_dispatch_;
  bool use_seqlen_k_;
};

Status TurboQuantCopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                             const Tensor* K, const Tensor* past_key, Tensor* present_key,
                             const Tensor* V, const Tensor* past_value, Tensor* present_value,
                             uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
