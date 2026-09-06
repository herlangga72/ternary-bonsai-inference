//! Minimal hand-rolled FFI bindings to the PrismML llama.cpp fork.
//!
//! Struct layouts and signatures are transcribed from `include/llama.h` of
//! https://github.com/PrismML-Eng/llama.cpp (branch `prism`). The fork is the only
//! runtime that understands the ternary `Q2_0` (ggml type 42) weights used by
//! Ternary-Bonsai-27B.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_void};

pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;

// opaque handle types
#[repr(C)]
pub struct llama_model {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_context {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_vocab {
    _p: [u8; 0],
}
#[repr(C)]
pub struct llama_sampler {
    _p: [u8; 0],
}

// ---------------------------------------------------------------------------
// struct llama_model_params
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_model_params {
    // NULL-terminated list of devices to use for offloading
    pub devices: *mut *mut c_void,
    // NULL-terminated list of buffer types to use for tensors that match a pattern
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32, // enum llama_split_mode
    pub load_mode: i32,  // enum llama_load_mode
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *const c_void, // llama_progress_callback
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: u8,
    pub check_tensors: u8,
    pub use_extra_bufts: u8,
    pub no_host: u8,
    pub no_alloc: u8,
    pub load_mtp: u8,
}

// ---------------------------------------------------------------------------
// struct llama_sampler_seq_config / struct llama_context_params
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_sampler_seq_config {
    pub seq_id: llama_seq_id,
    pub sampler: *mut llama_sampler,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,          // enum llama_context_type
    pub rope_scaling_type: i32, // enum llama_rope_scaling_type
    pub pooling_type: i32,      // enum llama_pooling_type
    pub attention_type: i32,    // enum llama_attention_type
    pub flash_attn_type: i32,   // enum llama_flash_attn_type
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *const c_void, // ggml_backend_sched_eval_callback
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32, // enum ggml_type
    pub type_v: i32, // enum ggml_type
    pub path_kv_mean_center: *const c_char,
    pub abort_callback: *const c_void, // ggml_abort_callback
    pub abort_callback_data: *mut c_void,
    pub embeddings: u8,
    pub offload_kqv: u8,
    pub no_perf: u8,
    pub op_offload: u8,
    pub swa_full: u8,
    pub kv_unified: u8,
    pub samplers: *mut llama_sampler_seq_config,
    pub n_samplers: usize,
    pub ctx_other: *mut llama_context,
}

// ---------------------------------------------------------------------------
// struct llama_batch
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut f32,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

// ---------------------------------------------------------------------------
// struct llama_sampler_chain_params
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_sampler_chain_params {
    pub no_perf: u8,
}

// ---------------------------------------------------------------------------
// struct llama_chat_message
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_chat_message {
    pub role: *const c_char,
    pub content: *const c_char,
}

// LLAMA_DEFAULT_SEED
pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

// ggml log levels used by the log callback
pub const GGML_LOG_LEVEL_ERROR: i32 = 4;
pub const GGML_LOG_LEVEL_WARN: i32 = 3;
pub const GGML_LOG_LEVEL_INFO: i32 = 2;
pub const GGML_LOG_LEVEL_CONT: i32 = 5;

extern "C" {
    pub fn llama_backend_init();
    pub fn llama_log_set(callback: *const c_void, user_data: *mut c_void);
    pub fn llama_version() -> *const c_char;

    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);

    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> i32;
    pub fn llama_model_desc(model: *const llama_model, buf: *mut c_char, buf_size: usize) -> i32;
    pub fn llama_model_size(model: *const llama_model) -> u64;
    pub fn llama_model_n_params(model: *const llama_model) -> u64;
    pub fn llama_model_chat_template(
        model: *const llama_model,
        name: *const c_char,
    ) -> *const c_char;

    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_set_n_threads(
        ctx: *mut llama_context,
        n_threads: i32,
        n_threads_batch: i32,
    );

    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: u8,
        parse_special: u8,
    ) -> i32;

    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: u8,
    ) -> i32;

    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> u8;
    pub fn llama_vocab_get_text(
        vocab: *const llama_vocab,
        token: llama_token,
    ) -> *const c_char;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> u8;

    pub fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const llama_chat_message,
        n_msg: usize,
        add_ass: u8,
        buf: *mut c_char,
        length: i32,
    ) -> i32;

    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut f32;
    pub fn llama_kv_cache_clear(ctx: *mut llama_context);

    pub fn llama_sampler_chain_init(
        params: llama_sampler_chain_params,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
}
