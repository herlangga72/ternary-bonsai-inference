//! Minimal Vulkan compute host (G0, ash).
//!
//! Creates an instance + logical device on the first suitable GPU (prefers a
//! discrete card - the RX 7600 - over the integrated Vega), sets up one
//! compute pipeline for the PQ2_0 matvec, and runs it against host-visible
//! buffers. Mirrors the OpenCL path in opencl.rs one-to-one so the same
//! CPU-vs-GPU row checks validate both backends.
//!
//! Everything is explicit and synchronous (submit + wait per launch). That is
//! fine for G0 validation; G2 will keep weights in device-local VRAM and reuse
//! command buffers/descriptors in an in-order queue.

#![allow(dead_code)]

use ash::vk;
use ash::{Device, Entry, Instance};
use std::ffi::CStr;

/// SPIR-V for the PQ2_0 matvec shader, compiled from `shaders/pq2_matvec.comp`
/// by build.rs (glslangValidator) into OUT_DIR.
pub const PQ2_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/pq2_matvec.spv"));

/// SPIR-V for the RMSNorm shader (`shaders/rms_norm.comp`).
pub const RMS_NORM_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/rms_norm.spv"));

/// SPIR-V for the fused elementwise shader (`shaders/elem.comp`).
pub const ELEM_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/elem.spv"));

/// SPIR-V for the row-wise norm shader (`shaders/norm_rows.comp`).
pub const NORM_ROWS_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/norm_rows.spv"));

/// SPIR-V for the masked softmax shader (`shaders/softmax_row.comp`).
pub const SOFTMAX_ROW_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/softmax_row.spv"));

/// SPIR-V for the IMROPE shader (`shaders/rope_imrope.comp`).
pub const ROPE_IMROPE_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/rope_imrope.spv"));

/// SPIR-V for the gated-delta-net step (`shaders/gdn_step.comp`).
pub const GDN_STEP_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/gdn_step.spv"));

/// SPIR-V for the attention trio (`shaders/attn_scores.comp`,
/// `softmax_inplace.comp`, `attn_out.comp`).
pub const ATTN_SCORES_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/attn_scores.spv"));
pub const SOFTMAX_INPLACE_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/softmax_inplace.spv"));
pub const ATTN_OUT_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/attn_out.spv"));

/// SPIR-V for the two-pass PQ2_0 matvec (`shaders/pq2_partial.comp`,
/// `shaders/pq2_rowsum.comp`).
pub const PQ2_PARTIAL_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/pq2_partial.spv"));
pub const PQ2_ROWSUM_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/pq2_rowsum.spv"));

/// SPIR-V for device arena ops (slice 1 of the full decode):
/// residual add, fused q|gate split, KV append.
pub const ADD_RESIDUAL_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/add_residual.spv"));
pub const SPLIT_QGATE_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/split_qgate.spv"));
pub const KV_STORE_SPV: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/kv_store.spv"));

/// Fixed gdn shapes for qwen35 (must match the shader constants).
pub const GDN_DK: usize = 2048; // H_K * S
pub const GDN_DI: usize = 6144; // H_V * S
pub const GDN_HV: usize = 48;
pub const GDN_INS_LEN: usize = 2 * GDN_DK + GDN_DI + 2 * GDN_HV; // q,k,v,gate,beta
pub const GDN_STATE_LEN: usize = GDN_HV * 128 * 128;

/// Shader workgroup width; must match `layout(local_size_x = ...)` in the
/// GLSL source.
pub const LOCAL_X: u32 = 256;

struct RawBuf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: usize,
    ptr: *mut u8,
}

unsafe impl Send for RawBuf {}

impl RawBuf {
    unsafe fn data(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
    unsafe fn data_mut(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }
}

pub struct Gpu {
    _entry: Entry,
    _instance: Instance,
    device: Device,
    physical: vk::PhysicalDevice,
    qfi: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_layout: vk::DescriptorSetLayout,
    /// Extra compute pipelines on the shared layout (all use the same 3-SSBO
    /// set layout + 8-byte push range, so any shader that fits can register).
    kernels: std::collections::HashMap<String, vk::Pipeline>,
    staging: Option<RawBuf>,
    mv: Option<MvCache>,
    // per-token recording state (full device decode)
    rec_pool: Option<vk::DescriptorPool>,
    rec_cb: Option<vk::CommandBuffer>,
    rec_fence: Option<vk::Fence>,
    rec_dummy: Option<DevBuf>,
    rec_open: bool,
    pub name: String,
    pub discrete: bool,
}

/// Device buffer (typically device-local VRAM). Never mapped directly; upload
/// through a staging buffer (`Gpu::upload`).
pub struct DevBuf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub len: usize,
}

/// Reusable per-matvec resources (G3): one x/y host pair, one descriptor set
/// (re-pointed at the current weight buffer with vkUpdateDescriptorSets), one
/// command buffer and one fence. Eliminates the per-call allocation + driver
/// object churn that made single-shot matvecs ~100x slower than the kernel.
struct MvCache {
    xb: RawBuf,
    yb: RawBuf,
    wdummy: DevBuf,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    max_ne0: usize,
    max_rows: usize,
}

fn err<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e}"))
}

fn pick_queue_family(
    instance: &Instance,
    physical: vk::PhysicalDevice,
) -> Option<u32> {
    // Prefer a pure-compute family; fall back to a graphics+compute one.
    let props = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let mut fallback = None;
    for (i, q) in props.iter().enumerate() {
        if q.queue_count == 0 {
            continue;
        }
        if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            if !q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                return Some(i as u32);
            }
            fallback.get_or_insert(i as u32);
        }
    }
    fallback
}

impl Gpu {
    pub fn open() -> Result<Gpu, String> {
        let entry = err(unsafe { Entry::load() }, "Entry::load")?;

        let app_name = std::ffi::CString::new("bonsai-run").unwrap();
        let engine_name = std::ffi::CString::new("bonsai-vk").unwrap();
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_2);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = err(
            unsafe { entry.create_instance(&instance_info, None) },
            "create_instance",
        )?;

        let physicals =
            err(unsafe { instance.enumerate_physical_devices() }, "enumerate_physical_devices")?;
        if physicals.is_empty() {
            return Err("no Vulkan physical devices".into());
        }

        // Pick the first discrete GPU, else the first device of any kind.
        let mut pick: Option<vk::PhysicalDevice> = None;
        let mut pick_discrete = false;
        let mut pick_fallback: Option<vk::PhysicalDevice> = None;
        for &p in &physicals {
            let props = unsafe { instance.get_physical_device_properties(p) };
            let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            if discrete && !pick_discrete {
                pick = Some(p);
                pick_discrete = true;
            }
            if pick_fallback.is_none() {
                pick_fallback = Some(p);
            }
        }
        let physical = pick.or(pick_fallback).unwrap();
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let qfi = pick_queue_family(&instance, physical)
            .ok_or_else(|| format!("{name}: no compute queue family"))?;

        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        let device = err(
            unsafe { instance.create_device(physical, &device_info, None) },
            "create_device",
        )?;
        let queue = unsafe { device.get_device_queue(qfi, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qfi)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = err(
            unsafe { device.create_command_pool(&pool_info, None) },
            "create_command_pool",
        )?;

        // ---- descriptor set layout: 3 storage buffers -----------------------
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_layout = err(
            unsafe { device.create_descriptor_set_layout(&layout_info, None) },
            "create_descriptor_set_layout",
        )?;

        // ---- pipeline layout: shared push constant range --------------------
        // All kernels on the shared layout declare their own push-constant
        // block at offset 0; 64 bytes covers every current and planned shader
        // (matvec 8B, rms_norm 8B, elem 8B, row norms 16B, gdn up to 64B).
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(64);
        let set_layouts = [desc_layout];
        let push_ranges = [push_range];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = err(
            unsafe { device.create_pipeline_layout(&pl_info, None) },
            "create_pipeline_layout",
        )?;

        // ---- compute pipeline --------------------------------------------------
        let words: Vec<u32> = PQ2_MATVEC_SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = err(
            unsafe { device.create_shader_module(&module_info, None) },
            "create_shader_module",
        )?;
        let main_name = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&main_name);
        let pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipelines = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None) }
            .map_err(|(_, e)| format!("create_compute_pipelines: {e}"))?;
        let pipeline = pipelines[0];
        unsafe { device.destroy_shader_module(module, None) };

        Ok(Gpu {
            _entry: entry,
            _instance: instance,
            device,
            physical,
            qfi,
            queue,
            command_pool,
            pipeline,
            pipeline_layout,
            desc_layout,
            kernels: std::collections::HashMap::new(),
            staging: None,
            mv: None,
            rec_pool: None,
            rec_cb: None,
            rec_fence: None,
            rec_dummy: None,
            rec_open: false,
            name,
            discrete,
        })
    }

    /// Pre-allocate the reusable matvec cache. `max_ne0`/`max_rows` must cover
    /// every matvec that will run through `matvec_on` (x and y are re-used and
    /// never resized).
    pub fn prep_matvec_cache(&mut self, max_ne0: usize, max_rows: usize) -> Result<(), String> {
        if self.mv.is_some() {
            return Ok(());
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        unsafe {
            let xb = self.create_host_buffer(max_ne0.max(1) * 4, usage)?;
            let yb = self.create_host_buffer(max_rows.max(1) * 4, usage)?;
            let wdummy = self.create_dev_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER)?;

            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes);
            let pool = err(
                self.device.create_descriptor_pool(&pool_info, None),
                "prep_matvec_cache: pool",
            )?;
            let layouts = [self.desc_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let set = self
                .device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("prep_matvec_cache: set: {e}"))?[0];

            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = self
                .device
                .allocate_command_buffers(&cb_info)
                .map_err(|e| format!("prep_matvec_cache: cb: {e}"))?[0];
            let fence = err(
                self.device.create_fence(&vk::FenceCreateInfo::default(), None),
                "prep_matvec_cache: fence",
            )?;
            self.mv = Some(MvCache {
                xb,
                yb,
                wdummy,
                pool,
                set,
                cb,
                fence,
                max_ne0,
                max_rows,
            });
        }
        Ok(())
    }

    /// Index of a memory type that satisfies `required` flags (host-visible +
    /// coherent for G0 buffers), or None.
    fn memory_type_index(
        &self,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        let props = unsafe { self._instance.get_physical_device_memory_properties(self.physical) };
        for (i, mt) in props.memory_types.iter().enumerate() {
            if mt.property_flags.contains(required) {
                return Ok(i as u32);
            }
        }
        Err(format!(
            "no memory type with flags {required:?} ({} types)",
            props.memory_type_count
        ))
    }

    /// Create a host-visible, host-coherent buffer and map it.
    unsafe fn create_host_buffer(
        &self,
        len: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<RawBuf, String> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(len as vk::DeviceSize)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = err(
            self.device.create_buffer(&buffer_info, None),
            "create_buffer",
        )?;
        let req = self.device.get_buffer_memory_requirements(buffer);
        let mem_idx = self.memory_type_index(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_idx);
        let memory = err(self.device.allocate_memory(&alloc_info, None), "allocate_memory")?;
        err(
            self.device.bind_buffer_memory(buffer, memory, 0),
            "bind_buffer_memory",
        )?;
        let ptr = err(
            self.device.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty()),
            "map_memory",
        )? as *mut u8;
        Ok(RawBuf { buffer, memory, len, ptr })
    }

    unsafe fn destroy_host_buffer(&self, buf: RawBuf) {
        self.device.unmap_memory(buf.memory);
        self.device.destroy_buffer(buf.buffer, None);
        self.device.free_memory(buf.memory, None);
    }

    // ---- G1: device-local buffers + staging uploads -------------------------

    /// Memory type that is device-local (preferred for weights), falling back
    /// to host-visible if the device has no pure device-local heap (iGPU
    /// carve-outs usually still do).
    fn device_local_mem_index(&self) -> Result<u32, String> {
        let props = unsafe {
            self._instance
                .get_physical_device_memory_properties(self.physical)
        };
        for (i, mt) in props.memory_types.iter().enumerate() {
            if mt.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                return Ok(i as u32);
            }
        }
        self.memory_type_index(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// Device buffer for weights: storage + transfer-dst, allocated on the
    /// preferred device-local heap. Convenience used by the Weights accel.
    pub fn create_weight_buffer(&mut self, len: usize) -> Result<DevBuf, String> {
        self.create_dev_buffer(
            len,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
    }

    pub fn create_dev_buffer(
        &self,
        len: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<DevBuf, String> {
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(len as vk::DeviceSize)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = err(self.device.create_buffer(&info, None), "create_dev_buffer")?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let idx = self.device_local_mem_index()?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(idx);
            let memory = err(self.device.allocate_memory(&alloc, None), "alloc dev memory")?;
            err(
                self.device.bind_buffer_memory(buffer, memory, 0),
                "bind dev buffer",
            )?;
            Ok(DevBuf { buffer, memory, len })
        }
    }

    pub fn destroy_dev_buffer(&self, b: DevBuf) {
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// Grow (or create) the reusable host-visible staging buffer to at least
    /// `len` bytes. Takes no long-lived borrow so callers can touch other
    /// fields afterwards.
    fn ensure_staging_size(&mut self, len: usize) -> Result<(), String> {
        if let Some(s) = &self.staging {
            if s.len >= len {
                return Ok(());
            }
        }
        unsafe {
            if let Some(old) = self.staging.take() {
                self.destroy_host_buffer(old);
            }
            let s = self.create_host_buffer(len, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.staging = Some(s);
        }
        Ok(())
    }

    /// Copy `data` into a device buffer through the staging buffer with a
    /// dedicated one-shot command buffer + fence.
    pub fn upload(&mut self, dev: &DevBuf, data: &[u8]) -> Result<(), String> {
        if data.len() > dev.len {
            return Err(format!("upload {} > device buffer {}", data.len(), dev.len));
        }
        self.ensure_staging_size(data.len())?;
        unsafe {
            let staging_ptr = self.staging.as_ref().unwrap().ptr;
            std::ptr::copy_nonoverlapping(data.as_ptr(), staging_ptr, data.len());
            let staging_buf = self.staging.as_ref().unwrap().buffer;

            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cbs = err(
                self.device.allocate_command_buffers(&cb_info),
                "upload: alloc cmd",
            )?;
            let cb = cbs[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            err(self.device.begin_command_buffer(cb, &begin), "upload: begin")?;
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(data.len() as vk::DeviceSize);
            self.device
                .cmd_copy_buffer(cb, staging_buf, dev.buffer, &[region]);
            err(self.device.end_command_buffer(cb), "upload: end")?;

            let fence = err(
                self.device.create_fence(&vk::FenceCreateInfo::default(), None),
                "upload: fence",
            )?;
            let cbs2 = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs2);
            err(
                self.device.queue_submit(self.queue, &[submit], fence),
                "upload: submit",
            )?;
            err(
                self.device.wait_for_fences(&[fence], true, u64::MAX),
                "upload: wait",
            )?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            Ok(())
        }
    }

    /// Create a descriptor set for one [w, x, y] storage-buffer binding from a
    /// fresh pool (max_sets 1). Used by the bench and, later, per launch.
    unsafe fn make_set(
        &self,
        w: vk::Buffer,
        x: vk::Buffer,
        y: vk::Buffer,
    ) -> Result<(vk::DescriptorPool, vk::DescriptorSet), String> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        let pool = err(
            self.device.create_descriptor_pool(&pool_info, None),
            "create_descriptor_pool",
        )?;
        let layouts = [self.desc_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = err(
            self.device.allocate_descriptor_sets(&alloc_info),
            "allocate_descriptor_sets",
        )?;
        let set = sets[0];
        let infos = [
            vk::DescriptorBufferInfo::default().buffer(w).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(x).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(y).range(vk::WHOLE_SIZE),
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[2..3]),
        ];
        self.device.update_descriptor_sets(&writes, &[]);
        Ok((pool, set))
    }

    /// Record the bind + push + one dispatch for the matvec (shared by bench
    /// and the future decode loop).
    unsafe fn record_matvec(
        &self,
        cb: vk::CommandBuffer,
        set: vk::DescriptorSet,
        ne0: u32,
        base_row: u32,
        n_rows: usize,
    ) {
        self.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        self.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[set],
            &[],
        );
        let pc = [ne0, base_row];
        let pc_bytes = std::slice::from_raw_parts(
            pc.as_ptr() as *const u8,
            std::mem::size_of_val(&pc),
        );
        self.device.cmd_push_constants(
            cb,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n_rows as u32).div_ceil(LOCAL_X).max(1);
        self.device.cmd_dispatch(cb, groups, 1, 1);
    }

    /// Benchmark + validate matvec with the G1 layout: weights live in a
    /// device-local buffer uploaded once; x and y are persistent host-visible
    /// buffers; `iters` dispatches are recorded into one command buffer so the
    /// reported time is kernel throughput, not per-launch host overhead.
    /// Returns (seconds per iteration, final y).
    pub fn matvec_bench(
        &mut self,
        payload: &[u8],
        ne0: usize,
        base_row: u32,
        n_rows: usize,
        x: &[f32],
        iters: u32,
    ) -> Result<(f64, Vec<f32>), String> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let w = self.create_dev_buffer(payload.len(), usage)?;
        self.upload(&w, payload)?;
        let mut x_host = unsafe { self.create_host_buffer(ne0 * 4, vk::BufferUsageFlags::STORAGE_BUFFER) }?;
        let y_len = n_rows * 4;
        let y_host =
            unsafe { self.create_host_buffer(y_len, vk::BufferUsageFlags::STORAGE_BUFFER) }?;
        unsafe {
            x_host.data_mut().copy_from_slice(bytemuck_slice(x));
        }

        let (pool, set) = unsafe { self.make_set(w.buffer, x_host.buffer, y_host.buffer) }?;

        let cb_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe { self.device.allocate_command_buffers(&cb_info) }
            .map_err(|e| format!("bench: alloc cmd: {e}"))?[0];

        let t0 = std::time::Instant::now();
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cb, &begin)
                .map_err(|e| format!("bench: begin: {e}"))?;
            // bind + push once, then dispatch `iters` times back to back
            self.record_matvec(cb, set, ne0 as u32, base_row, n_rows);
            for _ in 1..iters {
                self.device.cmd_dispatch(cb, (n_rows as u32).div_ceil(LOCAL_X).max(1), 1, 1);
            }
            self.device
                .end_command_buffer(cb)
                .map_err(|e| format!("bench: end: {e}"))?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("bench: fence: {e}"))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| format!("bench: submit: {e}"))?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("bench: wait: {e}"))?;
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            self.device.destroy_descriptor_pool(pool, None);

            let out = y_host.data()[..y_len]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();

            self.destroy_host_buffer(x_host);
            self.destroy_host_buffer(y_host);
            self.destroy_dev_buffer(w);
            Ok((dt, out))
        }
    }

    // ---- G2a: generic kernels on the shared layout ---------------------------

    /// Benchmark the pq2_atom (bandwidth-oriented) matvec kernel: weights in a
    /// device-local buffer, x/y host-visible, `iters` dispatches recorded into
    /// one command buffer for kernel-only timing, then a clean single dispatch
    /// for the returned y. Returns (seconds per iteration, final y).
    pub fn matvec_bench_atom(
        &mut self,
        payload: &[u8],
        ne0: usize,
        base_row: u32,
        n_rows: usize,
        x: &[f32],
        iters: u32,
    ) -> Result<(f64, Vec<f32>), String> {
        self.ensure_kernel("pq2_partial", PQ2_PARTIAL_SPV)?;
        self.ensure_kernel("pq2_rowsum", PQ2_ROWSUM_SPV)?;
        let pipeline_p = *self
            .kernels
            .get("pq2_partial")
            .ok_or("pq2_partial not registered")?;
        let pipeline_r = *self
            .kernels
            .get("pq2_rowsum")
            .ok_or("pq2_rowsum not registered")?;
        let nblocks = ne0.div_ceil(128);
        let n_partials = n_rows.saturating_mul(nblocks).max(1);
        let groups_p = (n_partials as u32).div_ceil(256).max(1);
        let groups_r = n_rows.max(1) as u32;

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let w = self.create_dev_buffer(payload.len(), usage)?;
        self.upload(&w, payload)?;
        let partial_len = n_partials * 4;
        let partials =
            self.create_dev_buffer(partial_len, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let mut x_host =
            unsafe { self.create_host_buffer(ne0 * 4, vk::BufferUsageFlags::STORAGE_BUFFER) }?;
        let mut y_host =
            unsafe { self.create_host_buffer(n_rows * 4, vk::BufferUsageFlags::STORAGE_BUFFER) }?;
        let mut dummy =
            unsafe { self.create_host_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER) }?;
        unsafe {
            x_host.data_mut().copy_from_slice(bytemuck_slice(x));
            dummy.data_mut().fill(0);
        }
        let (pool_p, set_p) =
            unsafe { self.make_set(w.buffer, x_host.buffer, partials.buffer) }?;
        let (pool_r, set_r) =
            unsafe { self.make_set(partials.buffer, dummy.buffer, y_host.buffer) }?;

        let cb_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe { self.device.allocate_command_buffers(&cb_info) }
            .map_err(|e| format!("matvec_atom: alloc: {e}"))?[0];

        let pc_p = [ne0 as u32, base_row];
        let pc_pb = unsafe {
            std::slice::from_raw_parts(pc_p.as_ptr() as *const u8, std::mem::size_of_val(&pc_p))
        };
        let pc_r = [nblocks as u32, 0u32];
        let pc_rb = unsafe {
            std::slice::from_raw_parts(pc_r.as_ptr() as *const u8, std::mem::size_of_val(&pc_r))
        };
        let barrier_bufs = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                )
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(partials.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                )
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(y_host.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE),
        ];

        let t0 = std::time::Instant::now();
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cb, &begin)
                .map_err(|e| format!("matvec_atom: begin: {e}"))?;
            for _ in 0..iters {
                self.device
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline_p);
                self.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[set_p],
                    &[],
                );
                self.device.cmd_push_constants(
                    cb,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_pb,
                );
                self.device.cmd_dispatch(cb, groups_p, 1, 1);
                self.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barrier_bufs,
                    &[],
                );
                self.device
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline_r);
                self.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[set_r],
                    &[],
                );
                self.device.cmd_push_constants(
                    cb,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_rb,
                );
                self.device.cmd_dispatch(cb, groups_r, 1, 1);
                self.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barrier_bufs,
                    &[],
                );
            }
            self.device
                .end_command_buffer(cb)
                .map_err(|e| format!("matvec_atom: end: {e}"))?;
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("matvec_atom: fence: {e}"))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| format!("matvec_atom: submit: {e}"))?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("matvec_atom: wait: {e}"))?;
            let dt = t0.elapsed().as_secs_f64() / iters as f64;

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            self.device.destroy_descriptor_pool(pool_p, None);
            self.device.destroy_descriptor_pool(pool_r, None);

            let out = y_host.data()[..n_rows * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(x_host);
            self.destroy_host_buffer(y_host);
            self.destroy_host_buffer(dummy);
            self.destroy_dev_buffer(w);
            self.destroy_dev_buffer(partials);
            Ok((dt, out))
        }
    }

    /// Build (once) a compute pipeline from SPIR-V bytes on the shared layout.
    /// Kernels must use at most bindings 0..3 of the descriptor set and an
    /// 8-byte push constant block.
    pub fn ensure_kernel(&mut self, name: &str, spv: &[u8]) -> Result<(), String> {
        if self.kernels.contains_key(name) {
            return Ok(());
        }
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        unsafe {
            let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = err(
                self.device.create_shader_module(&module_info, None),
                "ensure_kernel: module",
            )?;
            let main_name = std::ffi::CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&main_name);
            let pipe_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.pipeline_layout);
            let pipes = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
                .map_err(|(_, e)| format!("ensure_kernel({name}): {e}"))?;
            self.device.destroy_shader_module(module, None);
            self.kernels.insert(name.to_string(), pipes[0]);
        }
        Ok(())
    }

    /// Submit a one-shot dispatch of a registered kernel and wait. `set` must
    /// already describe the kernel's buffers.
    fn run_registered(
        &self,
        name: &str,
        set: vk::DescriptorSet,
        pc: &[u8],
        local: u32,
        groups_x: u32,
    ) -> Result<(), String> {
        let &pipeline = self
            .kernels
            .get(name)
            .ok_or_else(|| format!("kernel '{name}' not registered"))?;
        unsafe {
            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = self
                .device
                .allocate_command_buffers(&cb_info)
                .map_err(|e| format!("run_registered: alloc: {e}"))?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cb, &begin)
                .map_err(|e| format!("run_registered: begin: {e}"))?;
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc,
            );
            self.device.cmd_dispatch(cb, groups_x.max(1), 1, 1);
            self.device
                .end_command_buffer(cb)
                .map_err(|e| format!("run_registered: end: {e}"))?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("run_registered: fence: {e}"))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| format!("run_registered: submit: {e}"))?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("run_registered: wait: {e}"))?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            Ok(())
        }
    }

    /// RMSNorm over one vector: o = rms_norm(x, w). Mirrors kernels::rms_norm
    /// (single workgroup of 256, reduction in shared memory). x and w must be
    /// the same length; the norm weight `w` is typically small (n_embd).
    pub fn rms_norm(&mut self, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, String> {
        let n = x.len();
        if n == 0 || w.len() != n {
            return Err(format!("rms_norm: x len {n}, w len {}", w.len()));
        }
        self.ensure_kernel("rms_norm", RMS_NORM_SPV)?;
        unsafe {
            let mut xb = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut wb = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let ob = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            xb.data_mut().copy_from_slice(bytemuck_slice(x));
            wb.data_mut().copy_from_slice(bytemuck_slice(w));

            let (pool, set) = self.make_set(xb.buffer, wb.buffer, ob.buffer)?;
            let pc = [n as u32, eps.to_bits()];
            let pc_bytes =
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc));
            self.run_registered("rms_norm", set, pc_bytes, 256, 1)?;

            let out = ob.data()[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(xb);
            self.destroy_host_buffer(wb);
            self.destroy_host_buffer(ob);
            self.device.destroy_descriptor_pool(pool, None);
            Ok(out)
        }
    }

    /// Fused elementwise op over `a` (and optional `b`), modes as documented
    /// in shaders/elem.comp. Mirrors kernels.rs activation helpers.
    pub fn elem(&mut self, a: &[f32], b: Option<&[f32]>, mode: u32) -> Result<Vec<f32>, String> {
        let n = a.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if let Some(bb) = b {
            if bb.len() != n {
                return Err(format!("elem: a len {n}, b len {}", bb.len()));
            }
        }
        self.ensure_kernel("elem", ELEM_SPV)?;
        unsafe {
            let mut ab = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut bb = self.create_host_buffer(n.max(1) * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let ob = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            ab.data_mut().copy_from_slice(bytemuck_slice(a));
            if let Some(bbval) = b {
                bb.data_mut().copy_from_slice(bytemuck_slice(bbval));
            } else {
                bb.data_mut().fill(0);
            }

            let (pool, set) = self.make_set(ab.buffer, bb.buffer, ob.buffer)?;
            let pc = [n as u32, mode];
            let pc_bytes =
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc));
            let groups = (n as u32).div_ceil(256).max(1);
            self.run_registered("elem", set, pc_bytes, 256, groups)?;

            let out = ob.data()[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(ab);
            self.destroy_host_buffer(bb);
            self.destroy_host_buffer(ob);
            self.device.destroy_descriptor_pool(pool, None);
            Ok(out)
        }
    }

    /// Row-wise RMS (mode 0) or L2 (mode 1) normalization. `x` holds
    /// `rows * row_len` floats laid out row-major; `w` (row_len long, rms
    /// only) is applied per element. Mirrors kernels::rms_norm_rows /
    /// l2_norm_rows.
    pub fn norm_rows(
        &mut self,
        x: &[f32],
        w: Option<&[f32]>,
        mode: u32,
        row_len: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let total = x.len();
        if total == 0 || total % row_len != 0 {
            return Err(format!(
                "norm_rows: total {total} not divisible by row_len {row_len}"
            ));
        }
        if mode == 0 && w.map(|w| w.len()) != Some(row_len) {
            return Err("norm_rows rms: w must be row_len long".into());
        }
        self.ensure_kernel("norm_rows", NORM_ROWS_SPV)?;
        let rows = total / row_len;
        unsafe {
            let mut xb = self.create_host_buffer(total * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let w_len = if mode == 0 { row_len } else { 1 };
            let mut wb = self.create_host_buffer(w_len * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let ob = self.create_host_buffer(total * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            xb.data_mut().copy_from_slice(bytemuck_slice(x));
            if let Some(wv) = w {
                wb.data_mut().copy_from_slice(bytemuck_slice(wv));
            } else {
                wb.data_mut().fill(0);
            }
            let (pool, set) = self.make_set(xb.buffer, wb.buffer, ob.buffer)?;
            let pc = [total as u32, row_len as u32, mode, eps.to_bits()];
            let pc_bytes =
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc));
            self.run_registered("norm_rows", set, pc_bytes, 256, rows as u32)?;
            let out = ob.data()[..total * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(xb);
            self.destroy_host_buffer(wb);
            self.destroy_host_buffer(ob);
            self.device.destroy_descriptor_pool(pool, None);
            Ok(out)
        }
    }

    /// Masked softmax over one row. Mirrors kernels::softmax_rows for a single
    /// row; -inf entries become zero via exp(-inf).
    pub fn softmax_row(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
        let n = x.len();
        if n == 0 {
            return Err("softmax_row: empty input".into());
        }
        self.ensure_kernel("softmax_row", SOFTMAX_ROW_SPV)?;
        unsafe {
            let mut xb = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut dummy = self.create_host_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let ob = self.create_host_buffer(n * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            xb.data_mut().copy_from_slice(bytemuck_slice(x));
            dummy.data_mut().fill(0);

            let (pool, set) = self.make_set(xb.buffer, dummy.buffer, ob.buffer)?;
            let pc = [n as u32];
            let pc_bytes =
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc));
            self.run_registered("softmax_row", set, pc_bytes, 256, 1)?;

            let out = ob.data()[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(xb);
            self.destroy_host_buffer(dummy);
            self.destroy_host_buffer(ob);
            self.device.destroy_descriptor_pool(pool, None);
            Ok(out)
        }
    }

    /// Apply IMROPE to `nheads` contiguous head vectors of `hd` floats each,
    /// rotating the first `n_dims` elements in place (pure-text positions all
    /// equal `pos`). Mirrors rope::rope_imrope.
    pub fn rope_imrope(
        &mut self,
        heads: &[f32],
        hd: usize,
        n_dims: usize,
        pos: u32,
        freq_base: f32,
        sections: [u32; 4],
    ) -> Result<Vec<f32>, String> {
        let total = heads.len();
        if total == 0 || total % hd != 0 || n_dims > hd || n_dims % 2 != 0 {
            return Err(format!(
                "rope_imrope: total {total}, hd {hd}, n_dims {n_dims} invalid"
            ));
        }
        self.ensure_kernel("rope_imrope", ROPE_IMROPE_SPV)?;
        let nheads = total / hd;
        unsafe {
            let mut xb = self.create_host_buffer(total * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut dummy = self.create_host_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            xb.data_mut().copy_from_slice(bytemuck_slice(heads));
            dummy.data_mut().fill(0);

            // descriptor needs three buffers for the shared layout; binding 1
            // (dummy) and 2 are unused by this shader
            let (pool, set) = self.make_set(xb.buffer, dummy.buffer, dummy.buffer)?;
            let pc = [
                hd as u32,
                n_dims as u32,
                pos,
                freq_base.to_bits(),
                sections[0],
                sections[1],
                sections[2],
                sections[3],
            ];
            let pc_bytes =
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc));
            self.run_registered("rope_imrope", set, pc_bytes, 256, nheads as u32)?;

            let out = xb.data()[..total * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(xb);
            self.destroy_host_buffer(dummy);
            self.device.destroy_descriptor_pool(pool, None);
            Ok(out)
        }
    }

    /// One gated-delta-net recurrent step for a single layer. `ins` is the
    /// packed blob (q, k, v, gate, beta), `state` the H_V transposed matrices;
    /// returns (attn out, updated state). Mirrors gdn::gdn_step.
    pub fn gdn_step(
        &mut self,
        ins: &[f32],
        state: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if ins.len() != GDN_INS_LEN {
            return Err(format!(
                "gdn_step: ins len {} != {GDN_INS_LEN}",
                ins.len()
            ));
        }
        if state.len() != GDN_STATE_LEN {
            return Err(format!(
                "gdn_step: state len {} != {GDN_STATE_LEN}",
                state.len()
            ));
        }
        self.ensure_kernel("gdn_step", GDN_STEP_SPV)?;
        unsafe {
            let mut inb = self.create_host_buffer(ins.len() * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut sb = self.create_host_buffer(state.len() * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let ob = self.create_host_buffer(GDN_DI * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            inb.data_mut().copy_from_slice(bytemuck_slice(ins));
            sb.data_mut().copy_from_slice(bytemuck_slice(state));

            let (pool, set) = self.make_set(inb.buffer, sb.buffer, ob.buffer)?;
            self.run_registered("gdn_step", set, &[], 128, GDN_HV as u32)?;

            let attn = ob.data()[..GDN_DI * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            let state_out = sb.data()[..state.len() * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(inb);
            self.destroy_host_buffer(sb);
            self.destroy_host_buffer(ob);
            self.device.destroy_descriptor_pool(pool, None);
            Ok((attn, state_out))
        }
    }

    /// Full-attention inner product for one token over its per-layer KV
    /// cache: scores -> masked softmax -> weighted-v (GQA). `q` is the token's
    /// `nheads*hd` per-head vectors; `kcache`/`vcache` hold `n_pos` cached
    /// tokens as [pos][kv_head][hd]. Returns the `nheads*hd` attention output
    /// before the gate and output projection (mirrors forward.rs step 4).
    pub fn attn_step(
        &mut self,
        q: &[f32],
        kcache: &[f32],
        vcache: &[f32],
        n_pos: usize,
        nheads: usize,
        kvheads: usize,
        hd: usize,
    ) -> Result<Vec<f32>, String> {
        let kv_stride = kvheads * hd;
        if q.len() != nheads * hd
            || kcache.len() != n_pos * kv_stride
            || vcache.len() != n_pos * kv_stride
            || hd > 256
        {
            return Err(format!(
                "attn_step: q {} ({}x{}), k/v {} ({} pos x {kv_stride})",
                q.len(),
                nheads,
                hd,
                kcache.len(),
                n_pos
            ));
        }
        self.ensure_kernel("attn_scores", ATTN_SCORES_SPV)?;
        self.ensure_kernel("softmax_inplace", SOFTMAX_INPLACE_SPV)?;
        self.ensure_kernel("attn_out", ATTN_OUT_SPV)?;

        let score_len = nheads * n_pos;
        let scale = 1.0 / (hd as f32).sqrt();
        let pc = [
            nheads as u32,
            n_pos as u32,
            hd as u32,
            kv_stride as u32,
            scale.to_bits(),
            0,
        ];
        let pc_bytes =
            unsafe { std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc)) };

        unsafe {
            let mut qb = self.create_host_buffer(q.len() * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut kb = self.create_host_buffer(kcache.len() * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut vb = self.create_host_buffer(vcache.len() * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut sb = self.create_host_buffer(score_len.max(1) * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let ob = self.create_host_buffer(nheads * hd * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let mut dummy = self.create_host_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            qb.data_mut().copy_from_slice(bytemuck_slice(q));
            kb.data_mut().copy_from_slice(bytemuck_slice(kcache));
            vb.data_mut().copy_from_slice(bytemuck_slice(vcache));
            dummy.data_mut().fill(0);

            // 1) scores per (head, pos)
            let (p1, set1) = self.make_set(qb.buffer, kb.buffer, sb.buffer)?;
            self.run_registered("attn_scores", set1, pc_bytes, 256, nheads as u32)?;
            self.device.destroy_descriptor_pool(p1, None);

            // 2) softmax each head's score row in place
            let (p2, set2) = self.make_set(sb.buffer, dummy.buffer, dummy.buffer)?;
            self.run_registered("softmax_inplace", set2, pc_bytes, 256, nheads as u32)?;
            self.device.destroy_descriptor_pool(p2, None);

            // 3) weighted-v into out
            let (p3, set3) = self.make_set(sb.buffer, vb.buffer, ob.buffer)?;
            self.run_registered("attn_out", set3, pc_bytes, 256, nheads as u32)?;
            self.device.destroy_descriptor_pool(p3, None);

            let out = ob.data()[..nheads * hd * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(qb);
            self.destroy_host_buffer(kb);
            self.destroy_host_buffer(vb);
            self.destroy_host_buffer(sb);
            self.destroy_host_buffer(ob);
            self.destroy_host_buffer(dummy);
            Ok(out)
        }
    }

    /// PQ2_0 matvec against an already-uploaded device buffer `w`
    /// (rows `base_row + gid`). Host x in, host y out.
    ///
    /// When the persistent cache is prepared (`prep_matvec_cache`), the call
    /// reuses its x/y buffers, descriptor set, command buffer and fence, so
    /// only a memcpy + descriptor update + submit are per-call; otherwise it
    /// falls back to allocating per call (slow, for ad-hoc use).
    pub fn matvec_on(
        &mut self,
        w: &DevBuf,
        ne0: usize,
        base_row: u32,
        n_rows: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        if x.len() != ne0 {
            return Err(format!("matvec_on: x length {} != ne0 {ne0}", x.len()));
        }
        if let Some(c) = self.mv.as_mut() {
            if ne0 <= c.max_ne0 && n_rows <= c.max_rows {
                let device = &self.device;
                let queue = self.queue;
                let pipeline = self.pipeline;
                let layout = self.pipeline_layout;
                return Self::matvec_cached_impl(device, queue, pipeline, layout, w, ne0, base_row, n_rows, x, c);
            }
        }
        self.matvec_oneshot(w, ne0, base_row, n_rows, x)
    }

    /// Cached path: reuse x/y/desc/cmd/fence, re-point descriptors at `w`.
    fn matvec_cached_impl(
        device: &Device,
        queue: vk::Queue,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        w: &DevBuf,
        ne0: usize,
        base_row: u32,
        n_rows: usize,
        x: &[f32],
        c: &mut MvCache,
    ) -> Result<Vec<f32>, String> {
        unsafe {
            c.xb.data_mut()[..ne0 * 4].copy_from_slice(bytemuck_slice(x));
            let infos = [
                vk::DescriptorBufferInfo::default().buffer(w.buffer).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default()
                    .buffer(c.xb.buffer)
                    .range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default()
                    .buffer(c.yb.buffer)
                    .range(vk::WHOLE_SIZE),
            ];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(c.set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[0..1]),
                vk::WriteDescriptorSet::default()
                    .dst_set(c.set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[1..2]),
                vk::WriteDescriptorSet::default()
                    .dst_set(c.set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[2..3]),
            ];
            device.update_descriptor_sets(&writes, &[]);

            device
                .reset_command_buffer(c.cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("matvec cached: reset cb: {e}"))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(c.cb, &begin)
                .map_err(|e| format!("matvec cached: begin: {e}"))?;
            device
                .cmd_bind_pipeline(c.cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                c.cb,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[c.set],
                &[],
            );
            let pc = [ne0 as u32, base_row];
            let pc_bytes = std::slice::from_raw_parts(
                pc.as_ptr() as *const u8,
                std::mem::size_of_val(&pc),
            );
            device.cmd_push_constants(
                c.cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            let groups = (n_rows as u32).div_ceil(LOCAL_X).max(1);
            device.cmd_dispatch(c.cb, groups, 1, 1);
            device
                .end_command_buffer(c.cb)
                .map_err(|e| format!("matvec cached: end: {e}"))?;

            device
                .reset_fences(&[c.fence])
                .map_err(|e| format!("matvec cached: reset fence: {e}"))?;
            let cbs = [c.cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            device
                .queue_submit(queue, &[submit], c.fence)
                .map_err(|e| format!("matvec cached: submit: {e}"))?;
            device
                .wait_for_fences(&[c.fence], true, u64::MAX)
                .map_err(|e| format!("matvec cached: wait: {e}"))?;

            Ok(c.yb.data()[..n_rows * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
    }

    /// Fallback single-shot path (allocs per call; only used when the cache is
    /// not prepared or the shapes outgrow it).
    fn matvec_oneshot(
        &mut self,
        w: &DevBuf,
        ne0: usize,
        base_row: u32,
        n_rows: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        unsafe {
            let mut xb = self.create_host_buffer(ne0 * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let yb = self.create_host_buffer(n_rows * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            xb.data_mut().copy_from_slice(bytemuck_slice(x));
            let (pool, set) = self.make_set(w.buffer, xb.buffer, yb.buffer)?;

            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = self
                .device
                .allocate_command_buffers(&cb_info)
                .map_err(|e| format!("matvec_on: alloc: {e}"))?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cb, &begin)
                .map_err(|e| format!("matvec_on: begin: {e}"))?;
            self.record_matvec(cb, set, ne0 as u32, base_row, n_rows);
            self.device
                .end_command_buffer(cb)
                .map_err(|e| format!("matvec_on: end: {e}"))?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("matvec_on: fence: {e}"))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| format!("matvec_on: submit: {e}"))?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("matvec_on: wait: {e}"))?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            self.device.destroy_descriptor_pool(pool, None);

            let out = yb.data()[..n_rows * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();
            self.destroy_host_buffer(xb);
            self.destroy_host_buffer(yb);
            Ok(out)
        }
    }

    /// Run the PQ2_0 matvec over `n_rows` of a payload (row `base_row` +
    /// global id) and return the `n_rows` output floats.
    ///
    /// Buffers are created per call for G0; G2 will cache them.
    pub fn pq2_matvec(
        &self,
        payload: &[u8],
        ne0: usize,
        base_row: u32,
        n_rows: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(x.len(), ne0);
        let y_len = n_rows * 4;
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
            let mut wb = self.create_host_buffer(payload.len(), usage)?;
            let mut xb = self.create_host_buffer(ne0 * 4, usage)?;
            let yb = self.create_host_buffer(y_len, usage)?;
            wb.data_mut().copy_from_slice(payload);
            xb.data_mut().copy_from_slice(bytemuck_slice(x));

            // descriptor set for this run
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let pool = err(
                self.device.create_descriptor_pool(&pool_info, None),
                "create_descriptor_pool",
            )?;
            let set_layouts = [self.desc_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts);
            let sets = err(
                self.device.allocate_descriptor_sets(&alloc_info),
                "allocate_descriptor_sets",
            )?;
            let set = sets[0];

            let infos = [
                vk::DescriptorBufferInfo::default()
                    .buffer(wb.buffer)
                    .range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default()
                    .buffer(xb.buffer)
                    .range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default()
                    .buffer(yb.buffer)
                    .range(vk::WHOLE_SIZE),
            ];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[0..1]),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[1..2]),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[2..3]),
            ];
            self.device.update_descriptor_sets(&writes, &[]);

            // command buffer
            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cbs = err(
                self.device.allocate_command_buffers(&cb_info),
                "allocate_command_buffers",
            )?;
            let cb = cbs[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            err(self.device.begin_command_buffer(cb, &begin), "begin_command_buffer")?;
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            let pc = [ne0 as u32, base_row];
            let pc_bytes = std::slice::from_raw_parts(
                pc.as_ptr() as *const u8,
                std::mem::size_of_val(&pc),
            );
            self.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            let groups = (n_rows as u32).div_ceil(LOCAL_X).max(1);
            self.device.cmd_dispatch(cb, groups, 1, 1);
            err(self.device.end_command_buffer(cb), "end_command_buffer")?;

            let fence = err(
                self.device.create_fence(&vk::FenceCreateInfo::default(), None),
                "create_fence",
            )?;
            let command_buffers = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            err(
                self.device.queue_submit(self.queue, &[submit], fence),
                "queue_submit",
            )?;
            err(
                self.device.wait_for_fences(&[fence], true, u64::MAX),
                "wait_for_fences",
            )?;

            // read back y (host-coherent: visible after the fence)
            let out = yb.data()[..y_len]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>();

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            self.device.destroy_descriptor_pool(pool, None);
            self.destroy_host_buffer(wb);
            self.destroy_host_buffer(xb);
            self.destroy_host_buffer(yb);
            Ok(out)
        }
    }
    // ---- Per-token command recording (full device decode) -------------------

    /// Start recording one token: reset the descriptor pool + command buffer.
    pub fn rec_begin(&mut self) -> Result<(), String> {
        if self.rec_pool.is_none() {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(8192 * 3)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(8192)
                .pool_sizes(&sizes);
            let pool = err(
                unsafe { self.device.create_descriptor_pool(&pool_info, None) },
                "rec_begin: pool",
            )?;
            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = unsafe { self.device.allocate_command_buffers(&cb_info) }
                .map_err(|e| format!("rec_begin: cb: {e}"))?[0];
            let fence = err(
                unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) },
                "rec_begin: fence",
            )?;
            let dummy = unsafe { self.create_dev_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER) }?;
            self.rec_pool = Some(pool);
            self.rec_cb = Some(cb);
            self.rec_fence = Some(fence);
            self.rec_dummy = Some(dummy);
        }
        unsafe {
            self.device
                .reset_descriptor_pool(self.rec_pool.unwrap(), vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| format!("rec_begin: reset pool: {e}"))?;
            self.device
                .reset_command_buffer(self.rec_cb.unwrap(), vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("rec_begin: reset cb: {e}"))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.rec_cb.unwrap(), &begin)
                .map_err(|e| format!("rec_begin: begin: {e}"))?;
        }
        self.rec_open = true;
        Ok(())
    }

    /// Allocate one descriptor set from the per-token pool and point it at the
    /// three buffers (bindings 0,1,2).
    fn rec_set(&mut self, a: &DevBuf, b: &DevBuf, c: &DevBuf) -> Result<vk::DescriptorSet, String> {
        let pool = self.rec_pool.ok_or("rec_set: pool not created")?;
        unsafe {
            let layouts = [self.desc_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let sets = self
                .device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("rec_set: alloc: {e}"))?;
            let set = sets[0];
            let infos = [
                vk::DescriptorBufferInfo::default().buffer(a.buffer).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(b.buffer).range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default().buffer(c.buffer).range(vk::WHOLE_SIZE),
            ];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[0..1]),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[1..2]),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[2..3]),
            ];
            self.device.update_descriptor_sets(&writes, &[]);
            Ok(set)
        }
    }

    /// Record one kernel dispatch over buffers (a,b,c) with a memory barrier
    /// afterwards. Kernels must be registered via `ensure_kernel`.
    pub fn rec_dispatch(
        &mut self,
        kernel: &str,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        pc: &[u8],
        groups: u32,
    ) -> Result<(), String> {
        if !self.rec_open {
            return Err("rec_dispatch outside rec_begin/rec_end".into());
        }
        let &pipeline = self
            .kernels
            .get(kernel)
            .ok_or_else(|| format!("kernel '{kernel}' not registered"))?;
        let set = self.rec_set(a, b, c)?;
        unsafe {
            let cb = self.rec_cb.unwrap();
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc,
            );
            self.device.cmd_dispatch(cb, groups.max(1), 1, 1);
            // full compute->compute memory barrier over the touched buffers
            let mem: Vec<vk::BufferMemoryBarrier> = [a, b, c]
                .iter()
                .map(|d| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(d.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                })
                .collect();
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &mem,
                &[],
            );
        }
        Ok(())
    }

    /// Record the two-pass PQ2_0 matvec into the current token's command
    /// buffer: pq2_partial into `partials`, then pq2_rowsum writing rows
    /// `ybase..ybase+rows` of `y`.
    pub fn rec_matvec(
        &mut self,
        w: &DevBuf,
        x: &DevBuf,
        y: &DevBuf,
        ybase: u32,
        partials: &DevBuf,
        ne0: u32,
        base_row: u32,
        rows: u32,
    ) -> Result<(), String> {
        self.ensure_kernel("pq2_partial", PQ2_PARTIAL_SPV)?;
        self.ensure_kernel("pq2_rowsum", PQ2_ROWSUM_SPV)?;
        let nblocks = ne0.div_ceil(128);
        let total = (rows as u64 * nblocks as u64).max(1);
        let groups = (total as u32).div_ceil(256).max(1);
        let pc_p = [ne0, base_row];
        let pc_pb = unsafe {
            std::slice::from_raw_parts(pc_p.as_ptr() as *const u8, std::mem::size_of_val(&pc_p))
        };
        self.rec_dispatch("pq2_partial", w, x, partials, pc_pb, groups)?;
        let pc_r = [nblocks, ybase];
        let pc_rb = unsafe {
            std::slice::from_raw_parts(pc_r.as_ptr() as *const u8, std::mem::size_of_val(&pc_r))
        };
        // rowsum only uses bindings 0 (partials) and 2 (y); bind y twice
        self.rec_dispatch("pq2_rowsum", partials, y, y, pc_rb, rows.max(1))
    }

    /// End the token's command buffer, submit once and wait.
    pub fn rec_end_submit(&mut self) -> Result<(), String> {
        if !self.rec_open {
            return Ok(());
        }
        unsafe {
            let cb = self.rec_cb.unwrap();
            self.device
                .end_command_buffer(cb)
                .map_err(|e| format!("rec_end: {e}"))?;
            let fence = self.rec_fence.unwrap();
            self.device
                .reset_fences(&[fence])
                .map_err(|e| format!("rec_end: reset fence: {e}"))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| format!("rec_end: submit: {e}"))?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("rec_end: wait: {e}"))?;
        }
        self.rec_open = false;
        Ok(())
    }

    /// Copy `len` bytes from a device buffer back to the host (small reads:
    /// final hidden vector, logits later).
    pub fn read_dev(&mut self, dev: &DevBuf, len: usize) -> Result<Vec<u8>, String> {
        if len > dev.len {
            return Err(format!("read_dev: {len} > device {}", dev.len));
        }
        self.ensure_staging_size(len)?;
        unsafe {
            let staging = self.staging.as_ref().unwrap().buffer;
            let cb_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = self
                .device
                .allocate_command_buffers(&cb_info)
                .map_err(|e| format!("read_dev: cb: {e}"))?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cb, &begin).map_err(|e| format!("read_dev: begin: {e}"))?;
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(len as vk::DeviceSize);
            self.device.cmd_copy_buffer(cb, dev.buffer, staging, &[region]);
            self.device.end_command_buffer(cb).map_err(|e| format!("read_dev: end: {e}"))?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("read_dev: fence: {e}"))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device.queue_submit(self.queue, &[submit], fence)
                .map_err(|e| format!("read_dev: submit: {e}"))?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("read_dev: wait: {e}"))?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cb]);
            let ptr = self.staging.as_ref().unwrap().ptr;
            Ok(std::slice::from_raw_parts(ptr, len).to_vec())
        }
    }
}

fn bytemuck_slice(x: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}



impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(s) = self.staging.take() {
                self.device.unmap_memory(s.memory);
                self.device.destroy_buffer(s.buffer, None);
                self.device.free_memory(s.memory, None);
            }
            if let Some(pool) = self.rec_pool.take() {
                self.device.destroy_descriptor_pool(pool, None);
            }
            if let Some(cb) = self.rec_cb.take() {
                self.device.free_command_buffers(self.command_pool, &[cb]);
            }
            if let Some(fence) = self.rec_fence.take() {
                self.device.destroy_fence(fence, None);
            }
            if let Some(d) = self.rec_dummy.take() {
                self.device.destroy_buffer(d.buffer, None);
                self.device.free_memory(d.memory, None);
            }
            if let Some(c) = self.mv.take() {
                self.device.destroy_fence(c.fence, None);
                self.device.free_command_buffers(self.command_pool, &[c.cb]);
                self.device.destroy_descriptor_pool(c.pool, None);
                self.device.destroy_buffer(c.wdummy.buffer, None);
                self.device.free_memory(c.wdummy.memory, None);
                self.device.unmap_memory(c.xb.memory);
                self.device.destroy_buffer(c.xb.buffer, None);
                self.device.free_memory(c.xb.memory, None);
                self.device.unmap_memory(c.yb.memory);
                self.device.destroy_buffer(c.yb.buffer, None);
                self.device.free_memory(c.yb.memory, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            for (_, p) in self.kernels.drain() {
                self.device.destroy_pipeline(p, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self._instance.destroy_instance(None);
        }
    }
}
