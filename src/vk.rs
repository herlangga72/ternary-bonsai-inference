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
    include_bytes!(concat!(env!("OUT_DIR"), "/pq2_matvec.spv"));

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
    pub name: String,
    pub discrete: bool,
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
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
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

        // ---- pipeline layout: one 8-byte push constant range ----------------
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8);
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
            name,
            discrete,
        })
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
}

fn bytemuck_slice(x: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self._instance.destroy_instance(None);
        }
    }
}
