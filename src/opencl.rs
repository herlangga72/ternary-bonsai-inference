//! Minimal dynamic OpenCL 1.2 host binding (G0).
//!
//! Loads libOpenCL.so.1 at runtime with dlopen/dlsym - no build-time link, no
//! ROCm headers, no cargo dependency. Binaries that never touch the GPU never
//! dlopen anything. Mirrors the hand-rolled `llama.rs` FFI style used for the
//! llama.cpp comparison tools.
//!
//! Exposes just enough surface for the engine's kernels: platform/device
//! discovery, a context + command queue, program build from source, kernel
//! launch, and buffer read/write. Everything is read-only host-side until G2
//! wires the decode loop, so buffers stay simple.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// OpenCL constants used here
// ---------------------------------------------------------------------------
const CL_SUCCESS: c_int = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_TYPE_DEFAULT: u64 = 1 << 0;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
const CL_DEVICE_MAX_CLOCK_FREQUENCY: u32 = 0x100C;
const CL_DEVICE_LOCAL_MEM_SIZE: u32 = 0x1016;
const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x1020;
const CL_CONTEXT_PLATFORM: isize = 0x1084;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_WRITE_ONLY: u64 = 1 << 1;
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;
const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
pub const CL_MEM_HOST_WRITE_ONLY: u64 = 1 << 7;
pub const CL_MEM_HOST_READ_ONLY: u64 = 1 << 8;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_KERNEL_WORK_GROUP_SIZE: u32 = 0x11B0;

// type aliases: OpenCL objects are opaque pointers
pub type cl_platform_id = *mut c_void;
pub type cl_device_id = *mut c_void;
pub type cl_context = *mut c_void;
pub type cl_command_queue = *mut c_void;
pub type cl_program = *mut c_void;
pub type cl_kernel = *mut c_void;
pub type cl_mem = *mut c_void;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type CreateBuffer = unsafe extern "C" fn(
    cl_context,
    u64,           // cl_mem_flags
    usize,         // size
    *const c_void, // host_ptr
    *mut c_int,    // errcode_ret
) -> cl_mem;

type ReleaseMem = unsafe extern "C" fn(cl_mem) -> c_int;

type EnqueueWriteBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    u32, // cl_bool blocking_write
    usize,
    usize,
    *const c_void,
    u32,
    *const c_void,
    *const c_void,
) -> c_int;

type EnqueueReadBuffer = unsafe extern "C" fn(
    cl_command_queue,
    cl_mem,
    u32,
    usize,
    usize,
    *mut c_void,
    u32,
    *const c_void,
    *const c_void,
) -> c_int;

type EnqueueNDRange = unsafe extern "C" fn(
    cl_command_queue,
    cl_kernel,
    u32,
    *const usize,
    *const usize,
    *const usize,
    u32,
    *const c_void,
    *const c_void,
) -> c_int;

type Finish = unsafe extern "C" fn(cl_command_queue) -> c_int;

/// All resolved entry points.
struct Fns {
    get_platforms: unsafe extern "C" fn(u32, *mut cl_platform_id, *mut u32) -> c_int,
    get_devices: unsafe extern "C" fn(cl_platform_id, u64, u32, *mut cl_device_id, *mut u32) -> c_int,
    get_device_info: unsafe extern "C" fn(cl_device_id, u32, usize, *mut c_void, *mut usize) -> c_int,
    create_context: unsafe extern "C" fn(*const c_void, u32, *const c_void, *const c_void, *mut c_void, *mut c_int) -> cl_context,
    create_queue: unsafe extern "C" fn(cl_context, cl_device_id, u64, *mut c_int) -> cl_command_queue,
    create_program: unsafe extern "C" fn(cl_context, u32, *const *const c_char, *const usize, *mut c_int) -> cl_program,
    build_program: unsafe extern "C" fn(cl_program, u32, *const c_void, *const c_char, *const c_void, *const c_void) -> c_int,
    program_build_info: unsafe extern "C" fn(cl_program, cl_device_id, u32, usize, *mut c_void, *mut usize) -> c_int,
    create_kernel: unsafe extern "C" fn(cl_program, *const c_char, *mut c_int) -> cl_kernel,
    set_kernel_arg: unsafe extern "C" fn(cl_kernel, u32, usize, *const c_void) -> c_int,
    get_kernel_work_group_info: unsafe extern "C" fn(cl_kernel, cl_device_id, u32, usize, *mut c_void, *mut usize) -> c_int,
    release_queue: unsafe extern "C" fn(cl_command_queue) -> c_int,
    release_context: unsafe extern "C" fn(cl_context) -> c_int,
    release_program: unsafe extern "C" fn(cl_program) -> c_int,
    release_kernel: unsafe extern "C" fn(cl_kernel) -> c_int,
    create_buffer: CreateBuffer,
    release_mem: ReleaseMem,
    enqueue_write: EnqueueWriteBuffer,
    enqueue_read: EnqueueReadBuffer,
    enqueue_ndrange: EnqueueNDRange,
    finish: Finish,
}

impl Fns {
    /// dlsym one symbol, panicking on a missing required entry point (this is
    /// a host-call bug, not a runtime condition).
    unsafe fn sym<T: Copy>(handle: *mut c_void, name: &str) -> T {
        let cname = std::ffi::CString::new(name).unwrap();
        let p = dlsym(handle, cname.as_ptr());
        assert!(!p.is_null(), "libOpenCL is missing required symbol {name}");
        let raw = p as usize;
        std::mem::transmute_copy(&raw)
    }
}

/// dlopen handle wrapper: the raw pointer is only ever used as an argument to
/// dlsym from the single load() call, so it is safe to share across threads.
#[derive(Clone, Copy)]
struct LibHandle(*mut c_void);
unsafe impl Send for LibHandle {}
unsafe impl Sync for LibHandle {}

fn lib() -> &'static OnceLock<LibHandle> {
    static LIB: OnceLock<LibHandle> = OnceLock::new();
    &LIB
}

/// Load libOpenCL.so.1 and resolve every entry point once. `Ok(())` means the
/// GPU backend is available; errors are descriptive strings.
pub fn load() -> Result<(), String> {
    let handle = *lib().get_or_init(|| unsafe {
        let name = std::ffi::CString::new("libOpenCL.so.1").unwrap();
        LibHandle(dlopen(name.as_ptr(), 2 /* RTLD_NOW */))
    });
    if handle.0.is_null() {
        return Err("dlopen libOpenCL.so.1 failed: no OpenCL runtime installed".into());
    }
    let _ = fns();
    Ok(())
}

/// Access the resolved entry points (must call `load` first).
fn fns() -> &'static Fns {
    static FNS: OnceLock<Fns> = OnceLock::new();
    FNS.get_or_init(|| {
        let handle = lib()
            .get()
            .expect("opencl::load() must be called before opencl::fns()")
            .0;
        unsafe {
            Fns {
                get_platforms: Fns::sym(handle, "clGetPlatformIDs"),
                get_devices: Fns::sym(handle, "clGetDeviceIDs"),
                get_device_info: Fns::sym(handle, "clGetDeviceInfo"),
                create_context: Fns::sym(handle, "clCreateContext"),
                create_queue: Fns::sym(handle, "clCreateCommandQueue"),
                create_program: Fns::sym(handle, "clCreateProgramWithSource"),
                build_program: Fns::sym(handle, "clBuildProgram"),
                program_build_info: Fns::sym(handle, "clGetProgramBuildInfo"),
                create_kernel: Fns::sym(handle, "clCreateKernel"),
                set_kernel_arg: Fns::sym(handle, "clSetKernelArg"),
                get_kernel_work_group_info: Fns::sym(handle, "clGetKernelWorkGroupInfo"),
                release_queue: Fns::sym(handle, "clReleaseCommandQueue"),
                release_context: Fns::sym(handle, "clReleaseContext"),
                release_program: Fns::sym(handle, "clReleaseProgram"),
                release_kernel: Fns::sym(handle, "clReleaseKernel"),
                create_buffer: Fns::sym(handle, "clCreateBuffer"),
                release_mem: Fns::sym(handle, "clReleaseMemObject"),
                enqueue_write: Fns::sym(handle, "clEnqueueWriteBuffer"),
                enqueue_read: Fns::sym(handle, "clEnqueueReadBuffer"),
                enqueue_ndrange: Fns::sym(handle, "clEnqueueNDRangeKernel"),
                finish: Fns::sym(handle, "clFinish"),
            }
        }
    })
}

fn check(err: c_int, what: &str) -> Result<(), String> {
    if err == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("OpenCL error {err} in {what}"))
    }
}

/// A discovered OpenCL device with the handful of properties the engine uses.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_units: u32,
    pub max_clock_mhz: u32,
    pub local_mem: u64,
    pub global_mem: u64,
}

pub struct Context {
    pub ctx: cl_context,
    pub queue: cl_command_queue,
    pub dev: cl_device_id,
    pub info: DeviceInfo,
}

impl Drop for Context {
    fn drop(&mut self) {
        let f = fns();
        unsafe {
            (f.release_queue)(self.queue);
            (f.release_context)(self.ctx);
        }
    }
}

/// Enumerate GPUs, pick the first, create a context + queue, and report its
/// properties. This is the only setup call the engine needs.
pub fn open_gpu() -> Result<Context, String> {
    load()?;
    let f = fns();
    unsafe {
        let mut n_platforms: u32 = 0;
        check(
            (f.get_platforms)(0, ptr::null_mut(), &mut n_platforms),
            "clGetPlatformIDs count",
        )?;
        if n_platforms == 0 {
            return Err("no OpenCL platforms found".into());
        }
        let mut platforms: Vec<cl_platform_id> = vec![ptr::null_mut(); n_platforms as usize];
        check(
            (f.get_platforms)(n_platforms, platforms.as_mut_ptr(), &mut n_platforms),
            "clGetPlatformIDs",
        )?;

        let mut dev: cl_device_id = ptr::null_mut();
        let mut found = false;
        for &platform in &platforms {
            let mut n_dev = 0u32;
            if (f.get_devices)(platform, CL_DEVICE_TYPE_GPU, 0, ptr::null_mut(), &mut n_dev) == CL_SUCCESS
                && n_dev > 0
            {
                check(
                    (f.get_devices)(platform, CL_DEVICE_TYPE_GPU, 1, &mut dev, &mut n_dev),
                    "clGetDeviceIDs",
                )?;
                found = true;
                break;
            }
        }
        if !found || dev.is_null() {
            return Err("no OpenCL GPU device found".into());
        }

        // ---- device info (name + the numeric properties the engine uses) ---
        let read_device = |param: u32, out: *mut c_void, len: usize| -> Result<(), String> {
            check(
                (f.get_device_info)(dev, param, len, out, ptr::null_mut()),
                "clGetDeviceInfo",
            )
        };
        let read_str = |param: u32| -> Result<String, String> {
            let mut n = 0usize;
            check(
                (f.get_device_info)(dev, param, 0, ptr::null_mut(), &mut n),
                "clGetDeviceInfo len",
            )?;
            let mut buf = vec![0u8; n.max(1)];
            read_device(param, buf.as_mut_ptr() as *mut c_void, buf.len())?;
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
        };
        let read_u32 = |param: u32| -> Result<u32, String> {
            let mut v = 0u32;
            read_device(param, &mut v as *mut u32 as *mut c_void, 4)?;
            Ok(v)
        };
        let read_u64 = |param: u32| -> Result<u64, String> {
            let mut v = 0u64;
            read_device(param, &mut v as *mut u64 as *mut c_void, 8)?;
            Ok(v)
        };
        let info = DeviceInfo {
            name: read_str(CL_DEVICE_NAME)?,
            compute_units: read_u32(CL_DEVICE_MAX_COMPUTE_UNITS)?,
            max_clock_mhz: read_u32(CL_DEVICE_MAX_CLOCK_FREQUENCY)?,
            local_mem: read_u64(CL_DEVICE_LOCAL_MEM_SIZE)?,
            global_mem: read_u64(CL_DEVICE_GLOBAL_MEM_SIZE)?,
        };

        let platform = platforms[0];
        let props: [isize; 3] = [CL_CONTEXT_PLATFORM, platform as isize, 0];
        let mut err: c_int = 0;
        let ctx = (f.create_context)(
            props.as_ptr() as *const c_void,
            1,
            &dev as *const cl_device_id as *const c_void,
            ptr::null(),
            ptr::null_mut(),
            &mut err,
        );
        check(err, "clCreateContext")?;
        if ctx.is_null() {
            return Err("clCreateContext returned null".into());
        }
        let mut qerr: c_int = 0;
        let queue = (f.create_queue)(ctx, dev, 0, &mut qerr);
        check(qerr, "clCreateCommandQueue")?;
        if queue.is_null() {
            return Err("clCreateCommandQueue returned null".into());
        }
        Ok(Context { ctx, queue, dev, info })
    }
}

pub struct Program {
    prog: cl_program,
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { (fns().release_program)(self.prog) };
    }
}

impl Program {
    /// Compile `source` for the context's device; on failure return the build
    /// log so kernel bugs are debuggable without a separate toolchain.
    pub fn build(ctx: &Context, source: &str) -> Result<Program, String> {
        let t0 = std::time::Instant::now();
        eprintln!("[opencl] clBuildProgram start ({} bytes)...", source.len());
        let f = fns();
        unsafe {
            let csrc = std::ffi::CString::new(source).map_err(|_| "kernel source has NUL byte")?;
            let mut err: c_int = 0;
            let prog = (f.create_program)(ctx.ctx, 1, &csrc.as_ptr(), ptr::null(), &mut err);
            check(err, "clCreateProgramWithSource")?;
            if prog.is_null() {
                return Err("clCreateProgramWithSource returned null".into());
            }
            let opts = std::ffi::CString::new("").unwrap();
            let berr = (f.build_program)(prog, 0, ptr::null(), opts.as_ptr(), ptr::null(), ptr::null());
            if berr != CL_SUCCESS {
                // fetch the build log
                let get = |param: u32, out: *mut c_void, n: usize| -> c_int {
                    (f.program_build_info)(prog, ctx.dev, param, n, out, &mut (n as usize))
                };
                let mut len = 0usize;
                (f.program_build_info)(prog, ctx.dev, CL_PROGRAM_BUILD_LOG, 0, ptr::null_mut(), &mut len);
                let mut log = vec![0u8; len.max(1)];
                get(CL_PROGRAM_BUILD_LOG, log.as_mut_ptr() as *mut c_void, log.len());
                let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
                return Err(format!(
                    "clBuildProgram failed ({berr}) after {:.1}s: {}",
                    t0.elapsed().as_secs_f32(),
                    String::from_utf8_lossy(&log[..end])
                ));
            }
            eprintln!(
                "[opencl] clBuildProgram done in {:.1}s",
                t0.elapsed().as_secs_f32()
            );
            Ok(Program { prog })
        }
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel, String> {
        let f = fns();
        unsafe {
            let cname = std::ffi::CString::new(name).unwrap();
            let mut err: c_int = 0;
            let k = (f.create_kernel)(self.prog, cname.as_ptr(), &mut err);
            check(err, "clCreateKernel")?;
            if k.is_null() {
                return Err(format!("clCreateKernel({name}) returned null"));
            }
            Ok(Kernel { kernel: k })
        }
    }
}

pub struct Kernel {
    pub kernel: cl_kernel,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (fns().release_kernel)(self.kernel) };
    }
}

impl Kernel {
    /// `value` is any type; pass `&x` for scalars and structs, or a raw pointer
    /// for buffers (a `cl_mem` value, i.e. `&mem.0` as *const _ as *const c_void`).
    pub fn arg_raw(&self, index: u32, size: usize, value: *const c_void) -> Result<(), String> {
        let f = fns();
        unsafe { check((f.set_kernel_arg)(self.kernel, index, size, value), "clSetKernelArg") }
    }

    pub fn arg<T>(&self, index: u32, value: &T) -> Result<(), String> {
        self.arg_raw(index, std::mem::size_of::<T>(), value as *const T as *const c_void)
    }

    pub fn max_work_group_size(&self, dev: cl_device_id) -> Result<usize, String> {
        let f = fns();
        unsafe {
            let mut v = 0usize;
            check(
                (f.get_kernel_work_group_info)(self.kernel, dev, CL_KERNEL_WORK_GROUP_SIZE, 8, &mut v as *mut usize as *mut c_void, ptr::null_mut()),
                "clGetKernelWorkGroupInfo",
            )?;
            Ok(v)
        }
    }
}

/// Device memory buffer. Created with the given host bytes when `init` is
/// `Some`, otherwise uninitialized.
pub struct Buffer {
    mem: cl_mem,
    pub len_bytes: usize,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (fns().release_mem)(self.mem) };
    }
}

impl Buffer {
    pub fn create(ctx: &Context, read_only: bool, len_bytes: usize) -> Result<Buffer, String> {
        let flags = if read_only { CL_MEM_READ_ONLY } else { CL_MEM_READ_WRITE };
        Buffer::create_with_flags(ctx, flags, len_bytes)
    }

    pub fn create_with_flags(
        ctx: &Context,
        flags: u64,
        len_bytes: usize,
    ) -> Result<Buffer, String> {
        eprintln!("[opencl] clCreateBuffer({len_bytes} bytes, flags {flags:#x})...");
        let f = fns();
        unsafe {
            let mut err: c_int = 0;
            let mem = (f.create_buffer)(ctx.ctx, flags, len_bytes, ptr::null(), &mut err);
            check(err, "clCreateBuffer")?;
            if mem.is_null() {
                return Err("clCreateBuffer returned null".into());
            }
            eprintln!("[opencl] clCreateBuffer ok");
            Ok(Buffer { mem, len_bytes })
        }
    }

    pub fn write(&self, ctx: &Context, data: &[u8]) -> Result<(), String> {
        eprintln!("[opencl] clEnqueueWriteBuffer({} bytes)...", data.len());
        if data.len() > self.len_bytes {
            return Err(format!(
                "buffer write {} > buffer {}",
                data.len(),
                self.len_bytes
            ));
        }
        let f = fns();
        unsafe {
            check(
                (f.enqueue_write)(
                    ctx.queue,
                    self.mem,
                    1, // blocking
                    0,
                    data.len(),
                    data.as_ptr() as *const c_void,
                    0,
                    ptr::null(),
                    ptr::null(),
                ),
                "clEnqueueWriteBuffer",
            )?;
            check((f.finish)(ctx.queue), "clFinish after write")?;
            eprintln!("[opencl] clEnqueueWriteBuffer ok");
            Ok(())
        }
    }

    pub fn read(&self, ctx: &Context, out: &mut [u8]) -> Result<(), String> {
        eprintln!("[opencl] clEnqueueReadBuffer({} bytes)...", out.len());
        if out.len() > self.len_bytes {
            return Err("buffer read larger than buffer".into());
        }
        let f = fns();
        unsafe {
            check(
                (f.enqueue_read)(
                    ctx.queue,
                    self.mem,
                    1,
                    0,
                    out.len(),
                    out.as_mut_ptr() as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null(),
                ),
                "clEnqueueReadBuffer",
            )?;
            check((f.finish)(ctx.queue), "clFinish after read")?;
            eprintln!("[opencl] clEnqueueReadBuffer ok");
            Ok(())
        }
    }

    pub fn as_mem(&self) -> cl_mem {
        self.mem
    }
}

/// Launch `global` work items in `local`-sized workgroups and wait for the
/// queue to drain.
pub fn run(ctx: &Context, kernel: &Kernel, global: usize, local: usize) -> Result<(), String> {
    eprintln!("[opencl] clEnqueueNDRangeKernel({global} gid, {local} lid)...");
    let f = fns();
    let g = global.max(1);
    let l = local.clamp(1, g);
    unsafe {
        check(
            (f.enqueue_ndrange)(
                ctx.queue,
                kernel.kernel,
                1,
                ptr::null(),
                &g,
                &l,
                0,
                ptr::null(),
                ptr::null(),
            ),
            "clEnqueueNDRangeKernel",
        )?;
        check((f.finish)(ctx.queue), "clFinish")?;
        eprintln!("[opencl] clEnqueueNDRangeKernel ok");
        Ok(())
    }
}

/// Human-readable name of a cl error code (for the common ones).
pub fn err_name(code: c_int) -> &'static str {
    match code {
        -1 => "CL_DEVICE_NOT_FOUND",
        -2 => "CL_DEVICE_NOT_AVAILABLE",
        -6 => "CL_OUT_OF_HOST_MEMORY",
        -30 => "CL_INVALID_VALUE",
        -38 => "CL_INVALID_CONTEXT",
        -52 => "CL_INVALID_KERNEL_NAME",
        -59 => "CL_INVALID_PROGRAM_EXECUTABLE",
        -61 => "CL_INVALID_KERNEL_ARGS",
        -62 => "CL_INVALID_WORK_DIMENSION",
        -63 => "CL_INVALID_WORK_GROUP_SIZE",
        -64 => "CL_INVALID_WORK_ITEM_SIZE",
        0 => "CL_SUCCESS",
        _ => "unknown",
    }
}

/// Small sanity helper used by tools: `CStr::from_ptr` wrapper for device
/// strings is not needed since read_str handles it; kept for future use.
pub fn cstr_bytes(c: &CStr) -> &[u8] {
    c.to_bytes()
}
