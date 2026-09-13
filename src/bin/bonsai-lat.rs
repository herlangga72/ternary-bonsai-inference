//! bonsai-lat: measure true per-call matvec latency (cached single-shot
//! submits, one fence per call) so the decode architecture can be sized
//! correctly: kernel time vs submit+fence overhead per matvec.

#[path = "../gguf.rs"]
mod gguf;
#[path = "../kernels.rs"]
mod kernels;
#[path = "../kvquant.rs"]
mod kvquant;
#[path = "../vk.rs"]
mod vk;

fn rand_floats(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: bonsai-lat <model.gguf> <tensor> [calls]");
        std::process::exit(1);
    }
    let calls: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let g = gguf::GGUF::open(&args[0]).expect("open gguf");
    let info = g
        .tensors
        .iter()
        .find(|t| t.name == args[1])
        .expect("tensor not found")
        .clone();
    let ne0 = info.dims[0] as usize;
    let rows = kernels::n_rows(&info) as usize;
    let payload = g.payload_slice(&info).expect("payload");
    let x = rand_floats(0xABCD, ne0);

    let mut gpu = vk::Gpu::open().expect("gpu");
    gpu.prep_matvec_cache(ne0.max(17408), rows.max(248320))
        .expect("prep");
    let w = gpu.create_weight_buffer(payload.len()).expect("wbuf");
    gpu.upload(&w, payload).expect("upload");

    let _ = gpu.matvec_on(&w, ne0, 0, rows, &x).expect("warm");
    let t0 = std::time::Instant::now();
    for _ in 0..calls {
        let _ = gpu.matvec_on(&w, ne0, 0, rows, &x).expect("matvec");
    }
    let dt = t0.elapsed().as_secs_f64() / calls as f64;
    let macs = rows as u64 * ne0 as u64;
    println!(
        "{} {}x{}: {:.2} ms/call (kernel+submit+fence), {:.2} GMAC/s, {:.3} GB/s",
        args[1],
        rows,
        ne0,
        dt * 1e3,
        macs as f64 / dt / 1e9,
        payload.len() as f64 / dt / 1e9
    );
    gpu.destroy_dev_buffer(w);
}
