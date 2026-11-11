use std::{fs::File, time::Instant};
use symphonia::core::{
    codecs::audio::{AudioDecoder, AudioDecoderOptions},
    codecs::registry::CodecRegistry,
    formats::{FormatOptions, FormatReader, SeekMode, SeekTo, TrackType, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    units::Timestamp,
};

struct Open { reader: Box<dyn FormatReader>, decoder: Box<dyn AudioDecoder>, id: u32 }

fn open(path: &str) -> Open {
    let file = File::open(path).unwrap();
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new(); hint.with_extension("flac");
    let reader = symphonia::default::get_probe().probe(&hint, stream, FormatOptions::default(), MetadataOptions::default()).unwrap();
    let track = reader.default_track(TrackType::Audio).unwrap();
    let id = track.id;
    let params = track.codec_params.as_ref().unwrap().audio().unwrap().clone();
    let mut codecs = CodecRegistry::new();
    symphonia::default::register_enabled_codecs(&mut codecs);
    let decoder = codecs.make_audio_decoder(&params, &AudioDecoderOptions::default()).unwrap();
    Open { reader, decoder, id }
}

fn page(o: &mut Open, a: u64, n: usize) -> (Vec<f32>, u64) {
    o.reader.seek(SeekMode::Accurate, SeekTo::Timestamp { ts: Timestamp::new(a as i64), track_id: o.id }).unwrap();
    o.decoder.reset();
    let mut out: Vec<f32> = Vec::with_capacity(n * 2);
    let mut first: Option<i64> = None;
    let mut decoded_frames = 0u64;
    let mut tmp = Vec::new();
    loop {
        let p = o.reader.next_packet().unwrap().unwrap();
        let pts = p.pts.get();
        let buf = o.decoder.decode(&p).unwrap();
        decoded_frames += buf.frames() as u64;
        if first.is_none() { first = Some(pts); }
        buf.copy_to_vec_interleaved(&mut tmp);
        let start = pts;
        let lo = (a as i64 - start).max(0) as usize;
        let hi = ((a as i64 + n as i64) - start).min(buf.frames() as i64);
        if hi > lo as i64 { out.extend_from_slice(&tmp[lo * 2..hi as usize * 2]); }
        if start + buf.frames() as i64 >= a as i64 + n as i64 { break; }
    }
    (out, decoded_frames)
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut x: u64 = 0x1234_5678;
    let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    let t = Instant::now();
    let mut o = open(&path);
    let mut total = 0u64;
    while let Ok(Some(p)) = o.reader.next_packet() { let b = o.decoder.decode(&p).unwrap(); total += b.frames() as u64; }
    let e = t.elapsed().as_secs_f64();
    println!("whole decode: {total} frames in {:.1} ms = {:.0}x realtime", e * 1e3, total as f64 / 48000.0 / e);
    let pages = (total / 32768) as u64;
    for mode in ["warm-reader", "cold-open"] {
        let mut o = open(&path);
        let mut times = vec![]; let mut dec = vec![];
        for _ in 0..300 {
            let k = rnd() % pages; let a = k * 32768;
            let t = Instant::now();
            if mode == "cold-open" { o = open(&path); }
            let (out, d) = page(&mut o, a, 32768);
            times.push(t.elapsed().as_secs_f64() * 1e3); dec.push(d);
            assert_eq!(out.len(), 65536);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{mode}: page 32768 frames: median {:.2} ms p95 {:.2} ms max {:.2} ms; decoded frames/page median {}", times[150], times[285], times[299], { dec.sort(); dec[150] });
    }
    let mut o = open(&path); let mut whole: Vec<f32> = vec![]; let mut tmp = vec![];
    while let Ok(Some(p)) = o.reader.next_packet() { let b = o.decoder.decode(&p).unwrap(); b.copy_to_vec_interleaved(&mut tmp); whole.extend_from_slice(&tmp); }
    let mut o = open(&path); let mut bad = 0;
    for _ in 0..200 { let k = rnd() % pages; let a = (k * 32768 + rnd() % 3000) as usize; let (out, _) = page(&mut o, a as u64, 32768); if out[..] != whole[a * 2..(a + 32768) * 2] { bad += 1; } }
    println!("page != whole slice in {bad}/200 random unaligned windows");
}
