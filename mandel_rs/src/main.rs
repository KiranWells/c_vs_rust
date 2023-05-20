// A mandelbrot generation program written in Rust
// designed to run as efficiently as possible for double precision.
//
// Author: Griffith Thomas
//
// Sources for any external functions are annotated in the code

#![feature(portable_simd)]
#![feature(try_blocks)]

mod bitmap;

use bitmap::generate_bitmap_image;
use std::time::Instant;

// Normally I would put this in a separate file, but
// it is a bit unnecessary and it matches better with the C
mod mandel {
    use crate::bitmap::BYTES_PER_PIXEL;

    use std::f64::consts::PI;
    use std::hint::unreachable_unchecked;
    use std::io::Write;
    use std::num::NonZeroUsize;
    use std::simd::{Simd, SimdFloat, SimdPartialOrd, StdFloat};
    use std::thread::available_parallelism;

    const SIMD_WIDTH: usize = 4;
    type SimdFloatVec = std::simd::f64x4;

    /// Contains all necessary data for generating the image
    pub struct MandelImage {
        n_threads: NonZeroUsize,
        // file data
        filename: String,
        width: usize,
        height: usize,
        // image parameters
        max_iter: usize,
        scale: f64,
        zoom: f64,
        offset: (f64, f64),
        // colors
        saturation: f64,
        color_frequency: f64,
        color_offset: f64,
        glow_spread: f64,
        glow_strength: f64,
        brightness: f64,
        internal_brightness: f64,
    }

    /// converts hsl to rgb, modified from
    /// https://web.archive.org/web/20081227003853/http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
    fn hsl2rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
        let (r, g, b);

        let i = (h * 6.).floor();
        let f = h * 6. - i;
        let p = v * (1. - s);
        let q = v * (1. - f * s);
        let t = v * (1. - (1. - f) * s);

        match (i % 6.0) as u8 {
            0 => (r, g, b) = (v, t, p),
            1 => (r, g, b) = (q, v, p),
            2 => (r, g, b) = (p, v, t),
            3 => (r, g, b) = (p, q, v),
            4 => (r, g, b) = (t, p, v),
            5 => (r, g, b) = (v, p, q),
            _ => unsafe { unreachable_unchecked() },
        }

        [(b * 255.0) as u8, (g * 255.0) as u8, (r * 255.0) as u8]
    }

    fn print_progress(percentage: f64) {
        let mut stdout = std::io::stdout().lock();
        write!(stdout, "\x1b[2K\x1b[0GProgress: |").unwrap();

        for i in 0..40 {
            if i as f64 / 40.0 < percentage {
                write!(
                    stdout,
                    "\x1b[9{}m█\x1b[0m",
                    [5, 1, 3, 2, 6, 4][(i as f64 / 40. * 6.) as usize]
                )
            } else {
                write!(stdout, " ")
            }
            .unwrap()
        }
        write!(stdout, "| {:.2}%", percentage * 100.0).unwrap();
        stdout.flush().unwrap();
    }

    impl MandelImage {
        /// calculates the color for pixels starting at `[i,j]` and ending at `[i+4,j]`.
        ///
        /// `#[inline(always)]` is used to force the compiler to inline this function.
        /// This is necessary for the compiler to optimize the code with avx2 instructions
        /// (at least on the current nightly version)
        #[inline(always)]
        fn calc_pixel(&self, data: &mut [u8], i: usize, j: usize) {
            // TODO: add julia set capabilities

            // initialize values
            let mm_ones: SimdFloatVec = Simd::splat(1.0);
            let mm_zero: SimdFloatVec = Simd::splat(0.0);

            // c: complex number
            let c_real = Simd::from_array([
                (i as f64 / self.width as f64 - 0.5) * self.scale + self.offset.0,
                ((i + 1) as f64 / self.width as f64 - 0.5) * self.scale + self.offset.0,
                ((i + 2) as f64 / self.width as f64 - 0.5) * self.scale + self.offset.0,
                ((i + 3) as f64 / self.width as f64 - 0.5) * self.scale + self.offset.0,
            ]);

            let c_imag = Simd::splat(
                (j as f64 / self.height as f64 - 0.5)
                    * self.scale
                    * (self.height as f64 / self.width as f64)
                    + self.offset.1,
            );

            // z: complex number
            let mut z_real = mm_zero;
            let mut z_imag = mm_zero;

            // z': complex running derivative
            let mut z_prime_r = mm_ones;
            let mut z_prime_i = mm_ones;

            // z^2: temporary value for optimized computation
            let mut real_2 = mm_zero;
            let mut imag_2 = mm_zero;

            // value accumulators for coloring
            let mut step_acc = mm_zero;
            let mut orbit_acc = mm_ones;

            for _step in 0..self.max_iter {
                // iterate values, according to z = z^2 + c
                //
                // uses an optimized computation method from wikipedia for z:
                //   z.i := 2 × z.r × z.i + c.i
                //   z.r := r2 - i2 + c.r
                //   r2 := z.r × z.r
                //   i2 := z.i × z.i
                //
                // z' is calculated according to the standard formula (z' = 2*z*z' + 1):
                //   z'.r = 2 * (z.r * z'.r - z.i * z'.i) + 1
                //   z'.i = 2 * (z.i * z'.r + z.r * z'.i)

                let z_imag_tmp = (z_real + z_real) * z_imag + c_imag;
                let z_real_tmp = real_2 - imag_2 + c_real;

                // intermediate values for z'
                let ac_bd = z_real * z_prime_r - z_imag * z_prime_i;
                let bc_da = z_imag * z_prime_r + z_real * z_prime_i;

                let z_prime_r_tmp = ac_bd + ac_bd + mm_ones;
                let z_prime_i_tmp = bc_da + bc_da;

                let radius_2 = real_2 + imag_2;

                // select lanes which have not escaped
                // escape of 1000.0 used to smooth distance estimate
                let mask = radius_2.simd_lt(Simd::splat(1000.0));

                // conditionally iterate, only if the pixel has not escaped
                z_real = mask.select(z_real_tmp, z_real);
                z_imag = mask.select(z_imag_tmp, z_imag);
                z_prime_i = mask.select(z_prime_i_tmp, z_prime_i);
                z_prime_r = mask.select(z_prime_r_tmp, z_prime_r);

                real_2 = z_real * z_real;
                imag_2 = z_imag * z_imag;

                step_acc = mask.select(mm_ones, mm_zero) + step_acc;
                orbit_acc = orbit_acc.simd_min(real_2 + imag_2);

                // finish if all pixels have escaped
                if !mask.any() {
                    break;
                }
            }

            // calculate the absolute value (radius) of z for distance estimation
            let r = (real_2 + imag_2).sqrt();
            let dr = (z_prime_r * z_prime_r + z_prime_i * z_prime_i).sqrt();

            // extract values necessary for coloring
            let extracted_step = step_acc.as_array();
            let extracted_dr = dr.as_array();
            let extracted_r = r.as_array();
            let extracted_orbit = orbit_acc.as_array();

            // do coloring for all four pixels
            for v in 0..extracted_step.len() {
                // distance estimation: 0.5 * log(r) * r/dr
                let dist_est = 0.5 * (extracted_r[v]).ln() * extracted_r[v] / extracted_dr[v];
                // a 'glow' effect based on distance (manually adjusted to taste and to adjust to zoom level)
                let glow =
                    (-(dist_est).ln() + self.glow_spread) / (self.zoom + 15.1) * self.glow_strength;
                // a smoothed version of the iteration count: i + (1 - ln(ln(r))/ln(2))
                let smoothed_step =
                    extracted_step[v] + (1.0 - ((extracted_r[v]).ln()).ln() / f64::ln(2.0));

                if extracted_step[v] as usize >= self.max_iter {
                    // color the inside using orbit trap method
                    data[(i + v) * 3..(i + v) * 3 + 3].copy_from_slice(&hsl2rgb(
                        0.0,
                        0.0,
                        ((extracted_orbit[v]).sqrt() * self.brightness * self.internal_brightness)
                            .clamp(0.0, 1.0),
                    ));
                } else {
                    // color the outside
                    data[(i + v) * 3..(i + v) * 3 + 3].copy_from_slice(&hsl2rgb(
                        // color hue based on an sinusoidal step counter, offset to a [0,1] range
                        (((smoothed_step * 0.05 * self.color_frequency - self.color_offset * PI)
                            .sin())
                            * 0.5
                            + 0.5)
                            .clamp(0.0, 1.0),
                        // saturation decreased when glow is high to hide noise when hue oscillates quickly
                        (self.saturation * (1.0 - (glow * glow))).clamp(0.0, 1.0),
                        // use glow around edges for brightness
                        (glow * self.brightness).clamp(0.0, 1.0),
                    ));
                }
            }
        }

        // AVX2, AVX, and SSE4.1 implementations are identical to the scalar implementation
        // except for the target_feature annotations and the function names. (the portable
        // simd utilities provided by the std library do the heavy lifting)
        #[target_feature(enable = "avx2")]
        #[inline]
        #[allow(dead_code)]
        unsafe fn calc_pixel_avx2(&self, data: &mut [u8], i: usize, j: usize) {
            self.calc_pixel(data, i, j);
        }

        #[target_feature(enable = "avx")]
        #[inline]
        #[allow(dead_code)]
        unsafe fn calc_pixel_avx(&self, data: &mut [u8], i: usize, j: usize) {
            self.calc_pixel(data, i, j);
        }

        #[target_feature(enable = "sse4.1")]
        #[inline]
        #[allow(dead_code)]
        unsafe fn calc_pixel_sse4(&self, data: &mut [u8], i: usize, j: usize) {
            self.calc_pixel(data, i, j);
        }

        /// calculates one thread's portion of the image.
        /// also prints progress in the first thread (id=0)
        fn calc_image_region(&self, lines: Vec<(usize, &mut [u8])>, t: usize) {
            for (j, line) in lines {
                // print progress bar
                if t == 0 && j % 64 == 0 {
                    let percentage = j as f64 / self.height as f64;
                    print_progress(percentage);
                }
                for i in (0..self.width).step_by(SIMD_WIDTH) {
                    // select the correct implementation based on target features

                    // compile time feature detection first
                    #[cfg(target_feature = "avx2")]
                    unsafe {
                        self.calc_pixel_avx2(line, i, j);
                    }
                    #[cfg(all(target_feature = "avx", not(target_feature = "avx2")))]
                    unsafe {
                        self.calc_pixel_avx(line, i, j);
                    }
                    #[cfg(all(
                        target_feature = "sse4.1",
                        not(any(target_feature = "avx2", target_feature = "avx"))
                    ))]
                    unsafe {
                        self.calc_pixel_sse4(line, i, j);
                    }

                    // runtime feature detection as fallback (this does not seem to have any
                    // measurable performance impact compared to the compile time feature detection)
                    #[cfg(not(any(
                        target_feature = "avx2",
                        target_feature = "avx",
                        target_feature = "sse4.1"
                    )))]
                    {
                        if is_x86_feature_detected!("avx2") {
                            unsafe {
                                self.calc_pixel_avx2(line, i, j);
                            }
                        } else if is_x86_feature_detected!("avx") {
                            unsafe {
                                self.calc_pixel_avx(line, i, j);
                            }
                        } else if is_x86_feature_detected!("sse4.1") {
                            unsafe {
                                self.calc_pixel_sse4(line, i, j);
                            }
                        } else {
                            self.calc_pixel(line, i, j);
                        }
                    }
                }
            }
            // print complete progress bar
            if t == 0 {
                print_progress(1.0);
                println!();
            }
        }

        /// handles the dispatch of all threads
        pub fn run_threads(&self) -> Vec<u8> {
            // The image data is created here to prevent MandelImage
            // from owning the data, which would make it impossible
            // to pass to the threads without either cloning or making
            // the data immutable.

            // create the image buffer
            let mut data = vec![0_u8; self.width * self.height * BYTES_PER_PIXEL];
            // split image into lines for each thread
            let lines = data.chunks_exact_mut(self.width * BYTES_PER_PIXEL);
            // interleave the lines to distribute the work evenly
            let mut interleaved_lines: Vec<Vec<(usize, &mut [u8])>> = (0..self.n_threads.get())
                .map(|_| Vec::with_capacity(self.width * BYTES_PER_PIXEL))
                .collect();
            lines.enumerate().for_each(|(i, line)| {
                interleaved_lines[i % self.n_threads].push((i, line));
            });
            // thread::scope allows the threads to borrow data from the parent thread
            // because the threads are joined before the scope ends, ensuring that
            // the data lives for the entire duration of each thread
            std::thread::scope(|scope| {
                interleaved_lines
                    .into_iter()
                    .enumerate()
                    .map(|(t, lines)| scope.spawn(move || self.calc_image_region(lines, t)))
                    // force the threads to start by consuming the iterator
                    .collect::<Vec<_>>()
                    .into_iter()
                    .for_each(|t| t.join().unwrap());
            });
            data
        }

        /// the usage info for the program, printed on --help
        /// and on failure to parse arguments
        const USAGE: &'static str = " [options] <filename>\n\
                     \nOptions:\n\
                     \t-w | --width <int>\tThe width of the output in pixels\n\t\tDefault: 1920\n\
                     \t-h | --height <int>\tThe height of the output in pixels\n\t\tDefault: 1080\n\
                     \t-z | --zoom <float>\tThe scale of the image; larger is more zoomed in. Scale = 2^(-zoom)\n\t\tDefault: -2\n\
                     \t-x | --x-offset <float>\tThe real coordinate of the center of the image\n\t\tDefault: -0.5\n\
                     \t-y | --y-offset <float>\tThe imaginary coordinate of the center of the image\n\t\tDefault: 0.0\n\
                     \t-m | --max-iter <int>\tThe maximum number of fractal iterations\n\t\tDefault: 1000\n\
                     \t-s | --saturation <float>\tThe saturation of the image. [0,1]\n\t\tDefault: 1.0\n\
                     \t-f | --color-frequency <float>\tThe frequency of the cycling of colors\n\t\tDefault: 1.0\n\
                     \t-o | --color-offset <float>\tThe offset of the cycling period of colors [-1,1]\n\t\tDefault: 1.0\n\
                     \t-g | --glow-strength <float>\tThe strength of the glow around the edges \n\t\tDefault: 1.0\n\
                     \t-G | --glow-spread <float>\tHow much the glow bleeds out into surrounding area \n\t\tDefault: 1.0\n\
                     \t-b | --brightness <float>\tThe brightness multiplier for the image \n\t\tDefault: 2.0\n\
                     \t-B | --internal-brightness <float>\tThe brightness multiplier for the internal shading \n\t\tDefault: 1.0\n\
                     \t-t | --threads <int>\tThe number of threads used for computation. \n\t\tDefaults to number of logical cores on the machine.\n\
                     \n";

        /// Creates a new image instance from the process arguments
        pub fn create_from_args() -> Self {
            let mut image = MandelImage {
                n_threads: available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap()),
                filename: String::new(),
                width: 1920,
                height: 1080,
                max_iter: 1000, // this is way more than necessary for default
                scale: 4.0,
                zoom: -2.0,
                offset: (-0.5, 0.0),
                saturation: 1.0,
                color_frequency: 1.0,
                color_offset: 0.0,
                glow_spread: 1.0,
                glow_strength: 1.0,
                brightness: 2.0,
                internal_brightness: 1.0,
            };

            // parse arguments
            let mut args = std::env::args();
            // assume we at least have an argv[0]
            let exe_name = args.next().unwrap();
            while let Some(arg) = args.next() {
                if arg == "--help" {
                    println!("usage: {} {}", exe_name, Self::USAGE);
                    // this can be done more idiomatically in Rust,
                    // but this is more comparable to the C behavior
                    std::process::exit(0);
                }
                if let Some(c) = arg.chars().next() {
                    if c != '-' {
                        image.filename = arg;
                        continue;
                    }
                }
                // all other flags require an argument
                // note: this does not check if the flag is valid
                let next_arg = args.next();
                if next_arg.is_none() {
                    println!("Flag {} requires an argument.", arg);
                    println!("usage: {} {}", exe_name, Self::USAGE);
                    std::process::exit(1);
                }
                let next_arg = next_arg.unwrap();

                let r: Result<_, Box<dyn std::error::Error>> = try {
                    match arg.as_str() {
                        "-w" | "--width" => image.width = next_arg.parse()?,
                        "-h" | "--height" => image.height = next_arg.parse()?,
                        "-z" | "--zoom" => {
                            image.zoom = next_arg.parse()?;
                            image.scale = f64::powf(2.0, -image.zoom);
                        }
                        "-x" | "--x-offset" => image.offset.0 = next_arg.parse()?,
                        "-y" | "--y-offset" => image.offset.1 = -next_arg.parse()?,
                        "-m" | "--max-iter" => image.max_iter = next_arg.parse()?,
                        "-s" | "--saturation" => image.saturation = next_arg.parse()?,
                        "-f" | "--color-frequency" => image.color_frequency = next_arg.parse()?,
                        "-o" | "--color-offset" => image.color_offset = next_arg.parse()?,
                        "-g" | "--glow-strength" => image.glow_strength = next_arg.parse()?,
                        "-G" | "--glow-spread" => image.glow_spread = next_arg.parse()?,
                        "-b" | "--brightness" => image.brightness = next_arg.parse()?,
                        "-B" | "--internal-brightness" => {
                            image.internal_brightness = next_arg.parse()?
                        }
                        "-t" | "--threads" => image.n_threads = next_arg.parse()?,
                        _ => {
                            println!("Argument not recognized: {}", arg);
                            println!("usage: {} {}", exe_name, Self::USAGE);
                            std::process::exit(1);
                        }
                    }
                    Ok::<(), Box<dyn std::error::Error>>(())
                };
                if r.is_err() {
                    println!("Failed to read argument: {} {}", arg, next_arg);
                    println!("usage: {} {}", exe_name, Self::USAGE);
                    std::process::exit(1);
                }
            }
            if image.filename.is_empty() {
                println!("Output filename required.");
                println!("usage: {} {}", exe_name, Self::USAGE);
                std::process::exit(1);
            }

            // print final parsed options
            println!("Running with options: \n");
            println!("\tWidth:\t\t\t{}", image.width);
            println!("\tHeight:\t\t\t{}", image.height);
            // println!("\tScale:\t\t\t{}", image.scale);
            println!("\tZoom:\t\t\t{:.2}", image.zoom);
            println!(
                "\tCenter point:\t\t({:.9}, {:.9})",
                image.offset.0, image.offset.1
            );
            println!("\tMax Iterations:\t\t{}", image.max_iter);
            println!("\tSaturation:\t\t{:.2}", image.saturation);
            println!("\tColor Frequency:\t{:.2}", image.color_frequency);
            println!("\tColor Offset:\t\t{:.2}", image.color_offset);
            println!("\tGlow Strength:\t\t{:.2}", image.glow_strength);
            println!("\tGlow Spread:\t\t{:.2}", image.glow_spread);
            println!("\tBrightness:\t\t{:.2}", image.brightness);
            println!("\tInternal Brightness:\t{:.2}", image.internal_brightness);
            println!("\tThreads:\t\t{}", image.n_threads);
            println!("\tFilename:\t\t{}", image.filename);

            image
        }

        pub fn width(&self) -> usize {
            self.width
        }
        pub fn height(&self) -> usize {
            self.height
        }
        pub fn filename(&self) -> &str {
            self.filename.as_str()
        }
    }
}

fn main() {
    let total_start = Instant::now();

    let image = mandel::MandelImage::create_from_args();

    println!("Rendering...");
    let render_start = Instant::now();
    let image_data = image.run_threads();
    let render_end = Instant::now();

    println!("Saving...");
    let save_start = Instant::now();
    if let Err(e) =
        generate_bitmap_image(&image_data, image.height(), image.width(), image.filename())
    {
        println!("Failed to save image: {:?}", e);
        std::process::exit(1);
    }
    let save_end = Instant::now();

    println!("Done.");
    println!();

    // print elapsed time
    let total_end = Instant::now();
    println!(
        "Time:
    Rendering:\t{:.2} s
    Saving:\t{:.2} s
    Total:\t{:.2} s",
        render_end.duration_since(render_start).as_secs_f64(),
        save_end.duration_since(save_start).as_secs_f64(),
        total_end.duration_since(total_start).as_secs_f64()
    );
}
