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
use std::time::SystemTime;

// Normally I would put this in a separate file, but
// it is a bit unnecessary and it matches better with the C
mod mandel {
    use crate::bitmap::BYTES_PER_PIXEL;

    use std::alloc::{alloc, dealloc, Layout};
    use std::f64::consts::PI;
    use std::hint::unreachable_unchecked;
    use std::io::Write;
    use std::ptr;
    use std::simd::{f64x4, Simd, StdFloat};
    use std::slice;

    const MM_ONES: f64x4 = Simd::splat(1.0);
    const MM_ZERO: f64x4 = Simd::splat(0.0);

    /// Contains all necessary data for generating the image
    pub struct MandelImage {
        data: *mut u8,
        n_threads: usize,
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

    /// Contains the data necessary for the functions in threads.
    ///
    /// Ideally, this would be done in a cleaner way. This type is
    /// only necessary to prevent the pointer from being dropped multiple times
    /// and allow it to be shared mutably.
    #[derive(Clone)]
    struct PassedData {
        data: *mut u8,
        n_threads: usize,
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

    // this contains all functions which are inside a thread
    impl PassedData {
        /// converts hsl to rgb, modified from
        /// https://web.archive.org/web/20081227003853/http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
        ///
        /// ### Safety
        /// Assumes it is the only concurrent instance running with a certain
        /// `i` and `j`. Also assumes `i` and `j` are valid to use to dereference
        /// self.data
        unsafe fn set_hsl2rgb(&mut self, x: usize, y: usize, h: f64, s: f64, v: f64) {
            let r;
            let g;
            let b;

            let i = (h * 6.).floor();
            let f = h * 6. - i;
            let p = v * (1. - s);
            let q = v * (1. - f * s);
            let t = v * (1. - (1. - f) * s);

            match (i % 6.0) as u8 {
                0 => {
                    r = v;
                    g = t;
                    b = p;
                }
                1 => {
                    r = q;
                    g = v;
                    b = p;
                }
                2 => {
                    r = p;
                    g = v;
                    b = t;
                }
                3 => {
                    r = p;
                    g = q;
                    b = v;
                }
                4 => {
                    r = t;
                    g = p;
                    b = v;
                }
                5 => {
                    r = v;
                    g = p;
                    b = q;
                }
                _ => unreachable_unchecked(),
            }

            // This is an external function in the C version
            // this is just simpler
            ptr::write(
                self.data
                    .offset((x * BYTES_PER_PIXEL + y * self.width * BYTES_PER_PIXEL) as isize + 2),
                (r * 255.0) as u8,
            );
            ptr::write(
                self.data
                    .offset((x * BYTES_PER_PIXEL + y * self.width * BYTES_PER_PIXEL) as isize + 1),
                (g * 255.0) as u8,
            );
            ptr::write(
                self.data
                    .offset((x * BYTES_PER_PIXEL + y * self.width * BYTES_PER_PIXEL) as isize + 0),
                (b * 255.0) as u8,
            );
        }

        /// calculates the color for a pixel `[i,j]` of the image
        ///
        /// ### Safety
        /// Assumes `i` and `j` are valid to use in `set_hsl2rgb`
        #[target_feature(enable = "avx2")]
        unsafe fn calc_pixel_mm(&mut self, i: usize, j: usize) {
            // TODO: add julia set capabilities

            // initialize values

            // c: complex number
            let c_real = Simd::from_array([
                ((i + 0) as f64 / self.width as f64 - 0.5) * self.scale + self.offset.0,
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
            let mut z_real = MM_ZERO;
            let mut z_imag = MM_ZERO;

            // z': complex running derivative
            let mut z_prime_r = MM_ONES;
            let mut z_prime_i = MM_ONES;

            // z^2: temporary value for optimized computation
            let mut real_2 = MM_ZERO;
            let mut imag_2 = MM_ZERO;

            // value accumulators for coloring
            let mut step_acc = MM_ZERO;
            let mut orbit_acc = MM_ONES;

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

                let z_prime_r_tmp = ac_bd + ac_bd + MM_ONES;
                let z_prime_i_tmp = bc_da + bc_da;

                let radius_2 = real_2 + imag_2;

                // select lanes which have not escaped
                // escape of 1000.0 used to smooth distance estimate
                let mask = radius_2.lanes_lt(Simd::splat(1000.0));

                // conditionally iterate, only if the pixel has not escaped
                z_real = mask.select(z_real_tmp, z_real);
                z_imag = mask.select(z_imag_tmp, z_imag);
                z_prime_i = mask.select(z_prime_i_tmp, z_prime_i);
                z_prime_r = mask.select(z_prime_r_tmp, z_prime_r);

                real_2 = z_real * z_real;
                imag_2 = z_imag * z_imag;

                step_acc = mask.select(MM_ONES, MM_ZERO) + step_acc;
                orbit_acc = orbit_acc.min(real_2 + imag_2);

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
                    self.set_hsl2rgb(
                        i + v,
                        j,
                        0.0,
                        0.0,
                        ((extracted_orbit[v]).sqrt() * self.brightness * self.internal_brightness)
                            .clamp(0.0, 1.0),
                    );
                } else {
                    // color the outside
                    self.set_hsl2rgb(
                        i + v,
                        j,
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
                    );
                }
            }
        }

        /// calculates one thread's portion of the image.
        /// also prints progress in the first thread (id=0)
        ///
        /// ### Safety
        /// Assumes it is running in parallel with a unique thread_id,
        /// calling multiple concurrent instances with the same thread_id
        /// may lead to data races
        unsafe fn calc_image_region(mut self, start_height: usize, thread_id: usize) {
            for j in (start_height..self.height).step_by(self.n_threads) {
                // print progress bar
                if thread_id == 0 && j % (self.n_threads * 4) == 0 {
                    let percentage = j as f64 / self.height as f64;
                    print!("\x1b[2K\x1b[0GProgress: |");
                    for i in 0..40 {
                        if i as f64 / 40.0 < percentage {
                            print!(
                                "\x1b[9{}m█\x1b[0m",
                                [5, 1, 3, 2, 6, 4][(i as f64 / 40. * 6.) as usize]
                            );
                        } else {
                            print!(" ");
                        }
                    }
                    print!("| {:.2}%", percentage * 100.0);
                    std::io::stdout().flush().unwrap();
                }
                // actual calculation
                for i in (0..self.width).step_by(MM_ONES.lanes()) {
                    self.calc_pixel_mm(i, j);
                }
            }
            // print a 100% message. This is not strictly necessary, but it looks nicer
            if thread_id != 0 {
                return;
            }
            print!("\x1b[2K\x1b[0GProgress: |");
            for i in 0..40 {
                print!(
                    "\x1b[9{}m█\x1b[0m",
                    [5, 1, 3, 2, 6, 4][(i as f64 / 40. * 6.) as usize]
                );
            }
            println!("| {:.2}%", 100.0);
        }
    }

    // PassedData is only a mutable borrow into the MandelImage
    // all functions which use it know this, and it should not be used otherwise
    unsafe impl Send for PassedData {}

    impl MandelImage {
        /// handles the dispatch of all threads
        pub fn run_threads(&mut self) {
            (0..self.n_threads)
                .into_iter()
                .map(|t| {
                    let passable = PassedData {
                        data: self.data,
                        n_threads: self.n_threads,
                        width: self.width,
                        height: self.height,
                        max_iter: self.max_iter,
                        scale: self.scale,
                        zoom: self.zoom,
                        offset: self.offset,
                        saturation: self.saturation,
                        color_frequency: self.color_frequency,
                        color_offset: self.color_offset,
                        glow_spread: self.glow_spread,
                        glow_strength: self.glow_strength,
                        brightness: self.brightness,
                        internal_brightness: self.internal_brightness,
                    };
                    std::thread::spawn(move || unsafe { passable.calc_image_region(t, t) })
                })
                // force the threads to start by consuming the iterator
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|t| t.join().unwrap());
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
                data: ptr::null_mut(),
                n_threads: num_cpus::get(),
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
                };
                if r.is_err() {
                    println!("Failed to read argument: {} {}", arg, next_arg);
                    println!("usage: {} {}", exe_name, Self::USAGE);
                    std::process::exit(1);
                }
            }
            if image.filename.len() == 0 {
                println!("Output filename required.");
                println!("usage: {} {}", exe_name, Self::USAGE);
                std::process::exit(1);
            }

            // allocate necessary memory
            let layout = Layout::array::<u8>(image.width * image.height * BYTES_PER_PIXEL)
                .expect("Failed to make layout for array!");

            image.data = unsafe { alloc(layout) };

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

        /// Returns a slice referencing the internal data of this function
        pub fn get_data(&self) -> &[u8] {
            unsafe {
                &*slice::from_raw_parts(self.data, self.width * self.height * BYTES_PER_PIXEL)
            }
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

    impl Drop for MandelImage {
        fn drop(&mut self) {
            // deallocate owned memory
            let layout =
                std::alloc::Layout::array::<u8>(self.width * self.height * BYTES_PER_PIXEL)
                    .expect("Failed to make layout for array!");
            unsafe { dealloc(self.data, layout) };
        }
    }
}

fn main() {
    let total_start = SystemTime::now();

    let mut image = mandel::MandelImage::create_from_args();

    println!("Rendering...");
    let render_start = SystemTime::now();
    image.run_threads();
    let render_end = SystemTime::now();

    println!("Saving...");
    let save_start = SystemTime::now();
    if let Err(e) = generate_bitmap_image(
        image.get_data(),
        image.height(),
        image.width(),
        image.filename(),
    ) {
        println!("Failed to save image: {:?}", e);
        std::process::exit(1);
    }
    let save_end = SystemTime::now();

    println!("Done.");
    println!();

    // print elapsed time
    let total_end = SystemTime::now();
    println!(
        "Time:
    Rendering:\t{:.2} s
    Saving:\t{:.2} s
    Total:\t{:.2} s",
        render_end
            .duration_since(render_start)
            .unwrap()
            .as_secs_f64(),
        save_end.duration_since(save_start).unwrap().as_secs_f64(),
        total_end.duration_since(total_start).unwrap().as_secs_f64()
    );
}
