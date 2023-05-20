# C vs. Rust: Mandelbrot

This repository is a comparison between C and Rust, using a Mandelbrot fractal generator. The main goal is to see how close Rust can come to the efficiency of a highly optimized (at least in some respects) C program using threads and SIMD compiler intrinsics. ~~Because of this, the Rust version is implemented using unsafe code and raw pointers for the performance-critical portions.~~ The Rust code has been refactored to remove almost all unsafe code without significantly affecting performance (the only remaining unsafe is marking an unreachable branch, and has nothing to do with memory writes). 

## Results

In testing on my local machine, I was able to get the runtime of both implementations to be almost indistinguishable. The only noticeable performance difference is in the bitmap saving code, which makes up a very small portion of the run time.

Sample execution time for default image rendered at 7680×4320 pixels:

```
$ cargo run --release -- out.bmp -w 7680 -h 4320

Internal timing:
Time:
    Rendering:	3.32 s
    Saving:	0.15 s
    Total:	3.47 s

zsh time command:
cargo run --release -- out.bmp -w 7680 -h 4320  25.35s user 0.19s system 710% cpu 3.596 total

$ ./mandel out.bmp -w 7680 -h 4320

Internal timing:
Time:
    Rendering:	3.54 s
    Saving:	0.01 s
    Total:	3.55 s

zsh time command:
./mandel out.bmp -w 7680 -h 4320  28.20s user 0.21s system 728% cpu 3.901 total
```

## Compiling

### Requirements:

Both implementations require advanced vector extensions to run as intended, and the C code is only known to support Linux-based systems.

The C code can be compiled by executing `make` in the `mandel_c` folder. The rust code can be compiled by running `cargo build --release` in the `mandel_rs` folder.

### Usage

Both programs accept arguments describing the image to render, and produce a bitmap output with the specified filename. The arguments accepted by both programs are the same, but the binaries are in slightly different locations. Rust locates the binary in `./target/release/mandel_rs` (or can be run with `cargo run --release -- [args]`), whereas the C is compiled to `./mandel`.

Help is printed on failure to parse arguments, or if the `--help` flag is passed

```sh
./mandel --help
```

Example usage:

```
./mandel output.bmp
```

![A simple example](simple_example.jpg)

```
./mandel \
  -x -1.1641595843886 \
  -y -0.25057061578732 \
  -z 38.5 \
  -m 50000 \
  -w 1920 \
  -h 1080 \
  -s 0.8 \
  -f 0.05 \
  -o 0.0 \
  -g 1.35 \
  -G -4.0 \
  -b 1.2 \
  -B 10.2 \
  output.bmp
```

![A complex example](complex_example.jpg)
