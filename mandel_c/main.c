// A mandelbrot generation program written in pure (posix) c
// designed to run as efficiently as possible for double precision.
//
// Author: Griffith Thomas
//
// Sources for any external functions are annotated in the code

#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

// linux-only
#include <pthread.h>
#include <unistd.h>

#include "bitmap.h"

// for convenience
#define uint unsigned int

/// a conditional set operation. Sets a[i] to b[i] if m[i] is all ones,
/// otherwise does not modify a
#define mm256_mask_set(a, b, m) a = _mm256_or_pd(_mm256_andnot_pd(m, a), _mm256_and_pd(b, m));

/// Contains all necessary data for generating the image
struct MandelImage
{
  unsigned char *data;
  long n_threads;
  // file data
  char *filename;
  uint width;
  uint height;
  // image parameters
  uint max_iter;
  double scale;
  double zoom;
  double x_offset;
  double y_offset;
  // colors
  double saturation;
  double color_frequency;
  double color_offset;
  double glow_spread;
  double glow_strength;
  double brightness;
  double internal_brightness;
};

/// Contains the data necessary for the functions in threads.
///
/// Ideally, this would be done in a cleaner way. This type is
/// only necessary to allow for sending data in a single `void*`
struct PassedData
{
  struct MandelImage m;
  uint offset;
  uint thread_id;
};

/// Contains the thread information for convenience
struct ThreadPool
{
  pthread_t *threads;
  struct PassedData *data;
};

void set(struct MandelImage m, uint i, uint j, double r, double g, double b)
{
  m.data[i * BYTES_PER_PIXEL + j * m.width * BYTES_PER_PIXEL + 2] = (unsigned char)(r * 255);
  m.data[i * BYTES_PER_PIXEL + j * m.width * BYTES_PER_PIXEL + 1] = (unsigned char)(g * 255);
  m.data[i * BYTES_PER_PIXEL + j * m.width * BYTES_PER_PIXEL + 0] = (unsigned char)(b * 255);
}

/// converts hsl to rgb, modified from
/// https://web.archive.org/web/20081227003853/http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
void set_hsl2rgb(struct MandelImage m, uint x, uint y, double h, double s, double v)
{
  double r, g, b;

  double i = floor(h * 6.);
  double f = h * 6. - i;
  double p = v * (1. - s);
  double q = v * (1. - f * s);
  double t = v * (1. - (1. - f) * s);

  switch ((int)(fmod(i, 6.0)))
  {
  case 0:
    r = v, g = t, b = p;
    break;
  case 1:
    r = q, g = v, b = p;
    break;
  case 2:
    r = p, g = v, b = t;
    break;
  case 3:
    r = p, g = q, b = v;
    break;
  case 4:
    r = t, g = p, b = v;
    break;
  case 5:
    r = v, g = p, b = q;
    break;
  }

  set(m, x, y, r, g, b);
}

double clamp(double x)
{
  return x > 1.0 ? 1.0 : (x < 0.0 ? 0.0 : x);
}

/// calculates the color for a pixel [i,j] of the image
void calc_pixel_mm(struct MandelImage m, uint i, uint j)
{
  // TODO: add julia set capabilities
  const __m256d mm_ones = _mm256_set1_pd(1.0);

  // initialize values

  // c: complex number
  const __m256d c_real = _mm256_set_pd(
      ((double)(i + 0) / (double)m.width - 0.5) * m.scale + m.x_offset,
      ((double)(i + 1) / (double)m.width - 0.5) * m.scale + m.x_offset,
      ((double)(i + 2) / (double)m.width - 0.5) * m.scale + m.x_offset,
      ((double)(i + 3) / (double)m.width - 0.5) * m.scale + m.x_offset);
  const __m256d c_imag = _mm256_set1_pd(((double)j / (double)m.height - 0.5) * m.scale * ((double)m.height / (double)m.width) + m.y_offset);

  // z: complex number
  __m256d z_real = _mm256_setzero_pd();
  __m256d z_imag = _mm256_setzero_pd();

  // z': complex running derivative
  __m256d z_prime_r = mm_ones;
  __m256d z_prime_i = mm_ones;

  // z^2: temporary value for optimized computation
  __m256d real_2 = _mm256_setzero_pd();
  __m256d imag_2 = _mm256_setzero_pd();

  // value accumulators for coloring
  __m256d step_acc = _mm256_setzero_pd();
  __m256d orbit_acc = _mm256_set1_pd(1.0);

  for (uint step = 0; step < m.max_iter; step++)
  {
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

    // temp values are used as some pixels in the vector will not be iterated
    const __m256d z_imag_tmp = _mm256_add_pd(_mm256_mul_pd(_mm256_add_pd(z_real, z_real), z_imag), c_imag);
    const __m256d z_real_tmp = _mm256_add_pd(_mm256_sub_pd(real_2, imag_2), c_real);

    // intermediate values for z'
    const __m256d ac_bd = _mm256_sub_pd(_mm256_mul_pd(z_real, z_prime_r), _mm256_mul_pd(z_imag, z_prime_i));
    const __m256d bc_da = _mm256_add_pd(_mm256_mul_pd(z_imag, z_prime_r), _mm256_mul_pd(z_real, z_prime_i));

    // calculate the running derivative
    const __m256d z_prime_r_tmp = _mm256_add_pd(_mm256_add_pd(ac_bd, ac_bd), mm_ones);
    const __m256d z_prime_i_tmp = _mm256_add_pd(bc_da, bc_da);

    // calculate the square of the radius for boundary checking
    const __m256d radius_2 = _mm256_add_pd(real_2, imag_2);
    // create a mask which is true if the pixel has *not* escaped
    const __m256d mask = _mm256_cmp_pd(
        radius_2,
        _mm256_set1_pd(1000.0), _CMP_LT_OQ); // escape of 1000.0 used to smooth distance estimate

    // conditionally iterate, only if the pixel has not escaped
    mm256_mask_set(z_real, z_real_tmp, mask);
    mm256_mask_set(z_imag, z_imag_tmp, mask);
    mm256_mask_set(z_prime_r, z_prime_r_tmp, mask);
    mm256_mask_set(z_prime_i, z_prime_i_tmp, mask);

    // calculate the next z.r^2 and z.i^2 values
    real_2 = _mm256_mul_pd(z_real, z_real);
    imag_2 = _mm256_mul_pd(z_imag, z_imag);

    // update accumulators
    step_acc = _mm256_add_pd(_mm256_and_pd(mask, mm_ones), step_acc);
    orbit_acc = _mm256_min_pd(orbit_acc, _mm256_add_pd(real_2, imag_2));

    // finish if all pixels have escaped
    if (_mm256_movemask_pd(mask) == 0x0)
    {
      break;
    }
  }

  // calculate the absolute value (radius) of z for distance estimation
  __m256d r = _mm256_sqrt_pd(_mm256_add_pd(real_2, imag_2));
  __m256d dr = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(z_prime_r, z_prime_r), _mm256_mul_pd(z_prime_i, z_prime_i)));

  // extract values necessary for coloring
  double extracted_step[4];
  double extracted_r[4];
  double extracted_dr[4];
  double extracted_orbit[4];

  _mm256_store_pd(extracted_step, step_acc);
  _mm256_store_pd(extracted_dr, dr);
  _mm256_store_pd(extracted_r, r);
  _mm256_store_pd(extracted_orbit, orbit_acc);

  // do coloring for all four pixels
  for (uint v = 0; v < 4; v++)
  {
    // distance estimation: 0.5 * log(r) * r/dr
    double dist_est = (0.5 * log(extracted_r[v]) * extracted_r[v] / extracted_dr[v]);
    // a 'glow' effect based on distance (manually adjusted to taste and to adjust to zoom level)
    double glow = ((-log(dist_est) + m.glow_spread) / (m.zoom + 15.1) * m.glow_strength);
    // a smoothed version of the iteration count: i + (1 - ln(ln(r))/ln(2))
    double smoothed_step = extracted_step[v] + 1 - log(log(extracted_r[v])) / log(2);

    if ((uint)extracted_step[v] >= m.max_iter)
    {
      // color the inside using orbit trap method
      set_hsl2rgb(m, i + 3 - v, j,
                  0.0,                                                                     // hue
                  0.0,                                                                     // saturation
                  clamp(sqrt(extracted_orbit[v]) * m.brightness * m.internal_brightness)); // lightness
    }
    else
    {
      // color the outside
      set_hsl2rgb(m, i + 3 - v, j,
                  // color hue based on an sinusoidal step counter, offset to a [0,1] range
                  clamp((sin(smoothed_step * 0.05 * m.color_frequency - m.color_offset * 3.14159265)) * 0.5 + 0.5),
                  // saturation decreased when glow is high to hide noise when hue oscillates quickly
                  clamp(m.saturation * (1.0 - (glow * glow))),
                  // use glow around edges for brightness
                  clamp(glow * m.brightness));
    }
  }
  return;
}

/// calculates one thread's portion of the image.
/// also prints progress in the first thread (id=0)
void calc_image_region(struct MandelImage image, uint start_height, uint thread_id)
{
  for (uint j = start_height; j < image.height; j += image.n_threads)
  {
    // progress bar
    if (thread_id == 0 && j % (image.n_threads * 4) == 0)
    {
      double percentage = (double)j / (double)image.height;
      printf("\033[2K\033[0GProgress: |");
      int colors[] = {5, 1, 3, 2, 6, 4};
      // assume 40 characters free for progress bar
      for (double i = 0.; i < 40.; i++)
      {
        if (i / 40.0 < percentage)
          printf("\x1b[9%dm█\x1b[0m", colors[(uint)(i / 40. * 6.)]);
        else
          printf(" ");
      }
      printf("| %.2f%%", percentage * 100.0);
      fflush(stdout);
    }
    // actual calculation
    for (uint i = 0; i < image.width; i += 4)
    {
      calc_pixel_mm(image, i, j);
    }
  }
  // print a 100% message. This is not strictly necessary, but it looks nicer
  if (thread_id != 0)
  {
    return;
  }
  printf("\033[2K\033[0GProgress: |");
  int colors[] = {5, 1, 3, 2, 6, 4};
  // assume 40 characters free for progress bar
  for (double i = 0.; i < 40.; i++)
  {
    printf("\x1b[9%dm█\x1b[0m", colors[(uint)(i / 40. * 6.)]);
  }
  printf("| %.2f%%\n", 100.0);
}

/// extracts the data from the void* to call calc_image_region
void *calc_image_region_wrapper(void *p)
{
  struct PassedData *pd = (struct PassedData *)(p);

  struct MandelImage m = pd->m;
  uint offset = pd->offset;
  uint thread_id = pd->thread_id;

  calc_image_region(m, offset, thread_id);

  return NULL;
}

/// handles the dispatch of all threads
void run_threads(struct MandelImage image)
{
  struct ThreadPool tp;
  tp.data = (struct PassedData *)malloc(sizeof(struct PassedData) * image.n_threads);
  tp.threads = (pthread_t *)malloc(sizeof(pthread_t) * image.n_threads);

  for (uint t = 0; t < image.n_threads; t++)
  {
    tp.data[t].m = image;
    tp.data[t].offset = t;
    tp.data[t].thread_id = t;
    pthread_create(&tp.threads[t], NULL, &calc_image_region_wrapper, (void *)(&tp.data[t]));
  }

  for (uint t = 0; t < image.n_threads; t++)
  {
    pthread_join(tp.threads[t], NULL);
  }

  free(tp.data);
  free(tp.threads);
}

/// the usage info for the program, printed on --help
/// and on failure to parse arguments
const char USAGE[] = "usage: %s [options] <filename>\n"
                     "\nOptions:\n"
                     "\t-w | --width <int>\tThe width of the output in pixels\n\t\tDefault: 1920\n"
                     "\t-h | --height <int>\tThe height of the output in pixels\n\t\tDefault: 1080\n"
                     "\t-z | --zoom <float>\tThe scale of the image; larger is more zoomed in. Scale = 2^(-zoom)\n\t\tDefault: -2\n"
                     "\t-x | --x-offset <float>\tThe real coordinate of the center of the image\n\t\tDefault: -0.5\n"
                     "\t-y | --y-offset <float>\tThe imaginary coordinate of the center of the image\n\t\tDefault: 0.0\n"
                     "\t-m | --max-iter <int>\tThe maximum number of fractal iterations\n\t\tDefault: 1000\n"
                     "\t-s | --saturation <float>\tThe saturation of the image. [0,1]\n\t\tDefault: 1.0\n"
                     "\t-f | --color-frequency <float>\tThe frequency of the cycling of colors\n\t\tDefault: 1.0\n"
                     "\t-o | --color-offset <float>\tThe offset of the cycling period of colors [-1,1]\n\t\tDefault: 1.0\n"
                     "\t-g | --glow-strength <float>\tThe strength of the glow around the edges \n\t\tDefault: 1.0\n"
                     "\t-G | --glow-spread <float>\tHow much the glow bleeds out into surrounding area \n\t\tDefault: 1.0\n"
                     "\t-b | --brightness <float>\tThe brightness multiplier for the image \n\t\tDefault: 2.0\n"
                     "\t-B | --internal-brightness <float>\tThe brightness multiplier for the internal shading \n\t\tDefault: 1.0\n"
                     "\t-t | --threads <int>\tThe number of threads used for computation. \n\t\tDefaults to number of logical cores on the machine.\n"
                     "\n";

/// Creates a new image instance from the process arguments
struct MandelImage parse_options(char **argv, int argc)
{
  if (argc < 2)
  {
    printf(USAGE, argv[0]);
    exit(1);
  }
  // set defaults
  struct MandelImage image;
  image.width = 1920;
  image.height = 1080;
  image.scale = 4.0;
  image.zoom = -2;
  image.x_offset = -0.5;
  image.y_offset = 0.0;
  image.max_iter = 1000; // this is way more than necessary for default
  image.saturation = 1.0;
  image.color_frequency = 1.0;
  image.color_offset = 0.0;
  image.glow_strength = 1.0;
  image.glow_spread = 1.0;
  image.brightness = 2.0;
  image.internal_brightness = 1.0;
  image.n_threads = sysconf(_SC_NPROCESSORS_ONLN);
  image.filename = NULL;

  // loop through arguments and attempt to parse individually
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp("--help", argv[i]))
    {
      printf(USAGE, argv[0]);
      exit(0);
    }
    // last element case
    if (i + 1 >= argc)
    {
      if (argv[i][0] == '-')
      {
        printf("Flag %s requires an argument.\n", argv[i]);
        printf(USAGE, argv[0]);
        exit(1);
      }
      image.filename = argv[i];
      continue;
    }
    // there is at least one more argument
    if (!strcmp("-w", argv[i]) || !strcmp("--width", argv[i]))
    {
      image.width = atoi(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-h", argv[i]) || !strcmp("--height", argv[i]))
    {
      image.height = atoi(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-z", argv[i]) || !strcmp("--zoom", argv[i]))
    {
      image.zoom = atof(argv[i + 1]);
      image.scale = pow(2.0, -image.zoom);
      i++;
      continue;
    }
    if (!strcmp("-x", argv[i]) || !strcmp("--x-offset", argv[i]))
    {
      image.x_offset = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-y", argv[i]) || !strcmp("--y-offset", argv[i]))
    {
      image.y_offset = -atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-m", argv[i]) || !strcmp("--max-iter", argv[i]))
    {
      image.max_iter = atoi(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-s", argv[i]) || !strcmp("--saturation", argv[i]))
    {
      image.saturation = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-f", argv[i]) || !strcmp("--color-frequency", argv[i]))
    {
      image.color_frequency = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-o", argv[i]) || !strcmp("--color-offset", argv[i]))
    {
      image.color_offset = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-g", argv[i]) || !strcmp("--glow-strength", argv[i]))
    {
      image.glow_strength = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-G", argv[i]) || !strcmp("--glow-spread", argv[i]))
    {
      image.glow_spread = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-b", argv[i]) || !strcmp("--brightness", argv[i]))
    {
      image.brightness = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-B", argv[i]) || !strcmp("--internal-brightness", argv[i]))
    {
      image.internal_brightness = atof(argv[i + 1]);
      i++;
      continue;
    }
    if (!strcmp("-t", argv[i]) || !strcmp("--threads", argv[i]))
    {
      image.n_threads = atoi(argv[i + 1]);
      i++;
      continue;
    }
    if (argv[i][0] == '-')
    {
      printf("Argument not recognized: %s\n", argv[i]);
      printf(USAGE, argv[0]);
      exit(1);
    }
    image.filename = argv[i];
    continue;
  }

  if (image.filename == NULL)
  {
    printf("Output filename required.");
    printf(USAGE, argv[0]);
    exit(1);
  }

  // allocate image data
  image.data = (unsigned char *)malloc(sizeof(unsigned char) * image.width * image.height * BYTES_PER_PIXEL);

  // print final parsed options
  printf("Running with options: \n\n");
  printf("\tWidth:\t\t\t%d\n", image.width);
  printf("\tHeight:\t\t\t%d\n", image.height);
  // printf("\tScale:\t\t\t%.2f\n", image.scale);
  printf("\tZoom:\t\t\t%.2f\n", image.zoom);
  printf("\tCenter point:\t\t(%.9f, %.9f)\n", image.x_offset, image.y_offset);
  printf("\tMax Iterations:\t\t%d\n", image.max_iter);
  printf("\tSaturation:\t\t%.2f\n", image.saturation);
  printf("\tColor Frequency:\t%.2f\n", image.color_frequency);
  printf("\tColor Offset:\t\t%.2f\n", image.color_offset);
  printf("\tGlow Strength:\t\t%.2f\n", image.glow_strength);
  printf("\tGlow Spread:\t\t%.2f\n", image.glow_spread);
  printf("\tBrightness:\t\t%.2f\n", image.brightness);
  printf("\tInternal Brightness:\t%.2f\n", image.internal_brightness);
  printf("\tThreads:\t\t%ld\n", image.n_threads);
  printf("\tFilename:\t\t%s\n", image.filename);

  return image;
}

/// a simple min function
int min(int i, int j)
{
  return i > j ? j : i;
}

int main(int argc, char **argv)
{
  clock_t total_start, total_end;
  total_start = clock();

  struct MandelImage image = parse_options(argv, argc);

  printf("Rendering...\n");
  clock_t render_start, render_end;
  render_start = clock();
  run_threads(image);
  render_end = clock();

  printf("Saving...\n");
  clock_t save_start, save_end;
  save_start = clock();
  generateBitmapImage(image.data, image.height, image.width, image.filename);
  save_end = clock();

  printf("Done.\n\n");

  free(image.data);

  // print elapsed time
  int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
  total_end = clock();
  printf("Time:\n    Rendering:\t%.2f s\n    Saving:\t%.2f s\n    Total:\t%.2f s\n",
         (double)(render_end - render_start) / (double)(CLOCKS_PER_SEC) / (double)(min(numCPU, image.n_threads)),
         (double)(save_end - save_start) / (double)(CLOCKS_PER_SEC) / (double)(min(numCPU, image.n_threads)),
         (double)(total_end - total_start) / (double)(CLOCKS_PER_SEC) / (double)(min(numCPU, image.n_threads)));
}
