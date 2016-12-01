#include <stdio.h>
#include <algorithm>
#include <getopt.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "CycleTimer.h"

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

/*extern void mandelbrotThread(
    float* x0, float* y0, float* x1, float* y1,
    int* width, int* height,
    int* maxIterations,
    int output[]);*/

// Ugly hack to deal with linker issues


__device__ void mandel(float c_re, float c_im, int count, int *counter)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    *counter = i;
}



//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
__global__ void mandelbrotThread(
    int output[])
{
    const unsigned int width = 1200;
    const unsigned int height = 800;
    const int maxIterations = 256;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int perBlockXQuota = (int)ceilf(width/(float)gridDim.x);
    int perBlockYQuota = (int)ceilf(height/(float)gridDim.y);

    int perThreadXQuota = (int)ceilf(perBlockXQuota/(float)blockDim.x);
    int perThreadYQuota = (int)ceilf(perBlockYQuota/(float)blockDim.y);

    int startBlockX = blockIdx.x * perBlockXQuota;
    int endBlockX = (blockIdx.x + 1) * perBlockXQuota;
    endBlockX = endBlockX > width ? width : endBlockX;

    int startBlockY = blockIdx.y * perBlockYQuota;
    int endBlockY = (blockIdx.y + 1) * perBlockYQuota;
    endBlockY = endBlockY > height ? height : endBlockY;

    int startX = startBlockX + (threadIdx.x * perThreadXQuota);
    int endX = startBlockX + ((threadIdx.x + 1) * perThreadXQuota);
    endX = endX > endBlockX ? endBlockX : endX;

    int startY = startBlockY + (threadIdx.y * perThreadYQuota);
    int endY = startBlockY + ((threadIdx.y + 1) * perThreadYQuota);
    endY = endY > endBlockY ? endBlockY : endY;
    


    for (int j = startY; j < endY; j++) {
        for (int i = startX; i < endX; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * (width) + i);

            mandel(x, y, maxIterations, &output[index]);
        }

    }
    
}


extern void writePPMImage(
    int* data,
    int width, int height,
    const char *filename,
    int maxIterations);

void
scaleAndShift(float& x0, float& x1, float& y0, float& y1,
              float scale,
              float shiftX, float shiftY)
{

    x0 *= scale;
    x1 *= scale;
    y0 *= scale;
    y1 *= scale;
    x0 += shiftX;
    x1 += shiftX;
    y0 += shiftY;
    y1 += shiftY;

}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -t  --threads <N>  Use N threads\n");
    printf("  -v  --view <INT>   Use specified view settings\n");
    printf("  -?  --help         This message\n");
}

bool verifyResult (int *gold, int *result, int width, int height) {

    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (gold[i * width + j] != result[i * width + j]) {
                printf ("Mismatch : [%d][%d], Expected : %d, Actual : %d\n",
                            i, j, gold[i * width + j], result[i * width + j]);
            }
        }
    }

    return 1;
}

int main(int argc, char** argv) {

    const unsigned int width = 1200;
    const unsigned int height = 800;
    const int maxIterations = 256;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"threads", 1, 0, 't'},
        {"view", 1, 0, 'v'},
        {"help", 0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "v:?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'v':
        {
            int viewIndex = atoi(optarg);
            // change view settings
            if (viewIndex == 2) {
                float scaleValue = .015f;
                float shiftX = -.986f;
                float shiftY = .30f;
                scaleAndShift(x0, x1, y0, y1, scaleValue, shiftX, shiftY);
            } else if (viewIndex > 1) {
                fprintf(stderr, "Invalid view index\n");
                return 1;
            }
            break;
        }
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options


    int* output_serial = new int[width*height];
    int* output_thread = new int[width*height];

    //
    // Run the serial implementation.  Run the code three times and
    // take the minimum to get a good estimate.
    //
    memset(output_serial, 0, width * height * sizeof(int));
    printf("Height: %d, Width: %d\n", height, width);
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        mandelbrotSerial(x0, y0, x1, y1, width, height, 0, height, maxIterations, output_serial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[mandelbrot serial]:\t\t[%.3f] ms\n", minSerial * 1000);
    writePPMImage(output_serial, width, height, "mandelbrot-serial.ppm", maxIterations);

    //
    // Run the threaded version
    //
    memset(output_thread, 0, width * height * sizeof(int));
    double minThread = 1e30;
    int *x;
    cudaMalloc((void**)&x, sizeof(int));

    
    double startTime = CycleTimer::currentSeconds();
    int *d_output_thread;
    /*    , *d_width, *d_height, *d_maxIterations;
    float *d_x0, *d_y0, *d_x1, *d_y1;*/

    cudaMalloc((void**)&d_output_thread, width * height * sizeof(int));


    mandelbrotThread<<<50000, 1>>>(d_output_thread);
    cudaMemcpy(output_thread, d_output_thread, width * height * sizeof(int), cudaMemcpyDeviceToHost);
double endTime = CycleTimer::currentSeconds();
    

    
    minThread = endTime - startTime;

    printf("[mandelbrot thread]:\t\t[%.3f] ms\n", minThread * 1000);
    writePPMImage(output_thread, width, height, "mandelbrot-thread2.ppm", maxIterations);

    if (! verifyResult (output_serial, output_thread, width, height)) {
        printf ("Error : Output from threads does not match serial output\n");

        /*delete[] output_serial;
        delete[] output_thread;

        return 1;*/
    }

    // compute speedup
    printf("\t\t\t\t(%.2fx speedup)\n", minSerial/minThread);

    return 0;
}
