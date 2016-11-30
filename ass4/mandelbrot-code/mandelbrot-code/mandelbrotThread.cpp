#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;



static inline int mandel(float c_re, float c_im, int count)
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

    return i;
}



//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
__global__ void mandelbrotThread(
    float* x0, float* y0, float* x1, float* y1,
    int* width, int* height,
    int* maxIterations, int output[])
{
    float dx = (*x1 - *x0) / *width;
    float dy = (*y1 - *y0) / *height;

    int perBlockXQuota = ceil(*width/gridDim.x);
    int perBlockYQuota = ceil(*height/gridDim.y);

    int startX = blockId.x * perBlockXQuota;
    int endX = (blockId.x + 1) * perBlockXQuota;
    endX = endX > *width ? *width : endX;

    int startY = blockId.y * perBlockYQuota;
    int endY = (blockId.y + 1) * perBlockYQuota;
    endY = endY > *height ? *height : endY;

    for (int j = startY; j < endY; j++) {
        for (int i = startX; i < endX; ++i) {
            float x = *x0 + i * dx;
            float y = *y0 + j * dy;

            int index = (j * (*width) + i);
            output[index] = mandel(x, y, *maxIterations);
        }

    }
    
}
