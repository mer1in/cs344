/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

// for 1024 threads per block
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

__global__ void find_min_max(
        const float* const d_logLuminance,
        float* d_limits,
        const unsigned int count)
{
    float __min = d_logLuminance[count-1];
    float __max = __min;
    const unsigned int tid = threadIdx.x;
    extern __shared__ float sdata[];

    for (unsigned int base = 0; base < count; base += blockDim.x)
    {
        if (tid+base >= count)
            return;
        sdata[tid] = d_logLuminance[base + tid];
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s; s /= 2)
        {
            if (tid<s)
            {
                float _max = max(
                    sdata[blockDim.x - 1 - tid],
                    sdata[blockDim.x - 1 - tid - s]
                );
                sdata[tid] = min(sdata[tid], sdata[tid+s]);
                sdata[blockDim.x - 1 - tid] = _max;
            }
            // take uneven case into account
            if (s&1 && s!=1)
                s++;
            __syncthreads();
        }
        __syncthreads();
        if (tid==0)
        {
            if (__min > sdata[0])
                __min = sdata[0];
            if (__max < sdata[blockDim.x - 1])
                __max = sdata[blockDim.x - 1];
        }
    }
    if (tid==0)
    {
        d_limits[0] = __min;
        d_limits[1] = __max;
    }
}

// Naive imlementation with atomicAdd.
// TODO: compare to per-thread histogram calculation and reduce
__global__ void calc_hist(const float* const d_logLuminance,
                          const float* d_limits,
                          unsigned int *d_histogram,
                          const size_t count,
                          const size_t numBins)
{
    extern __shared__ unsigned int s_histogram[];
    float lumRange = d_limits[1]-d_limits[0];
    float lumMin = d_limits[0];
    unsigned int tid = threadIdx.x;

    // it works because numBins == blockDim.x
    s_histogram[tid] = 0;
    __syncthreads();

    for (unsigned int base = 0; base < count; base += blockDim.x)
    {
        unsigned int bin = (d_logLuminance[base + tid] - lumMin) / lumRange * numBins;
        if (bin == numBins)
            bin--;
        atomicAdd(&(s_histogram[bin]), 1);
    }
    __syncthreads();
    d_histogram[tid] = s_histogram[tid];
}

__global__ void scan(const unsigned int* d_histogram,
                     unsigned int* d_cdf)
{
    extern __shared__ unsigned int s_cdf[];
    unsigned int tid = threadIdx.x;
    s_cdf[tid] = d_histogram[tid];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1)
    {
        int src_idx = tid - s;
        if (src_idx >= 0)
            s_cdf[tid] += s_cdf[src_idx];
        __syncthreads();
    }
    if (tid == 0)
        d_cdf[0] = 0;
    else
        d_cdf[tid] = s_cdf[tid-1];
    __syncthreads();
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    const dim3 blockSize = dim3(BLOCK_WIDTH, BLOCK_HEIGHT);
    const dim3 gridSize = dim3(1+numCols/BLOCK_WIDTH, 1+numRows/BLOCK_HEIGHT);

    float *d_limits;
    unsigned int *d_histogram;
    checkCudaErrors(cudaMalloc(&d_limits, sizeof(float) * 2));
    checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * numBins));

    find_min_max<<<1, BLOCK_WIDTH * BLOCK_HEIGHT, sizeof(float) * BLOCK_WIDTH * BLOCK_HEIGHT>>>
        (d_logLuminance, d_limits, numRows * numCols);

    calc_hist<<<1, numBins, sizeof(unsigned int) * numBins>>>
        (d_logLuminance, d_limits, d_histogram, numRows*numCols, numBins);

    scan<<<1, numBins, sizeof(unsigned int) * numBins>>>
        (d_histogram, d_cdf);

    checkCudaErrors(cudaFree(d_limits));
    checkCudaErrors(cudaFree(d_histogram));
}
