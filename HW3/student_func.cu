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

__global__ void find_max_min_grid(const float* const d_logLuminance,
                                  float *d_min,
                                  float *d_max,
                                  const size_t numRows,
                                  const size_t numCols)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int _gridIdx = blockIdx.y * gridDim.x + blockIdx.x;
    
    // load into shared mem
    sdata[tid] = d_logLuminance[thread_1D_pos];
    __syncthreads();

    // loop over the data
    for (unsigned int s = blockDim.x * blockDim.y / 2; s; s >>= 1)
    {
        if (tid<s)
        {
            float left = sdata[tid];
            float right = sdata[tid+s];
            sdata[tid] = min(left, right);
            sdata[blockDim.x * blockDim.y - 1 - tid] = max(left, right);
        }
        __syncthreads();
    }

    // write the results into corresponding grid cell
    if (tid==0)
    {
        d_min[_gridIdx] = sdata[0];
        d_max[_gridIdx] = sdata[blockDim.x * blockDim.y - 1];
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    // 0) init variables
    const dim3 blockSize = dim3(BLOCK_WIDTH, BLOCK_WIDTH);
    const dim3 gridSize = dim3(1+numCols/BLOCK_WIDTH, 1+numRows/BLOCK_WIDTH);

    float *d_min;
    float *d_max;
    checkCudaErrors(cudaMalloc(&d_min, sizeof(float) * gridSize.x * gridSize.y));
    checkCudaErrors(cudaMalloc(&d_max, sizeof(float) * gridSize.x * gridSize.y));

    // 1) find the minimum and maximum value in the input logLuminance channel
    //    scan with grid of 32x32 blocks
    find_max_min_grid<<<gridSize, blockSize, sizeof(float) * BLOCK_WIDTH * BLOCK_WIDTH>>>
        (d_logLuminance, d_min, d_max, numRows, numCols);
    //    reduce the grid
    
}
