//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define BITS 4
#define BINS_COUNT (1<<BITS)
#define MASK (BINS_COUNT-1)
#define test(val,bin,shift) ((val>>shift) & (~bin))

__global__ void r_sort(unsigned int* const d_inputVals,
                       unsigned int* const d_inputPos,
                       unsigned int* const d_outputVals,
                       unsigned int* const d_outputPos,
                       const size_t numElems)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int threadsCount = blockDim.x;
    extern __shared__ float s_hist[];

    unsigned int* const sourceVals = d_inputVals;
    unsigned int* const sourcePos = d_inputPos;
    unsigned int* const destVals = d_outputVals;
    unsigned int* const destPos = d_outputPos;

    int prev = 0; // scan results on previous iteration
    for (int block = 0; block < numElems; block += threadsCount)
    {
        if (!tid)
            sourceVals[block] += prev;
        for (int neighborDelta = 1; neighborDelta < threadsCount; neighborDelta = neighborDelta<<1)
        {
            int neighborIdx = tid - neighborDelta;
            if (neighborIdx >= 0)
                result[tid+block] += source[neighborIdx+block];
            else
                result[tid+block] = source[tid+block];
            __syncronize();
        }
        if (!tid)
            prev = dest[last];
        __syncronize();
    }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Get device info, use max avail threads
    r_sort<<<1, prop.maxThreadsPerBlocki, sizeof(unsigned int) * 16>>>
        (d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
}
