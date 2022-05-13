#include <random>

#define SIZE 1024
#define BITS 4
#define BINS_COUNT (1<<BITS)
#define MASK (BINS_COUNT-1)

int main(int argc, char **argv) {
    uint32_t arr [SIZE];
    uint32_t bin [BINS_COUNT][SIZE];
    uint32_t bin_count [BINS_COUNT] = {0};

    // Init 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 1000000);
    for (int i = 0; i<SIZE; i++)
        arr[i] = distr(gen); // generate numbers

    // Sort
    for (int shift = 0; shift < sizeof(uint32_t)*8; shift += BITS)
    {
        for (int i = 0; i<SIZE; i++)
        {
            int bin_idx = (arr[i] >> shift) & MASK;
            bin[bin_idx][bin_count[bin_idx]] = arr[i];
            bin_count[bin_idx]++;
        }
        int pos = 0;
        for (int i = 0; i<BINS_COUNT; i++)
        {
            for (int j = 0; j<bin_count[i]; j++)
                arr[pos++] = bin[i][j];
            bin_count[i] = 0;
        }
    }

    // Test
    for (int i = 1; i<SIZE; i++)
    {
        if (arr[i-1]>arr[i])
            return 1;
    }
}

