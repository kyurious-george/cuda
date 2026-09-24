**Q1**: 
- a. 128/32 = 4
- b. (9)(128)(4) = 4608
- c. 
  - i. 3
  - ii. 2 
  - iii. 100
  - iv. 8/32 = 25% 
  - v. 24/32 = 75% 
- d. 
  - i. 4608
  - ii. 4608
  - iii. 50%
- e. 
  - i. 3
  - ii. 2

**Q2**: ((2000 + 512 - 1) / 512) 512 = 2048

**Q3**: 1

**Q4**: 17.083%

**Q5**: Bad idea. This can cause race conditions or undefined behavior if these threads were supposed to consolidate. 

**Q6**: a = 512, b = 1024, c = 1536 (with only 3 thread blocks), d = 1024 (with only 1 thread block)

**Q7**: 
- a. Not possible
- b. 50% 
- c. 50% 
- d. 100% 
- e. 100%

**Q8**: 
- a. yes. 16 blocks of 128 threads => 2048 threads/SM => 61,440 registers/SM used total (safe)
- b. no. 32 blocks of 32 threads => 1024 threads (limiting factor is number of threads/block is too low)
- c. no. 7 blocks of 256 threads (-1 block due to limiting factor of 34 registers/thread) => 2028 threads * 34 

**Q9**: Couple of issues with this. 1. we can't actually fit a 32x32 thread block on the CUDA device as it only has 512 threads per block while the proposed design doesn't allow for that. Even if they did use different thread block sizing, they would still have to have done 256 iterative operations in order to do matmul where each thread represents one output element.