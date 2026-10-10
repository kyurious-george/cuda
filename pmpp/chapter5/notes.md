### Chapter 5: GPU Memory 

**Arithmetic/Computational Intensity**: $(FLOPS) / (B/s) = FLOP / B$
- Enables you to calculate the maximum computational intensity for the hardware device

**Speed-of-light Analysis**: `memory_accessed/E2E_latency = memory_bandwidth`

*Types of GPU Memory*:
- **Global Memory**: off-chip GPU memory (variant of HBM) that allows for R/W access from all threads
	- **Constant Memory**: variant of global memory that gets cached onto the SM's on-chip constant memory cache (`__device__ __constant__`)
	- **Local Memory**: variant of global memory that can only be accessed by threads, so not shared
- **Registers**: on-chip memory for scratchpad computations and variables that are allocated to individual threads and allow for fast R access latency
- **Shared Memory**: on-chip memory that is allocated by thread blocks and allows for efficient sharing of memory to avoid too much memory latency (`__device__ __shared__`)

off-chip memory lookup = ~500 clock cycles vs. on-chip memory lookup = ~1 clock cycle

#### Tiled Matrix-Matrix Multiplication

