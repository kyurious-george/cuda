**Q1**: Matrix addition does not share memory because each thread's index corresponds to the other matrix inputs. 

**Q2**: For 2x2, instead of each thread loading n from row and n from col, each tile loads n/2 from row and n/2 from col. For 4x4, each tile loads n/4 from row and n/4 from col. 

**Q3**: First, `__syncthreads()` this would result in possibly reading from shared memory address that hasn't been read, which would lead to undefined behavior. 

**Q4**: Shared memory can be acccessed by other threads. 

**Q5**: M*N/32

**Q6**: 512,000 (thread grain)

**Q7**: 1000 (block grain)

**Q8**: 
- a. N 
- b. N/T

**Q9**: 
- a. Memory bound
- b. Compute bound 

**Q10**:  
- a. `BLOCK_SIZE = 1`
- b. The primary issue is block A is shared memory and overwrites or undefined behavior can occur since threads may read and load without coordination. We need `__syncthreads()` @ L10

**Q11**: 
- a. 1024
- b. 1024
- c. 8
- d. 8 
- e. (128 + 1) / 4 = 516
- f. 10 / 4 = 2.5 OPS/B

**Q12**: 
- a. No, limited by the amount of shared memory / SM
- b. Yes, can acheive full occupancy