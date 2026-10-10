**Q1**: 
- a. coalesced
- b. N/A - shared memory does not need coalescing
- c. coalesced
- d. not coalesced
- e. N/A - shared memory does not need coalescing 
- f. N/A - shared memory does not need coalescing 
- g. coalesced
- h. N/A - shared memory does not need coalescing 
- i. not coalesced

**Q2**: See folder for code - Corner Turning Mat Mul

**Q3**: In order to completely avoid uncoalesced accesses to global memory, the `BLOCK_SIZE` should be multiples of 32 (aka 32 or 64 because more than that would likely go over bounds of shared memory)

**Q4**: See folder for code - Vector Addition with Vector Loads

**Q5**:
- a. 1 slot - 32 bank conflicts (all land on 0)
- b. 32 slots - 0 bank conflicts 
- c. 4 slots - 8 bank conflicts
- d. 2 slots - 16 bank conflicts
- e. 8 slots - 4 bank conflicts 
- f. 4 slots - 8 bank conflicts
- g. 32 slots - 0 bank conflicts 

The solution to this problem lies in finding the greatest common denominator (`gcd`) between 32 and the stride value `s`. `32 / gcd = # of banks` and each of these banks end up with `gcd` conflicts.