using namespace cooperative_groups ;

__global__ void GB_cuda_rowscale_kernel
(
    GrB_Matrix C,
    GrB_Matrix D,
    GrB_Matrix B
)
{
    const GB_A_TYPE *__restrict__ Dx = (GB_A_TYPE *) D->x ;
    const GB_B_TYPE *__restrict__ Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) C->x ;

    #ifdef GB_JIT_KERNEL
    #define D_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool D_iso = D->iso ;
    const bool B_iso = B->iso ;
    #endif

    const int64_t *__restrict__ Bi = B->i ;
    GB_B_NVALS (bnz) ;
    const int64_t bvlen = B->vlen ;

    int ntasks = gridDim.x * blockDim.x;
    // ntasks = GB_IMIN (bnz, ntasks) ;     take care of this when setting gridsz/blocksz
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t pstart, pend ;
    GB_PARTITION (pstart, pend, bnz, tid, ntasks) ;
    for (int64_t p = pstart ; p < pend ; p++)
    {
        int64_t i = GBI_B (Bi, p, bvlen) ;      // get row index of B(i,j)
        GB_DECLAREA (dii) ;
        GB_GETA (dii, Dx, i, D_iso) ;           // dii = D(i,i)
        GB_DECLAREB (bij) ;
        GB_GETB (bij, Bx, p, B_iso) ;           // bij = B(i,j)
        GB_EWISEOP (Cx, p, dii, bij, 0, 0) ;    // C(i,j) = dii*bij
    }
}

extern "C" {
    GB_JIT_CUDA_KERNEL_ROWSCALE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_ROWSCALE_PROTO (GB_jit_kernel)
{
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!C->iso) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    
    GB_cuda_rowscale_kernel <<<grid, block, 0, stream>>> (C, D, B) ;
    cudaStreamSynchronize (stream) ;

    return (GrB_SUCCESS) ;
}