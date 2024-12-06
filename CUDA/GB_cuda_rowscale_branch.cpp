#include "GB_cuda.hpp"

bool GB_cuda_rowscale_branch
(
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    if (D->static_header)
    {
        return false ;
    }
    if (B->static_header)
    {
        return false ;
    }

    if (!GB_cuda_type_branch (D->type) || 
        !GB_cuda_type_branch (B->type) ||
        !GB_cuda_type_branch (semiring->multiply->ztype))
    {
        return false;
    }

    double work = GB_nnz_held (B) ;
    int ngpus_to_use = GB_ngpus_to_use (work) ;
    GBURBLE (" work:%g gpus:%d ", work, ngpus_to_use) ;
    if (ngpus_to_use > 0)
    {
        // FIXME: gpu_id = GB_Context_gpu_id_get ( ) ;
        // cudaSetDevice (gpu_id) ;
        return (true) ;
    }
    else
    {
        return (false) ;
    }
}
