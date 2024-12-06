#include "GB_cuda.hpp"

bool GB_cuda_apply_unop_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op
)
{
    if (op == NULL)
    {
        return false ;
    }

    if (A->static_header)
    {
        return false ;
    }
    
    bool ok = (GB_cuda_type_branch (ctype) && GB_cuda_type_branch (A->type)) ;

    if (op->xtype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->xtype)) ;
    }
    if (op->ytype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->ytype)) ;
    }
    if (op->ztype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->ztype)) ;
    }
    
    ok = ok && (op->hash != UINT64_MAX) ;

    double work = GB_nnz_held (A) ;
    int ngpus_to_use = GB_ngpus_to_use (work) ;
    GBURBLE (" work:%g gpus:%d ", work, ngpus_to_use) ;
    if (ngpus_to_use > 0)
    {
        // FIXME: gpu_id = GB_Context_gpu_id_get ( ) ;
        // cudaSetDevice (gpu_id) ;
        return (ok) ;
    }
    else
    {
        return (false) ;
    }
}