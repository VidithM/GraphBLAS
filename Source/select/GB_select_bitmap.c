//------------------------------------------------------------------------------
// GB_select_bitmap:  select entries from a bitmap or full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "select/GB_select.h"
#include "jitifyer/GB_stringify.h"
#ifndef GBCOMPACT
#include "FactoryKernels/GB_sel__include.h"
#endif

#define GB_FREE_ALL         \
    GB_phybix_free (C) ;

static int tot_hits = 0 ;
static int cuda_hits = 0 ;

GrB_Info GB_select_bitmap
(
    GrB_Matrix C,               // output matrix, static header
    const bool C_iso,           // if true, C is iso
    const GrB_IndexUnaryOp op,
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const int64_t ithunk,       // (int64_t) Thunk, if Thunk is NULL
    const GB_void *restrict athunk,     // (A->type) Thunk
    const GB_void *restrict ythunk,     // (op->ytype) Thunk
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A for bitmap selector", GB0) ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for bitmap selector", GB0) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    GB_Opcode opcode = op->opcode ;
    ASSERT (opcode != GB_NONZOMBIE_idxunop_code) ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz_held (A) ;
    const size_t asize = A->type->size ;
    const GB_Type_code acode = A->type->code ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    // C->b and C->x are malloc'd, not calloc'd
    GB_OK (GB_new_bix (&C, // always bitmap, existing header
        A->type, A->vlen, A->vdim, GB_ph_calloc, true,
        GxB_BITMAP, false, A->hyper_switch, -1, anz, true, C_iso,
        /* OK: */ false, false)) ;

    ASSERT (GxB_BITMAP == GB_sparsity (C)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // copy values of A into C
    //--------------------------------------------------------------------------

    // C and A have the same type, since C is the T matrix from GB_select,
    // not the final user's C matrix.  In the future A could be typecasted
    // into C in the JIT kernel.

    if (C_iso)
    { 
        // Cx [0] = Ax [0] or (A->type) thunk
        GB_select_iso (C->x, opcode, athunk, A->x, asize) ;
    }
    else
    { 
        // Cx [0:anz-1] = Ax [0:anz-1]
        // Fixme for CUDA: do this on the GPU if appropriate
        GB_memcpy (C->x, A->x, anz * asize, nthreads) ;
    }

    //--------------------------------------------------------------------------
    // bitmap selector kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    OPEN_STATS ("select_bitmap", &tot_hits, &cuda_hits) ;
    
    #define TRIAL_FREE                                               \
    {                                                                \
        GB_phybix_free (C) ;                                         \
        GB_OK (GB_new_bix (&C,                                       \
            A->type, A->vlen, A->vdim, GB_ph_calloc, true,           \
            GxB_BITMAP, false, A->hyper_switch, -1, anz, true, C_iso,\
            false, false)) ;                                         \
         if (C_iso)                                                  \
         {                                                           \
            GB_select_iso (C->x, opcode, athunk, A->x, asize) ;      \
         }                                                           \
         else                                                        \
         {                                                           \
            GB_memcpy (C->x, A->x, anz * asize, nthreads) ;          \
         }                                                           \
        info = GrB_NO_VALUE ;                                        \
    }
    #define STATS_RESET TRIAL_FREE

    #if defined ( GRAPHBLAS_HAS_CUDA )
    if (GB_cuda_select_branch (A, op))
    {
        BEGIN_STATS ("gpu", anz) ;
        BEGIN_TRIALS (5) ;
        START_TIME ;

        info = GB_cuda_select_bitmap (C, A, flipij, ythunk, op) ;

        STOP_TIME ;
        END_TRIALS ;
        END_STATS ;
    }
    #endif

    if (info == GrB_NO_VALUE)
    {
        BEGIN_STATS ("cpu", anz) ;
        BEGIN_TRIALS (5) ;

        if (GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode))
        {
            START_TIME_NAMED ("positional") ;

            //------------------------------------------------------------------
            // bitmap selector for positional ops
            //------------------------------------------------------------------
            info = GB_select_positional_bitmap (C, A, ithunk, op, nthreads) ;

            STOP_TIME ;
        }
        else
        { 

            //------------------------------------------------------------------
            // bitmap selector for VALUE* and user-defined ops
            //------------------------------------------------------------------

            #ifndef GBCOMPACT
            GB_IF_FACTORY_KERNELS_ENABLED
            {
                START_TIME_NAMED ("factory") ;

                //--------------------------------------------------------------
                // via the factory kernel 
                //--------------------------------------------------------------

                #define GB_selbit(opname,aname) \
                    GB (_sel_bitmap_ ## opname ## aname)
                #define GB_SEL_WORKER(opname,aname)                            \
                {                                                              \
                    info = GB_selbit (opname, aname) (C, A, ythunk, nthreads) ;\
                }                                                              \
                break ;

                #include "select/factory/GB_select_entry_factory.c"

                STOP_TIME ;
            }
            #endif

            //------------------------------------------------------------------
            // via the JIT or PreJIT kernel
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            {
                START_TIME_NAMED ("jit") ;

                info = GB_select_bitmap_jit (C, A, flipij, ythunk, op,
                    nthreads) ;

                STOP_TIME ;
            }

            //------------------------------------------------------------------
            // via the generic kernel 
            //------------------------------------------------------------------

            if (info == GrB_NO_VALUE)
            { 
                START_TIME_NAMED ("generic") ;

                GBURBLE ("(generic select) ") ;
                info = GB_select_generic_bitmap (C, A, flipij, ythunk, op,
                    nthreads) ;

                STOP_TIME ;
            }
        }
        
        END_TRIALS ;
        END_STATS ;
    }

    CLOSE_STATS ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_OK (info) ;  // check for out-of-memory, or other error
    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C from bitmap selector", GB0) ;
    return (GrB_SUCCESS) ;
}

