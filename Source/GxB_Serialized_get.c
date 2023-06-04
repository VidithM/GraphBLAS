//------------------------------------------------------------------------------
// GxB_Serialized_get_*: query the contents of a serialized blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"
#include "GB_serialize.h"

//------------------------------------------------------------------------------
// GB_blob_header_get: get all properties of the blob
//------------------------------------------------------------------------------

static GrB_Info GB_blob_header_get
(
    // output:
    char *type_name,            // name of the type (char array of size at
                                // least GxB_MAX_NAME_LEN)
    int *type_code,             // type code of the matrix
    int *sparsity_status,       // sparsity status
    int *sparsity_ctrl,         // sparsity control
    double *hyper_sw,           // hyper_switch
    double *bitmap_sw,          // bitmap_switch
    int *storage,               // GrB_COLMAJOR or GrB_ROWMAJOR

    // input, not modified:
    const void *blob,       // the blob
    GrB_Index blob_size     // size of the blob
)
{

    //--------------------------------------------------------------------------
    // read the content of the header (160 bytes)
    //--------------------------------------------------------------------------

    size_t s = 0 ;

    if (blob_size < GB_BLOB_HEADER_SIZE)
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT)  ;
    }

    GB_BLOB_READ (blob_size2, uint64_t) ;
    GB_BLOB_READ (typecode, int32_t) ;
    uint64_t blob_size1 = (uint64_t) blob_size ;

    if (blob_size1 != blob_size2
        || typecode < GB_BOOL_code || typecode > GB_UDT_code
        || (typecode == GB_UDT_code &&
            blob_size < GB_BLOB_HEADER_SIZE + GxB_MAX_NAME_LEN))
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT)  ;
    }

    GB_BLOB_READ (version, int32_t) ;
    GB_BLOB_READ (vlen, int64_t) ;
    GB_BLOB_READ (vdim, int64_t) ;
    GB_BLOB_READ (nvec, int64_t) ;
    GB_BLOB_READ (nvec_nonempty, int64_t) ;     ASSERT (nvec_nonempty >= 0) ;
    GB_BLOB_READ (nvals, int64_t) ;
    GB_BLOB_READ (typesize, int64_t) ;
    GB_BLOB_READ (Cp_len, int64_t) ;
    GB_BLOB_READ (Ch_len, int64_t) ;
    GB_BLOB_READ (Cb_len, int64_t) ;
    GB_BLOB_READ (Ci_len, int64_t) ;
    GB_BLOB_READ (Cx_len, int64_t) ;
    GB_BLOB_READ (hyper_switch, float) ;
    GB_BLOB_READ (bitmap_switch, float) ;
    GB_BLOB_READ (sparsity_control, int32_t) ;
    GB_BLOB_READ (sparsity_iso_csc, int32_t) ;
    GB_BLOB_READ (Cp_nblocks, int32_t) ; GB_BLOB_READ (Cp_method, int32_t) ;
    GB_BLOB_READ (Ch_nblocks, int32_t) ; GB_BLOB_READ (Ch_method, int32_t) ;
    GB_BLOB_READ (Cb_nblocks, int32_t) ; GB_BLOB_READ (Cb_method, int32_t) ;
    GB_BLOB_READ (Ci_nblocks, int32_t) ; GB_BLOB_READ (Ci_method, int32_t) ;
    GB_BLOB_READ (Cx_nblocks, int32_t) ; GB_BLOB_READ (Cx_method, int32_t) ;

    (*sparsity_status) = sparsity_iso_csc / 4 ;
    bool iso = ((sparsity_iso_csc & 2) == 2) ;
    bool is_csc = ((sparsity_iso_csc & 1) == 1) ;
    (*sparsity_ctrl) = sparsity_control ;
    (*hyper_sw)  = (double) hyper_switch ;
    (*bitmap_sw) = (double) bitmap_switch ;
    (*storage) = (is_csc) ? GrB_COLMAJOR : GrB_ROWMAJOR ;

    //--------------------------------------------------------------------------
    // determine the matrix type_code and C type_name
    //--------------------------------------------------------------------------

    (*type_code) = (int) GB_type_code_get (typecode) ;
    memset (type_name, 0, GxB_MAX_NAME_LEN) ;

    if (typecode >= GB_BOOL_code && typecode < GB_UDT_code)
    { 
        // blob has a built-in type; the name is not in the blob
        strcpy (type_name, GB_code_string (typecode)) ;
    }
    else if (typecode == GB_UDT_code)
    { 
        // blob has a user-defined type
        // get the GxB_JIT_C_NAME of the user type from the blob
        memcpy (type_name, ((GB_void *) blob) + GB_BLOB_HEADER_SIZE,
            GxB_MAX_NAME_LEN) ;
    }

    // this should already be in the blob, but set it to null just in case
    type_name [GxB_MAX_NAME_LEN-1] = '\0' ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_Scalar
(
    const void * blob,
    GrB_Scalar value,
    GrB_Field field,
    size_t blob_size
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Serialized_get_Scalar (blob, value, field, blobsize)") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    char type_name [GxB_MAX_NAME_LEN] ;
    int sparsity_status, sparsity_ctrl, type_code, storage ;
    double hyper_sw, bitmap_sw ;

    GrB_Info info = GB_blob_header_get (type_name, &type_code, &sparsity_status,
        &sparsity_ctrl, &hyper_sw, &bitmap_sw, &storage, blob, blob_size) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    double dvalue = 0 ;
    int ivalue = 0 ;
    bool is_double = false ;

    if (info == GrB_SUCCESS)
    {
        switch (field)
        {
            case GrB_STORAGE_ORIENTATION_HINT : 

                ivalue = storage ;
                break ;

            case GrB_ELTYPE_CODE : 

                ivalue = type_code ;
                break ;

            case GxB_SPARSITY_CONTROL : 

                ivalue = sparsity_ctrl ;
                break ;

            case GxB_SPARSITY_STATUS : 

                ivalue = sparsity_status ;
                break ;

            case GxB_FORMAT : 

                ivalue = (storage == GrB_COLMAJOR) ? GxB_BY_COL : GxB_BY_ROW ;
                break ;

            case GxB_HYPER_SWITCH : 
                dvalue = hyper_sw ;
                is_double = true ;
                break ;

            case GxB_BITMAP_SWITCH : 
                dvalue = bitmap_sw ;
                is_double = true ;
                break ;

            default : 
                return (GrB_INVALID_VALUE) ;
        }

        if (is_double)
        { 
            // field specifies a double: assign it to the scalar
            info = GB_setElement ((GrB_Matrix) value, NULL, &dvalue, 0, 0,
                GB_FP64_code, Werk) ;
        }
        else
        { 
            // field specifies an int: assign it to the scalar
            info = GB_setElement ((GrB_Matrix) value, NULL, &ivalue, 0, 0,
                GB_INT32_code, Werk) ;
        }
    }

    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_String
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_String
(
    const void * blob,
    char * value,
    GrB_Field field,
    size_t blob_size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Serialized_get_String (blob, value, field, blobsize)") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    char type_name [GxB_MAX_NAME_LEN] ;
    int sparsity_status, sparsity_ctrl, type_code, storage ;
    double hyper_sw, bitmap_sw ;

    GrB_Info info = GB_blob_header_get (type_name, &type_code, &sparsity_status,
        &sparsity_ctrl, &hyper_sw, &bitmap_sw, &storage, blob, blob_size) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    (*value) = '\0' ;
    const char *name ;

    if (info == GrB_SUCCESS)
    {
        switch (field)
        {

            case GrB_NAME : 
                // FIXME: give the blob a name
                break ;

            case GxB_JIT_C_NAME : 
                strcpy (value, type_name) ;
                break ;

            case GrB_ELTYPE_STRING : 
                // FIXME: return the user_name of user-defined type
                name = GB_code_name_get (type_code, NULL) ;
                if (name != NULL)
                {
                    strcpy (value, name) ;
                }
                break ;

            default : 
                return (GrB_INVALID_VALUE) ;
        }
    }

    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_ENUM
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_ENUM
(
    const void * blob,
    int * value,
    GrB_Field field,
    size_t blob_size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Serialized_get_ENUM (blob, value, field, blobsize)") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    char type_name [GxB_MAX_NAME_LEN] ;
    int sparsity_status, sparsity_ctrl, type_code, storage ;
    double hyper_sw, bitmap_sw ;

    GrB_Info info = GB_blob_header_get (type_name, &type_code, &sparsity_status,
        &sparsity_ctrl, &hyper_sw, &bitmap_sw, &storage, blob, blob_size) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    {
        switch (field)
        {
            case GrB_STORAGE_ORIENTATION_HINT : 

                (*value) = storage ;
                break ;

            case GrB_ELTYPE_CODE : 

                (*value) = type_code ;
                break ;

            case GxB_SPARSITY_CONTROL : 

                (*value) = sparsity_ctrl ;
                break ;

            case GxB_SPARSITY_STATUS : 

                (*value) = sparsity_status ;
                break ;

            case GxB_FORMAT : 

                (*value) = (storage == GrB_COLMAJOR) ? GxB_BY_COL : GxB_BY_ROW ;
                break ;

            default : 
                return (GrB_INVALID_VALUE) ;
        }
    }

    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_SIZE
(
    const void * blob,
    size_t * value,
    GrB_Field field,
    size_t blob_size
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Serialized_get_SIZE (blob, value, field, blobsize)") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    char type_name [GxB_MAX_NAME_LEN] ;
    int sparsity_status, sparsity_ctrl, type_code, storage ;
    double hyper_sw, bitmap_sw ;

    GrB_Info info = GB_blob_header_get (type_name, &type_code, &sparsity_status,
        &sparsity_ctrl, &hyper_sw, &bitmap_sw, &storage, blob, blob_size) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    const char *name ;

    if (info == GrB_SUCCESS)
    {
        switch (field)
        {

            case GrB_NAME :     
    GB_GOTCHA ;
                (*value) = 1 ;      // FIXME : name of blob
                break ;

            case GxB_JIT_C_NAME : 
                (*value) = strlen (type_name) + 1 ;
                break ;

            case GrB_ELTYPE_STRING : 
                // FIXME: return the user_name of user-defined type
                name = GB_code_name_get (type_code, NULL) ;
                (*value) = (name == NULL) ? 1 : (strlen (name) + 1) ;
                break ;

            default : 
                return (GrB_INVALID_VALUE) ;
        }
    }
    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_VOID
(
    const void * blob,
    void * value,
    GrB_Field field,
    size_t blob_size
)
{ 
    return (GrB_NOT_IMPLEMENTED) ;
}

