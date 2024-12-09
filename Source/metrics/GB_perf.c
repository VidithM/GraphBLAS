#include "GB.h"

static const char *allow_timing [] = {
    "select_sparse",
    "select_bitmap",
    "rowscale",
    "colscale",
    "apply_bind1st",
    "apply_bind2nd",
    "apply_unop",
    NULL
} ;