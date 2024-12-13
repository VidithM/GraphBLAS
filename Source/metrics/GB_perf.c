#include "GB.h"
#include "GB_timing_context.h"

struct timing_context timing_ctx = {
    .stats_file = NULL,
    .kern_name = NULL,
    .subtrial_name = NULL,
    .loc = NULL,
    .cuda_hits = NULL,
    .tot_hits = NULL,
    .ntrials = 0,
    .curr_trial = 0,
    .do_timing = false
} ;

const char *allow_timing [] = {
    "select_sparse",
    "select_bitmap",
    "rowscale",
    "colscale",
    "apply_bind1st",
    "apply_bind2nd",
    "apply_unop",
    NULL
} ;