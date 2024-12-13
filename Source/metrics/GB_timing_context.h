#ifndef GB_TIMING_CONTEXT_H
#define GB_TIMING_CONTEXT_H

struct timing_context
{
    FILE *stats_file ;
    char *kern_name ;
    char *subtrial_name ;
    char *loc ;
    int *cuda_hits ;
    int *tot_hits ;
    double t_start ;
    int ntrials ;
    int curr_trial ;
    bool do_timing ;
} ;

#endif