#ifndef GB_PERF_H
#define GB_PERF_H

char *allow_timing [] = {
    "select_sparse",
    "select_bitmap",
    "rowscale",
    "colscale",
    "apply_bind1st",
    "apply_bind2nd",
    "apply_unop",
    NULL
} ;

#include <ctime>

#define RESULTS_LOCAL

#ifdef RESULTS_LOCAL
    #define RESULTS_DIR "/home/vidith/Desktop/cuda-results"
#else
    #define RESULTS_DIR "/home/grads/v/vidithm/cuda-results"
#endif

#define OPEN_STATS(fname, tot_hits, cuda_hits)                       \
    FILE *_stats_file ;                                              \
    int *_tot_hits = tot_hits ;                                      \
    int *_cuda_hits = cuda_hits ;                                    \
    char *_kern_name = fname ;                                       \
    bool _do_timing = false ;                                        \
{                                                                    \
    for (int i = 0 ; allow_timing [i] != NULL ; i++) {               \
        if (!strcmp (allow_timing [i], _kern_name)) {                \
            _do_timing = true ;                                      \
            break ;                                                  \
        }                                                            \
    }                                                                \
    if (_do_timing) {                                                \
        char results_path [PATH_MAX] ;                               \
        snprintf (results_path, PATH_MAX, "%s%s_timing.txt",         \
            RESULTS_DIR, fname) ;                                    \
        _stats_file = fopen (fname, "a") ;                           \
        if (_stats_file == NULL) {                                   \
            printf ("Failed to open stats file\n") ;                 \
            exit (-1) ;                                              \
        }                                                            \
    }
}

// For manual conditions
#define DO_TIMING _do_timing

// Hit counts not provided
#define OPEN_STATS(fname) OPEN_STATS (fname, NULL, NULL)

#define INIT_STATS(loc, work)                                        \
{                                                                    \
    char *_loc = loc ;                                               \
    } else if (!strcmp (_loc, "gpu"))) {                             \
        if (_cuda_hits != NULL) {                                    \
            (*_cuda_hits) += ((*_cuda_hits) != -1) ;                 \
        }                                                            \
    } else if (strcmp (_loc, "cpu")) {                               \
        printf ("Invalid loc specified\n") ;                         \
        exit (-1) ;                                                  \
    }                                                                \
    if (tot_hits != NULL) {                                          \
        (*_tot_hits) += ((*_tot_hits) != -1) ;                       \
    }                                                                \
    if (_do_timing) {                                                \
        char timestamp [64] ;                                        \
        time_t = time (NULL) ;                                       \
        struct tm *tm = localtime (&t) ;                             \
        size_t ret = strftime (timestamp, 64, "%c", tm) ;            \
        if (_cuda_hits == 0) {                                       \
            fprintf (_stats_file, "\n\nBatch at: %s\n\n"             \
                "======== [Kernel: %s] "                             \
                "[%s] [Start run: %d] "                              \
                "[tot_hits: %d (ratio: %0.3f)] [work: %ld] "         \
                "========\n", timestamp,                             \
                _kern_name, _loc, _cuda_hits,                        \
                _tot_hits, ((double) _cuda_hits) / _tot_hits,        \
                work) ;                                              \
        } else {                                                     \
            fprintf (_stats_file, "======== [Kernel: %s] "           \
                "[%s] [Start run: %d] [tot_hits: %d] "               \
                "(ratio: %0.3f) [work: %ld] ========\n",             \
                _kern_name, _loc, _cuda_hits, _tot_hits,             \
                ((double) _cuda_hits) / _tot_hits, work) ;           \
        }                                                            \
        fflush (_stats_file) ;
    }

#define END_STATS                                                    \
    {                                                                \
        if (_do_timing) {                                            \
            fprintf (stats_file, "======== [Kernel: %s] "            \
                "[%s] [End run: %d] ========\n\n",                   \
                _kern_name, _loc, cuda_hits) ;                       \
            fflush (stats_file) ;                                    \
            if (!strcmp (_loc, "gpu")) {                             \
                info = GrB_NO_VALUE ;                                \
            }                                                        \
        }                                                            \
    }                                                                \
}

#define CLOSE_STATS             \
{                               \
    if (_do_timing) {           \
        fclose (_stats_file) ;  \
    }
}

#define INIT_TRIALS(name, ntrials)                                   \
    int _ntrials = (_do_timing ? ntrials : 1) ;                      \
    char *_trial_name = name ;                                       \
    for (int i = 0 ; i < _ntrials ; i++) {                           \


#define END_TRIALS                  \
        if (i < _ntrials - 1) {     \
            TRIAL_FREE ;            \
        }                           \
    }

#define TRIAL_RETURN(info)          \
{                                   \
    if (i < _ntrials - 1) {         \
        TRIAL_FREE ;                \
        continue ;                  \
    } else {                        \
        return info ;               \
    }                               \
}

#if defined(_OPENMP) && defined(_OMP_H)
// Use omp_get_wtime()
#define START_TIME                          \
    double _t_start = omp_get_wtime () ;    \

#else
// We are in CUDA, so OpenMP is not available. 
// Can use C++ std::high_resolution_clock
#define START_TIME
// ...

#endif

#endif