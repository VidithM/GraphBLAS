#ifndef GB_PERF_H
#define GB_PERF_H

// #define RESULTS_LOCAL

#ifdef RESULTS_LOCAL
    #define RESULTS_DIR "/home/vidith/Desktop/cuda-results"
#else
    #define RESULTS_DIR "/home/grads/v/vidithm/cuda-results"
#endif

#include <time.h>
#include "GB_timing_context.h"

extern struct timing_context timing_ctx ;

#define GET(field) timing_ctx.field

#define OPEN_STATS(_kern_name, _tot_hits, _cuda_hits)                \
{                                                                    \
    extern char *allow_timing [] ;                                   \
    GB_memset (&timing_ctx, 0,                                       \
        sizeof(struct timing_context), 1) ;                          \
    for (int i = 0 ; allow_timing [i] != NULL ; i++) {               \
        if (!strcmp (allow_timing [i], _kern_name)) {                \
            GET (do_timing) = true ;                                 \
            break ;                                                  \
        }                                                            \
    }                                                                \
    if (GET (do_timing)) {                                           \
        char results_path [PATH_MAX] ;                               \
        snprintf (results_path, PATH_MAX, "%s/%s_timing.txt",        \
            RESULTS_DIR, _kern_name) ;                               \
        GET (stats_file) = fopen (results_path, "a") ;               \
        if (GET (stats_file) == NULL) {                              \
            printf ("Failed to open stats file\n") ;                 \
            exit (-1) ;                                              \
        }                                                            \
        GET (kern_name) = _kern_name ;                               \
        GET (cuda_hits) = _cuda_hits ;                               \
        GET (tot_hits) = _tot_hits ;                                 \
    }                                                                \
}

// For manual checking if timing is enabled
#define DO_TIMING                                                    \
    GET (do_timing)

#define BEGIN_STATS(_loc, _work)                                     \
{                                                                    \
                                                                     \
    if (GET (do_timing)) {                                           \
        int _cuda_hits, _tot_hits ;                                  \
        GET (loc) = _loc ;                                           \
        _cuda_hits = _tot_hits = -1 ;                                \
        if (GET (cuda_hits) != NULL) {                               \
            if (!strcmp (GET (loc), "gpu")) {                        \
                (*GET (cuda_hits))++ ;                               \
            }                                                        \
            _cuda_hits = *GET (cuda_hits) ;                          \
        }                                                            \
        if (GET (tot_hits) != NULL) {                                \
            if (!strcmp (GET (loc), "cpu")) {                        \
                (*GET (tot_hits))++ ;                                \
            }                                                        \
            _tot_hits = *GET (tot_hits) ;                            \
        }                                                            \
        char _timestamp [64] ;                                       \
        time_t _t = time (NULL) ;                                    \
        struct tm *_tm = localtime (&_t) ;                           \
        strftime (_timestamp, 64, "%c", _tm) ;                       \
        if (_tot_hits + _cuda_hits == 1) {                           \
            fprintf (GET (stats_file), "\n\nBatch at: %s\n\n"        \
                "======== [Kernel: %s] "                             \
                "[%s] [cuda_hits: %d] "                              \
                "[tot_hits: %d (ratio: %0.3f)] [work: %ld] "         \
                "========\n", _timestamp,                            \
                GET (kern_name), GET (loc), _cuda_hits,              \
                _tot_hits, ((double) _cuda_hits) / _tot_hits,        \
                _work) ;                                             \
        } else {                                                     \
            fprintf (GET (stats_file), "======== [Kernel: %s] "      \
                "[%s] [cuda_hits: %d] [tot_hits: %d] "               \
                "(ratio: %0.3f) [work: %ld] ========\n",             \
                GET (kern_name), GET (loc), _cuda_hits, _tot_hits,   \
                ((double) _cuda_hits) / _tot_hits, _work) ;          \
        }                                                            \
        fflush (GET (stats_file)) ;                                  \
    }                                                                \
}

// TODO: Do a stats summary (min/max work, %GPU beat, ...)
#define END_STATS                                                    \
{                                                                    \
    if (GET (do_timing)) {                                           \
        int _cuda_hits ;                                             \
        if (GET (cuda_hits) != NULL) {                               \
            _cuda_hits = *GET (cuda_hits) ;                          \
        }                                                            \
        fprintf (GET (stats_file), "======== [Kernel: %s] "          \
            "[%s] [End run: %d] ========\n\n",                       \
            GET (kern_name), GET (loc), _cuda_hits) ;                \
        fflush (GET (stats_file)) ;                                  \
        if (!strcmp (GET (loc), "gpu")) {                            \
            STATS_RESET ;                                            \
            info = GrB_NO_VALUE ;                                    \
        }                                                            \
    }                                                                \
}

#define CLOSE_STATS                                                  \
{                                                                    \
    if (GET (do_timing)) {                                           \
        fclose (GET (stats_file)) ;                                  \
    }                                                                \
}

#define BEGIN_TRIALS(_ntrials)                                       \
    GET (ntrials) = (GET (do_timing) ? _ntrials : 1) ;               \
    GET (curr_trial) = 0 ;                                           \
    for (; GET (curr_trial) < GET (ntrials) ; GET (curr_trial)++) {  \

#define END_TRIALS                                                   \
        if (GET (curr_trial) < GET (ntrials) - 1) {                  \
            TRIAL_FREE ;                                             \
        }                                                            \
    }

#define TRIAL_RETURN(_info)                                          \
{                                                                    \
    if (GET (curr_trial) < GET (ntrials) - 1) {                      \
        TRIAL_FREE ;                                                 \
        continue ;                                                   \
    } else {                                                         \
        return _info ;                                               \
    }                                                                \
}

#if defined(_OPENMP) && defined(_OMP_H)
// Use omp_get_wtime()
#define START_TIME                          \
{                                           \
    if (GET (do_timing)) {                  \
        GET (subtrial_name) = "N/A" ;       \
        GET (t_start) = omp_get_wtime () ;  \
    }                                       \
}

#define START_TIME_NAMED(name)              \
{                                           \
    if (GET (do_timing)) {                  \
        GET (subtrial_name) = name ;        \
        GET (t_start) = omp_get_wtime () ;  \
    }                                       \
}

#define STOP_TIME                                                    \
{                                                                    \
    if (GET (do_timing)) {                                           \
        double _t_end = omp_get_wtime () ;                           \
        fprintf (GET (stats_file), "[trial: %-3d] "                  \
            "[subtrial: %-15s] "                                     \
            "wall clock: %0.8fs\n", GET (curr_trial),                \
            GET (subtrial_name), _t_end - GET (t_start)) ;           \
        fflush (GET (stats_file)) ;                                  \
    }                                                                \
}

#else

#if defined(__cplusplus)
// In non-JIT CUDA host code; can use
// C++ chrono::system_clock
#define START_TIME                                                   \
{                                                                    \
    if (GET (do_timing)) {                                           \
        GET (subtrial_name) = "N/A" ;                                \
        auto _t_start = std::chrono::system_clock::now () ;          \
        GET (t_start) = _t_start.time_since_epoch().count () ;       \
    }                                                                \
}

#define START_TIME_NAMED(name)                                       \
{                                                                    \
    if (GET (do_timing)) {                                           \
        GET (subtrial_name) = name ;                                 \
        auto _t_start = std::chrono::system_clock::now () ;          \
        GET (t_start) = _t_start.time_since_epoch().count () ;       \
    }                                                                \
}

#define STOP_TIME                                                    \
{                                                                    \
    if (GET (do_timing)) {                                           \
        auto t_end = std::chrono::system_clock::now()                \
            .time_since_epoch().count() ;                            \
        double duration = t_end - GET (t_start) ;                    \
        duration /= 1e9 ;                                            \
        fprintf (GET (stats_file), "[trial: %-3d] "                  \
            "[subtrial: %-15s] "                                     \
            "wall clock: %0.8fs\n", GET (curr_trial),                \
            GET (subtrial_name), duration) ;                         \
        fflush (GET (stats_file)) ;                                  \
    }                                                                \
}
#endif // ifdef __cplusplus

#endif // ifdef _OPENMP and _OMP_H

#endif // ifndef GB_PERF_H
