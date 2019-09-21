function gbtest62
%GBTEST62 test ldivide, rdivide, mldivide, mrdivide

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

n = 10 ;
for trial = 1:40

    fprintf ('.') ;

    A = 100 * rand (n) ;
    B = 100 * rand (n) ;
    b = rand (n, 1) ;

    r = rand ;
    s = gb (r) ;

    GA = gb (A) ;
    GB = gb (B) ;

    C0 = A ./ r ;
    C1 = GA ./ s ;
    assert (isequal (C0, C1)) ;

    C0 = A / r ;
    C1 = GA / s ;
    assert (isequal (C0, C1)) ;

    C0 = A ./ 0 ;
    C1 = GA ./ 0 ;
    assert (isequal (C0, C1)) ;

    C0 = A / 0 ;
    C1 = GA / 0 ;
    assert (isequal (C0, C1)) ;

    C0 = 0 .\ A ;
    C1 = 0 .\ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 0 \ A ;
    C1 = 0 \ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 2 ./ r ;
    C1 = gb (2) ./ s ;
    assert (isequal (C0, C1)) ;

    C0 = 2 ./ A ;
    C1 = 2 ./ GA ;
    assert (isequal (C0, C1)) ;

    C0 = 2 .\ r ;
    C1 = gb (2) .\ s ;
    assert (isequal (C0, C1)) ;

    C0 = 2 \ r ;
    C1 = gb (2) \ s ;
    assert (isequal (C0, C1)) ;

    C0 = A ./ B ;
    C1 = GA ./ GB ;
    assert (isequal (C0, C1)) ;

    C0 = A .\ B ;
    C1 = GA .\ GB ;
    assert (isequal (C0, C1)) ;

    x = A \ b ;
    y = GA \ b ;
    assert (isequal (x, y)) ;

    x = b' / A ;
    y = b' / GA ;
    assert (norm (x - y) < 1e-12) ;

    A = sprand (n, n, 0.5) ;
    B = rand * A ;
    GA = gb (A) ;
    GB = gb (B) ;

    C0 = A ./ B ;
    C1 = GA ./ GB ;
    assert (isequal (isnan (C0), isnan (C1))) ;
    assert (isequal (gb.prune (C0, nan), gb.prune (C1, nan))) ;

    C0 = A .\ B ;
    C1 = GA .\ GB ;
    assert (isequal (isnan (C0), isnan (C1))) ;
    assert (isequal (gb.prune (C0, nan), gb.prune (C1, nan))) ;

end

fprintf ('\ngbtest62: all tests passed\n') ;

