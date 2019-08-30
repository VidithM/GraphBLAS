function C = diag (G, k)
% DIAG Diagonal matrices and diagonals of a matrix.
% C = diag (v,k) when v is a GraphBLAS vector with n components is a square
% sparse GarphBLAS matrix of dimension n+abs(k), with the elements of v on the
% kth diagonal. The main diagonal is k = 0; k > 0 is above the diagonal, and k
% < 0 is below the main diagonal.  C = diag (v) is the same as C = diag (v,0).
%
% c = diag (G,k) when G is a GraphBLAS matrix returns a GraphBLAS column
% vector c formed the entries on the kth diagonal of G.  The main diagonal
% is c = diag(G).
%
% The GraphBLAS diag function always constructs a GraphBLAS sparse matrix,
% unlike the the MATLAB diag, which always constructs a MATLAB full matrix.
%
% Examples:
%
%   C1 = diag (gb (1:10, 'uint8'), 2)
%   C2 = sparse (diag (1:10, 2))
%   nothing = double (C1-C2)
%
%   A = magic (8)
%   full (double ([diag(A,1) diag(gb(A),1)]))
%
%   m = 5 ;
%   f = ones (2*m,1) ;
%   A = diag(-m:m) + diag(f,1) + diag(f,-1)
%   G = diag(gb(-m:m)) + diag(gb(f),1) + diag(gb(f),-1)
%   nothing = double (A-G)
%
% See also diag, spdiags, tril, triu, gb.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    k = 0 ;
end
[am, an] = size (G) ;
isvec = (am == 1) || (an == 1) ;

if (isvec)
    % C = diag (v,k) is an m-by-m matrix
    if (am == 1)
        % convert G to a column vector
        v = G.' ;
    else
        v = G ;
    end
    n = length (v) ;
    m = n + abs (k) ;
    if (k >= 0)
        [I, ~, X] = gb.extracttuples (v, struct ('kind', 'zero-based')) ;
        J = I + int64 (k) ;
    else
        [J, ~, X] = gb.extracttuples (v, struct ('kind', 'zero-based')) ;
        I = J - int64 (k) ;
    end
    C = gb.build (I, J, X, m, m) ;
else
    % C = diag (G,k) is a column vector formed from the elements of the kth
    % diagonal of G
    C = gb.select ('diag', G, k) ;
    if (k >= 0)
        [I, ~, X] = gb.extracttuples (C, struct ('kind', 'zero-based')) ;
        m = min (an-k, am) ;
    else
        [~, I, X] = gb.extracttuples (C, struct ('kind', 'zero-based')) ;
        m = min (an, am+k) ;
    end
    J = zeros (length (X), 1, 'int64') ;
    C = gb.build (I, J, X, m, 1) ;
end

