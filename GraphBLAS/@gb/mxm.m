function Cout = mxm (varargin)
%GB.MXM sparse matrix-matrix multiplication
%
% gb.mxm computes C<M> = accum (C, A*B) using a given semiring.
%
% Usage:
%
%   Cout = gb.mxm (semiring, A, B)
%   Cout = gb.mxm (semiring, A, B, desc)
%
%   Cout = gb.mxm (Cin, accum, semiring, A, B)
%   Cout = gb.mxm (Cin, accum, semiring, A, B, desc)
%
%   Cout = gb.mxm (Cin, M, semiring, A, B)
%   Cout = gb.mxm (Cin, M, semiring, A, B, desc)
%
%   Cout = gb.mxm (Cin, M, accum, semiring, A, B)
%   Cout = gb.mxm (Cin, M, accum, semiring, A, B, desc)
%
% Not all inputs are required.
%
% Cin is an optional input matrix.  If Cin is not present or is an empty
% matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of the
% right size (which depends on A, B, and the descriptor).  Its type is the
% output type of the accum operator, if it is present; otherwise, its type
% is the type of the additive monoid of the semiring.
%
% M is the optional mask matrix.  If not present, or if empty, then no mask
% is used.  If present, M must have the same size as C.
%
% If accum is not present, then the operation becomes C<...> = A*B.
% Otherwise, accum (C,A*B) is computed.  The accum operator acts like a
% sparse matrix addition (see gb.eadd).
%
% The semiring is a required string defining the semiring to use, in the
% form 'add.mult.type', where '.type' is optional.  For example,
% '+.*.double' is the conventional semiring for numerical linear algebra,
% used in MATLAB for C=A*B when A and B are double.  If A or B are complex,
% then the '+.*.complex' semiring is used (once complex matrice are
% supported).  GraphBLAS has many more semirings it can use.  See 'help
% gb.semiringinfo' for more details.
%
% A and B are the input matrices.  They are transposed on input if
% desc.in0 = 'transpose' (which transposes A), and/or
% desc.in1 = 'transpose' (which transposes B).
%
% The descriptor desc is optional.  If not present, all default settings are
% used.  Fields not present are treated as their default values.  See
% 'help gb.descriptorinfo' for more details.
%
% The input matrices Cin, M, A, and B can be MATLAB matrices or GraphBLAS
% objects, in any combination.
%
% Examples:
%
%   A = sprand (4,5,0.5) ;
%   B = sprand (5,3,0.5) ;
%   C = gb.mxm ('+.*', A, B) ;
%   norm (sparse(C)-A*B,1)
%   E = sprand (4,3,0.7) ;
%   M = logical (sprand (4,3,0.5)) ;
%   C2 = gb.mxm (E, M, '+', '+.*', A, B) ;
%   C3 = E ; AB = A*B ; C3 (M) = C3 (M) + AB (M) ;
%   norm (sparse(C2)-C3,1)
%
%
% See also gb.descriptorinfo, gb.add, mtimes.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbmxm (args {:})) ;
else
    Cout = gbmxm (args {:}) ;
end

