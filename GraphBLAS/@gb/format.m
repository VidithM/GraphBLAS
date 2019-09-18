function f = format (arg)
%GB.FORMAT get/set the default GraphBLAS matrix format.
%
% In its ANSI C interface, SuiteSparse:GraphBLAS stores its matrices by
% row, by default, since that format tends to be fastest for graph
% algorithms, but it can also store its matrices by column.  MATLAB
% sparse and dense sparse matrices are always stored by column.  For
% better compatibility with MATLAB sparse matrices, the default for the
% MATLAB interface for SuiteSparse:GraphBLAS is to store matrices by
% column.  This has performance implications, and algorithms should be
% designed accordingly.  The default format can be can changed via:
%
%   gb.format ('by row')
%   gb.format ('by col')
%
% which changes the format of all subsequent GraphBLAS matrices.
% Existing gb matrices are not affected.
%
% The current default global format can be queried with
%
%   f = gb.format ;
%
% which returns the string 'by row' or 'by col'.
%
% Since MATLAB sparse and dense matrices are always 'by col', converting
% them to a gb matrix 'by row' requires an internal transpose of the
% format.  That is, if A is a MATLAB sparse or dense matrix,
%
%   gb.format ('by row')
%   G = gb (A)
%
% constructs a double gb matrix G that is held by row, but this takes
% more work than if G is held by column, as follows:
%
%   gb.format ('by col')
%   G = gb (A)
%
% If a subsequent algorithm works better with its matrices held by row,
% then this transformation can save significant time in the long run.
% Graph algorithms tend to be faster with their matrices held by row,
% since the edge (i,j) is typically the entry G(i,j) in the matrix G, and
% most graph algorithms need to know the outgoing edges of node i.  This
% is G(i,:), which is very fast if G is held by row, but very slow if G
% is held by column.
%
% When the gb.format (f) is changed, it becomes the default format for
% all subsequent matrices.  All prior matrices created before gb.format
% (f) are kept in their same format; this setting only applies to new
% matrices.  Operations on matrices can be done with any mix of with
% different formats.  The format only affects time and memory usage, not
% the results.
%
% The format of the output C of a GraphBLAS method is defined using the
% following rules.  The first rule that holds is used:
%
%   (1) If the format is determined by the descriptor to the method, then
%       that determines the format of C.
%   (2) If C is a column vector then C is stored by column.
%   (3) If C is a row vector then C is stored by row.
%   (4) If the method has a first matrix input (usually called A), and it
%       is not a row or column vector, then its format is used for C.
%   (5) If the method has a second matrix input (usually called B), and
%       it is not a row or column vector, then its format is used for C.
%   (6) Otherwise, the global default format is used for C.
%
% The gb.format setting is reset to 'by col', by 'clear all' or by
% gb.clear.
%
% To query the format for a given GraphBLAS matrix G, use the following
% (which does not affect the global format setting):
%
%   f = gb.format (G)
%
% Use G = gb (G, 'by row') or G = gb (G, 'by col') to change the format
% of G after it is constructed.
%
% Examples:
%
%   A = sparse (rand (4))
%   gb.format ('by row') ;
%   G = gb (A)
%   gb.format (G)
%   gb.format ('by col') ;      % set the default format to 'by col'
%   G = gb (A)
%   gb.format (G)               % query the format of G
%
% See also gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin == 0)
    % f = gb.format ; get the global format
    f = gbformat ;
else
    if (isa (arg, 'gb'))
        % f = gb.format (G) ; get the format of the matrix G
        f = gbformat (arg.opaque) ;
    else
        % f = gb.format (f) ; set the global format for all future matrices
        f = gbformat (arg) ;
    end
end

