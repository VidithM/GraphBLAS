function descriptorinfo (d)
%GB.DESCRIPTOR list the contents of a SuiteSparse:GraphBLAS descriptor
%
% Usage:
%
%   gb.descriptorinfo
%   gb.descriptorinfo (d)
%
% The GraphBLAS descriptor is a MATLAB struct that can be used to modify
% the behavior of GraphBLAS operations.  It contains the following
% components, each of which are a string or a number.  Any component of
% struct that is not present is set to the default value.  If the
% descriptor d is empty, or not present, in a GraphBLAS function, all
% default settings are used.
%
% The following descriptor values are strings:
%
%   d.out   'default' or 'replace'      determines if C is cleared before
%                                         the accum/mask step
%   d.mask  'default' or 'complement'   determines if M or !M is used
%   d.in0   'default' or 'transpose'    determines A or A' is used
%   d.in1   'default' or 'transpose'    determines B or B' is used
%   d.axb   'default', 'Gustavson', 'heap', or 'dot'
%            determines the method used in gb.mxm.  The default is to let
%            GraphBLAS determine the method automatically, via a
%            heuristic.
%   d.kind   For most gb.methods, this is a string equal to 'default',
%            'gb', 'sparse', or 'full'.  The default is d.kind = 'gb',
%            where the GraphBLAS operation returns an object, which is
%            preferred since GraphBLAS sparse matrices are faster and can
%            represent many more data types.  However, if you want a
%            standard MATLAB sparse matrix, use d.kind='sparse'.  Use
%            d.kind='full' for a MATLAB dense matrix.  For any gb.method
%            that takes a descriptor, the following uses are the same, but
%            the first method is faster and takes less temporary workspace:
%
%               d.kind = 'sparse' ;
%               S = gb.method (..., d) ;
%
%               % with no d, or d.kind = 'default'
%               S = double (gb.method (...)) :
%
%           [I, J, X] = gb.extracttuples (G,d) uses d.kind = 'one-based' or
%           'zero-based' to determine the type of I and J.
%
% These descriptor values are scalars:
%
%   d.nthreads  max # of threads to use; default is omp_get_max_threads.
%   d.chunk     controls # of threads to use for small problems.
%
% gb.descriptorinfo (d) lists the contents of a GraphBLAS descriptor and
% checks if its contents are valid.  Also refer to the
% SuiteSparse:GraphBLAS User Guide for more details.
%
% See also gb, gb.unopinfo, gb.binopinfo, gb.monoidinfo, gb.semiringinfo.

if (nargin == 0)
    help gb.descriptorinfo
else
    gbdescriptorinfo (d) ;
end

