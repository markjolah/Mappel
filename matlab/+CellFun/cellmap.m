function varargout = cellmap( fun, data, varargin )
%cellmap This function maps one cell array to anouther with cellfun w/ Uniform=0.
%
%     Will accept either a cell-array or a object/numeric array as input and always returns a cell array.
%
%     This eliminates the ugly matlab ('Uniform', 0) syntax everywhere, and makes the cellfun more
%     useful, for cases when a cellarray output is necessary.
    if iscell(data)
        [varargout{1:nargout}] = cellfun(fun, data, varargin{:}, 'Uniform', 0);
    else
        [varargout{1:nargout}] = arrayfun(fun, data, varargin{:},'Uniform', 0);
    end
end

