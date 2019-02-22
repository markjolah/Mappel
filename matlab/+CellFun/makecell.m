function cell_arg=makecell(arg)
    % If arg is not already a cell, Try to make it into one.  This is a convienient way to ensure
    % passed arguments can be safely treated as cell arrays.
    if iscell(arg)
        cell_arg = arg;
    elseif isscalar(arg) || ischar(arg)
        cell_arg = {arg};
    elseif isempty(arg)
        cell_arg = {}; % any empty type input becomes empty cell
    else
        cell_arg = {arg};
    end
end
