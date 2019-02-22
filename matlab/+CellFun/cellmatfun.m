function mat = cellmatfun( fun, cellarray )
%cellmatfun This function makes a matrix out of a cellarray by calling fun on each cell.
    if iscell(cellarray)
        mat=cell2mat(cellfun(fun, cellarray, 'Uniform', 0));
    else
        mat=cell2mat(arrayfun(fun, cellarray, 'Uniform', 0));
    end
end

