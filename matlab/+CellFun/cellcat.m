function C = cellcat(Cs)
    %cellcat This function concatenates a cell array of cell arrays down 1-level
    % This is equivelent to [Cs{:}] statement, but now in a function instead of a statement.
    if isrow(Cs)
        C = horzcat(Cs{:});
    elseif isempty(Cs)
        C = {};
    else
        C = vertcat(Cs{:});
    end 
end

