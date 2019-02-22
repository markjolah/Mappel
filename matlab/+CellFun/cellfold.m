function val = cellfold( fun, base, C )
%cellfold This function folds over a cell array.  
% fun - function handle of type: @(base, cell_item) -> base
% base - The 'zero' for the type of output.  This is the element used to start the recursion.
% C - cell array of the type that should be the second argument of 'fun'
    val=base;
    for n=1:length(C)
        val=fun(val,C{n});
    end
end

