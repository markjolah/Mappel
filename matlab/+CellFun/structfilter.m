% Mark J. Olah (mjo@cs.unm.edu)
% 03-2015

function fst = structfilter( st, keep_fieldnames)
    %  Make a new structure containing only selected fieldnames.  If a field name from keep_fieldnames
    % does not exist in st it will also be missing in fst output with no warnings or errors.
    % [IN]
    %  st - A structure
    %  keep_fieldnames - cell array of field names to keep
    % [OUT]
    %  fst - The filtered structure with only the selected fieldnames (if they exitsted in st).
    fst = rmfield(st, setdiff(fieldnames(st),keep_fieldnames));
end
