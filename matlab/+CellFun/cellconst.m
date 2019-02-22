function newcell = cellconst( c, N)
    %Return a cellarray with the constant const repeated n times.
    newcell=cellmap(@(~) c, 1:N);
end
