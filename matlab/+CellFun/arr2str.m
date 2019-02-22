%Mark J. Olah
% Novermber 2014
function s=arr2str(in)
    % Chooses the best way to turn something into a string.
    if isempty(in)
        s = '';
    elseif ischar(in)
        s = in;
    elseif isscalar(in)
        s = num2str(in);
    elseif isvector(in)
        if isinteger(in) || all(fix(in)==in)
            nums = cellmap(@(v) sprintf('%i',v),in);
        else
            nums = cellmap(@(v) sprintf('%#10.5g',v),in);
        end
        s = sprintf('[%s]',strjoin(nums,', '));
    else
        s = mat2str(in,5);
    end
end
