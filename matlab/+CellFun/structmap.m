function varargout = structmap(fun, st)
    % a shorter name for a non-uniform sturctfun mapping.
    [varargout{1:nargout}] = structfun(fun,st,'UniformOutput',0);
end
