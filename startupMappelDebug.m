function startupMappelDebug()
    mex_sub_dir='mex.glnxa64.debug';
    startup_git_path = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(startup_git_path,'matlab')));
    addpath(genpath(fullfile(startup_git_path,'..','MexIFace','matlab')));
    addpath(fullfile(startup_git_path,'_install','lib','Mappel','mex',mex_sub_dir));
end
