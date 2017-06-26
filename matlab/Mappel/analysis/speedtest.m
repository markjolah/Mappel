
function results=speedtest(nproblemsizes, maxproblemsize, maximagesize, ntrials)
    % ntrials - How many
    results.date=datestr(clock);
    results.ntrials=ntrials;
    results.maxproblemsize=maxproblemsize;
    results.maximagesize=maximagesize;
    results.imagesize=imagesize(maximagesize, ntrials);
    results.problemsize=problemsize(nproblemsizes, maxproblemsize, ntrials);
end

function results=imagesize(maxsize,count)
    minsize=4;
    psfsigma=1;
    problemsize=1e3;
    sizes=minsize:maxsize;
    models=cellmap(@(size) Gauss2DMAP(size,psfsigma), sizes); 
    thetas=cellmap(@(model) model.samplePrior(problemsize), models);
    images=cellmap(@(model,theta) model.simulateImage(theta), models, thetas);

    funs=cellmap(@(model) @model.samplePrior, models);
    disp('ImageSize:samplePrior');
    out.methods.samplePrior=timeit(count, funs, problemsize);
  
    funs=cellmap(@(model) @model.modelImage, models);
    disp('ImageSize:modelImage');
    out.methods.modelImage=timeit(count, funs, thetas);

    funs=cellmap(@(model) @model.simulateImage, models);
    disp('ImageSize:simulateImage');
    out.methods.simulateImage=timeit(count, funs, thetas);

    funs=cellmap(@(model) @model.LLH, models);
    disp('ImageSize:LLH');
    out.methods.LLH=timeit(count, funs, images, thetas);
    disp('ImageSize:LLHsingletheta');
    out.methods.LLHsingletheta=timeit(count, funs, images, cellmap(@(th) th(:,1),thetas));
    disp('ImageSize:LLHsingleimage');
    out.methods.LLHsingleimage=timeit(count, funs, cellmap(@(im) im(:,:,1), images), thetas);
    
    funs=cellmap(@(model) @model.CRLB, models);
    disp('ImageSize:CRLB');
    out.methods.CRLB=timeit(count, funs, thetas);

    funs=cellmap(@(model) @model.estimateMAP, models);
    for i=1:numel(Gauss2DMAP.EstimationMethods)
        name=Gauss2DMAP.EstimationMethods{i};
        if strfind(name,'CGauss'); continue; end
        disp(['ImageSize:Estimator: ' name]);
        name_input=cellconst(name,numel(sizes));
        out.estimators.(name)=timeit(count, funs, images, name_input);
    end
    
    cellfun(@(model) delete(model), models);

    results.methods.mean=makeTimesTable(sizes','mean',out.methods);
    results.methods.std=makeTimesTable(sizes','std',out.methods);
    results.estimators.mean=makeTimesTable(sizes','mean', out.estimators);
    results.estimators.std=makeTimesTable(sizes','std', out.estimators);
end

function results=problemsize(npoints,maxsize, count)
    %results is a table
    size=8;
    psfsigma=1;
    results.N=int32(logspace(0,log10(maxsize),npoints));

    model=Gauss2DMAP(size, psfsigma);
    thetas=cellmap(@model.samplePrior,results.N);
    images=cellmap(@model.simulateImage,thetas);
    
    disp('ProblemSize:samplePrior');
    out.methods.samplePrior=timeit(count, @model.samplePrior, results.N);
    disp('ProblemSize:modelImage');
    out.methods.modelImage=timeit(count, @model.modelImage, thetas);
    disp('ProblemSize:simulateImage');
    out.methods.simulateImage=timeit(count, @model.simulateImage, thetas);
    disp('ProblemSize:LLH');
    out.methods.LLH=timeit(count, @model.LLH, images, thetas);
    disp('ProblemSize:sLLHsingletheta');
    out.methods.LLHsingletheta=timeit(count, @model.LLH, images, cellconst(thetas{1},npoints));
    disp('ProblemSize:LLHsingleimage');
    out.methods.LLHsingleimage=timeit(count, @model.LLH, cellconst(images{1},npoints), thetas);
    disp('ProblemSize:CRLB');
    out.methods.CRLB=timeit(count, @model.CRLB, thetas);

    for i=1:numel(Gauss2DMAP.EstimationMethods)
        name=model.EstimationMethods{i};
        if strfind(name,'CGauss'); continue; end
        disp(['ProblemSize:Estimator: ' name]);
        name_input=cellconst(name,npoints);
        out.estimators.(name)=timeit(count, @model.estimateMAP, images, name_input);
    end
    delete(model);

    results.size=size;
    results.psfsigma=psfsigma;

    results.methods.mean=makeTimesTable(results.N','mean',out.methods);
    results.methods.std=makeTimesTable(results.N','std',out.methods);
    results.estimators.mean=makeTimesTable(results.N','mean', out.estimators);
    results.estimators.std=makeTimesTable(results.N','std', out.estimators);
end

function tab=makeTimesTable(N, name, time_struct)
    tab=table(N);
    tab=[tab struct2table(structmap(@(f) f.(name), time_struct))];
end
