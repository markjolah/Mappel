function mapEstimatorAccuracyCompare(obj,estimators, paramNames, gridsize, nTrials, intensity, background)
    if nargin<7
        background=5;
    end
    if nargin<6
        intensity=100;
    end
    if nargin<5
        nTrials=100;
    end
    if nargin<4
        gridsize=10;
    end
    if nargin<3
        paramNames={'x','y'};
    elseif isa(paramNames,'char')
        paramNames={paramNames};
    end
    mask=obj.paramMask(paramNames);
    if isa(estimators, 'char')
        estimators={estimators};
    end
    Nestimators=length(estimators);
    f=figure('PaperSize',[11 8.5]);
    
    sz=double(obj.imsize(1));
    ticks=linspace(1,gridsize,sz);
    ticks_labels=linspace(1,sz,sz);
    set(f,'Colormap',jet());
    set(f,'PaperType','usletter', 'PaperOrientation', 'landscape', 'PaperPositionMode', 'manual');
    set(f,'PaperPosition',[0.5 0.5 10.5,8.0]);
    grids=cell(1,Nestimators);
    gmax=0;
    for i=1:Nestimators
        estimator=estimators{i};
        fprintf('Mapping estimator accuracy: %s\n',estimator);
        grid=obj.mapEstimatorAccuracy(estimator, mask, gridsize, nTrials, intensity);
        gmax=max(gmax,max(grid(:)));
        grids{i}=grid;
    end
    for i=1:Nestimators
        if Nestimators>3
            subplot(ceil(Nestimators/2),2,i);
        else
            subplot(Nestimators,1,i);
        end
        imagesc(grids{i},[0,gmax]);
        set(gca,'XTick',ticks);
        set(gca,'XTickLabel',ticks_labels);
        set(gca,'YTick',ticks);
        set(gca,'YTickLabel',ticks_labels);
        xlabel('x (pixels)');
        ylabel('y (pixels)');
        title(sprintf('%s $I=%.3f$, $bg=%.2f$',estimators{i},intensity, background),'Interpreter' ,'latex');
        c=colorbar();
        ylabel(c, sprintf('RMSE(px) [%s]',strjoin(paramNames)));
    end
    tightfig;
end
