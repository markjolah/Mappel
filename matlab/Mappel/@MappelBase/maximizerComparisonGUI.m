% MappelBase.maximizerComparisonGUI()
% This GUI allows for comparison of the accuracy and efficiency of each of the estimation routines.
% The accuracies are measured based on a sample of size N images all sampled from the same theta.
% Optionally a theta_init can be specified.  If unset, the butilin theta initialization will be used.

function guiFig = maximizerComparisonGUI(obj, theta, theta_init)
    gui_name = sprintf('[%s] Maximizer Algorithm Comparison',class(obj));
    fig_sz = [1600 800]; %figure size
    uH = 25;
    guiFig = figure('Units','pixels','Position',[20 20 fig_sz],'Resize','off',...
                    'MenuBar','none','ToolBar','figure','NumberTitle','off',...
                    'Name',gui_name,'Visible','on');
    if nargin<2
        theta = obj.samplePrior();
    end
    if nargin<3
        theta_init=[];
    end
    crlb = obj.CRLB(theta);
    nSamples = 30;
    useThetaInit=false;
    resultsPlotted=false;
    model_im=[];
    ims = [];
    llh_ims = [];
    run = struct();
    
    Es = {'Newton','NewtonDiagonal'};
%     Es = {'CGauss','Newton'};
    nEs = numel(Es);
    colors.Newton = [1,0,0];
    colors.NewtonDiagonal = [0,1,0];
    colors.QuasiNewton = [0,0,1];
    colors.Heuristic = [.8,.8,.8];
    colors.CGauss = [0, 1, 1];
    
    mainFontSz =10;
    boarder = 5;
    sp = 2;
    panh = [500, 1100];
    paramPan_pos = [boarder, boarder, panh(1)-sp-boarder,  fig_sz(2)-2*boarder];
    paramPan_sz = paramPan_pos(3:4) - paramPan_pos(1:2);
    accPan_pos = [panh(1)+sp,  boarder, panh(2)-panh(1)-2*sp, fig_sz(2)-2*boarder];
    accPan_sz = accPan_pos(3:4) - accPan_pos(1:2);
    effPan_pos = [panh(2)+sp,  boarder, fig_sz(1)-boarder-panh(2)-sp, fig_sz(2)-2*boarder];
    effPan_sz = effPan_pos(3:4) - effPan_pos(1:2);
    handles.paramPan = uipanel(guiFig,'Units','pixels','Position',paramPan_pos,'Title','Parameters','FontSize',mainFontSz);
    handles.accPan = uipanel(guiFig,'Units','pixels','Position',accPan_pos,'Title','Accuracy','FontSize',mainFontSz);
    handles.effPan = uipanel(guiFig,'Units','pixels','Position',effPan_pos,'Title','Efficiency','FontSize',mainFontSz);

    %axes initialization
    
    model_fig_pos = [15, 10,paramPan_sz(1)-boarder-15,450];
    ax.model = axes('Parent',handles.paramPan,'Units','pixels','Position',model_fig_pos,...
                    'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');
    acc_axes_pos = [sp, sp, paramPan_sz(1)-2*sp, paramPan_sz(2)-2*sp];
%     ax.acc = axes('Parent',handles.accPan,'Units','pixels','Position',acc_axes_pos);


    %Initialization
    createControls();
    initializeAxes();
    setTheta(theta);
% 
    function createControls()
        control_h = 130;
        panel1_pos=[sp, paramPan_sz(2)-control_h, paramPan_sz(1)-2*sp, paramPan_sz(2)-sp];
        hNames={'nSamples','theta','thetaInit','thetaCRLB'};
        labels={'#Samples','Theta','Theta Init','Theta SE (CRLB):'};
        values={nSamples, theta,theta_init,sqrt(crlb) };
        CBs={@setNSamplesEdit_CB,@setThetaEdit_CB,@setThetaInitEdit_CB,[]};
        handles.edits = GUIBuilder.labeledHEdits(handles.paramPan, panel1_pos, uH, hNames, labels, values, CBs);
        useInit_toggle_pos = [sp, paramPan_sz(2)-control_h-uH, 80, uH];
        handles.useInitToggle = uicontrol('Parent',handles.paramPan,'Style','checkbox','Position',useInit_toggle_pos,'String',...
                                      'Use Init','Selected','off','Callback',@setUseInit_CB);
        handles.edits.thetaInit.Enable='off';
        eval_but_pos = [paramPan_sz(1)-80-boarder, paramPan_sz(2)-control_h-uH, 80, uH];
        handles.evalButton = uicontrol('Parent',handles.paramPan,'Style','pushbutton','String','Evaluate','Position',eval_but_pos,'Callback',@evaluate_CB);
    
    end

    function setNSamplesEdit_CB(~,~)
        N=int32(str2num(handles.edits.nSamples.String)); %#ok<ST2NM>
        if isempty(N) || ~isscalar(N) || N<10
            error('MappelBase:maximizeComparisonGui','nSamples: min value=10');
        end
        nSamples=N;
    end

    function setUseInit_CB(~,~)
        useThetaInit = logical(handles.useInitToggle.Value);
        if useThetaInit
            handles.edits.thetaInit.Enable='on';
        else
            handles.edits.thetaInit.Enable='off';
        end
        if isempty(theta_init)
            if isempty(ims)
                theta_init = obj.estimate(obj.simulateImage(theta,1),'Heuristic');
            else
                theta_init = mean(obj.estimate(ims,'Heuristic'),2);
            end
        end
        handles.edits.thetaInit.String = arr2str(theta_init(:)');
    end

    function initializeAxes()
        %Sim Position axes
        xlabel(ax.model,'X (px)');
        ylabel(ax.model,'Y (px)');
        axis(ax.model,'tight');
        title(ax.model,'Model Image');
        colormap(ax.model,hot());
        colorbar(ax.model);
        GUIBuilder.positionImageAxes(ax.model,obj.imsize,model_fig_pos,[00 00 40 00]);
    end


    function evaluate_CB(~,~)
        evaluateMaximizers();
    end

    function setThetaEdit_CB(~,~)
        theta_edit = str2num(handles.edits.theta.String)';
        if any(theta~=theta_edit)
            setTheta(theta_edit);
        end
    end

    function setThetaInitEdit_CB(~,~)
        theta_init_edit = str2num(handles.edits.thetaInit.String)';
        if isempty(theta_init) || any(theta_init~=theta_init_edit)
            setThetaInit(theta_init_edit);
        end
    end

    function setThetaInit(new_theta_init)
        if numel(new_theta_init)~=obj.nParams
            error('MappelBase:maximizerComparisonGUI','bad theta_init size');
        end
        theta_init = new_theta_init;
        handles.edits.theta_init.String = arr2str(theta_init(:)');
        handles.theta_init_pt = impoint(ax.model,theta_init(1),theta_init(2));
        clearResults();
    end

    function setTheta(new_theta)
        if numel(new_theta)~=obj.nParams
            error('MappelBase:maximizerComparisonGUI','bad theta size');
        end
        theta = new_theta(:);
        model_im = obj.modelImage(theta);
        crlb = obj.CRLB(theta);
        handles.edits.thetaCRLB.String = arr2str(crlb(:)');
        handles.edits.theta.String = arr2str(theta(:)');
        axes(ax.model);
        imagesc(model_im, [0, max(model_im(:))] );
        colorbar(ax.model);
        xlabel(ax.model,'X (px)');
        ylabel(ax.model,'Y (px)');
        title(ax.model,'Model Image');
        clearResults();
    end

    function evaluateMaximizers()
        ims = obj.simulateImage(theta,nSamples);
        llh_ims = obj.LLH(ims,theta);
        for name_idx = Es
            name = name_idx{1};
            if useThetaInit
                [run.(name).est_theta, ~, ~, run.(name).stats] = obj.estimate(ims,name,theta_init);
            else
                [run.(name).est_theta, ~, ~, run.(name).stats] = obj.estimate(ims,name);
            end
            run.(name).est_errors = run.(name).est_theta - repmat(theta,1,nSamples);
            run.(name).est_error_std = std(run.(name).est_errors,[],2);
            run.(name).est_error_mean = mean(run.(name).est_errors,2);
            run.(name).est_theta_mean = mean(run.(name).est_theta,2);
        end
        plotResults()
    end

    function plotResults()
        resultsPlotted = true;
        norm = 'countdensity';
        nbins = ceil(max(sqrt(double(nSamples)),15));
        tlw = 3; % true line width
        tls = 'k--'; % true line style
        handles.plots.accX=subplot(3,2,1,'Parent',handles.accPan);
        cla(handles.plots.accX,'reset');
        hold('on');
        bin_width=[];
        for name_idx = Es            
            name = name_idx{1};
            desc = sprintf('%s $\\bar{x}$=%.6g SE=%.4f',name, run.(name).est_theta_mean(1), run.(name).est_error_std(1));
            h = histogram(run.(name).est_theta(1,:),'NumBins',nbins,'EdgeColor','none','FaceColor',colors.(name), 'DisplayName',desc,'DisplayStyle','bar','Normalization',norm);
%             if isempty(bin_width)
%                 bin_width = h.BinWidth;
%             else
%                 h.BinWidth = bin_width;
%             end
        end
        yl=ylim();
        plot(theta([1,1]),yl([1,2]),tls,'LineWidth',tlw,'DisplayName',sprintf('true x=%.6f crlbSE=%.4f', theta(1), sqrt(crlb(1))));
        xl=xlim();
        xl(1)=max(xl(1),0);
        xl(2)=min(xl(2),obj.imsize(1));
        xlim(xl);
        title('X-distribution');
        h=legend('location','best');
        h.Interpreter='latex';


        handles.plots.accY=subplot(3,2,2,'Parent',handles.accPan);
        cla(handles.plots.accY,'reset');
        hold('on');
        bin_width=[];
        for name_idx = Es            
            name = name_idx{1};
            desc = sprintf('%s $\\bar{y}$=%.6g SE=%.4f',name, run.(name).est_theta_mean(2), run.(name).est_error_std(2));
            h = histogram(run.(name).est_theta(2,:),'EdgeColor','none','FaceColor',colors.(name),'NumBins',nbins, 'DisplayName',desc,'DisplayStyle','bar','Normalization',norm);
%             if isempty(bin_width)
%                 bin_width = h.BinWidth;
%             else
%                 h.BinWidth = bin_width;
%             end
        end
        yl=ylim();
        plot(theta([2,2]),yl([1,2]),tls,'LineWidth',tlw,'DisplayName',sprintf('true y=%.6f crlbSE=%.4f', theta(2), sqrt(crlb(2))));
        xl=xlim();
        xl(1)=max(xl(1),0);
        xl(2)=min(xl(2),obj.imsize(2));
        xlim(xl);
        title('Y-distribution');
        h=legend('location','best');
        h.Interpreter='latex';

        handles.plots.accI=subplot(3,2,3,'Parent',handles.accPan);
        cla(handles.plots.accI,'reset');
        hold('on');
        bin_width=[];
        for name_idx = Es            
            name = name_idx{1};
            desc = sprintf('%s $\\bar{I}$=%.6g SE=%.4f',name, run.(name).est_theta_mean(3), run.(name).est_error_std(3));
            h = histogram(run.(name).est_theta(3,:),'EdgeColor','none','FaceColor',colors.(name),'NumBins',nbins, 'DisplayName',desc,'DisplayStyle','bar','Normalization',norm);
            if isempty(bin_width)
                bin_width = h.BinWidth;
            else
                h.BinWidth = bin_width;
            end
        end
        yl=ylim();
        plot(theta([3,3]),yl([1,2]),tls,'LineWidth',tlw,'DisplayName',sprintf('true I=%.6f crlbSE=%.4f', theta(3), sqrt(crlb(3))));
        title('I-distribution');
        h=legend('location','best');
        h.Interpreter='latex';

        handles.plots.accbg=subplot(3,2,4,'Parent',handles.accPan);
        cla(handles.plots.accbg,'reset');
        hold('on');
        bin_width=[];
        for name_idx = Es            
            name = name_idx{1};
            desc = sprintf('%s $\\bar{bg}$=%.6g SE=%.4f',name, run.(name).est_theta_mean(4), run.(name).est_error_std(4));
            h = histogram(run.(name).est_theta(4,:),'EdgeColor','none','FaceColor',colors.(name),'NumBins',nbins, 'DisplayName',desc,'DisplayStyle','bar','Normalization',norm);
            if isempty(bin_width)
                bin_width = h.BinWidth;
            else
                h.BinWidth = bin_width;
            end
        end
        yl=ylim();
        plot(theta([4,4]),yl([1,2]),tls,'LineWidth',tlw,'DisplayName',sprintf('true bg=%.6f crlbSE=%.4f', theta(4), sqrt(crlb(4))));
        xl=xlim();
        xlim([0,xl(2)]);
        title('bg-distribution');
        h=legend('location','best');
        h.Interpreter='latex';


        handles.plots.accbias=subplot(3,2,5,'Parent',handles.accPan);
        cla(handles.plots.accbias,'reset');
        hold('on');
        bar_data = zeros(obj.nParams, nEs);
        for n=1:nEs
            name = Es{n};
            bar_data(:,n)= mean(run.(Es{n}).est_errors ./ run.(name).est_theta,2);
        end
        bs=bar(bar_data);
        for n=1:nEs
            name = Es{n};
            bs(n).DisplayName=name;
            bs(n).FaceColor=colors.(name);
        end
       
        ylim([-.2,.2]);
        title('Relative Estimator Bias');
        h=legend('location','best');
        h.Interpreter='latex';
    end


    function clearResults()
        if ~resultsPlotted
            return
        end
        ns=fieldnames(handles.plots);
        for n=1:numel(ns)
            h = handles.plots.(ns{n});
            if ishandle(h)
                cla(h);
            end
        end
        resultsPlotted=false;
    end
end

