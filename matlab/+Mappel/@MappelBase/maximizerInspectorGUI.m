% MappelBase.maximizerInspectorGUI()
% This GUI allows for inspection of the trajectroies of the optimization algorithms run in debug mode
% for a single image and given starting point.  The GUI inspects the trajectory of a single optimization
% method run on a single image with a single starting theta.

function guiFig = maximizerInspectorGUI(obj)
    HasSigmaParam = obj.NumParams > 4; % Boolean if this model has a free sigma parameter or not

    gui_name = sprintf('[%s] Maximizer Trajectory Inspector',class(obj));
    
    uH = 25; % unit height for elements
    boarder = 10;%Boarder width around the outside of the gui
    sp = 2; %spacing between elements.
    but_sz = 80; %Button size
    fig_sz = [1400 850]; %figure size
    halfw = 550;
    bot_buf = 400; %buffer on bottom.  Images will start above here
    slider_area_w = 200;
    controlAxH = 210; %Height for the axes for the intensity and sigma controls
    simIntensityAx_pos = [boarder, bot_buf, slider_area_w, controlAxH];
    simSigmaAx_pos =     [boarder, bot_buf+controlAxH, slider_area_w, controlAxH];
    simFig_pos = [slider_area_w, bot_buf-20, halfw-slider_area_w+70, fig_sz(2)-bot_buf-boarder+70];
    estSubAxesDivide =(fig_sz(1)-halfw-20)/3;
    seqAxH = 180;
    estPosAx_pos =       [halfw, boarder+controlAxH, estSubAxesDivide, seqAxH];
    estIntensityAx_pos = [halfw+boarder+estSubAxesDivide, boarder+controlAxH+2*sp, estSubAxesDivide, seqAxH];
    estLLHAx_pos =       [halfw+boarder+2*estSubAxesDivide, boarder+controlAxH+2*sp, estSubAxesDivide, seqAxH];
    estBGAx_pos =    [halfw, boarder, estSubAxesDivide, seqAxH];
    estSigmaAx_pos = [halfw+boarder+estSubAxesDivide, boarder, estSubAxesDivide, seqAxH];
    estFig_pos = [halfw+boarder, bot_buf, fig_sz(1)-halfw-2*boarder, fig_sz(2)-bot_buf-boarder];
    guiFig = figure('Units','pixels','Position',[10 0 fig_sz],'Resize','off',...
                    'MenuBar','none','ToolBar','figure','NumberTitle','off',...
                    'Name',gui_name,'Visible','on');
    smallFontSz=8;
    intensity_bounds = [10,1e6];
    background_bounds = [0.1,100];
    sigma_bounds = [0.5, 5];
    method = 'Newton';
    theta = obj.samplePrior(); %true theta
    sim=[]; %simulated image
    imageBounds = {[.5,double(obj.ImageSize(1))-.5],[.5,double(obj.ImageSize(2))-.5]};
    
    theta_init=theta;
    theta_est=[];
    obsI=[];
    theta_est_rllh=[];
    crlb=[];
    estimator_stats=[];
    evaluated_seq=[]; %Sequence of evaluated
    evaluated_seq_rllh=[];
    bounds_condfidence=0.95;
    post_sample=[]; %Posterior sampling sequence from MCMC
    post_sample_rllh=[]; %Posterior sampling sequence 
    post_candidates=[];
    post_candidates_rllh=[];
    
    nIter =0;
    nBacktracks=0;
    nFunEvals=0;
    nDerEvals=0;
    backtrack_theta=[];
    backtrack_theta_rllh=[];
    backtrack_theta_iter=[];
    backtrack_idxs=[];
    est_im=[];
    uniform_llh=[];
    uniform_bg_mle=[];
    noise_llh=[];
    theoreticalSE=[];
%     observedSE=[];
    estimatorMethodList = [obj.EstimationMethods, {'Posterior2000', 'Posterior10000'}];
    set_theta_init_active=false;
    set_theta_active=false;
        

    %Timer to draw Phase2 (more time consuming) plots only when the estimated image is unchaneged for
    plot_phase2_delay = 0.75; %secs to delay phase2 plotting after last update
    handles.timers.runPhase2 = timer('ExecutionMode','singleShot','StartDelay',plot_phase2_delay,'TimerFcn',@(~,~) plotEstPhase2());
    
    ax.sim_intensity = axes('Units','pixels','Position',simIntensityAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    if HasSigmaParam
        ax.sim_sigma = axes('Units','pixels','Position',simSigmaAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    end
    ax.sim = axes('Units','pixels','Position',simFig_pos,'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');

    ax.est_pos = axes('Units','pixels','Position',estPosAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est_bg = axes('Units','pixels','Position',estBGAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est_intensity = axes('Units','pixels','Position',estIntensityAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    if HasSigmaParam
        ax.est_sigma = axes('Units','pixels','Position',estSigmaAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    end
    ax.est_llh = axes('Units','pixels','Position',estLLHAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est = axes('Units','pixels','Position',estFig_pos,'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');
    

    createControls();
    initializeAxes();
    generateImage();
    createImpointCBs()
   

    function initializeAxes()
        %Sim Position axes
        xlabel(ax.sim,'X (px)');
        ylabel(ax.sim,'Y (px)');
        axis(ax.sim,'tight');
        colormap(ax.sim,hot());
        colorbar(ax.sim);
        GUIBuilder.positionImageAxes(ax.sim,double(obj.ImageSize),simFig_pos,[20 20 20 20]);
        title(ax.sim,'Simulated Image')

        %Sim Intensity axes
        xlabel(ax.sim_intensity,'I (photons)','FontSize',smallFontSz);
        ylabel(ax.sim_intensity,'bg (photons/px)','FontSize',smallFontSz);
        xlim(ax.sim_intensity,intensity_bounds);
        ylim(ax.sim_intensity,background_bounds);
        grid(ax.sim_intensity,'on');
        ax.sim_intensity.YScale='log';
        ax.sim_intensity.XScale='log';
        ax.sim_intensity.XMinorGrid='on';
        ax.sim_intensity.YMinorGrid='on';
        GUIBuilder.positionAxes(ax.sim_intensity,simIntensityAx_pos,[20 20 0 0]);

        if HasSigmaParam
            %Sim Sigma axes
            axes(ax.sim_sigma);
            hold('on');
            plot([sigma_bounds(1),sigma_bounds(2)],[sigma_bounds(1),sigma_bounds(2)],'r-');
            xlabel(ax.sim_sigma,'sigmaX ratio','FontSize',smallFontSz);
            ylabel(ax.sim_sigma,'sigmaY ratio','FontSize',smallFontSz);
            xlim(ax.sim_sigma,sigma_bounds);
            ylim(ax.sim_sigma,sigma_bounds);
            grid(ax.sim_sigma,'on');
            GUIBuilder.positionAxes(ax.sim_sigma,simSigmaAx_pos,[20 20 0 0]);
        end

        %Est Position
        xlabel(ax.est,'X (px)');
        ylabel(ax.est,'Y (px)');
        axis(ax.est,'tight');
        colormap(ax.est,hot());
        colorbar(ax.est);
        GUIBuilder.positionImageAxes(ax.est,double(obj.ImageSize),estFig_pos,[20 20 20 20]);
        title(ax.est,'Estimated Image')
        
        %Est Position axes
        xlabel(ax.est_pos,'step','FontSize',smallFontSz);
        ylabel(ax.est_pos,'position','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_pos,estPosAx_pos,[0 0 0 0]);

        %Est BG axes
        xlabel(ax.est_bg,'step','FontSize',smallFontSz);
        ylabel(ax.est_bg,'Background','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_bg,estBGAx_pos,[0 0 0 0]);
        
        %Est Intensity axes
        xlabel(ax.est_intensity,'step','FontSize',smallFontSz);
        ylabel(ax.est_intensity,'Intensity','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_intensity,estIntensityAx_pos,[0 0 0 0]);

        %Est Sigma axes
        if HasSigmaParam
            xlabel(ax.est_sigma,'step','FontSize',smallFontSz);
            ylabel(ax.est_sigma,'sigma ratio','FontSize',smallFontSz);
            GUIBuilder.positionAxes(ax.est_sigma,estSigmaAx_pos,[0 0 0 0]);
        end

        %LLH axes
        xlabel(ax.est_llh,'step','FontSize',smallFontSz);
        ylabel(ax.est_llh,'LLH','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_llh,estLLHAx_pos,[0 0 0 0]);

        labels = {'Set Theta Position','Set Theta Init Position'};
        CBs = {@setThetaPosition_CB, @setThetaInitPosition_CB};
        handles.simMenu = GUIBuilder.makeContextMenu(labels,CBs);
        

        %Method name selector
        method_pos = [100, bot_buf-uH, 300, uH];
        handles.methodSelect = GUIBuilder.horzLabeledSelectBox(guiFig,'Method',estimatorMethodList,method_pos);
        handles.methodSelect.Value=find(strcmpi(method,obj.EstimationMethods));
        handles.methodSelect.Callback = @methodSelect_CB;
        %Generate Button
        gen_button_pos = [halfw-140-sp, bot_buf-uH, 130, uH];
        handles.genButton = uicontrol('Parent',guiFig,'Style','pushbutton','String','Generate!','Position',gen_button_pos,'Callback',@generate_CB);
        %Fix init toggle
        fix_toggle_pos = [20, bot_buf-uH, but_sz, uH];
        handles.fixToggle = uicontrol('Parent',guiFig,'Style','checkbox','Position',fix_toggle_pos,'String',...
                                      'FixInit','Selected','off','Callback',[]);
    end

    function createControls()
        panel1_pos=[boarder,boarder+30,halfw-20-boarder,bot_buf-boarder];
        hNames={'theta','thetaInit','thetaEst','thetaSE','thetaErr','exit','niter','nback','nfuneval','ndereval'};
        labels={'Theta:','Theta Init:','Estimated Theta:', 'Theta SE:', 'Abs. theta Error:',...
                'Est. Status:','Num. iter:','Num. backtracks:','Num fun eval:','Num der eval:',};
        values={theta,theta_init,theta_est,sqrt(crlb), [],'Invalid','0','0','0','0'};
        CBs={@setThetaEdit_CB,@setThetaInitEdit_CB,[],[],[],[],[],[],[],[]};
        handles.edits = GUIBuilder.labeledHEdits(guiFig, panel1_pos, uH, hNames, labels, values, CBs);
    end

    function createImpointCBs()
        handles.pos_pt = impoint(ax.sim,theta(1),theta(2));
        handles.pos_pt.addNewPositionCallback(@moveThetaPosition_CB);
        handles.pos_pt.setPositionConstraintFcn(makeConstrainToRectFcn('impoint',double([0, obj.ImageSize(1)]),...
                                                                                 double([0, obj.ImageSize(2)])));

        handles.init_pos_pt = impoint(ax.sim,theta_init(1),theta_init(2));
        handles.init_pos_pt.addNewPositionCallback(@moveThetaInitPosition_CB);
        handles.init_pos_pt.setPositionConstraintFcn(makeConstrainToRectFcn('impoint',double([0, obj.ImageSize(1)]), ...
                                                                                      double([0, obj.ImageSize(2)])));
        handles.init_pos_pt.setColor([0.3 0.3 0]);

        handles.intensity_pt = impoint(ax.sim_intensity,theta(3),theta(4));
        handles.intensity_pt.addNewPositionCallback(@moveThetaIntensity_CB);
        handles.intensity_pt.setPositionConstraintFcn(makeConstrainToRectFcn('impoint',intensity_bounds, background_bounds));

        if HasSigmaParam
            handles.sigma_pt = impoint(ax.sim_sigma,theta(5),theta(5));
            handles.sigma_pt.addNewPositionCallback(@moveThetaSigma_CB);
            handles.sigma_pt.setPositionConstraintFcn(@(new_pos) max(0.5,new_pos([1,1]))); 
        end
    end
    
    function methodSelect_CB(~,~)
        method = estimatorMethodList{handles.methodSelect.Value};
        runEstimator();
    end

    function generate_CB(~,~)
        generateImage();
    end

    function setTheta(new_theta)
        disp(['setTheta: ',mat2str(new_theta),' set_theta_active: ', mat2str(set_theta_active)])
        if set_theta_active
            return;
        end
        set_theta_active=true;
        new_theta = obj.boundTheta(new_theta(:));
        new_theta_str = CellFun.arr2str(new_theta);
        if ~strcmp(new_theta_str,handles.edits.theta.String)
            handles.edits.theta.String = CellFun.arr2str(new_theta);
        end
        
        theta = new_theta;
        generateImage();
        if isfield(handles,'pos_pt')
            pos = handles.pos_pt.getPosition();
            if ~all(new_theta(1:2) == pos(:))
                handles.pos_pt.setPosition(new_theta(1:2));
            end
        end
        set_theta_active=false;
    end

    function setThetaInit(new_theta)
        disp(['setThetaInit: ',mat2str(new_theta),' set_theta_init_active: ', mat2str(set_theta_init_active)])
        if set_theta_init_active
            return;
        end
        set_theta_init_active=true;
        new_theta = new_theta(:);
        new_theta_str = CellFun.arr2str(new_theta);
        if ~strcmp(new_theta_str,handles.edits.thetaInit.String)
            handles.edits.thetaInit.String = CellFun.arr2str(new_theta);
        end
        theta_init = new_theta;
        if isfield(handles,'init_pos_pt') 
            pos = handles.init_pos_pt.getPosition();
            if ~all(new_theta(1:2) == pos(:))
                handles.init_pos_pt.setPosition(new_theta(1:2));    
            end
        end
        runEstimator()
        set_theta_init_active=false;
    end
    
    function setThetaPosition_CB(~,~)
        new_theta = theta;
        new_theta(1:2) = ax.sim.CurrentPoint(1,1:2);
        setTheta(new_theta);
    end

    function setThetaInitPosition_CB(~,~)
        new_theta = theta_init;
        new_theta(1:2) = ax.sim.CurrentPoint(1,1:2);
        setThetaInit(new_theta);
    end

    function setThetaInitEdit_CB(~,~)
        setThetaInit(str2num(handles.edits.thetaInit.String)'); %#ok<ST2NM>
    end

    function setThetaEdit_CB(~,~)
        setTheta(str2num(handles.edits.theta.String)'); %#ok<ST2NM>
    end

    function moveThetaPosition_CB(hObj,~)
        new_theta=theta;
        new_theta(2) = hObj(2);
        new_theta(1) = hObj(1);
        setTheta(new_theta);
    end

    function moveThetaInitPosition_CB(hObj,~)
        new_theta=theta_init;
        new_theta(2) = hObj(2);
        new_theta(1) = hObj(1);
        if ~set_theta_init_active
            setThetaInit(new_theta);
        end
    end

    function generateImage()
        sim = obj.simulateImage(theta);
%         sim_stack = obj.simulateImage(theta, Nstack);        
        plotSimFig();
        if ~handles.fixToggle.Value
            theta_init = obj.estimate(sim,'Heuristic');
            setThetaInit(theta_init);
        else
            runEstimator()
        end
    end

    function plotSimFig()
        axes(ax.sim);
        hold('on');
        if isfield(handles,'sim_imsc') && ishandle(handles.sim_imsc)
            delete(handles.sim_imsc);
        end
        handles.sim_imsc=imagesc(imageBounds{:},sim);
        colorbar(ax.sim);
        handles.sim_imsc.UIContextMenu=handles.simMenu;
        xlabel(ax.sim,'X (px)');
        ylabel(ax.sim,'Y (px)');
        ax.sim.XTick = double(0:obj.ImageSize(1));
        ax.sim.XTickLabels = 0:obj.ImageSize(1);
        ax.sim.YTick = double(0:obj.ImageSize(2));
        ax.sim.YTickLabels = 0:obj.ImageSize(2);
        ax.sim.Children = ax.sim.Children(end:-1:1); %make impoint come first in draw order
%         uistack(handles.pos_pt,'top');
    end

    function moveThetaIntensity_CB(hObj,~)
        new_theta=theta;
        new_theta(4) = hObj(2);
        new_theta(3) = hObj(1);
        setTheta(new_theta);
    end

    function moveThetaSigma_CB(hObj,~)
        new_theta=theta;
        new_theta(5) = hObj(1);
        setTheta(new_theta);
    end

    function plotEstPhase1()
        handles.edits.thetaEst.String = CellFun.arr2str(theta_est);
        handles.edits.thetaSE.String = CellFun.arr2str(sqrt(crlb));
        handles.edits.thetaErr.String = CellFun.arr2str(abs(theta-theta_est));
               
        maxc=max(32,max(max(est_im(:)),max(sim(:))));
        axes(ax.est);
        hold('off');
        handles.est_imsc = imagesc(imageBounds{:},est_im);
        hold('on');
        
        colormap(ax.est,hot());
        colorbar(ax.est);
        ax.est.CLim=[0,maxc];
        ax.sim.CLim=[0,maxc];
        ax.est.XTick = double(0:obj.ImageSize(1));
        ax.est.XTickLabels = 0:obj.ImageSize(1);
        ax.est.YTick = double(0:obj.ImageSize(2));
        ax.est.YTickLabels = 0:obj.ImageSize(2);
        title(ax.est,'Estimated Image')
%         xlabel(ax.est,'X (px)');
        ylabel(ax.est,'Y (px)');
        stop(handles.timers.runPhase2);
        start(handles.timers.runPhase2);
%         plotEstPhase2()
    end

    function plotEstPhase2()
        image_ms =5;
        seq_ms =6;
        init_marker='^';
        est_marker='o';
        true_marker='s';
        btrack_ms = 3; % backtrack marker size
        %locations at which to plot backtracks
        backtrackPlotIdxs = backtrack_theta_iter - .5 + (1/(2*(nBacktracks+1))) * (1:nBacktracks);
        if ~isempty(backtrack_theta)
            backtrackRLLH = obj.modelRLLH(sim,backtrack_theta);
        end
        %Estimated Image Axes
        if strncmpi(method,'posterior',9)
            plot_seq = post_sample;
        else
            plot_seq = [evaluated_seq,theta_est];
        end
        seq_len = size(plot_seq,2);
        xs=(1:seq_len);
        N = numel(xs);
        
        %Draw on estimated image
        axes(ax.est);
        hold('on');
        plot(plot_seq(1,1), plot_seq(2,1),'Marker',init_marker,'LineWidth',2,'MarkerEdgeColor', [0 1 0],'MarkerSize',image_ms); %theta_init
        plot(theta(1),theta(2),'Marker',true_marker,'LineWidth',2,'MarkerEdgeColor',[0 0 1],'MarkerSize',image_ms);
        plot(theta_est(1),theta_est(2),'Marker',est_marker,'LineWidth',2,'MarkerEdgeColor',[1 0 1],'MarkerSize',image_ms);
        if(N>1)
            plot(plot_seq(1,:), plot_seq(2,:),'LineWidth',2,'LineStyle','-','Color', [0 1 0]);
        end
        niter = max(1,size(plot_seq,2))-1;
        %plot backtracks
        if ~isempty(backtrack_idxs) 
            for n=1:nBacktracks
                iter = backtrack_theta_iter(n);
                plot([plot_seq(1,iter),backtrack_theta(1,n)],[plot_seq(2,iter),backtrack_theta(2,n)],...
                     'LineWidth',1.5,'LineStyle','-','Color',[1,1,.3]);
            end
        end
        
        
        
        %Plot pos sequence
        axes(ax.est_pos);
        cla(ax.est_pos);
        hold('on');
        hs=[];
        %plot backtracks
        plot(1, plot_seq(1,1),'Marker',init_marker,'MarkerEdgeColor', [1 0 0],'MarkerFaceColor', [.5 0 0],'MarkerSize',seq_ms); %theta_init x
        plot(1, plot_seq(2,1),'Marker',init_marker,'MarkerEdgeColor', [0 0 1],'MarkerFaceColor', [0 0 .5],'MarkerSize',seq_ms); %theta_init y
        plot(seq_len, theta_est(1),'Marker',est_marker,'MarkerEdgeColor', [1 0 0],'MarkerFaceColor', [.5 0 0],'MarkerSize',seq_ms-2); %theta_init x
        plot(seq_len, theta_est(2),'Marker',est_marker,'MarkerEdgeColor', [0 0 1],'MarkerFaceColor', [0 0 .5],'MarkerSize',seq_ms-2); %theta_init y
        plot(seq_len, theta(1),'Marker',true_marker,'MarkerEdgeColor', [.6 0 0 ],'MarkerSize',seq_ms+1,'LineWidth',1.5); %theta_init x
        plot(seq_len, theta(2),'Marker',true_marker,'MarkerEdgeColor', [0 0 .6],'MarkerSize',seq_ms+1,'LineWidth',1.5); %theta_init y
        
        hs(1)=plot(xs,plot_seq(1,:),'r-','LineWidth',1,'DisplayName','Est x');
        hs(2)=plot(xs,plot_seq(2,:),'b-','LineWidth',1,'DisplayName','Est y');
        if ~isempty(backtrackPlotIdxs)
            hs(3)=plot(backtrackPlotIdxs, backtrack_theta(1,:), 'o','MarkerSize',btrack_ms,'MarkerEdgeColor','k','MarkerFaceColor',[1,.3,.3],'DisplayName','Backtracks (x)');
            hs(4)=plot(backtrackPlotIdxs, backtrack_theta(2,:), 'o','MarkerSize',btrack_ms,'MarkerEdgeColor','k','MarkerFaceColor',[.3,.3,1],'DisplayName','Backtracks (y)');
            legend(hs,{'Est x','Est y','Backtracks (x)','Backtracks (y)'},'location','best');
        else
            legend(hs,{'Est x','Est y'},'location','best');
        end
        if N<10
            xticks(xs);
        else
            xticks('auto');
        end

        xlim([1,seq_len]);
        yl = ylim();
        yl_min=min([yl, min(theta_est(1:2)),min(theta(1:2)),min(plot_seq(1,:)),min(plot_seq(2,:))]);
        yl_max=max([yl, max(theta_est(1:2)),max(theta(1:2)),max(plot_seq(1,:)),max(plot_seq(2,:))]);
        if(backtrack_theta)
            yl_min=min([yl_min, min(backtrack_theta(1,:)), min(backtrack_theta(2,:))]);
            yl_max=max([yl_max, max(backtrack_theta(1,:)), max(backtrack_theta(2,:))]);
        end
        ylim([yl_min,yl_max]);
        title('Position Seq.')
        xlabel('Iteration')
        
        %Plot bg sequence
        axes(ax.est_bg)
        hs_bg=[];
        hs_bg(1)=plot(xs,plot_seq(4,:),'m-','LineWidth',1,'DisplayName','Est BG');
        hold('on');
        plot(1, plot_seq(4,1),'Marker',init_marker,'MarkerEdgeColor', [1 0 1],'MarkerFaceColor', [.5 0 .5],'MarkerSize',seq_ms); %theta_init x
        plot(seq_len, theta_est(4),'Marker',est_marker,'MarkerEdgeColor', [1 0 1],'MarkerFaceColor', [.5 0 .5],'MarkerSize',seq_ms-2); %theta_init x
        plot(seq_len, theta(4),'Marker',true_marker,'MarkerEdgeColor', [.6 0 .6],'MarkerSize',seq_ms+1,'LineWidth',1.5); %theta_init x
        yl = ylim();
        ylim([0 1.2*yl(2)]);
        if ~isempty(backtrackPlotIdxs)
            hs_bg(2)=plot(backtrackPlotIdxs, backtrack_theta(4,:), 'o','MarkerSize',btrack_ms,'MarkerEdgeColor','k','MarkerFaceColor',[1,.3,1],'DisplayName','Backtracks (bg)');
            legend(hs_bg,{'Est background','Backtracks (bg)'},'location','best');
        else
            legend(hs_bg,{'Est background'},'location','best');
        end
        if N<10
            xticks(xs);
        else
            xticks('auto');
        end
        xlim([1,seq_len]);
        hold('off');
        title('Background Seq.')
        xlabel('Iteration')
        ylabel('Background')

        %Plot intensity sequence
        axes(ax.est_intensity)
        hs_I=[];
        hs_I(1)=plot(xs,plot_seq(3,:),'r-','LineWidth',1,'DisplayName','Est Intensity sequence');
        hold('on');
        plot(1, plot_seq(3,1),'Marker',init_marker,'MarkerEdgeColor', [1 .5 0],'MarkerFaceColor', [.5 0 .5],'MarkerSize',seq_ms); %theta_init x
        plot(seq_len, theta_est(3),'Marker',est_marker,'MarkerEdgeColor', [1 .5 0],'MarkerFaceColor', [.5 0 .5],'MarkerSize',seq_ms-2); %theta_init x
        plot(seq_len, theta(3),'Marker',true_marker,'MarkerEdgeColor', [1 .5 0],'MarkerSize',seq_ms+1,'LineWidth',1.5); %theta_init x
        
        if ~isempty(backtrackPlotIdxs)
            hs_I(2)=plot(backtrackPlotIdxs, backtrack_theta(3,:), 'o','MarkerSize',btrack_ms,'MarkerEdgeColor','k','MarkerFaceColor',[1,0,0],'DisplayName','Backtracks (I)');
            legend(hs_I,{'Est Intensity','Backtracks (I)'},'location','best');
        else
            legend(hs_I,{'Est Intensity'},'location','best');
        end
  %         yl = ylim();
%         ylim([0 1.2*yl(2)]);
        if N<10
            xticks(xs);
        else
            xticks('auto');
        end
        xlim([1,seq_len]);
        legend('location','best');
        hold('off');
        title('Intensity Seq.')
        xlabel('Iteration')
        ylabel('Intensity')

% 
%         %Plot sigma sequence
%         if obj.NumParams > 4
%             axes(ax.est_sigma)
%             plot(xs,plot_seq(5,:),'b-','DisplayName','Est Sigma');
%             hold('on');
%             yl = ylim();
%             ylim([0 1.2*yl(2)]);
%             plot([0, seq_len-1],[theta(5) theta(5)], 'k--','DisplayName','True Sigma');
%             if ~isempty(backtrackPlotIdxs)
%                 plot(backtrackPlotIdxs, backtrack_theta(5,:), '*','MarkerSize',btrack_ms,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',[0,0,1],'DisplayName','Backtracks (sigma)');
%             end
%             legend('location','best');
%             xticks([xs,seq_len]);
%             xlim([0,seq_len]);
%             hold('off');
%             title('Sigma Seq.')
%         end
% 
        %Plot LLH sequence
        axes(ax.est_llh)
        hold('off');
        hs_llh=[];
        if strncmpi(method,'posterior',9)
            plot(xs,post_sample_rllh,'LineStyle','-','Color',[0.5, 0.5 ,0],'DisplayName','Sampled LLH');
            hold('on');
            plot(xs,post_candidates_rllh,'LineStyle','None','MarkerStyle','o','MarkerFaceColor',[0, 0 ,0.5],'MarkerSize',backtrack_ms,'DisplayName','Candidate LLH');
        else
            hs_llh(1)=plot(xs,[evaluated_seq_rllh; theta_est_rllh],'LineStyle','-','Color',[0.5, 0.5 ,0],'DisplayName','LLH');
        end
        hold('on');
        if ~isempty(backtrackPlotIdxs)
            hs_llh(2)=plot(backtrackPlotIdxs, backtrackRLLH, 'o','MarkerSize',btrack_ms,'Color',[0.5,0.5,0],'DisplayName','Backtracks');
            legend(hs_llh,{'RLLH','Backtracks'},'location','best');
        else
            legend(hs_llh,{'RLLH'},'location','best');
        end
        hold('off');
        if N<10
            xticks(xs);
        else
            xticks('auto');
        end
        xlim([1,seq_len]);
        yl = ylim();
        ylim([0.8*min(yl) 1.2*max(yl)]);
        legend('location','best');
        xlabel('Iteration')
        ylabel('llh')
        title('LLH Seq.')
    end

    function runEstimator()
        if strncmpi(method,'posterior',9)
            count = str2double(method(10:end));
            [post_sample,post_sample_rllh, post_candidates, post_candidates_rllh] =obj.estimatePosteriorDebug(sim,theta_init,count); %#ok<*SETNU>
            theta_est = mean(post_sample,2);
            estimator_stats = struct();
            estimator_stats.total_iterations = count;
            estimator_stats.total_fun_evals = count;
            estimator_stats.total_der_evals = 0;
        else
%             if handles.fixToggle.Value
                [theta_est, theta_est_rllh, obsI, evaluated_seq, evaluated_seq_rllh,estimator_stats] = obj.estimateDebug(sim,method,theta_init);
%             else
%                 [theta_est, theta_est_rllh, obsI, evaluated_seq, evaluated_seq_rllh,estimator_stats] = obj.estimateDebug(sim,method);
%             end
        end
        crlb = obj.CRLB(theta_est);
%         theta_est_stack = obj.estimate(sim_stack,method,theta_init);
        if isfield(estimator_stats,'total_iterations')
            nIter = estimator_stats.total_iterations;
        elseif isfield(estimator_stats,'num_iterations')
            nIter = estimator_stats.num_iterations;
        else
            nIter = 0;
        end
        if isfield(estimator_stats,'total_backtracks')
            nBacktracks = estimator_stats.total_backtracks;
        else
            nBacktracks = 0;
        end
        if isfield(estimator_stats,'total_fun_evals')
            nFunEvals = estimator_stats.total_fun_evals;
        elseif isfield(estimator_stats,'num_fun_evals')
            nFunEvals = estimator_stats.num_fun_evals;
        else
            nFunEvals = 0;
        end
        if isfield(estimator_stats,'total_der_evals')
            nDerEvals = estimator_stats.total_der_evals;
        elseif isfield(estimator_stats,'num_der_evals')
            nDerEvals = estimator_stats.num_der_evals;
        else
             nDerEvals = 0;
        end
        if isfield(estimator_stats,'backtrack_idxs') && ~isempty(estimator_stats.backtrack_idxs)
            backtrack_idxs = estimator_stats.backtrack_idxs;
            backtrack_theta = evaluated_seq(:,logical(backtrack_idxs));
            backtrack_theta_rllh = evaluated_seq_rllh(logical(backtrack_idxs));
            
            evaluated_seq=evaluated_seq(:,~backtrack_idxs);
            evaluated_seq_rllh=evaluated_seq_rllh(~backtrack_idxs);
            assert(nBacktracks == size(backtrack_theta,2));
            backtrack_theta_iter = zeros(1,nBacktracks); %Iteration to which this backtrack corresponds 0=init step
            iter=0;
            idx=1;
            for n=1:nBacktracks
                while ~backtrack_idxs(idx)
                    idx = idx+1;
                    iter = iter+1;
                end
                backtrack_theta_iter(n)=iter;                
                idx = idx+1;
            end
        else
            backtrack_idxs = [];
            backtrack_theta=[];
            backtrack_theta_iter=[];
        end
        if isfield(estimator_stats,'num_exit_error') && estimator_stats.num_exit_error
            handles.edits.exit.String = 'Error';
        elseif isfield(estimator_stats,'num_exit_function_value') && estimator_stats.num_exit_function_value
            handles.edits.exit.String = 'OK: Function value change.';
        elseif isfield(estimator_stats,'num_exit_grad_ratio') && estimator_stats.num_exit_grad_ratio
            handles.edits.exit.String = 'OK: Grad ratio.';
        elseif isfield(estimator_stats,'num_exit_max_backtracks') && estimator_stats.num_exit_max_backtracks
            handles.edits.exit.String = '?: Max backtracks.';
        elseif isfield(estimator_stats,'num_exit_max_iter') && estimator_stats.num_exit_max_iter
            handles.edits.exit.String = '?: Max Iter.';
        elseif isfield(estimator_stats,'num_exit_step_size') && estimator_stats.num_exit_step_size
            handles.edits.exit.String = 'OK: Step size ratio.';
        elseif isfield(estimator_stats,'num_exit_trust_region_radius') && estimator_stats.num_exit_trust_region_radius
            handles.edits.exit.String = 'OK: Trust region radius.';
        elseif isfield(estimator_stats,'num_exit_model_improvement') && estimator_stats.num_exit_model_improvement
            handles.edits.exit.String = 'OK: Predicted Model Improvement.';
        elseif isfield(estimator_stats,'num_exit_success') && estimator_stats.num_exit_success
            handles.edits.exit.String = 'OK: Success.';
        else
            handles.edits.exit.String = 'Unset';            
        end
        handles.edits.niter.String = mat2str(nIter);
        handles.edits.nback.String = mat2str(nBacktracks);
        handles.edits.nfuneval.String = mat2str(nFunEvals);
        handles.edits.ndereval.String = mat2str(nDerEvals);
        
        [uniform_llh, uniform_bg_mle] = obj.uniformBackgroundModelLLH(sim);
        est_im = obj.modelImage(theta_est);
        noise_llh = obj.noiseBackgroundModelLLH(sim);
        theoreticalSE = sqrt(crlb);
        plotEstPhase1();
    end

%     function runSimulation()
%         observedSE = obj.evaluateEstimatorAt(method,theta,1e4,theta_init);
%     end

end
