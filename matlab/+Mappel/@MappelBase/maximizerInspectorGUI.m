% MappelBase.maximizerInspectorGUI()
% This GUI allows for inspection of the trajectroies of the optimization algorithms run in debug mode
% for a single image and given starting point.  The GUI inspects the trajectory of a single optimization
% method run on a single image with a single starting theta.

function guiFig = maximizerInspectorGUI(obj)
    gui_name = sprintf('[%s] Maximizer Trajectory Inspector',class(obj));
    
    uH = 25; % unit height for elements
    boarder = 10;%Boarder width around the outside of the gui
    sp = 2; %spacing between elements.
    but_sz = 80; %Button size
    fig_sz = [1400 800]; %figure size
    halfw = 550;
    bot_buf = 350; %buffer on bottom.  Images will start above here
    slider_area_w = 200;
    controlAxH = 190; %Hiehgt for the axes for the intensity and sigma controls
    simIntensityAx_pos = [boarder, bot_buf, slider_area_w, controlAxH];
    simSigmaAx_pos =     [boarder, bot_buf+controlAxH, slider_area_w, controlAxH];
    simFig_pos = [boarder+slider_area_w, bot_buf, halfw-2*boarder-slider_area_w, fig_sz(2)-bot_buf-boarder];
    estSubAxesDivide =(fig_sz(1)-halfw-40)/3;
    estPosAx_pos = [halfw+boarder, boarder+controlAxH+2*sp, estSubAxesDivide, controlAxH];
    estBGAx_pos =  [halfw+boarder, boarder, estSubAxesDivide, controlAxH];
    estIntensityAx_pos = [halfw+2*boarder+estSubAxesDivide, boarder+controlAxH+2*sp, estSubAxesDivide, controlAxH];
    estSigmaAx_pos =     [halfw+2*boarder+estSubAxesDivide, boarder, estSubAxesDivide, controlAxH];
    estLLHAx_pos = [halfw+3*boarder+2*estSubAxesDivide, boarder+controlAxH+2*sp, estSubAxesDivide, controlAxH];
    estFig_pos = [halfw+boarder, bot_buf, fig_sz(1)-halfw-2*boarder, fig_sz(2)-bot_buf-boarder];
    guiFig = figure('Units','pixels','Position',[10 0 fig_sz],'Resize','off',...
                    'MenuBar','none','ToolBar','figure','NumberTitle','off',...
                    'Name',gui_name,'Visible','on');
    smallFontSz=8;
    intensity_bounds = [10,1e6];
    background_bounds = [0.1,100];
    sigma_bounds = [0.5, 5];
    method = 'Newton';
    theta = obj.samplePrior();
    sim=[];
    Nstack=1e4;
    sim_stack=[];
    theta_est_stack=[];
    theta_init=theta;
    theta_est=[];
    crlb=[];
    emitter_llh=[];
    estimator_stats=[];
    
    theta_seq=[];
    llh_seq=[];
    nIter =0;
    nBacktracks=0;
    nFunEvals=0;
    nDerEvals=0;
    backtrack_theta=[];
    backtrack_theta_iter=[];
    backtrack_idxs=[];
    est_im=[];
    uniform_llh=[];
    uniform_bg_mle=[];
    noise_llh=[];
    theoreticalSE=[];
    observedSE=[];
    estimatorMethodList = [obj.EstimationMethods, {'Posterior2000', 'Posterior10000'}];
        
    status.plotPhase2=false;

    handles=[];
    
    ax.sim_intensity = axes('Units','pixels','Position',simIntensityAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.sim_sigma = axes('Units','pixels','Position',simSigmaAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.sim = axes('Units','pixels','Position',simFig_pos,'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');

    ax.est_pos = axes('Units','pixels','Position',estPosAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est_bg = axes('Units','pixels','Position',estBGAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est_intensity = axes('Units','pixels','Position',estIntensityAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est_sigma = axes('Units','pixels','Position',estSigmaAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est_llh = axes('Units','pixels','Position',estLLHAx_pos,'Box','on','BoxStyle','full','FontSize',smallFontSz);
    ax.est = axes('Units','pixels','Position',estFig_pos,'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');
    imageBounds = {[.5,obj.imsize(2)-.5],[.5,obj.imsize(1)-.5]};


    createControls();
    initializeAxes();
    generateImage();
    runEstimator();
    initializeTimers();

    function initializeAxes()
        %Sim Position axes
        xlabel(ax.sim,'X (px)');
        ylabel(ax.sim,'Y (px)');
        axis(ax.sim,'tight');
        colormap(ax.sim,hot());
        colorbar(ax.sim);
        GUIBuilder.positionImageAxes(ax.sim,obj.imsize,simFig_pos,[20 20 20 20]);

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

        %Sim Sigma axes
        axes(ax.sim_sigma);
        hold('on');
        plot([sigma_bounds(1),sigma_bounds(2)],[sigma_bounds(1),sigma_bounds(2)],'r-');
        xlabel(ax.sim_sigma,'sigmaX ratio','FontSize',smallFontSz);
        ylabel(ax.sim_sigma,'sigmaY ratio','FontSize',smallFontSz);
        xlim(ax.sim_sigma,sigma_bounds);
        ylim(ax.sim_sigma,sigma_bounds);
        grid(ax.sim_sigma,'on');
%         axis(ax.sim_sigma,'square');
        GUIBuilder.positionAxes(ax.sim_sigma,simSigmaAx_pos,[20 20 0 0]);


        %Est Position
        xlabel(ax.est,'X (px)');
        ylabel(ax.est,'Y (px)');
        axis(ax.est,'tight');
        colormap(ax.est,hot());
        colorbar(ax.est);
        GUIBuilder.positionImageAxes(ax.est,obj.imsize,estFig_pos,[20 20 20 20]);

        %Est Position axes
        xlabel(ax.est_pos,'step','FontSize',smallFontSz);
        ylabel(ax.est_pos,'Background','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_pos,estPosAx_pos,[20 20 0 0]);

        %Est BG axes
        xlabel(ax.est_bg,'step','FontSize',smallFontSz);
        ylabel(ax.est_bg,'Background','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_bg,estBGAx_pos,[20 20 0 0]);
        
        %Est Intensity axes
        xlabel(ax.est_intensity,'step','FontSize',smallFontSz);
        ylabel(ax.est_intensity,'Intensity','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_intensity,estIntensityAx_pos,[20 20 0 0]);

        %Est Sigma axes
        xlabel(ax.est_sigma,'step','FontSize',smallFontSz);
        ylabel(ax.est_sigma,'sigma ratio','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_sigma,estSigmaAx_pos,[20 20 0 0]);

        %LLH axes
        xlabel(ax.est_llh,'step','FontSize',smallFontSz);
        ylabel(ax.est_llh,'LLH','FontSize',smallFontSz);
        GUIBuilder.positionAxes(ax.est_llh,estLLHAx_pos,[20 20 0 0]);

%         labels = {'SetPosition','SetInitPosition'};
%         CBs = {@setPosition, @setInitPosition};
%         handles.simMenu = GUIBuilder.makeContextMenu(labels,CBs);
        

        handles.pos_pt = impoint(ax.sim,theta(1),theta(2));
        handles.pos_pt.addNewPositionCallback(@moveThetaPosition_CB);
        handles.pos_pt.setPositionConstraintFcn(makeConstrainToRectFcn('impoint',[0, obj.imsize(1)],[0, obj.imsize(2)]));

        handles.init_pos_pt = impoint(ax.sim,theta(1),theta(2));
        handles.init_pos_pt.addNewPositionCallback(@moveThetaInitPosition_CB);
        handles.init_pos_pt.setPositionConstraintFcn(makeConstrainToRectFcn('impoint',[0, obj.imsize(1)],[0, obj.imsize(2)]));
        handles.init_pos_pt.setColor([0.3 0.3 0]);

        handles.intensity_pt = impoint(ax.sim_intensity,theta(3),theta(4));
        handles.intensity_pt.addNewPositionCallback(@moveThetaIntensity_CB);
        handles.intensity_pt.setPositionConstraintFcn(makeConstrainToRectFcn('impoint',intensity_bounds, background_bounds));

        if obj.nParams>4
            handles.sigma_pt = impoint(ax.sim_sigma,theta(5),theta(5));
            handles.sigma_pt.addNewPositionCallback(@moveThetaSigma_CB);
            handles.sigma_pt.setPositionConstraintFcn(@(new_pos) max(0.5,new_pos([1,1]))); 
        end
        %Generate Button
        method_pos = [slider_area_w, bot_buf-uH, 300, uH];
        handles.methodSelect = GUIBuilder.horzLabeledSelectBox(guiFig,'Method',estimatorMethodList,method_pos);
        handles.methodSelect.Value=find(strcmpi(method,obj.EstimationMethods));
        handles.methodSelect.Callback = @methodSelect_CB;
        gen_button_pos = [halfw-150-sp, bot_buf-uH, 150, uH];
        handles.genButton = uicontrol('Parent',guiFig,'Style','pushbutton','String','Generate!','Position',gen_button_pos,'Callback',@generate_CB);
        fix_toggle_pos = [halfw-450-sp, bot_buf-uH, 80, uH];
        handles.fixToggle = uicontrol('Parent',guiFig,'Style','checkbox','Position',fix_toggle_pos,'String',...
                                      'FixInit','Selected','off','Callback',[]);
    end

    function createControls()
        panel1_pos=[boarder,boarder+100,halfw-100-boarder,bot_buf-boarder];
        hNames={'theta','thetaInit','thetaEst','thetaSE','thetaErr'};
        labels={'Theta:','Theta Init:','Estimated Theta:', 'Theta SE:', 'Abs. theta Error:'};
        values={theta,theta_init,theta_est,sqrt(crlb), [] };
        CBs={@setThetaEdit_CB,@setThetaInitEdit_CB,[],[],[]};
        handles.edits = GUIBuilder.labeledHEdits(guiFig, panel1_pos, uH, hNames, labels, values, CBs);
    end

    function initializeTimers()
        handles.timers.runPhase2 = timer('BusyMode','drop','ExecutionMode','fixedSpacing','Period',1,...
                                         'TimerFcn',@timerPhase2_CB);
%         start(handles.timers.runPhase2);
    end

%     function setPosition(~,evt)
%         ax.sim.CurrentPoint
%     end

    function methodSelect_CB(~,~)
        method = estimatorMethodList{handles.methodSelect.Value};
        runEstimator();
    end

    function generate_CB(~,~)
        generateImage();
    end

%     function fixToggle_CB(~,~)
%     end

    function setTheta(new_theta)
        theta = new_theta;
        handles.edits.theta.String = arr2str(theta(:)');
        handles.pos_pt.setPosition(theta(1:2));
        generateImage();
    end

    function setThetaInit(new_theta)
        theta_init = new_theta;
        handles.edits.thetaInit.String = arr2str(theta_init(:)');
        handles.init_pos_pt.setPosition(theta_init(1:2));
        runEstimator();
    end
    
    function setThetaInitEdit_CB(~,~)
        setThetaInit(str2num(handles.edits.thetaInit.String)'); %#ok<ST2NM>
    end

    function setThetaEdit_CB(~,~)
        setTheta(str2num(handles.edits.theta.String)'); %#ok<ST2NM>
    end

    function generateImage()
        sim = obj.simulateImage(theta);
%         sim_stack = obj.simulateImage(theta, Nstack);        
        plotSimFig();
        if ~handles.fixToggle.Value
            theta_init = obj.estimate(sim,'Heuristic');
        end
        setThetaInit(theta_init);
    end

    function plotSimFig()
        axes(ax.sim);
        hold('on');
        if isfield(handles,'sim_imsc') && ishandle(handles.sim_imsc)
            delete(handles.sim_imsc);
        end
        handles.sim_imsc=imagesc(imageBounds{:},sim);
        colorbar(ax.sim);
%         H.UIContextMenu=handles.simMenu;
        xlabel(ax.sim,'X (px)');
        ylabel(ax.sim,'Y (px)');
        ax.sim.Children = ax.sim.Children(end:-1:1); %make impoint come first in draw order
%         uistack(handles.pos_pt,'top');
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
        setThetaInit(new_theta);
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

    function plotEstFig()
        plotEstPhase1()
        status.runPhase2=true;
        plotEstPhase2()
    end

    function plotEstPhase1()
        handles.edits.thetaEst.String = arr2str(theta_est);
        handles.edits.thetaSE.String = arr2str(sqrt(crlb));
        handles.edits.thetaErr.String = arr2str(abs(theta-theta_est));
               
        axes(ax.est);
        hold('off');
        imagesc(imageBounds{:},est_im);
        hold('on');
        maxc=max(32,max(max(est_im(:)),max(sim(:))));
        colorbar(ax.est);
        ax.est.CLim=[0,maxc];
        ax.sim.CLim=[0,maxc];
    end

    function timerPhase2_CB(~,~)
        plotEstPhase2()
    end

    function plotEstPhase2()
        if ~status.runPhase2
            return
        end
        btrack_ms = 5; % backtrack marker size
        delta = 1/(2*(nBacktracks+1));
            
        seq_len = size(theta_seq,2);
        xs=(1:seq_len)-1;       
        axes(ax.est);
        hold('on');
        xlabel(ax.est,'X (px)');
        ylabel(ax.est,'Y (px)');
        plot(theta(1),theta(2),'Marker','s','MarkerEdgeColor',[0 0 1]);
        plot(theta_seq(1,:), theta_seq(2,:),'LineWidth',2,'LineStyle','-','Color', [0 1 0]);
        plot(theta_est(1),theta_est(2),'Marker','o','MarkerEdgeColor',[0 0 1]);
%         scatter(theta_est_stack(1,:),theta_est_stack(2,:),'k.');
        niter = max(1,size(theta_seq,2))-1;
        %plot backtracks
        if ~isempty(backtrack_idxs) 
            for n=1:nBacktracks
                iter = backtrack_theta_iter(n);
                plot([theta_seq(1,iter),backtrack_theta(1,n)],[theta_seq(2,iter),backtrack_theta(2,n)],...
                     'LineWidth',2,'LineStyle','-','Color',[1,1,.3]);
            end
        end
        
        %Plot pos sequence
        axes(ax.est_pos);
        cla(ax.est_pos);
        plot(xs,theta_seq(1,:),'r-','DisplayName','Est X');
        hold('on');
        plot(xs,theta_seq(2,:),'b-','DisplayName','Est Y');
        yl = ylim();
        ylim([0 1.2*yl(2)]);
        plot([0, seq_len-1],[theta(1) theta(1)], 'r--','DisplayName','True X');
        plot([0, seq_len-1],[theta(2) theta(2)], 'b--','DisplayName','True Y');
        legend('location','best');
        %plot backtracks
        for n=1:nBacktracks
            iter = backtrack_theta_iter(n);
            h=plot(iter-0.5+delta*n, backtrack_theta(1,n), '*','MarkerSize',btrack_ms,'Color',[1,.3,.3]);
            h.Annotation.LegendInformation.IconDisplayStyle='off';
            h=plot(iter-0.5+delta*n, backtrack_theta(2,n), '*','MarkerSize',btrack_ms,'Color',[.3,.3,1]);
            h.Annotation.LegendInformation.IconDisplayStyle='off';
        end
        ylim([0,max(obj.imsize)]);

        %Plot bg sequence
        axes(ax.est_bg)
        plot(xs,theta_seq(4,:),'m-','DisplayName','Est BG');
        hold('on');
        yl = ylim();
        ylim([0 1.2*yl(2)]);
        plot([0, seq_len-1],[theta(4) theta(4)], 'm--','DisplayName','True BG');
        legend('location','best');
        hold('off');

        %Plot intensity sequence
        axes(ax.est_intensity)
        plot(xs,theta_seq(3,:),'r-','DisplayName','Est Intensity sequence');
        hold('on');
        plot([0, niter],[theta(3) theta(3)], 'k--','DisplayName','True Intensity');
        plot(niter,theta_est(3),'ro','DisplayName','Est Intensity');
        yl = ylim();
        ylim([0 1.2*yl(2)]);
        legend('location','best');
        hold('off');

        %Plot sigma sequence
        if obj.nParams > 4
            axes(ax.est_sigma)
            plot(xs,theta_seq(5,:),'b-','DisplayName','Est Sigma');
            hold('on');
            yl = ylim();
            ylim([0 1.2*yl(2)]);
            plot([0, seq_len-1],[theta(5) theta(5)], 'k--','DisplayName','True Sigma');
            legend('location','best');
            hold('off');
        end

        %Plot LLH sequence
        axes(ax.est_llh)
        hold('off');
        plot(xs,llh_seq,'LineStyle','-','Color',[0.5, 0.5 ,0],'DisplayName','LLH');
        yl = ylim();
        ylim([1.2*yl(1) 0.9*yl(2)]);
        legend('location','best');
        status.runPhase2=false;
    end

    function runEstimator()
        if strncmpi(method,'posterior',9)
            count = str2double(method(10:end));
            [theta_est, cov, theta_seq, llh_seq]=obj.estimatePosteriorDebug(sim,count,theta_init);
            crlb = diag(cov);
            estimator_stats = struct();
            estimator_stats.total_iterations=count;
            estimator_stats.total_fun_evals=count;
            estimator_stats.total_der_evals=0;
        else
            [theta_est, crlb, emitter_llh, estimator_stats, theta_seq, llh_seq] = obj.estimateDebug(sim,method,theta_init);
        end
%         theta_est_stack = obj.estimate(sim_stack,method,theta_init);
        if isfield(estimator_stats,'total_iterations')
            nIter = estimator_stats.total_iterations;
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
        else
            nFunEvals = 0;
        end
        if isfield(estimator_stats,'total_total_der_evals')
            nDerEvals = estimator_stats.total_der_evals;
        else
             nDerEvals = 0;
        end
        if isfield(estimator_stats,'backtrack_idxs') && ~isempty(estimator_stats.backtrack_idxs)
            backtrack_idxs = estimator_stats.backtrack_idxs;
            backtrack_theta = theta_seq(:,logical(backtrack_idxs));
            backtrack_theta_llh = llh_seq(logical(backtrack_idxs));
            theta_seq=theta_seq(:,~backtrack_idxs);
            llh_seq=llh_seq(~backtrack_idxs);
            nbacktracks = size(backtrack_theta,2);
            backtrack_theta_iter = zeros(1,nbacktracks);
            iter=0;
            idx=1;
            for n=1:nbacktracks
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
        [uniform_llh, uniform_bg_mle] = obj.uniformBackgroundModelLLH(sim);
        est_im = obj.modelImage(theta_est);
        noise_llh = obj.noiseBackgroundModelLLH(sim);
        theoreticalSE = sqrt(crlb);
        plotEstFig();
    end

    function runSimulation()
        observedSE = obj.evaluateEstimatorAt(method,theta,1e4,theta_init);
    end

end
