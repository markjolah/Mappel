%
% QDBlinkSim.m  - A class for simulating and plotting QD blinking
%
% Mark J. Olah (mjo@cs.unm.edu)
% 04-30-2014

classdef QDBlinkSim < handle


    properties
        frame_time=1.0E-3; %capture time for a single frame (s)
        num_frames=8; %number of sequential frames caputured in an image: log-log slope will be -(1+alpha)
        burnin_time;
        tmin_on=1.0E-4; % The minimum time scale for 'on' state lifetimes
        alpha_on=0.64; % The scaling exponent for the 'on' state lifetime distribution: log-log slope will be -(1+alpha)
        tmin_off=1.0E-4; % The minimum time scale for 'on' state lifetimes
        alpha_off=0.64; % The scaling exponent for the 'on' state lifetime distribution        
    end
    properties (Access=public)
        pre_comp_size=100000;
        times;
        cumtimes;
        times_idx;

        ensemble_size=1000;        
        duty_ratios;
    end

    methods
        function obj=QDBlinkSim(ensemble_size, frame_time, num_frames, burnin_time)
            if nargin<4
                burnin_time=1E4*max(obj.tmin_on,obj.tmin_off);
            end
            if nargin<3
                num_frames=8;
            end
            if nargin<2
                frame_time=1e-3;
            end
            obj.ensemble_size=ensemble_size;
            obj.precompute_times();
            obj.burnin_time=burnin_time;
            obj.frame_time=frame_time;
            obj.num_frames=num_frames;
            fprintf('Begin Burnin Time:%.3g s\n',obj.burnin_time);
            obj.precompute_duty_ratios();
        end

        function [state_seq,transition_times]=simulateQD(obj, max_time, start_state)
            if nargin<3 || start_state<0
                start_state=int64(rand(1));
            end
            transition_times=obj.sample_transition_times(max_time,start_state);
            if isempty(transition_times) %There were no transitions we stay in the same state
                transition_times=[0 max_time];
                state_seq=[start_state start_state];
            elseif transition_times(end)==max_time %There was a transition right at the end
                transition_times=[0 transition_times];
                state_seq=ones(1,length(transition_times));
                state_seq(start_state+1:2:end)=0;
            else %Normal case.  Last state lasts from final transition_time to max_time so last state is repeated
                transition_times=[0 transition_times max_time];
                state_seq=ones(1,length(transition_times));
                state_seq(start_state+1:2:end)=0;
                state_seq(end)=state_seq(end-1);
            end
        end

        

        function mean_state=sampleEnsembleMean(obj, sample_times, ensemble_size, start_state)
            if nargin<4
                start_state=-1;
            end
            nts=length(sample_times);
            max_time=sample_times(end);
            mean_state=zeros(1,nts);
            for i=1:ensemble_size
                [state_seq,transition_times]=obj.simulateQD(max_time, start_state);
                k=1;
                for j=1:nts
                    while transition_times(k+1)<=sample_times(j) && k+1<length(transition_times)
                        k=k+1;
                    end
                    mean_state(j)=mean_state(j)+state_seq(k);
                end
            end
            mean_state=mean_state./ensemble_size;
        end

        function [duty_ratios,state_seq,transition_times]=simulateImage(obj, frame_time, num_frames, burnin)
            if nargin<4
                burnin=obj.burnin_time;
            end
            if nargin<3
                num_frames=obj.num_frames;
            end
            if nargin<2
                frame_time=obj.frame_time;
            end
            max_time=burnin+frame_time*num_frames;
            [state_seq,transition_times]=obj.simulateQD(max_time);
            duty_ratios=zeros(num_frames,1);
            state_idx=find(transition_times>burnin,1)-1;
            for frame_idx=1:num_frames
                frame_begin=burnin+frame_time*(frame_idx-1);
                frame_end=burnin+frame_time*frame_idx;
                while true
                    current_state_begin=transition_times(state_idx);
                    current_state_end=transition_times(state_idx+1);
                    if state_seq(state_idx) %only add contribution from 'on' states
                        current_state_duration=min(frame_end,current_state_end)-max(frame_begin, current_state_begin);
                        duty_ratios(frame_idx)=duty_ratios(frame_idx)+current_state_duration;
                    end
                    if frame_end<=current_state_end
                        break;
                    else
                        state_idx=state_idx+1;
                    end
                end
            end
            duty_ratios=duty_ratios./frame_time;
        end

        function f=plotQDStates(obj, max_time,start_state)
            if nargin<3
                start_state=int64(rand(1));
            end
            f=figure(1);
            clf;
            ax=axes();
            [state_seq,transition_times]=obj.simulateQD(max_time, start_state);

            set(ax,'YTick',[0,1]);
            hold on;
            stairs(transition_times, state_seq,'ok-','markerfacecolor','y');
            ylim([-0.1,1.1]);
            xlim([-0.1,max_time+0.1]);
            xlabel('time (s)');
            ylabel('state');
            hold off;
        end


        function f=plotSingleFrameDutyRates(obj,frame_time, num_frames, burnin, select_interesting)
            if nargin<5
                select_interesting=1;
            end
            if nargin<4
                burnin=obj.burnin_time;
            end
            if nargin<3
                num_frames=obj.num_frames;
            end
            if nargin<2
                frame_time=obj.frame_time;
            end
            
            f=figure();
%             ax=axes();
            total_time=frame_time*num_frames;
            [duty_ratios,state_seq,transition_times]=obj.simulateImage(frame_time, num_frames, burnin);
            while select_interesting && (mean(duty_ratios)<0.5 || mean(duty_ratios)>0.99)
                [duty_ratios,state_seq,transition_times]=obj.simulateImage(frame_time, num_frames, burnin);
            end
            t0=find(transition_times > burnin);
            assert(transition_times(end)==burnin+total_time);
            tt=[0, transition_times(t0:end)-burnin];
            ss=state_seq(t0-1:end);
            
            hold on;
            stairs(tt, ss,'ok-','markerfacecolor','y');
            for n=1:num_frames
                patch([(n-1) n n (n-1)]*frame_time,...
                      [0, 0, duty_ratios(n),  duty_ratios(n)],ones(1,4),...
                      'FaceColor',[1, 0.3, 0], 'FaceAlpha',0.1, 'EdgeColor','none');
                text((n-1+0.1)*frame_time,0.5,sprintf('%.2f',duty_ratios(n)),'BackgroundColor',[1 0.9 1]);
                plot([1 1]*frame_time*n,[-0.1 1.1],'b:','LineWidth',0.5);
            end
            plot([0 0]*frame_time*n,[-0.1 1.1],'b:','LineWidth',0.5);
            ylim([-0.1,1.1]);
            xlim([-0.1*total_time,total_time*1.1]);
            xlabel('time (s)');
            ylabel('Duty Ratio');
            title('Simulated Duty Ratio');
            hold off;
        end


        function f=plotEnsembleMean(obj, max_time, ensemble_size)
            f=figure();
            ax=axes();
            nts=1000;
            sample_times=linspace(0,max_time,nts);
            mean_state=obj.sampleEnsembleMean(sample_times, ensemble_size);
            set(ax,'YTick',[0,1]);
            hold on;
            plot(sample_times,0.5*ones(1,nts),'r','linewidth',2);
            plot(sample_times,mean_state,'k-','linewidth',2);
            ylim([-0.1,1.1]);
            xlim([-0.1,max_time+0.1]);
            xlabel('time (s)');
            ylabel('state');
            hold off;
        end

        function f=plotDutyRatioDist(obj)   
            nbins=100;
            bin_edges=linspace(0,1+1e-6,nbins+1);
            bin_centers=0.5*(bin_edges(2:end)+bin_edges(1:end-1));
            f=figure();
            axes('XScale','linear','YScale','log');
            hold on
            colormap('jet')
            cmap=colormap;
            ps=zeros(1,obj.num_frames);
            lw=3;
            for col=1:obj.num_frames
                N=histc(obj.duty_ratios(col,:),bin_edges);
                N(end-1)=N(end-1)+N(end);
                N(end)=[];
                assert(sum(N)==obj.ensemble_size);
                name=sprintf('Col:%i',col);
                color=cmap(int64(col/obj.num_frames*64),:);
                ps(col)=plot(bin_centers,N/obj.ensemble_size,'linestyle','-','Color', color, 'LineWidth', lw,'DisplayName',name);

            end
%             beta_a=1.25E-3;
%             beta_b=1.25E-3;
%             nxs=linspace(0,1,10*nbins);
%             ps(num_frames+1)=plot(nxs,betapdf(nxs,beta_a,beta_b),'Color','k','LineStyle','--','LineWidth',2, 'DisplayName', sprintf('Beta(%g,%g)',beta_a,beta_b));
            t=sprintf('\\alpha^+=%g, \\quad t^+_{\\min}=%g, \\alpha^-=%g, \\quad t^-_{\\min}=%g, \\quad T_f=%g, \\quad N=%i,\\quad s=%i',...
                        obj.alpha_on,obj.tmin_on,  obj.alpha_off,obj.tmin_off,obj.frame_time,obj.ensemble_size,obj.num_frames);
            title(['$' t '$'],'interpreter','latex')
            xlabel('Duty Ratio');
            ylabel('Probability Density');
            legend(ps,'Location','North');
            hold off
        end

        function f=plotConditionalDist2D(obj, cond_col, sampled_col)
            nbins=100;
            f=figure();
            clf;
            axes('XScale','linear','YScale','linear','ZScale','log');
            hold on;
            
            cnt=hist3(obj.duty_ratios(1:2,:)'./obj.ensemble_size,[nbins nbins]);
            min_val=min(cnt(:));
            max_val=max(cnt(:));
            hist3(obj.duty_ratios(1:2,:)'./obj.ensemble_size,[nbins nbins]);
            colorbar();
            colormap(jet(256));
            caxis([min_val, max_val]);
            xlabel(sprintf('Col:%i',cond_col));
            ylabel(sprintf('Col:%i',sampled_col));
            t=sprintf('\\mathrm{P}(\\theta_{D_%i}, \\theta_{D_%i})', sampled_col, cond_col);
            title(['$' t '$'],'interpreter','latex');
            colorbar();
            set(f,'renderer','opengl');
            set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
            hold off;
        end

        function f=plotConditionalDist1D(obj, cond_col, cond_val)
            nbins=20;
            f=figure();
            axes('XScale','linear','YScale','log');
            hold on;
            cond_val_idx=floor(cond_val*nbins-1e-6);
            min_cond_val=cond_val_idx/nbins;
            max_cond_val=(cond_val_idx+1)/nbins;
            ok=obj.duty_ratios(cond_col,:)>=min_cond_val & obj.duty_ratios(cond_col,:)<=max_cond_val+1e-6;
            colormap('jet');
            cmap=colormap;
            ps=zeros(1,obj.num_frames-1);
            lw=3;
            for col=1:obj.num_frames
                if col==cond_col
                    continue;
                end
                col_duty_ratios=obj.duty_ratios(col,ok);
                [N,X]=hist(col_duty_ratios,nbins);
                name=sprintf('Col:%i',col);
                color=cmap(int64(col/obj.num_frames*64),:);
                ps(col)=plot(X,N./length(col_duty_ratios),'linestyle','-','Color', color, 'LineWidth', lw,'DisplayName',name);
            end
            ps(cond_col)=[];
            xlabel('Duty Ratio');
            ylabel('Probability');
            t=sprintf('\\mathrm{P}(\\theta_{D_i} \\mid %f\\leq \\theta_{D_%i}\\leq %f)', min_cond_val, cond_col, max_cond_val);
            title(['$' t '$'],'interpreter','latex');
            legend(ps,'Location','North');
            hold off;
        end

        function f=plotTimesHist(obj)
            %Check that the on and off times histograms match the theoretical
            %Pareto pdf.
            xs=logspace(-5,2,100);
            on_times=obj.times(1:2:end);
            off_times=obj.times(2:2:end);
            n=obj.pre_comp_size/2;
            Non=histc(on_times,xs);
            Noff=histc(off_times,xs);
            e_tmin_on=min(on_times);
            e_tmin_alpha=n./sum(log(on_times)-log(e_tmin_on));
            fprintf('On: etmin=%f ealpha=%f\n',e_tmin_on, e_tmin_alpha);
            f=figure();
            ax=axes();
            hold on;
            set(ax,'XScale','log')
            set(ax,'YScale','log')
            xmids=.5*(xs(2:end)+xs(1:end-1));
            xwidths=xs(2:end)-xs(1:end-1);
            display_name=sprintf('Simulated On Times PDF: [Tmin=%.3g alpha=%.3g]',obj.tmin_on,obj.alpha_on);
            LS(1)=plot(xmids,Non(1:end-1)./(n*xwidths),'r-','DisplayName',display_name);
            display_name=sprintf('Simulated Off Times PDF: [Tmin=%.3g alpha=%.3g]',obj.tmin_off,obj.alpha_off);
            LS(2)=plot(xmids,Noff(1:end-1)./(n*xwidths),'b-','DisplayName',display_name);
            on_pdf=obj.alpha_on*(obj.tmin_on^obj.alpha_on).*(xs.^(-obj.alpha_on-1));
            display_name=sprintf('Expected On Times PDF: [Tmin=%.3g alpha=%.3g]',obj.tmin_on,obj.alpha_on);
            LS(3)=plot(xs,on_pdf,':r','DisplayName',display_name);
            off_pdf=obj.alpha_off*(obj.tmin_off^obj.alpha_off).*(xs.^(-obj.alpha_off-1));
            display_name=sprintf('Expected Off Times PDF: [Tmin=%.3g alpha=%.3g]',obj.tmin_off,obj.alpha_off);
            LS(4)=plot(xs,off_pdf,':b','DisplayName',display_name);

            display_name=sprintf('Matlab gppdf On Times: [Tmin=%.3g alpha=%.3g]',obj.tmin_on,obj.alpha_on);
            LS(5)=plot(xs,gppdf(xs,1/obj.alpha_on,obj.tmin_on/obj.alpha_on,obj.tmin_on),'--r','DisplayName',display_name);
            display_name=sprintf('Matlab gppdf Off Times: [Tmin=%.3g alpha=%.3g]',obj.tmin_off,obj.alpha_off);
            LS(6)=plot(xs,gppdf(xs,1/obj.alpha_off,obj.tmin_off/obj.alpha_off,obj.tmin_off),'--b','DisplayName',display_name);
            legend(LS,'Location','NorthEast');
            set(gca(),'XGrid','on','YGrid','on');
            title('Quantum dot simulated blinking On/Off times histogram');
            hold off;
            xlabel('Duration (s)');
            ylabel('Occurences');

        end
    end

    methods (Access=private)
        function precompute_times(obj)
            %Paramaters for matlabs generalized pareto distribution mapped from
            %our regular pareto distribution.
            shape_on=1/obj.alpha_on;
            scale_on=obj.tmin_on/obj.alpha_on;
            location_on=obj.tmin_on;

            shape_off=1/obj.alpha_off;
            scale_off=obj.tmin_off/obj.alpha_off;
            location_off=obj.tmin_off;

            n=obj.pre_comp_size/2;
            on_times=gprnd(shape_on, scale_on, location_on, 1, n);
            off_times=gprnd(shape_off, scale_off, location_off, 1, n);

            %Times alternate with on/off times
            obj.times=zeros(1,obj.pre_comp_size);
            obj.times(1:2:end)=on_times;
            obj.times(2:2:end)=off_times;
            %cumulative transitions times forms the global simulation time from which we
            % can simulate individual QD's by taking a subsequence from the
            % global sequence
            obj.cumtimes=cumsum(obj.times);
            obj.times_idx=1;
        end

        function precompute_duty_ratios(obj)
            obj.duty_ratios=zeros(obj.num_frames, obj.ensemble_size);
%             wh=waitbar(0,sprintf('Simulating %i/%i',0,obj.ensemble_size));
            for i=1:obj.ensemble_size
                obj.duty_ratios(:,i)=obj.simulateImage(obj.frame_time, obj.num_frames, obj.burnin_time);
%                 if mod(i,1000)==1
%                     waitbar(i/obj.ensemble_size,wh,sprintf('Simulating %i/%i',i,obj.ensemble_size));
%                 end
            end 
%             close(wh);
        end

        function transition_times=sample_transition_times(obj, max_time, start_state)
            %
            % Given a max_time (total simulation time) and a start_state: 
            % (0 or 1), return a sequence of transition times.  This function is
            % recursive, so it is seperated from simulateQD to isolate the
            % recursive part.
            %

            % Check we start with an appropriate on or off time depending on the
            % start_state.  Odd idxs are on_times and even idxs are off_times
            if mod(obj.times_idx,2)~= start_state
                obj.times_idx=obj.times_idx+1;
                if obj.times_idx>obj.pre_comp_size
                    obj.precompute_times();
                    if mod(obj.times_idx,2)~= start_state
                        obj.times_idx=obj.times_idx+1;
                    end
                end
            end
            %start_t will be the global time we start at, in terms of the
            %precomputed obj.cumtimes.  We subtract this off of global times to
            %get the local time for this QD.
            if obj.times_idx==1
                start_t=0.0;
            else
                start_t=obj.cumtimes(obj.times_idx-1);
            end
            last_t=obj.cumtimes(obj.pre_comp_size); %last simulated time in the global sequence
            goal_t=start_t+max_time; %Global time we end at to simulate this QD.
            if  last_t <= start_t+max_time 
                % There are not enough simulated transitions in the sequence
                ts=obj.cumtimes(obj.times_idx:end)-start_t;
                elapsed_time=last_t-start_t;
                obj.precompute_times();
                next_state=mod(obj.pre_comp_size,2)+1;
                %recurse
                nts=obj.sample_transition_times(max_time-elapsed_time, next_state)+elapsed_time;
                transition_times=[ts nts];
            else
                end_idx=find(obj.cumtimes(obj.times_idx:end) > goal_t ,1)+obj.times_idx-1;
                transition_times=obj.cumtimes(obj.times_idx:end_idx-1)-start_t;
                obj.times_idx=end_idx+1;
            end
        end

    end

end
