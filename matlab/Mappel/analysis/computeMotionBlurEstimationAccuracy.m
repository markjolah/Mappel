

function Data=computeMotionBlurEstimationAccuracy()
    % At 100nm pixel size a .1um^2/s D of a typical membrane protein is
    %  10 px^2/s
    %  at 100 frames/s, this is .1 px^2/frame
    Data.Ntrials=500;
    Data.Nsubsteps=100;
    Data.ND=50; % Number of diffusion constants to try
    Data.Ds=logspace(0, 3, Data.ND);
    Data.psf=1.0;
    Data.I=2000;
    Data.bg=0;
    Data.dT=3e-2;

    Data.traj=simulateTrajectories();
    
    Data.maxDisplacement=max(reshape(abs(Data.traj),[],Data.ND));
    Data.meanPos=squeeze(mean(Data.traj,1));
    Data.endPos=squeeze(Data.traj(Data.Nsubsteps,:,:,:));
    
    Data.boxSize=ceil(Data.maxDisplacement+6*Data.psf+1);
    Data.models=arrayfun(@(bs) Gauss2DMAP(bs,Data.psf), Data.boxSize, 'Uniform',0);
    Data.modelImages=computeModelImages();
    Data.simulatedImages=cellfun(@poissrnd,Data.modelImages,'Uniform',0);
    
    Data.estMeanError=zeros(2,Data.ND);
    Data.meanError=zeros(2,Data.ND);
    Data.meanError0=zeros(2,Data.ND);
    Data.meanError1=zeros(2,Data.ND);
    for di=1:Data.ND
        bs=Data.boxSize(di);
        model=Data.models{di};
        [theta_est, crlb_est]=model.estimateMAP(Data.simulatedImages{di},'Newton');
        Data.estMeanError(:,di)=sqrt(mean(crlb_est(1:2,:),2));
        err=theta_est(1:2,:)-bs/2*ones(2,Data.Ntrials);
        err0=theta_est(1:2,:)-(bs/2*ones(2,Data.Ntrials)-Data.meanPos(:,:,di));
        err1=theta_est(1:2,:)-(bs/2*ones(2,Data.Ntrials)-Data.meanPos(:,:,di))-Data.endPos(:,:,di);
        Data.meanError(:,di)=sqrt(mean(err.*err,2));
        Data.meanError0(:,di)=sqrt(mean(err0.*err0,2));
        Data.meanError1(:,di)=sqrt(mean(err1.*err1,2));
    end
    
    
    
    function traj=simulateTrajectories()
        traj=zeros(Data.Nsubsteps,2,Data.Ntrials,Data.ND); %Trajectories
        for di=1:Data.ND
            sigma=sqrt(2*Data.Ds(di)*Data.dT/Data.Nsubsteps); % Variance ofr each substep
            for ti=1:Data.Ntrials
                traj(:,:,ti,di)=cumsum(randn(Data.Nsubsteps,2).*sigma);
            end
        end
    end

    function modelIms=computeModelImages()
        modelIms=cell(1,Data.ND);
        N=Data.Nsubsteps;
        for di=1:Data.ND
            bs=Data.boxSize(di);
            ims=zeros(bs,bs,Data.Ntrials);
            model=Data.models{di};
            for ti=1:Data.Ntrials
                center=repmat(bs/2*ones(2,1)-Data.meanPos(:,ti,di),1,N); %center so that mean pos of each track is at center of box
                thetas=[Data.traj(:,:,ti,di)'+center; repmat([Data.I/N;Data.bg],1,N)];
                ims(:,:,ti)=sum(model.modelImage(thetas),3); %Sum the individual images
            end
            modelIms{di}=ims;
        end
    end
end