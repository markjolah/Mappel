function plotMotionBlurEstimationAccuracy(Data)

    f=figure();
    
    micronDs=Data.Ds/100; % for a 100nm x 100nm pixel = .01 um^2
    micronMeanErr=Data.meanError(1,:)*0.1; %error from mean position (micron)
    micronMeanErr0=Data.meanError0(1,:)*0.1; %error from frame start position (micron)
    micronMeanErr1=Data.meanError1(1,:)*0.1;  %error from frame end position (micron)
    micronEstMeanErr=Data.estMeanError(1,:)*0.1;
    corrEstAcc=sqrt(micronEstMeanErr.^2+2/3*Data.dT*micronDs);
    
    loglog(micronDs,micronMeanErr,'b-','DisplayName','$ \sqrt{\mathrm{E}\left[(\theta_x-\bar{x})^2\right]} $');
    hold on;
    loglog(micronDs,micronMeanErr0,'m-','DisplayName','$\sqrt{\mathrm{E}\left[(\theta_x-x_0)^2\right]}$');
    loglog(micronDs,micronMeanErr1,'-','Color',[.5 .2 .5], 'DisplayName','$\sqrt{\mathrm{E}\left[(\theta_x-x_{\mathrm{end}})^2\right]}$');
%     hold on;
    loglog(micronDs,micronEstMeanErr,'k--','DisplayName','$\sqrt{\mathrm{CRLB}(\theta_x)}$');
    loglog(micronDs,corrEstAcc,'g--','DisplayName','$\sqrt{\mathrm{CRLB}+2/3 D \delta t}$');
    yl=ylim();
    plot([.1 .1],yl,'r--','DisplayName','Membrane Protein $0.1 (\mu^2/\mathrm{s})$');
    lh=legend('Location','NorthEast');
    set(lh,'interpreter','latex');
    xlabel('$D (\mu^2/\mathrm{s})$','interpreter','latex');
    ylabel('RMSE ($\mathrm{\mu}$)','interpreter','latex');
    title(sprintf('Motion Blur Effect on Localization Accuracy ($\\delta t=%.3g$ s)',Data.dT),'interpreter','latex');
end