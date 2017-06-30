function f=plotspeeds(results)
    f=figure();
    names=fieldnames(results);
    ax=axes();
    set(ax,'XScale','log')
    set(ax,'YScale','log')
    colormap('lines')
    cm=colormap;
    hold;
%     loglog(results.N,1e-7*double(results.N),'k-','DisplayName','[linear]');
    j=0;
    for i=1:numel(names)
        func=names{i};
        if strcmp(func,'N') || strcmp(func,'Properties')
            continue
        end
        j=j+1;
        color=cm(j,:);
        loglog(results.N, results.(func),'Color',color,'MarkerFaceColor',color,'MarkerEdgeColor','k',...
                    'Marker','o','DisplayName',char(func));
    end
    hold;
    title('Gauss2DSpeed scaling');
    xlabel('N');
    ylabel('Time (s)');
    grid();
    legend(ax,'Location','NorthWest');
end
