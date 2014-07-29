function plotcmc(ds, sampleToTest)
%% Plot Cumulative Matching Characteristic (CMC) Curves
 names = fieldnames(ds);
        for nameCounter=1:length(names)
            s = [ds.(names{nameCounter})];
            ms.(names{nameCounter}).cmc = cat(1,s.cmc)./sampleToTest;
            ms.(names{nameCounter}).roccolor = s(1).roccolor;
        end
        
        subplot(1,2,1);
        % h = figure;
        names = fieldnames(ms);
        for nameCounter=1:length(names)
            hold on;
            plot(median(ms.(names{nameCounter}).cmc,1),'LineWidth',2, ...
                'Color',ms.(names{nameCounter}).roccolor);
        end
        
        title('Cumulative Matching Characteristic (CMC) Curves - VIPeR dataset');
        box('on');
        %         set(gca,'XTick',[0 10 20 30 40 50 100 150 200 250 300 350]);
        ylabel('Matches');
        xlabel('Rank');
        ylim([0 1]);
        hold off;
        grid on;
        legend(upper(names),'Location','SouthEast');
       
 

end