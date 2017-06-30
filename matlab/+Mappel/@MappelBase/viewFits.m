function f = viewFits(obj, ims)
% GUI Window to view emitter fits
% ims = sequence of images to fit


gui_name = sprintf('[%s] Emitter Fit Browser',class(obj));
    
uH = 25; % unit height for elements
boarder = 10;%Boarder width around the outside of the gui
sp = 2; %spacing between elements.
but_sz = 80; %Button size
fig_sz = [1500 800]; %figure size
halfw = 750;

guiFig = figure('Units','pixels','Position',[10 0 fig_sz],'Resize','off',...
                'MenuBar','none','ToolBar','figure','NumberTitle','off',...
                'Name',gui_name,'Visible','on');

imagePanel_pos = [boarder, boarder, halfw-boarder-sp, fig_sz(2)-2*boarder];
fitPanel_pos =   [halfw+sp, boarder, fig_sz(1)-halfw-sp, fig_sz(2)-2*boarder];
handles.imagePanel = uipanel('Parent',guiFig,'Units','Pixels','Position',imagePanel_pos,'Title','Images');
handles.fitPanel   = uipanel('Parent',guiFig,'Units','Pixels','Position',fitPanel_pos,'Title','Fit');

Nims = size(ims,ndims(ims));
curIm = []; % The current image to fit
imageBounds = {[.5,obj.imsize(2)-.5],[.5,obj.imsize(1)-.5]}; % bounds for plotting images
imageMax = max(ims(:));
pos=struct();


method = 'Newton';
theta_est=[];
crlb=[];
emitter_llh=[];
estimator_stats=[];
theta_seq=[];
llh_seq=[];
est_im=[];
uniform_llh=[];
uniform_bg_mle=[];
noise_llh=[];
theoreticalSE=[];
observedSE=[];

populateImagePanel()
populateFitPanel()
setImage(1);

function populateImagePanel()
    panel_sz = handles.imagePanel.Position(3:4);
    
    pos.slider = [boarder, boarder, panel_sz(1)-2*boarder, uH];
    top = pos.slider(2)+pos.slider(4)+2*sp;
    pos.imAxes =  [boarder, top, panel_sz(1)-2*boarder, panel_sz(2)-top-boarder];
    
    handles.imSlider = uicontrol('Parent',handles.imagePanel,'Style','Slider',  'Position', pos.slider,...
                                 'Min',1,'Max',Nims,'Value',1,'SliderStep',[1/Nims,1/Nims],'Callback',@imageSlider_CB);
    handles.imAxes = axes('Parent',handles.imagePanel, 'Units','pixels','Position',pos.imAxes,...
                           'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');

    colormap(hot());
%     GUIBuilder.positionImageAxes(handles.imAxes,obj.imsize,pos.imAxes,[10 10 40 10]);
end

function populateFitPanel()
    panel_sz = handles.fitPanel.Position(3:4);
    pos.edits = [boarder, boarder, panel_sz(1)-2*boarder, uH];
    edits.hNames={'eTheta','SE','LLH','uniformLLH','noiseLLH'};
    edits.labels={'Est. Theta', 'SE', 'LLH','Uniform Model LLH:','noiseModelLLH'};
    edits.values={[],[],[],[],[]};
    edits.CBs={[],[],[],[],[]};
    handles.fitEdits = GUIBuilder.labeledHEdits(handles.fitPanel, pos.edits, uH, edits.hNames, edits.labels, edits.values, edits.CBs);
    edits_pos = handles.fitEdits.(edits.hNames{1}).Position;
    top = edits_pos(2)+edits_pos(4)+sp;
    pos.modelSelect = [boarder, top, panel_sz(1)-2*boarder, uH];
    top = pos.modelSelect(2)+pos.modelSelect(4)+sp;
    pos.fitAxes = [boarder, top+sp, panel_sz(1)-2*boarder, panel_sz(2)-boarder-sp-top];
    
    handles.modelSelect = GUIBuilder.horzLabeledSelectBox(handles.fitPanel,'Method: ',obj.EstimationMethods,pos.modelSelect);
    handles.modelSelect.Value = find(strcmp(method,obj.EstimationMethods));
    handles.modelSelect.Callback = @modelSelect_CB;
    handles.fitAxes = axes('Parent',handles.fitPanel, 'Units','pixels','Position',pos.fitAxes,...
                           'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');

end



function setImage(idx)
    if idx<1 || idx>Nims
        warning('MappelBase:viewFits','Invalid image index: %i',idx);
    end
    curIm = ims(:,:,idx);
    axes(handles.imAxes);        
    imagesc(imageBounds{:},curIm);
    handles.imAxes.CLim=[0,imageMax];
    axis('tight');
    colorbar();
    xlabel('X (px)');
    ylabel('Y (px)');
    GUIBuilder.positionImageAxes(handles.imAxes,obj.imsize,pos.imAxes,[10 10 60 10]);
    fitImage();
end

function fitImage()
    [theta_est, crlb, emitter_llh, estimator_stats, theta_seq, llh_seq] = obj.estimateDebug(curIm,method);
    [uniform_llh, uniform_bg_mle] = obj.uniformBackgroundModelLLH(curIm);
    est_im = obj.modelImage(theta_est);
    noise_llh = obj.noiseBackgroundModelLLH(curIm);
    theoreticalSE = sqrt(crlb);
    handles.fitEdits.eTheta.String=arr2str(theta_est');
    handles.fitEdits.SE.String=arr2str(theoreticalSE');
    handles.fitEdits.LLH.String=num2str(emitter_llh);
    handles.fitEdits.uniformLLH.String=num2str(uniform_llh);
    handles.fitEdits.noiseLLH.String=num2str(noise_llh);
    
    plotEstFig();
end

function plotEstFig()
    axes(handles.fitAxes);        
    imagesc(imageBounds{:},est_im);
    handles.fitAxes.CLim=[0,imageMax];
    axis('tight');
    colorbar();
    xlabel('X (px)');
    ylabel('Y (px)');
    GUIBuilder.positionImageAxes(handles.fitAxes,obj.imsize,pos.fitAxes,[10 10 60 10]);
end


function imageSlider_CB(H,~)
    setImage(round(H.Value));
end
function modelSelect_CB(~,~)
    method = obj.EstimationMethods{handles.modelSelect.Value};
    fitImage();
end
% 
% 
% 
% 
%     method = 'Newton';
%     theta = obj.samplePrior();
%     sim=[];
%     theta_init=theta;
%     theta_est=[];
%     crlb=[];
%     emitter_llh=[];
%     estimator_stats=[];
%     theta_seq=[];
%     llh_seq=[];
%     est_im=[];
%     uniform_llh=[];
%     uniform_bg_mle=[];
%     noise_llh=[];
%     theoreticalSE=[];
%     observedSE=[];
% 
%     handles=[];
% 
%     simAx = axes('Units','pixels','Position',simFig_pos,'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');
%     estAx = axes('Units','pixels','Position',estFig_pos,'YDir','reverse','TickDir','out','Box','on','BoxStyle','full');
%     imageBounds = {[.5,obj.imsize(2)-.5],[.5,obj.imsize(1)-.5]};
% 


end

