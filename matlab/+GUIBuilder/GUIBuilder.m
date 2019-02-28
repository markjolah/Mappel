% GUIBuilder.m - A base class for making GUIs for class interaction
% 05/2015
% Mark J. Olah (mjo@cs.unm.edu)
%
classdef GUIBuilder < handle  
    properties (Access=protected, Constant=true)
        default_button_size=[90 30]
        default_spacing=3;
        default_boarder=5;
        default_font_size=10;
        default_unitHeight=25; %Default height of uicontrol elements
        default_buttonSize=[100 25];

        default_waitBar_loc=[500,500];
        
        color_editBG=[0.94 0.94 0.94];
        color_preservedBG=[0.5 0.5 1.0];
        color_unsavedBG=[1. 0.7 0.7];
        colors=[1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1;
                1 0.5 0; 0.3 1 0.5; 0.7 1 0.1; 1 0 0.4; 1.0 0.2 0.7;...
                0 1 0.7; 0 0.2 0.9; 0.2 0.8 1.0];
        colormaps={'gray','parula','jet','hsv','hot','spring','summer','winter','autumn','bone','copper','pink'};
    end

    properties (Transient=true, Hidden=true)
        disableWaitbar = false; % Set to true to diable the waitbar
    end

    properties (Access=protected, Transient=true)
        inGui=false; %Marks that GUI is running and we should pop up waitbars etc.
        waitbarH; %The handle for the waitbar
        keepWaitBar = false; %Keep the waitbar open
        guiFig; %The handle for the gui
        guiOpenFigs; %The open handles that we should be closing
        screenSize;
        dualMonitor; % Boolean if we are operating in a dual monitor setup
        color_figureBG=[0.94 0.94 0.94];
    end
    
    methods
        function gui(obj)
            obj.inGui=true;
            obj.updateScreenSize();
            obj.color_figureBG=get(0,'defaultFigureColor');
        end
        
        function closeGUI(obj)
            if ishandle(obj.guiFig)
                delete(obj.guiFig);
            end
            obj.guiFig=[];            
            obj.waitbarH = GUIBuilder.GUIBuilder.closeCloseableHandles(obj.waitbarH);
            obj.guiOpenFigs = GUIBuilder.GUIBuilder.closeCloseableHandles(obj.guiOpenFigs);
            obj.inGui=false;            
        end
    end
    
    methods (Access=protected)       
        function appendOpenFigs(obj, figH)
            figH=figH(ishandle(figH));
            obj.guiOpenFigs=[obj.guiOpenFigs figH(:)'];
        end
        
        function updateWaitbar(obj,frac,msg)
            if obj.disableWaitbar; return; end
            if frac==0
                if ishandle(obj.waitbarH)
                    obj.waitbarH=waitbar(0.05,obj.waitbarH,strrep(msg,'_','\_'));
                else
                    obj.waitbarH=waitbar(0.05,strrep(msg,'_','\_'));
                end
                figure(obj.waitbarH);
            elseif frac==1
                if ~isempty(obj.waitbarH) && ishandle(obj.waitbarH) && ~obj.keepWaitBar
                    close(obj.waitbarH);
                end
                if ~obj.keepWaitBar
                    obj.waitbarH=[];
                end
            else
                if isempty(obj.waitbarH) || ~ishandle(obj.waitbarH)
                    obj.waitbarH=waitbar(frac,strrep(msg,'_','\_'));
                else
                    if nargin==2 || isempty(msg)
                        waitbar(frac,obj.waitbarH);
                    else
                        waitbar(frac,obj.waitbarH,strrep(msg,'_','\_'));
                    end
                end
            end
        end
        
        function updateScreenSize(obj)
            % Update the saved screen size
            ss=get(0,'ScreenSize');
            ss=ss(3:4); %The 3rd and 4th elements are the x and y screen size
            obj.dualMonitor= ss(1)/ss(2)>2.15; %this is a dual monitor setup in linux
            if obj.dualMonitor
                ss(1)=ss(1)/2;
            end
            obj.screenSize=ss;
        end
        
        function h=viewMaximizedDipFig(obj,im,varargin)
            % Open a new figure window and display im as large as possible on the screen
            % while preserving the aspect ratio
            % [IN]
            % im - A dipimage
            % varargin - [optional] arguments to pass to set for the new figure window handle
            % [OUT]
            % h - handle of new figure window
            h=figure('Visible', 'off');
            dipshow(h,im);
            h.Visible = 'off';
            if isnumeric(im)
                im=dip_image(im); %convert to dip_image if a matlab array
            end
            scale=[1.1 1.1];
            if isa(im,'dip_image')
                im_size=[size(im,1), size(im,2)].*scale;
            elseif isa(im,'dip_image_array')
                im_size=[size(im{1},1), size(im{1},2)].*scale;
            else
                error('GUIBuilder:viewMaximizedDipFig','Unknown image format for im: %s',class(im));
            end
            ss=obj.screenSize;
            if isempty(ss)
                obj.updateScreenSize();
                ss=obj.screenSize;
            end
            sz=min(floor(ss./(im_size)));
            diptruesize(h,sz*100);
            movegui(h,'west');
            obj.guiOpenFigs(end+1)=h;
            set(h,varargin{:});
            h.Visible = 'on';
        end

        function setDefaultableControl(obj,h,prop)
            % This is to highlight the background of the entry uicontrols for "preserved properties"
            if isempty(obj.(prop)) && isfield(obj.preservedProperties,prop) && ~isempty(obj.preservedProperties.(prop))
                h.String = CellFun.arr2str(obj.preservedProperties.(prop));
                h.BackgroundColor = obj.color_preservedBG;
            else
                h.String = CellFun.arr2str(obj.(prop));
                h.BackgroundColor = obj.color_editBG;
            end
        end        
    end %protected methods

    methods (Static=true)
        function h=viewDipFig(im,varargin)
            % A static version of viewMaximizedDipFig
            % Open a new figure window and display im as large as possible on the screen
            % while preserving the aspect ratio
            % [IN]
            % im - A dipimage
            % varargin - [optional] arguments to pass to set for the new figure window handle
            % [OUT]
            % h - handle of new figure window
            h=figure('Visible', 'off');
            dipshow(h,im);
            h.Visible = 'off';
            if isnumeric(im)
                im=dip_image(im); %convert to dip_image if a matlab array
            end
            scale=[1.1 1.1];
            if isa(im,'dip_image')
                im_size=[size(im,1), size(im,2)].*scale;
            elseif isa(im,'dip_image_array')
                im_size=[size(im{1},1), size(im{1},2)].*scale;
            else
                error('GUIBuilder:viewMaximizedDipFig','Unknown image format for im: %s',class(im));
            end
            ss=[1920, 1280];
            sz=min(floor(ss./(im_size)));
            diptruesize(h,sz*100);
            movegui(h,'west');
            set(h,varargin{:});
            h.Visible = 'on';
        end

        function icon = readMatlabIcon(filename)
            iconsFolder = fullfile(matlabroot,'toolbox','matlab','icons');
            icon = imread(fullfile(iconsFolder,filename),'BackgroundColor',[1 1 1]);
            icon = double(icon) ./ double(intmax(class(icon)));
        end
        
        function editH=horzLabeledEditBox(parent,name,val,loc)
            %Make a labeled edit box
            uicontrol('Parent',parent,'Style','text','String',name,...
                'Position',[loc label_sz],...
                'HorizontalAlignment','left','FontSize',font_size);
            editH=uicontrol('Parent',parent,'Style','edit','String',CellFun.arr2str(val),...
                'Position',[loc(1)+label_sz(1)+sp, loc(2), edit_sz],...
                'HorizontalAlignment','right','FontSize',font_size);
        end
        
        function editH=horzLabeledSelectBox(parent,name,vals,pos)
            %Make a labeled edit box
            txtH=uicontrol('Parent',parent,'Style','text','String',name,...
                'Position',pos,'HorizontalAlignment','left');
            ext = txtH.Extent;
            txt_pos = pos;
            txt_pos(3) = ext(3);
            txtH.Position = txt_pos;
            edit_pos = pos;
            edit_pos(1) = edit_pos(1)+ext(3)+2;
            edit_pos(3) = pos(3) -ext(3)-2;
            editH=uicontrol('Parent',parent,'Style','popupmenu','String',CellFun.cellmap(@CellFun.arr2str,vals),...
                'Position',edit_pos,'HorizontalAlignment','right');
        end

        function editH=vertLabeledEditBox(parent,name,val,loc)
            %Make a labeled edit box
            uicontrol('Parent',parent,'Style','text','String',name,...
                'Position',[loc label_sz],...
                'HorizontalAlignment','left','FontSize',font_size);
            editH=uicontrol('Parent',parent,'Style','edit','String',CellFun.arr2str(val),...
                'Position',[loc(1)+label_sz(1)+sp, loc(2), edit_sz],...
                'HorizontalAlignment','right','FontSize',font_size);
        end

        function h=viewDip(im,varargin)
            % Open a new figure window and display im as large as possible on the screen
            % while preserving the aspect ratio
            % [IN]
            % im - A dipimage
            % varargin - [optional] arguments to pass to set for the new figure window handle
            % [OUT]
            % h - handle of new figure window
            h=figure();
            if isnumeric(im)
                im=dip_image(im); %convert to dip_image if a matlab array
            end
            scale=[1.05 1.07];
            if isa(im,'dip_image')
                im_size=[size(im,1), size(im,2)].*scale;
            elseif isa(im,'dip_image_array')
                im_size=[size(im{1},1), size(im{1},2)].*scale;
            else
                error('GUIBuilder:viewMaximizedDipFig','Unknown image format for im: %s',class(im));
            end
            dipshow(h,im);
            ss=[1920 1080];
            sz=min(floor(ss./im_size));
            diptruesize(h,sz*100);
            set(h,varargin{:});
        end

        function CB=safeCallback(cb_func)
            % useage: GUIBuilder.safeCallback(@myactual_CB)
            %
            % returns a function handle for callback useage that will check
            % for errors and report them as a popup errordlg, and still give you a text output to use
            % for tracking down the source.  This is very cool.
            %
            % 
            function wrappedCB(varargin)
                %this is the inner function that just wraps the actual callback and reports errors nicely
                try
                    cb_func(varargin{:});
                catch err
                    disp(getReport(err))
                    errordlg(err.message,err.identifier);
                end
            end
            if isempty(cb_func)
                CB = [];
            elseif isa(@sum,'function_handle')
                CB = @wrappedCB;
            else
                error('GUIBuilder:safeCallback','Got a non-function-handle argument');
            end
        end

        function pos = autoSizePanel(panelH)
            % Given a handle to a uipanel, expand the size to fit all elements, keeping the bottom
            % left corner [pos(1:2)] fixed.
            pos = panelH.Position;
            cpos = cellmatfun(@getpixelposition, panelH.Children); %children positions
            if ~isempty(cpos)
                max_sz = max(cpos(:,1:2)+cpos(:,3:4),[],1);
                pos = [pos(1:2) max_sz+[3 7]*GUIBuilder.GUIBuilder.default_spacing];
                panelH.Position = pos;
            end
        end

        function pos = autoSizeTabGroup(tabG)
            % Given a handle to a uitabgroup, expand the size to fit all elements for each of the tabs, keeping the bottom
            % left corner [pos(1:2)] fixed.
            pos = tabG.Position;
            tabs = tabG.Children;
            g_sz = [0,0];
            for n = 1:numel(tabs)
                cpos=cellmatfun(@getpixelposition, tabs(n).Children); %children positions
                if ~isempty(cpos)
                    tab_max_sz=max(cpos(:,1:2)+cpos(:,3:4),[],1);
                    g_sz = max(g_sz, tab_max_sz+[3 7]*GUIBuilder.GUIBuilder.default_spacing);
                end
            end
            g_sz(2) = g_sz(2)+20; %space for headers
            pos = [pos(1:2) g_sz];
            tabG.Position = pos;
        end

        function bbox=align(H, varargin)
            % Matlab has a very useful align function for aligning uicontrol elements.  It computes aligned
            % positions for each of a set of uicontrols, but does not actually move anything arround.  This is a
            % simple wrapper around align that allowsa version of align that actually moves around the
            % objects.
            % [IN]
            %   H - cellarray of handles
            %   [Align args] - All other argurments are passed directly to align.  In general you need to
            %   use the same agruments here as you would calling matlab's 'align' function
            % [OUT]
            %   bbox - [x y w h]: a bounding box in Matlab position format for the outer edge of the
            %                     aliged uicontrol elements.
            if isstruct(H)
                H = struct2cell(H);
            end
            if isscalar(H)
                bbox = H{1}.Position;
                return;
            end
            cur_pos=cell2mat(CellFun.cellmap(@(h) h.Position,H(:)));
            new_pos=align(cur_pos, varargin{:});
            for n=1:numel(H)
                H{n}.Position = new_pos(n,:); %Actually move the uicontrol elements.  Matlab 'align' doesnt do this for you.
            end
            bbox=[min(new_pos(:,1:2)), max(new_pos(:,1:2)+new_pos(:,3:4))]; %outer bbox for this group
        end
        
       function menuH = makeFigureMenu(figH, title, labels, CBs)
            %
            % Make a named menu at the top of a figure.
            %
            % NOTE: * This wraps all the items in a safe callback wrapper to bring up error messages
            %         nicely!
            %
            % [IN]
            %  figH - figure handle to add the menu to
            %  title - string - The title of the menu
            %  labels - cell array of strings - the labels on the menu, blanks [] make seperators
            %  CBs - cell array of callback function handles - the handles to call for each menu item.
            % [OUT]
            %  menuH - A handle to the new menu.
            if isempty(labels) || length(labels)~=length(CBs)
                error('GUIBuilder:makeFigureMenu','Got inconsitent list of labels and CBs');
            end
            menuH = uimenu(figH, 'Label',title);
            CBs = CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            for i = 1:length(labels)
                if isempty(labels{i})
                    uimenu(menuH, 'Separator','on');
                else
                    uimenu(menuH, 'Label', labels{i},'Callback',CBs{i});
                end
            end
        end
        
        function menuH = makeContextMenu(labels, CBs)
            %
            % Make a uicontextmenu which is not yet associated with any objects.
            %
            % NOTE: * This wraps all the items in a safe callback wrapper to bring up error messages
            %         nicely!
            %
            % [IN]
            %  labels - cell array of strings - the labels on the menu, blanks [] make seperators
            %  CBs - cell array of callback function handles - the handles to call for each menu item
            %        (also use [] for sperator).
            % [OUT]
            %  menuH - A handle to the new menu.
            if isempty(labels) || length(labels)~=length(CBs)
                error('GUIBuilder:makeContextMenu','Got inconsitent list of labels and CBs');
            end
            menuH = uicontextmenu();
            CBs = CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            seperator='off';
            for i = 1:length(labels)
                if isempty(labels{i})
                    seperator = 'on';
                    continue;
                else
                    uimenu(menuH, 'Label', labels{i},'Callback',CBs{i}, 'Separator',seperator);
                    seperator='off';
                end
            end
        end

        function menuH = makeJavaContextMenu(labels, CBs)
            if isempty(labels) || length(labels)~=length(CBs)
                error('GUIBuilder:makeJavaContextMenu','Got inconsitent list of labels and CBs');
            end
            menuH = javaObjectEDT('javax.swing.JPopupMenu');
            CBs = CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            for i = 1:length(labels)
                if isempty(labels{i})
                    menuH.addSeparator();
                else
                    item = javaObjectEDT('javax.swing.JMenuItem',labels{i});
                    set(item, 'ActionPerformedCallback', CBs{i});
                    menuH.add(item);
                end
            end
        end

        function clearJavaContextMenu(menuH)
            % Call this on exit from GUI to clear the callbacks
            items = menuH.getSubElements();
            for i = 1:length(items)
                set(items(i),'ActionPerformedCallback',[]);
            end
        end
        
        function makeTableRowSelectable(tableH, mousePressedCallback)
            % [IN]
            % tableH - a uitable handle of a table that should have the row-selection property turned on.
            % mousePressedCallback - Function handle to a callback for a mouse press
            %
            hJTable = GUIBuilder.GUIBuilder.getJTableHandle(tableH);
            hJTable.setNonContiguousCellSelection(false);
            hJTable.setColumnSelectionAllowed(false);
            hJTable.setRowSelectionAllowed(true);
            if nargin == 2
                CB = handle(hJTable,'CallbackProperties');
                CB.MousePressedCallback = mousePressedCallback;
            end
        end
        
        function hJTable=getJTableHandle(tableH)
            % Gets you the raw javaobject for this tableH.  BE CAREFUL!
            %
            %  This relies on the well estabilished Matlab fileexchange program findjobj
            %  http://www.mathworks.com/matlabcentral/fileexchange/14317-findjobj-find-java-handles-of-matlab-graphic-objects
            % [IN]
            %   tableH - handle to a uitable object
            % [OUT]
            %   hJTable - A handle to the java JTable object under the hood of the matlab uitable object
            hJScroll = findjobj(tableH,'nomenu');
            hJTable = hJScroll.getViewport.getView; % get the table component within the scroll object
        end
        
        function clearJTableCallbacks(tableH)
            % Should be called on GUI close to enmsure we clean up and can free the object.  This is
            % important if you have used the other 'JTable" methods in GUIBuilder
            hJTable = GUIBuilder.GUIBuilder.getJTableHandle(tableH);
            CB = handle(hJTable,'CallbackProperties');
            CB.MousePressedCallback = [];
        end
        
        function positionImageAxes(axesH, imageSize, area, labelMargin)
            % [IN]
            %  axH - handle to an axes to position
            %  imageSize - The size of an image to display with true isotropic scaling
            %  area - The area to draw into as a matlab Position [l b w h]
            %  labelMargin - A margin to move away from the area to account for labels
            if nargin==3
                labelMargin=zeros(1,4);
            end
            if ~isempty(imageSize)
                imageSize=double(imageSize);
                inset = axesH.TightInset;
                inset = inset + labelMargin;
                tot_inset = inset(1:2)+inset(3:4); %Total inset ammount in [x y]
                area = [area(1:2)+inset(1:2), area(3:4)-tot_inset];
                inset_ratio = imageSize./area(3:4);
                pos = area;
                if inset_ratio(1)>=inset_ratio(2)
                    %image is wider in aspect ratio than the area             
                    pos(4) = pos(3)*imageSize(2)/imageSize(1);
                    pos(2) = area(2)+(area(4)-pos(4))/2;
                else
                    %image is taller in aspect than wide
                    pos(3) = pos(4)*imageSize(1)/imageSize(2);
                    pos(1) = area(1)+(area(3)-pos(3))/2;
                end
                axesH.Position = pos;
            else
                axesH.OuterPosition = area; %No image, so give up and just set the outerposition
            end
        end
        
        function positionAxes(axesH, area, labelMargin)
            % [IN]
            %  axesH - handle to an axes
            %  area - [x y w h]: matlab position format - the maximum area the axes can conver
            %  labelMargin - [left bottom right top] - the ammount to pad the outside of the axes to
            %     allow for text labels and title.  These should be positive values if you need extra space
            %     along one of the sides of the figure because the labels are getting cut off.
            if nargin==2
                labelMargin=zeros(1,4);
            end
            inset = axesH.TightInset+labelMargin;

            axesH.Position = area+[inset(1:2), -inset(1:2)-inset(3:4)];
        end
        
        function handles=buttonCol(parent, area, button_sz, names, CBs, varargin)
            N=length(names);
            if N*(GUIBuilder.GUIBuilder.default_spacing+button_sz(2))>area(4)
                error('GUIBuilder:buttonRow', 'area too small');
            end
            handles=cell(1,N);
            pos=[area(1:2), button_sz];
            CBs=CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            for n=1:N
               handles{n}=uicontrol('Parent', parent, 'Style','pushbutton','String',names{n},...
                                    'Position', pos, 'Callback', CBs{n},varargin{:});
            end
            align([handles{:}],'Left','Fixed',GUIBuilder.default_spacing);
        end

        function Hs=buttonRow(parent, area, button_sz, names, CBs, varargin)
            N=length(names);
            sp=GUIBuilder.GUIBuilder.default_spacing;
            buts_per_row=floor(area(3)/(button_sz(1)+sp));
            nrows=ceil(N/buts_per_row);
            pos=[area(1:2), button_sz];
            Hs=cell(1,N);
            pos(2)=pos(2)+(nrows-1)*(button_sz(2)+sp);
            CBs=CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            for r=1:nrows
                row_st=1+(r-1)*buts_per_row;
                row_end=min(r*buts_per_row,N);
                for n=row_st:row_end
                   Hs{n}=uicontrol('Parent',parent,'Style','pushbutton','String',names{n},...
                                     'Position',pos,'Callback', CBs{n},varargin{:});
                end
                align([Hs{row_st:row_end}],'Fixed',GUIBuilder.GUIBuilder.default_spacing,'Bottom');
                pos(2)=pos(2)-button_sz(2)-sp;
            end
        end
        
        function handles=labeledHEdits(parent, area, height, hNames, labels, values, CBs)
            N=length(labels);
            H=cell(2,N);
            sp=GUIBuilder.GUIBuilder.default_spacing;
            label_pos=[area(1:2), (area(3)-sp)/2, height];
            exts=zeros(N,4);
            CBs=CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            for n=N:-1:1
                H{1,n}=uicontrol('Parent',parent,'Style','text','String',labels{n},...
                                 'Position',label_pos,'HorizontalAlignment','left');
                exts(n,:) = H{1,n}.Extent;
            end
            max_ext=max(exts,[],1);
            label_pos=[area(1:2), max_ext(3), height];
            for n=1:N
                H{1,n}.Position = label_pos;
            end
            align([H{1,end:-1:1}],'Left','Fixed',sp);
            label_pos=getpixelposition(H{1,1});
            edit_pos=[label_pos(1)+label_pos(3)+sp area(2), area(3)-label_pos(3)-sp, height];
            for n=N:-1:1
                if ischar(values{n})
                    H{2,n}=uicontrol('Parent',parent,'String',values{n},...
                                     'Position',edit_pos,'HorizontalAlignment','left');
                else
                    H{2,n}=uicontrol('Parent',parent,'String',CellFun.arr2str(values{n}),...
                                     'Position',edit_pos,'HorizontalAlignment','right');
                end
                handles.(hNames{n}) = H{2,n};
                if length(CBs)>=n && ~isempty(CBs{n})
                    H{2,n}.Callback=CBs{n};
                    H{2,n}.Style='edit';
                    H{1,n}.Position = H{1,n}.Position - [0 0 0 5];
                else
                    H{2,n}.Style='text';
                end
            end
            align([H{2,end:-1:1}] ,'Left','Fixed',sp);
        end
        
        function handles=labeledVEdits(parent, area, height, hNames, labels, values, CBs)
            N=length(labels);
            H=cell(2,N);
            sp=GUIBuilder.GUIBuilder.default_spacing;
            label_pos=[area(1:2), area(3)-sp, height];
            exts=zeros(N,4);
            CBs=CellFun.cellmap(@GUIBuilder.GUIBuilder.safeCallback,CBs); % Make callbacks safe
            for n=N:-1:1
                H{1,n}=uicontrol('Parent', parent, 'Style','text','String',labels{n},...
                                 'Position', label_pos,'HorizontalAlignment','left');
                exts(n,:) = H{1,n}.Extent;
            end
            max_ext=max(exts,[],1);
            label_pos=[area(1:2), max_ext(3:4)];
            for n=N:-1:1 %Resize labels
                H{1,n}.Position = label_pos;
            end
            edit_pos=[area(1:2), area(3)-sp, height];
            for n=N:-1:1
                H{2,n}=uicontrol('Parent', parent, 'Style','edit',...
                                 'String',CellFun.arr2str(values{n}),...
                                 'Position', edit_pos, 'HorizontalAlignment','left',...
                                 'BackgroundColor',GUIBuilder.GUIBuilder.color_editBG);
                if length(CBs)>=n && ~isempty(CBs{n})
                   H{2,n}.Callback=CBs{n};
                end
                handles.(hNames{n})=H{2,n};
            end
            align([H{end:-1:1}],'Left','Fixed',sp);
        end
               
        function gridH = makePropertyGrid(panH,pane_title,prop_struct, param_info, position)
            props = CellFun.cellmap(@makePropGridField, fieldnames(prop_struct));
            gridH = PropertyGrid(panH,'Properties', [props{:}], 'Units','pixels','Position',position);
            gridH.Control.Title=pane_title;
            function pgf = makePropGridField(name)
                val = prop_struct.(name);
                pgf=PropertyGridField(name,val);
                if isfield(param_info, name)
                    info=param_info.(name);
                    if isfield(info,'desc')
                        pgf.Description=info.desc;
                    end
                    if isfield(info,'disp')
                        pgf.DisplayName=info.disp;
                    end
                    if isfield(info,'range')
                        rng=info.range;
                        if isempty(rng)
                            if isscalar(val)
                                pgf.Type=PropertyType('denserealdouble','scalar');
                            elseif isvector(val)
                                pgf.Type=PropertyType('denserealdouble','row');
                            else
                                pgf.Type=PropertyType('denserealdouble','matrix');
                            end
                        elseif iscell(rng)
                            c=class(rng{1});
                            if isnumeric(rng{1})
                                caster=str2func(c);
                                pgf.Value=caster(pgf.Value);
                                pgf.Type=PropertyType(c,'scalar',rng);
                            else
                                pgf.Type=PropertyType(c,'row',rng);
                            end
                        elseif isnumeric(rng) && ~isscalar(rng)
                            pgf.Type=PropertyType('denserealdouble','scalar',rng);
                        end    
                    end
                end
            end
        end
                
        function [tableH, containerH] = makeTreeTable(parentH, ColSettings, data, position, varargin)
            tableH = treeTable(parentH, ColSettings.names, data, 'ColumnTypes',ColSettings.formats, varargin{:});
            tableH.setSelectionBackground(java.awt.Color(.4,.4,0.96));
            tableH.setShowGrid(true);
            tableH.setAutoResizeMode(tableH.java.AUTO_RESIZE_OFF);
            com.jidesoft.grid.TableUtils.autoResizeAllColumns(tableH,60,true); %This is the shit
            containerH = parentH.Children(1);
            containerH.Units = 'pixels';
            containerH.Position = position;
            jTextField = javax.swing.JTextField();
            jTextField.setEditable(false);
            jTextEdit = javax.swing.DefaultCellEditor(jTextField);
            jTextEdit.setClickCountToStart(intmax)
            colModelH = tableH.getColumnModel();
            N = tableH.getColumnCount();
            for n=1:N
                colModelH.getColumn(n-1).setCellEditor(jTextEdit);
            end
        end
        
        function modelH = setTreeTableData(tableH, ColSettings, data, groupcols)
            if nargin==3
                groupcols=[];
            end
            model = MultiClassTableModel(data, ColSettings.names);
            modelH = com.jidesoft.grid.DefaultGroupTableModel(model);% Wrap the standard model in a JIDE GroupTableModel
            modelH.setAutoExpand(true);
            if ~isempty(groupcols)
                for i=1:length(groupcols)
                    modelH.addGroupColumn(groupcols(i)-1);
                end
            end
            modelH.groupAndRefresh();
            tableH.setModel(modelH);
            tableH.setAutoResizeMode(tableH.java.AUTO_RESIZE_OFF);
            com.jidesoft.grid.TableUtils.autoResizeAllColumns(tableH,60,true); %This is the shit
            jTextField = javax.swing.JTextField();
            jTextField.setEditable(false);
            jTextEdit = javax.swing.DefaultCellEditor(jTextField);
            jTextEdit.setClickCountToStart(intmax)
            colModelH = tableH.getColumnModel();
            N = tableH.getColumnCount();
            for n=1:N
                colModelH.getColumn(n-1).setCellEditor(jTextEdit);
            end
        end
        
        function javaTableMouse_CB(H, eData, menuH)
            %This makes the context menus work for the properties tables
            if eData.isMetaDown
                menuH.show(H,eData.getX(),eData.getY());
                menuH.repaint();
            end
        end
       
        function setPropertyGridEnabled(pgrid_panelH, state)
            if ~isempty(pgrid_panelH)
                table = pgrid_panelH.Table;
                if ~state
                    table.clearSelection();
                end
                table.setEnabled(state);
            end
        end
        
        function setPropertyGridMenu(tableH, CB)
            ptH=tableH.Table;
            set(ptH,'MousePressedCallback',CB);
        end

        function clearPropertyGridMenus(tableHs)
            %This must be called on figure close or it will cause a memory leak
            %tableHs is a cell array of property grids to clear menue for
            tableHs = makecell(tableHs);
            for n = 1:numel(tableHs)
                tableH = tableHs{n};
                ptH = tableH.Table;
                set(ptH,'MousePressedCallback',[]);
            end
        end
       
        function safeCopyfile(src, dest, check_overwrite)
            if nargin==2
                check_overwrite=1;
            end
            [newpathn,~,~]=fileparts(dest);
            Pickle.createDirIfNonexistant(newpathn);
            if check_overwrite
                GUIBuilder.GUIBuilder.confirmOverwrite(dest);
            end
            if ~strcmp(src,dest)
                [success,msg,msgid]=copyfile(src,dest);
                if ~success
                    if strcmp(msgid,'MATLAB:COPYFILE:SourceAndDestinationSame')
                        return
                    else
                        error('GUIBuilder:safeCopyfile','Unable to copy file "%s"->"%s". Error %s - %s.',...
                                src,dest, msgid, msg);
                    end
                end
            end
        end
               
        function closeFigs = closeCloseableHandles(Hs)
            closeFigs = Hs( arrayfun(@ishandle,Hs) );
            if ~isempty(closeFigs)
                close(closeFigs);
            end
            closeFigs = Hs( arrayfun(@ishandle,Hs) );
        end
        
        function ok = confirmOverwrite(filename)
            if exist(filename,'file')
                button=questdlg(sprintf('File "%s" already exists.  Overwrite?', filename),'Overwrite file confirm');
                ok = strcmp(button,'Yes');
            else
                ok = true;
            end
        end

        function name=nextUnusedName(currNames,pattern, i)
            %Return a unique name using pattern that take integer i, and not matching any
            %in currNames
            if nargin==2
                i=1;
            end
            name=sprintf(pattern,i);
            while any(cellfun(@(s) strcmp(name,s),currNames))
                i=i+1;
                name=sprintf(pattern,i);
            end            
        end
    end % Static methods
end

