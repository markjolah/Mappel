
classdef MappelHSBase < MappelBase
    
    properties (Constant= true)
        %MinSize: The minimum imsize of an image in pixels
%         MinSize=4;
%         EstimationMethods={'Heuristic', 'Newton', 'NewtonRaphson', 'QuasiNewton', 'SimulatedAnnealing'};
    end

    properties (Access=protected)
        GPUGaussMLEFitType=0;
    end


    methods

      function obj = MappelHSBase(iface, imsize_,psf_sigma_)
            % obj = MappelBase(imsize,psf_sigma) - Make a new MappelBase for
            % point localization in 2D.
            % (in) imsize: scalar int - size of image in pixels on each side (min: obj.MinSize)
            % (in) psf_sigma: scalar double>0 - size of PSF in pixels
            % (out) obj - A new object
            
            if length(imsize_) ~= 3
                error('MappelHSBase:constructor','Invalid imsize')
            end
            if length(psf_sigma_) ~= 3
                error('MappelHSBase:constructor','Invalid psf_sigma')
            end
            obj@MappelBase(iface, imsize_,psf_sigma_);
            if any(obj.psf_sigma)<0.05
                error('MappelHSBase:constructor','psf_sigma too small')
            end
        end
    

        function viewModelDipImage(obj, theta)
            obj.viewDipImage(obj.modelImage(theta));
        end

        function viewSimulatedDipImage(obj, theta)
            obj.viewDipImage(obj.simulateImage(theta));
        end


%         function [X,Y,L]=imageGrid(obj)
%             sz=double(obj.imsize);
%             [X,Y,L]=meshgrid((1:sz(2)),(1:sz(1)),(1:sz(3)));
%         end

        function viewHSImage(obj, im, lambda)
            pixelSize=0.1;
            xs=double( (0:obj.imsize(1)).*pixelSize );
            ys=double( (0:obj.imsize(2)).*pixelSize );
            lambdas = double( lambda(1:obj.imsize(3)+1));
            f=figure();
            HSData.pixelVolumeView(xs,ys,lambdas,double(im), jet),'intensity';
            cbh=colorbar();
            set(get(cbh,'Label'),'String','Intensity (Photons)');
            fig_bg_color=[0.05 0.05 0.05];
            ax_txt_color=[1 1 1];
            ax_font_size=12;
            xlabel('x ($\mu$m)','interpreter','latex','Color',ax_txt_color,'FontSize',ax_font_size);
            ylabel('y ($\mu$m)','interpreter','latex','Color',ax_txt_color,'FontSize',ax_font_size);
            zlabel('$\lambda$ (nm)','interpreter','latex','Color',ax_txt_color,'FontSize',ax_font_size);
            set(f,'Color',fig_bg_color);
            set(cbh,'Color',ax_txt_color);
            set(get(cbh,'Label'),'interpreter','latex','FontSize',ax_font_size);
        end
        
        function viewSR_HSImage(obj, im)
            imsz=size(im);
            srfactor=imsz(1)./obj.imsize(1);
            pixelSize=0.1;
            xs=double( (0:imsz(3)).*pixelSize./srfactor);
            ys=double( (0:imsz(2)).*pixelSize./srfactor);
            lambdas=linspace(obj.lambda(1), obj.lambda(end), imsz(1)+1);
            f=figure();
            HSData.pixelVolumeView(xs,ys,lambdas,double(im), jet),'intensity';
            cbh=colorbar();
            set(get(cbh,'Label'),'String','Intensity (Photons)');
            fig_bg_color=[0.05 0.05 0.05];
            ax_txt_color=[1 1 1];
            ax_font_size=12;
            xlabel('x ($\mu$m)','interpreter','latex','Color',ax_txt_color,'FontSize',ax_font_size);
            ylabel('y ($\mu$m)','interpreter','latex','Color',ax_txt_color,'FontSize',ax_font_size);
            zlabel('$\lambda$ (nm)','interpreter','latex','Color',ax_txt_color,'FontSize',ax_font_size);
            set(f,'Color',fig_bg_color);
            set(cbh,'Color',ax_txt_color);
            set(get(cbh,'Label'),'interpreter','latex','FontSize',ax_font_size);
        end

        function srim=superResolutionModel(obj, theta, theta_err, res_factor)
            if nargin<3
                res_factor=3;
            end
            srimsize=obj.imsize*res_factor;
            theta(1:3)=theta(1:3)*res_factor;
            theta_err(1:3)=theta_err(1:3)*res_factor;

            xs=0:srimsize(1)-1;
            X=0.5*(erf(((xs+1)-theta(1))/(sqrt(2)*theta_err(1)))-erf((xs-theta(1))/(sqrt(2)*theta_err(1))));
            ys=0:srimsize(2)-1;
            Y=0.5*(erf(((ys+1)-theta(2))/(sqrt(2)*theta_err(2)))-erf((ys-theta(2))/(sqrt(2)*theta_err(2))));
            ls=0:srimsize(3)-1;
            L=0.5*(erf(((ls+1)-theta(3))/(sqrt(2)*theta_err(3)))-erf((ls-theta(3))/(sqrt(2)*theta_err(3))));
            srim=theta(4)* (repmat((X'*Y),1,1,srimsize(3)).*repmat(reshape(L,[1,1,srimsize(3)]),srimsize(1),srimsize(2),1));
            
        end


        function [theta_est_grid,est_var_grid]=mapEstimatorAccuracy(obj,estimator, sample_grid)
            nTrials=size(sample_grid,4);
            gridsize=[size(sample_grid,5), size(sample_grid,6)];
            theta_est_grid=zeros([obj.nParams, nTrials, gridsize]);
            est_var_grid=zeros([obj.nParams, nTrials, gridsize]);
            h=waitbar(0,sprintf('Maping Accuracy Model:%s Estimator%s gridsize:%s',obj.Name, estimator, mat2str(gridsize)));
            for x=1:gridsize(1)
                for y=1:gridsize(2)
                    [theta_est, est_var]=obj.evaluateEstimatorOn(estimator, sample_grid(:,:,:,:,x,y));
                    theta_est_grid(:,:,x,y)=theta_est;
                    est_var_grid(:,:,x,y)=est_var;
                end
                waitbar(x/gridsize(1),h);
            end
            close(h);
        end
        
        function [theta_grid,sample_grid]=makeThetaGridSamples(obj, theta, gridsize, nTrials)
            % (in) theta - A theta to test over a spatial grid
            % (in) gridsize - The size of the grid
            % (out) theta_grid - size:[obj.nParams, nTrials,gridsize(1),gridsize(2)]
            % (out) sample_grid - size:[obj.imsize(1),obj.imsize(2),nTrials,gridsize(1),gridsize(2)]
            theta=theta(:);
            if isscalar(gridsize)
                gridsize=[gridsize gridsize];
            end
            theta_grid=zeros([obj.nParams, nTrials, gridsize]);
            sample_grid=zeros([obj.imsize, nTrials, gridsize]);
            grid_edges.x=linspace(0,obj.imsize(1),gridsize(1)+1);
            grid_edges.y=linspace(0,obj.imsize(2),gridsize(2)+1);
            for x=1:gridsize(1)
                for y=1:gridsize(2)
                    pixel_thetas=repmat(theta,1,nTrials);
                    e0=[grid_edges.x(x) grid_edges.y(y)]';
                    e1=[grid_edges.x(x+1) grid_edges.y(y+1)]';
                    %Make pixel thetas uniformly distributied over the pixel
                    pixel_thetas(1:2,:)=rand(2,nTrials).*repmat(e1-e0,1,nTrials)+repmat(e0,1,nTrials);
                    theta_grid(:,:,x,y)=pixel_thetas;
                    sample_grid(:,:,:,:,x,y)=obj.simulateImage(pixel_thetas);
                end
            end
        end
    end

%     methods (Access=protected)
% %         function image=checkImage(obj, image)
% %             if size(image,1) ~= obj.imsize(1) || size(image,2) ~= obj.imsize(2) || size(image,3) ~= obj.imsize(3)
% %                 error('MAPPLEHSBase:checkImage', 'Invalid image shape');
% %             end
% %             if ~all(image>=0)
% %                 error('MAPPLEHSBase:checkImage', 'Invalid image');
% %             end
% %             image=double(image);
% %         end
%     end
%     methods (Static=true)
%         
%     end
    methods (Static=true)
        function viewRGBImage(image, lambda)
            if nargin==1
                lambda=125:-1:0;
            end
            colormap=HSData.hyperCM(lambda);
            colormap=colormap(1:size(image,1),:);
            RGB = HSData.makeRGB(image,colormap);
            dipshow(joinchannels('RGB',RGB));
            diptruesize(2000);
        end

        function viewDipImage(image)
            frames=permute(image,[2,3,1,4]);
            dipshow(dip_image(frames));
            diptruesize(2000);
            dipmapping('global');
            colormap('hot');
        end

    end
end
