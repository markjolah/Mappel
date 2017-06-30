
classdef MappelTest < matlab.unittest.TestCase
    properties (ClassSetupParameter)
        modelname = {'Gauss2DMAP','Gauss2DMLE','Gauss2DsMAP','Gauss2DsMLE'};
        imsize = {'[4,5]', '[7,5]', '[8, 8]', '[15, 12]'};
        psfsigma={'[1.0, 1.0]', '[0.8, 1.14]'};  
    end
    properties (TestParameter)
        estimationMethod=MappelBase.EstimationMethods;
        max_samples={10,333,1000};
    end

    properties
        nSamples=1000;
        imsz;
        model;
    end

    methods(TestClassSetup)
        function createModel(tc,modelname,imsize,psfsigma)
            tc.model=feval(str2func(modelname),str2num(imsize),str2num(psfsigma)); %#ok<*ST2NM>
            tc.imsz=[tc.model.getStats().sizeY tc.model.getStats().sizeX];
        end
    end

    methods(TestClassTeardown)
        function destroyModel(tc)
            delete([tc.model]);
        end
    end

    methods (Test)
        function testSamplePrior(tc)
            thetas=tc.model.samplePrior(tc.nSamples);
            tc.verifySize(thetas,[tc.model.nParams tc.nSamples]);
            tc.verifyTrue(tc.model.thetaInBounds(thetas)~=0);
        end

        function testModelImage(tc)
            thetas=tc.model.samplePrior(tc.nSamples);
            model_ims=tc.model.modelImage(thetas);
            tc.verifySize(model_ims,[tc.imsz tc.nSamples]);
            tc.verifyGreaterThanOrEqual(model_ims,0.0);
        end

        function testSimulateImage(tc)
            thetas=tc.model.samplePrior(tc.nSamples);
            ims=tc.model.simulateImage(thetas);
            tc.verifySize(ims,[tc.imsz tc.nSamples]);
            tc.verifyGreaterThanOrEqual(ims,0.0);
        end

        function testLLH(tc)
            thetas=tc.model.samplePrior(tc.nSamples);
            ims=tc.model.simulateImage(thetas);
            llh=tc.model.LLH(ims, thetas);
            tc.verifyEqual(length(llh),tc.nSamples);
            tc.verifyTrue(all(isfinite(llh)));
        end
        
        function testCRLB(tc)
            thetas=tc.model.samplePrior(tc.nSamples);
            crlb=tc.model.CRLB(thetas);
            tc.verifySize(crlb,[tc.model.nParams, tc.nSamples]);
            tc.verifyTrue(all(isfinite(crlb(:))));
            %tc.verifyTrue(all(crlb(:)>=0.)); %This is not true for some very low intensities
        end

        function testEstimateMAP(tc, estimationMethod)
            thetas=tc.model.samplePrior(tc.nSamples);
            ims=tc.model.simulateImage(thetas);
            [etheta, crlb, llh, stats]=tc.model.estimateMAP(ims, estimationMethod);
            tc.verifySize(etheta,[tc.model.nParams, tc.nSamples]);
            tc.verifySize(crlb, [tc.model.nParams, tc.nSamples]);
            tc.verifyEqual(length(llh),tc.nSamples);
            tc.verifyTrue(all(isfinite(llh)));
            tc.verifyTrue(all(all(isfinite(etheta))));
            tc.verifyInstanceOf(stats,'struct');
            if isempty(strfind('CGauss', estimationMethod))
                %CGauss fails to return finite values or valid theta estimates
                %in many cases so we can't assert that these sanity checks will
                %pass
                tc.verifyTrue(all(etheta(:)>=0.));
                tc.verifyTrue(all(isfinite(crlb(:))));
                %tc.verifyTrue(all(crlb(:)>=0.));
                llh2=tc.model.LLH(ims, etheta);
                tc.verifyEqual(llh, llh2);
                crlb2=tc.model.CRLB(etheta);
                tc.verifyEqual(crlb, crlb2);
            end
        end

        function testEstimateMAPDebug(tc, estimationMethod)
            theta=tc.model.samplePrior(1);
            im=tc.model.simulateImage(theta);
            [etheta, crlb, llh, stats, sample, sample_rllh]=tc.model.estimateMAPDebug(im, estimationMethod);
            tc.verifySize(etheta,[tc.model.nParams, 1]);
            tc.verifySize(crlb, [tc.model.nParams, 1]);
            nsamples=size(sample,2);
            tc.verifyEqual(length(sample_rllh),nsamples);
            tc.verifyTrue(isfinite(llh));
            tc.verifyTrue(all(isfinite(etheta)));
            tc.verifyInstanceOf(stats,'struct');
            if isempty(strfind(estimationMethod,'CGauss'))
                %CGauss fails to return finite values or valid theta estimates
                %in many cases so we can't assert that these sanity checks will
                %pass
                if isempty(strfind(tc.model.Name,'MLE'))
                    tc.verifyTrue(all(etheta>0.));
                else
                    tc.verifyTrue(all(etheta>=0.)); %MLE methods are allowed to estimate 0
                end
                tc.verifyTrue(all(isfinite(crlb(:))));
                %tc.verifyTrue(all(crlb(:)>=0.));
                llh2=tc.model.LLH(im, etheta);
                tc.verifyEqual(llh, llh2);
                crlb2=tc.model.CRLB(etheta);
                tc.verifyEqual(crlb, crlb2);
                tc.verifyTrue(all(isfinite(sample(:))));
                tc.verifyTrue(all(isfinite(sample_rllh)));
                if isempty(strfind(tc.model.Name,'MLE'))
                    tc.verifyTrue(all(sample(:)>0.));
                else
                    tc.verifyTrue(all(sample(:)>=0.)); %MLE methods are allowed to estimate 0
                end
            end
        end
        
        function testEstimatePosterior(tc, max_samples)
            thetas=tc.model.samplePrior(tc.nSamples);
            ims=tc.model.simulateImage(thetas);
            [mean, cov]=tc.model.estimatePosterior(ims, max_samples);
            tc.verifySize(mean,[tc.model.nParams, tc.nSamples]);
            tc.verifySize(cov, [tc.model.nParams, tc.model.nParams, tc.nSamples]);
            tc.verifyTrue(all(isfinite(mean(:))));
            tc.verifyTrue(all(isfinite(cov(:))));
            tc.verifyTrue(all(mean(:)>=0.));
        end

    end
end
