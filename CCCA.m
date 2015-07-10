classdef CCCA 
    
    properties(GetAccess = 'public', SetAccess = 'public')

        options
                                    % options.iterN
                                    % options.sel

        parameters
                                    % parameters.F
                                    % parameters.K
                                    % parameters.M
                                    % parameters.meancontext
                                    % parameters.stdcontext

        % input
        img_size                    % size of the images
        Xs                          % input rgb intensity values 
        context                     % context vectors
        masks                       % mask of the intensity from images of size img_size
        
        % variables
        X                           % rotated intensity values 
        A                           % precomp matrices
        chN                         % number of channels

        % output 
        Z                           % prototype context vectors
        theta                       % phi parameters
        h                           % component weights
        sigmasq                     % noise variance 
        colR                        % rotation matrices
        colt                        % translation vectors
    end

    methods(Static = false)

        function obj = CCCA()
            obj.chN = 3;
        end

        function obj = SetParameters(obj, param)
            obj.parameters = param;
        end

        function obj = SetOptions(obj, opt)
            obj.options = opt;
        end
    
        function obj = SetInput(obj, inp)    
            if numel(inp.Xs)~=numel(inp.context) ...
                || numel(inp.Xs)~=numel(inp.masks)
                error('Error in SetInput! Different number of elements in input.Xs, input.context and input.masks!');
            end

            if ~any(strcmp(properties(obj), 'options')) ...
                || ~isfield(obj.options, 'sel') ...
                || isempty(obj.options.sel)
                obj.options.sel = 1:numel(inp.Xs);
            end

            if max(obj.options.sel)>numel(inp.Xs) || min(obj.options.sel)<1
                error('Error with options.sel!');
            end

            obj.Xs = inp.Xs(obj.options.sel);
            obj.context = inp.context(obj.options.sel);
            obj.masks = inp.masks(obj.options.sel);
            obj.img_size = inp.img_size;

            % Reset variables (that are computed per image)
            obj.X = {};
            obj.h = {};
            obj.colR = {};
            obj.colt = {}
            obj.A = {};

            obj = obj.NormalizeContext();
        end

        function res = saveobj(obj)
            res = obj;
            % input matrices
            res.Xs= {};
            res.context = {};
            res.masks = {};
            
            % variable matrices
            res.X = {};
            res.A = {};
        end
        
        function [obj errscore] = SaveImages(obj)
            [success, message] = mkdir('Result');
            if ~success
                error('Failed to create directory');
            end
            errscore = [];
            if (1)
                thetamat = reshape(obj.theta, [obj.parameters.M*obj.chN obj.parameters.F+1]);
                for i=1:numel(obj.X)
                    Y = (obj.A{i}*thetamat*[1; obj.h{i}(:)]);

                    Yimg = get_visualization(Y, obj.masks{i}, obj.img_size, [0 1], 0, 1, obj.colR{i}, obj.colt{i});
                    Xsimg= get_visualization(obj.Xs{i}, obj.masks{i}, obj.img_size, [0 1], 0, 1);

                    imwrite(Yimg, ['Result/rec' num2str(i) '.png'], 'png');
                    imwrite(Xsimg, ['Result/src' num2str(i) '.png'], 'png');
                    
                    % this reconstruction is not truncated to [0, 1] range as Yimg
                    % rgbrec = bsxfun(@plus, (obj.colR{i}*((reshape(Y, [length(Y)/obj.chN obj.chN]))'))', obj.colt{i}');

                    curerr = (Yimg(repmat(obj.masks{i}, [1 1 obj.chN]))-Xsimg(repmat(obj.masks{i}, [1 1 obj.chN]))).^2;
                    errscore(i,:) = [sum(curerr(:),1) size(obj.Xs{i},1)];
                    
                    if (0)
                        dlmwrite(['Result/factors_i' num2str(i) '.mat'], obj.A{i}*thetamat);
                        dlmwrite(['Result/colR_i' num2str(i) '.mat'], obj.colR{i});
                        dlmwrite(['Result/colt_i' num2str(i) '.mat'], obj.colt{i});
                        dlmwrite(['Result/h_i' num2str(i) '.mat'], obj.h{i});
                        dlmwrite(['Result/factors_i' num2str(i) '.mat'], obj.A{i}*thetamat);
                    end
                end
                dlmwrite('Result/errscore.mat', errscore);
            end
        end

        function [obj dataset] = GetInpaintTargets(obj, inptarget)
            dataset.img_size = obj.img_size;
            dataset.masks = cell(1,numel(obj.masks));
            dataset.context = cell(1,numel(obj.context));
            dataset.Xs = cell(1,numel(obj.Xs));

            for i=1:numel(obj.Xs)
                observedmask = logical(zeros(obj.img_size));
                halfmask = logical(zeros(obj.img_size));
                fullmask = reshape(obj.masks{i}, obj.img_size);
                [yy xx] = find(fullmask);
                xmax = max(xx); xmin = min(xx);

                % Inpaint Target
                if inptarget==1
                    % Inpaint RHS
                    halfmask(:,1:round(xmax/2)) = 1;
                elseif inptarget==2
                    % Used for elephants, cats and facades:
                    halfmask(:,1:round(xmin+(xmax-xmin)/2)) = 1; 
                else
                    % Inpaint LHS
                    halfmask(:,round(xmax/2):end) = 1; 
                end

                observedmask(halfmask) = fullmask(halfmask);
                selection = observedmask(fullmask);

                observedcontext = obj.context{i}(selection,:);
                observedcontext = bsxfun(@plus, bsxfun(@times, observedcontext, obj.parameters.stdcontext), obj.parameters.meancontext);

                observedXs = obj.Xs{i}(selection,:);

                dataset.masks{i} = observedmask(:);
                dataset.context{i} = observedcontext;
                dataset.Xs{i} = observedXs;
            end
        end
        
        function obj = SaveAppTransImages(obj, tag, fh, fcolR, fcolt, targets)
            %% Appearance Transfer from source to all targets
            [success message] = mkdir('AppTrans');
            if ~success
                error('Failed to create directory');
            end

            thetamat = reshape(obj.theta, [obj.parameters.M*obj.chN obj.parameters.F+1]);
            
            if (0)
                dlmwrite(['AppTrans/fh_' tag '.mat'], fh);
                dlmwrite(['AppTrans/fcolR_' tag '.mat'], fcolR);
                dlmwrite(['AppTrans/fcolt_' tag '.mat'], fcolt);
                dlmwrite(['AppTrans/targets_' tag '.mat'], targets);
            end

            for i=targets
                Y = (obj.A{i}*thetamat*[1; obj.h{i}(:)]);

                Yimg = get_visualization(Y, obj.masks{i}, obj.img_size, [0 1], 0, 1, obj.colR{i}, obj.colt{i});
                Xsimg = get_visualization(obj.Xs{i}, obj.masks{i}, obj.img_size, [0 1], 0, 1);
                
                fixY = (obj.A{i}*thetamat*[1; fh(:)]);
                fixYimg = get_visualization(fixY, obj.masks{i}, obj.img_size, [0 1], 0, 1, fcolR, fcolt);

                imwrite(fixYimg, ['AppTrans/fix_' tag '_' num2str(i) '.png']);
                imwrite(Yimg, ['AppTrans/rec' num2str(i) '.png']);
                imwrite(Xsimg, ['AppTrans/src' num2str(i) '.png']);
            end
        end

        function obj = NormalizeContext(obj)
            if isempty(obj.parameters.meancontext) || isempty(obj.parameters.stdcontext)
                disp('Computing mean and std of the context vectors');
                allcontexts = cell2mat(obj.context');
                meancont = mean(allcontexts);
                stdcont = std(allcontexts, [], 1);
                stdcont(isnan(stdcont)) = 1;
                clear('allcontexts', 'var');

                if isempty(obj.parameters.meancontext)
                    obj.parameters.meancontext = meancont;
                end
                if isempty(obj.parameters.stdcontext)
                    obj.parameters.stdcontext = stdcont;
                end
            end

            obj.context = cellfun(@(x) bsxfun(@rdivide, bsxfun(@minus, x, obj.parameters.meancontext), obj.parameters.stdcontext), obj.context, 'UniformOutput', false);
        end

        function obj = InitPhi(obj)
            % TODO: do some clustering before sampling
            allcontexts = cell2mat(obj.context');
            if (size(allcontexts,1)<obj.parameters.M)
                error('Error, M is greater than all context vectors');
            end
            selection = randperm(size(allcontexts,1), obj.parameters.M);
            obj.Z = allcontexts(selection,:);
        end
        
        function [rows cols data] = GetKNN(obj, c)
            % TODO use knn datastructure or/and GPU
            cc = sum(c.*c, 2);
            zz = sum(obj.Z'.*obj.Z', 1);
            data = cc(:, ones(1,size(obj.Z, 1))) + zz(ones(1, size(c,1)), :) - 2*c*obj.Z';
            
            [vals, p] = sort(data,2);
            rows = repmat((1:size(c,1))', [1 obj.parameters.K]);
            cols = p(:,1:obj.parameters.K);
            data = vals(:,1:obj.parameters.K);
            %data = data.^(0.5);
            data = 1./(0.01+data);
            data = bsxfun(@rdivide, data, sum(data,2));
            rows = rows(:);
            cols = cols(:);
            data = data(:);
        end
        
        function obj = ComputeAllA(obj)
            if isempty(obj.Z)
                obj = obj.InitPhi();
            end
            disp('Computing A...');
            obj.A = cell(1,numel(obj.context));
            for i=1:numel(obj.context)
                obj.A{i} = obj.ComputeA(obj.context{i});
            end
            disp('Done!');
        end

        function A = ComputeA(obj, context)
            [currows curcols curdata] = obj.GetKNN(context);
            rows = [];
            cols = [];
            data = [];
            for j=1:obj.chN
                rows = [rows; (j-1)*size(context,1) + currows];
                cols = [cols; (j-1)*obj.parameters.M + curcols];
                data = [data; double(curdata)];
            end
            A = sparse(rows, cols, data, obj.chN*size(context,1), obj.chN*obj.parameters.M);
        end
        
        function obj = FitTheta(obj)
            sizes = cellfun(@(x) size(x,1), obj.A, 'UniformOutput', false);
            objh = cell2mat(obj.h)';
            x = cell2mat(obj.X');

            hh = [ones(size(objh,1),1) objh];
            AA = cell2mat(obj.A');

            S = cell2mat(sizes');
            cS = [0; cumsum(S)];
            
            % TODO: Change the 4th argument of lsqr function call to change the precision of solving for obj.theta.
%             tic;
%             profile on;
            obj.theta = lsqr(@bfun,x,[], 300, [], [], obj.theta);
%             profile viewer;
%             profile off;
%             toc;

                function y = bfun(xx,transp_flag)
                    if strcmp(transp_flag,'transp')      % y = B'*x
                        AA_x = zeros(obj.parameters.M*obj.chN, numel(obj.A));
                        for i=1:length(S)
                            AA_x(:, i) = full(xx((cS(i)+1):cS(i+1))'*obj.A{i});
                        end
                        yy = AA_x*hh;
                        y = yy(:);
                        
                    elseif strcmp(transp_flag,'notransp') % y = B*x
                        t = reshape(xx, obj.parameters.M*obj.chN, 1+obj.parameters.F);

                        % The next 3 lines do exactly the same thing. 
                        %AA_t = AA * t;                                                      % Runs on single core only (at least on my machine)
                        %AA_t = cell2mat(cellfun(@(x) x*t, obj.A, 'UniformOutput', false)'); % Runs on single core and multiple cores (depending on OS, and it is slow)
                        %AA_t = mex_omp_smm(AA,t);                                           % Runs on multiple cores

                        if exist('mex_omp_smm')==3
                            AA_t = mex_omp_smm(AA,t);
                        else
                            AA_t = AA * t;
                            %AA_t = cell2mat(cellfun(@(x) x*t, obj.A, 'UniformOutput', false)');
                        end

                        y = zeros(size(AA_t,1), 1);
                        for i = 1:length(S)
                            y((cS(i)+1):cS(i+1),:) = AA_t((cS(i)+1):cS(i+1),:) * hh(i,:)';
                        end
                    end
                end
        end
        
        function obj = FitH(obj)
            thetamat = reshape(obj.theta, [obj.parameters.M*obj.chN obj.parameters.F+1]);
            obj.h = cellfun(@func, obj.A, obj.X, 'UniformOutput', false);
            function y = func(A, X)
                mu = A*thetamat(:,1);
                phi = A*thetamat(:,2:end);
                AA = (phi'*phi + obj.sigmasq*eye(size(obj.parameters.F)));
                xx = (phi'*(X-mu));
                [curh flag] = lsqr(AA,xx, [], 100);
                y = curh(:);
            end
        end
       
        function obj = FitSigma(obj)
            thetamat = reshape(obj.theta, [obj.parameters.M*obj.chN obj.parameters.F+1]);
            func = @(x,A,h) (x-A*thetamat*[1; h(:)])'*(x-A*thetamat*[1; h(:)])/size(x,1);
            allsigmasq = cellfun(func, obj.X, obj.A, obj.h, 'UniformOutput', false);
            obj.sigmasq = sum(cell2mat(allsigmasq))./length(allsigmasq);
        end
       
        
        function obj = FitRt(obj)
            thetamat = reshape(obj.theta, [obj.parameters.M*obj.chN obj.parameters.F+1]);
            for i=1:numel(obj.colR)
                Y = (obj.A{i}*thetamat*[1; obj.h{i}(:)]);
                Y = reshape(Y, [length(Y)/obj.chN obj.chN]);
                mY = mean(Y);
                mXs = mean(obj.Xs{i});
                T = (bsxfun(@minus, obj.Xs{i}, mXs))'*(bsxfun(@minus, Y, mY));
                [u s v] = svd(T);
                obj.colR{i} = u*v';
                obj.colt{i} = mXs' - obj.colR{i}*mY';
            end
            
        end
        
        function obj = RtXs(obj)
            func = @(col,R,t) (R\((bsxfun(@minus, col, t'))'))';
            obj.X = cellfun(func, obj.Xs, obj.colR, obj.colt, 'UniformOutput', false);
            obj.X = cellfun(@(x) x(:), obj.X, 'UniformOutput', false);
        end
        
        function obj = InitUnknown(obj)
            if isempty(obj.h)
                obj.h = {};
                obj.h = cellfun(@(x) randn(obj.parameters.F,1), cell(1,numel(obj.Xs)), 'UniformOutput', false);
            end
            if isempty(obj.sigmasq)
                obj.sigmasq = 1;
            end
            if isempty(obj.theta)
                obj.theta = [];
                obj.theta = zeros(obj.parameters.M*obj.chN*(obj.parameters.F+1), 1);
            end
            if isempty(obj.colR)
                obj.colR = {};
                obj.colR = cell(1,numel(obj.Xs));
                for i=1:numel(obj.Xs)
                    obj.colR{i} = eye(obj.chN);
                end
            end
            if isempty(obj.colt)
                obj.colt = {};
                obj.colt = cell(1,numel(obj.Xs));
                for i=1:numel(obj.Xs)
                    obj.colt{i} = zeros(obj.chN, 1);
                end
            end
        end

        function obj = Learn(obj)
            disp(['Learning started']);
            for iter=1:obj.options.iterN
                disp(['Iteration: ' num2str(iter) ' out of ' num2str(obj.options.iterN)]);

                disp('Rotating Xs');
                obj = obj.RtXs();
                disp('Fitting Theta');
                obj = obj.FitTheta();
                disp('Fitting Sigma');
                obj = obj.FitSigma();
                disp('Fitting H');
                obj = obj.FitH();
                disp('Fitting Rt');
                obj = obj.FitRt();

                disp(['Sigmasq = ' num2str(obj.sigmasq)]);
                disp(['=================================']);
            end
            disp(['Learning finished']);
        end

        function obj = Fit(obj)
            disp(['Fitting started']);
            for iter=1:obj.options.iterN
                disp(['Iteration: ' num2str(iter) ' out of ' num2str(obj.options.iterN)]);

                disp('Rotating Xs');
                obj = obj.RtXs();
                disp('Fitting H');
                obj = obj.FitH();
                disp('Fitting Rt');
                obj = obj.FitRt();
                
                disp(['=================================']);
            end
            disp(['Fitting finished']);
        end

    end
end
