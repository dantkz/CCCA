datafile = 'elephant_data/elephant_data_rgb.mat';
if exist(datafile, 'file')~=2
    prep_dataset();
end

%% Parameters and Options
parameters.F = 4; % Number of components
parameters.K = 8; % Number of nearest neighbours
parameters.M = 2500; % Number of prototype context vectors Z
parameters.meancontext = []; % Estimate from training data
parameters.stdcontext = []; % Estimate from training data 

options.iterN = 3;
options.sel = 1:32; % The indices of the data that are used for learning or fitting

%% Input dataset
input = load(datafile);

%% Learning the model

obj = CCCA();
obj = obj.SetParameters(parameters);
obj = obj.SetOptions(options);
obj = obj.SetInput(input);
obj = obj.ComputeAllA();
obj = obj.InitUnknown();
obj = obj.Learn();

obj.SaveImages();
stamp = datestr((fix(clock)),'yyyy-mm-dd-HH-MM-SS');

system(['mv Result Result_' stamp]);
save(['model_' stamp '.mat'], 'obj', '-v7.3');

%% Reconstruction of the Training and Test sets
options.sel = 1:126;
options.iterN = 10;
obj = obj.SetOptions(options);
obj = obj.SetInput(input);
obj = obj.ComputeAllA();
obj = obj.InitUnknown();
obj = obj.Fit();

[~, error_score] = obj.SaveImages();
system(['mv Result Reconstruction_' stamp]);

%% Appearance Transfer from sources{i} to targetss{i}.
sources = {126, 32, 35};
targetss = {[1 12 106 100 102 105], [20 26 31 35 56], [104 22 33 38 52]};
for i = 1:numel(sources)
    source = sources{i};
    targets = targetss{i};
    tag = ['i' num2str(source)];
    fh = obj.h{source}; fcolR = obj.colR{source}; fcolt = obj.colt{source};
    obj = obj.SaveAppTransImages(tag, fh, fcolR, fcolt, targets);    
    
    Xsimg = get_visualization(obj.Xs{source}, obj.masks{source}, obj.img_size, [0 1], 0, 1);
    imwrite(Xsimg, ['AppTrans/src' num2str(source) '.png']);
    
    Y = (obj.A{source}*reshape(obj.theta, [obj.parameters.M*obj.chN obj.parameters.F+1])*[1; obj.h{source}(:)]);
    Yimg = get_visualization(Y, obj.masks{source}, obj.img_size, [0 1], 0, 1, obj.colR{source}, obj.colt{source});
    imwrite(Yimg, ['AppTrans/rec' num2str(source) '.png']);
end
system(['mv AppTrans AppTrans_' stamp]);


%% Structured Image Inpainting
[obj inpaintinput]= obj.GetInpaintTargets(2);

inpobj = obj;
inpobj = inpobj.SetInput(inpaintinput);
inpobj = inpobj.ComputeAllA();
inpobj = inpobj.InitUnknown();
inpobj = inpobj.Fit();
for i = 1:numel(inpobj.context)
    tag = ['i' num2str(i)];
    fh = inpobj.h{i}; fcolR = inpobj.colR{i}; fcolt = inpobj.colt{i};
    obj = obj.SaveAppTransImages(tag, fh, fcolR, fcolt, [i]);

    Obsimg = get_visualization(inpobj.Xs{i}, inpobj.masks{i}, inpobj.img_size, [0 1], 0, 1);
    mask = obj.masks{i} & ~inpobj.masks{i};
    mask = reshape(mask, obj.img_size);
    Obsimg(repmat(mask, [1 1 3])) = repmat([1 0 1]', [1 nnz(mask)])';
    imwrite(Obsimg, ['AppTrans/obs' num2str(i) '.png']);
end
system(['mv AppTrans Inpainted_' stamp]);
