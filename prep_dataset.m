function prep_dataset(varargin)
    if isempty(varargin)
        prep_dataset_horses();
        prep_dataset_elephants();
        prep_dataset_cats();
        prep_dataset_facades();
    else
        if strcmp(varargin{1}, 'horse')
            prep_dataset_horses();
        elseif strcmp(varargin{1}, 'elephant')
            prep_dataset_elephants();
        elseif strcmp(varargin{1}, 'cat')
            prep_dataset_cats();
        elseif strcmp(varargin{1}, 'facade')
            prep_dataset_facades();
        end
    end
end

function prep_dataset_horses()
    datafile = 'horse_data/horses_fullres.mat';
    filename = 'horse_data/horse_data_rgb.mat';

    if exist(filename, 'file')==2
        disp([filename ' already exists.']);
        return;
    end

    if exist(datafile, 'file')==2
        data = load(datafile);
        scalefactor = 0.25;
        scaled_filename = ['horse_data/horses_' num2str(scalefactor) '.mat'];
        scale_images(data, scalefactor, scaled_filename);

        data = load(scaled_filename); 
        filtsize = 45;
        targetparts = [1 2 3 4 5 6 7]; %yes, we skip the tail
        get_context(data, filtsize, targetparts, filename);
    else
        disp([datafile ' not found.']);
    end
end

function prep_dataset_elephants()
    datafile = 'elephant_data/elephants_fullres.mat';
    filename = 'elephant_data/elephant_data_rgb.mat';

    if exist(filename, 'file')==2
        disp([filename ' already exists.']);
        return;
    end
    
    if exist(datafile, 'file')==2
        data = load(datafile);
        scalefactor = 0.16;
        scaled_filename = ['elephant_data/elephants_' num2str(scalefactor) '.mat'];
        scale_images(data, scalefactor, scaled_filename);

        data = load(scaled_filename); 
        filtsize = 25;
        targetparts = [1 2 3 4 5 6 7]; 
        get_context(data, filtsize, targetparts, filename);
    else
        disp([datafile ' not found.']);
    end
end

function prep_dataset_cats()
    output_folder = 'cat_data/';
    filename = [output_folder 'cat_data_rgb.mat'];

    if exist(filename, 'file')==2
        disp([filename ' already exists.']);
        return;
    end

    dataset_path = 'C:/Users/dturmukh/Documents/Datasets/VOC_Parts_Dataset/ObjectsParts/';
    
    category_path = [dataset_path 'VOCdevkit/VOC2010/ImageSets/Main/'];
    img_path = [dataset_path 'VOCdevkit/VOC2010/JPEGImages/'];
    anno_path = [dataset_path 'trainval.tar/Annotations_Part/'];

    % We need 
    % part2ind.m and mat2map.m 
    % files that come with Pascal VOC Parts dataset
    path(path, [anno_path '../']);
    
    target = 'cat_trainval';
    target_cls = 8;

    imgsdir = 'cat_data/cat_trainval/imgs/';
    masksdir = 'cat_data/cat_trainval/instmask/';
    partsdir = 'cat_data/cat_trainval/partmask/';
    
    [imgsdir masksdir partsdir] = get_images_from_pascal_voc(category_path, anno_path, img_path, target, target_cls, output_folder);

    filtsize = 45;
    get_context_from_pascal_voc(imgsdir, masksdir, partsdir, filtsize, filename);
end

function prep_dataset_facades()
    output_folder = 'facade_data/';
    filename = [output_folder 'facade_data_rgb.mat'];

    if exist(filename, 'file')==2
        disp([filename ' already exists.']);
        return;
    end

    dataset_path = 'C:/Users/dturmukh/Documents/Datasets/ECP facade dataset/cvpr2010/';
    
    img_path = [dataset_path 'images/'];
    part_path = [dataset_path 'ground_truth_2011/'];

    imgsdir = [output_folder 'imgs/'];
    partsdir = [output_folder 'partmask/'];
    
    [imgsdir partsdir] = get_images_from_ecp_facades(part_path, img_path, output_folder);

    filtsize = 21;
    targetparts = [0 1 2 4 5];
    get_context_from_ecp_facades(imgsdir, partsdir, filtsize, targetparts, filename);
end

function [] = get_context_from_ecp_facades(imgsdir, partsdir, filtsize, targetparts, filename)
    imgsnames = dir([imgsdir, '/', '*.png']);
    sizes = cell(1,numel(imgsnames));
    for img_idx=1:numel(imgsnames)
        idxname = num2str(img_idx, '%.4d');
        img = imread([imgsdir idxname '.png']);
        sizes{img_idx} = size(img);
    end
    img_size = max(cell2mat(sizes'));
    img_size = img_size(1:2);

    imgN = numel(imgsnames);
    Xs = cell(1,imgN);
    masks = cell(1,imgN);
    context = cell(1,imgN);
    
    filts = makeFilters(filtsize);
    partsN = length(targetparts);
        
    for img_idx=1:numel(imgsnames)
        idxname = num2str(img_idx, '%.4d');
        timg = double(imread([imgsdir idxname '.png']))/255;
        img = zeros([img_size 3]);
        img(1:size(timg,1), 1:size(timg,2),:) = timg;
        
        tpart_mask = imread([partsdir idxname '.png']);
        part_mask = zeros(img_size);
        part_mask(1:size(tpart_mask,1), 1:size(tpart_mask,2)) = tpart_mask;
        
        tinst_mask = ones(size(timg(:,:,1)));
        curmask2d = zeros(img_size);
        curmask2d(1:size(tinst_mask,1), 1:size(tinst_mask,2)) = tinst_mask;
        curmask2d(part_mask==6) = 0;
        curmask2d(part_mask==3) = 0;
        
        masks{img_idx} = logical(curmask2d(:));
        img = reshape(img, [img_size(1)*img_size(2) 3]);
        Xs{img_idx} = img(masks{img_idx}, :);
        
        
        filtresp = zeros(size(masks{img_idx}(:),1), partsN*size(filts,3));
        
        for i=1:partsN
            bcurmask2d = zeros(size(curmask2d)+2*filtsize);
            bcurmask2d(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2)) = part_mask==targetparts(i);
            for j=1:size(filts,3)
                tmp = filter2(filts(:,:,j), bcurmask2d, 'same');
                tmp = tmp(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2));
                filtresp(:,(i-1)*size(filts,3)+j) = tmp(:);
            end
        end
        
        [xx yy] = meshgrid(1:img_size(2), 1:img_size(1));
        yy = yy-round(size(timg,1)/2);
        xx = xx-round(size(timg,2)/2);
    
        curcontext = [filtresp(masks{img_idx}, :) xx(masks{img_idx}) yy(masks{img_idx})];
        
        context{img_idx} = single(curcontext);
    end

    save(filename, '-v7.3', 'Xs', 'context', 'masks', 'img_size');
    
end

function [imgsdir partsdir] = get_images_from_ecp_facades(part_path, img_path, output_folder)
    imgs = dir([part_path, '/', '*.png']);
    
    imgsdir = [output_folder 'imgs/'];
    partsdir = [output_folder 'partmask/'];
    
    mkdir(output_folder);
    mkdir(imgsdir);
    mkdir(partsdir);
    
    idx = 1;
    for ii = 1:numel(imgs)
        imname = imgs(ii).name(1:end-4);
    
        img = double(imread([img_path, '/', imname '.jpg']))/255;
        part_mask = imread([part_path, '/', imname '.png'])/128;
        part_mask = part_mask(:,:,1) + part_mask(:,:,2)*3;% + 9*(part_mask(:,:,3)/2);
        part_mask(part_mask==8) = 4;
        part_mask(part_mask==7) = 3;
        
        idxname = num2str(idx, '%.4d');
    
        winlabels = bwlabel(part_mask==2); %window labels
        winN = max(winlabels(:));
        winwidths = zeros(winN,1);
        for i=1:winN
            tmp = winlabels==i;
            tmp = sum(tmp,2);
            tmp = tmp(tmp>0);
            winwidths(i) = median(tmp);
        end
        
        medW = mean(winwidths);
        
        % scale based on window width
        scaleval = (7/medW); % magic number
        cur_part_mask = imresize(part_mask, scaleval, 'nearest');
        cur_img = imresize(img, scaleval, 'bicubic');
    
        imwrite(cur_img, [imgsdir idxname '.png']);
        imwrite(cur_part_mask, [partsdir idxname '.png']);
    
        idx = idx+1;
    end
end

function F=makeFilters(SUP)
  NF = 4;
  F=zeros(SUP,SUP,2*NF);
  F(1:ceil(SUP/2),:,1) = 1;
  F(floor(SUP/2):end,:,2) = 1;
  F(:,1:ceil(SUP/2),3) = 1;
  F(:,floor(SUP/2):end,4) = 1;
  
  SUP2 = 2*round((SUP-1)/4)+1;
  shift = (SUP-SUP2)/2;
  
  cc = ceil(SUP/2);
  F((shift+1):cc, cc+(1:SUP2) - ceil(SUP2/2), 5) = 1;
  F(cc:(cc+floor(SUP2/2)), cc+(1:SUP2) - ceil(SUP2/2), 6) = 1;
  
  F(cc+(1:SUP2) - ceil(SUP2/2), (shift+1):cc, 7) = 1;
  F(cc+(1:SUP2) - ceil(SUP2/2), cc:(cc+floor(SUP2/2)), 8) = 1;
  
end


function [imgsdir instmaskdir partsdir] = get_images_from_pascal_voc(category_path, anno_path, img_path, target, target_cls, output_folder)
    category = [category_path target '.txt'];

    fio = fopen(category);
    filemask = textscan(fio, '%s %f');
    fclose(fio);
    filemask = filemask{1}(filemask{2}==1);

    pimap = part2ind();     % part index mapping

    mkdir(output_folder);
    mkdir([output_folder target]);
    imgsdir = [output_folder target '/imgs/'];
    instmaskdir = [output_folder target '/instmask/'];
    partsdir = [output_folder target '/partmask/'];

    mkdir(imgsdir);
    mkdir(instmaskdir);
    mkdir(partsdir);

    idx = 1; 
    for ii = 1:numel(filemask)
        imname = filemask{ii};

        if exist([anno_path, imname '.mat'], 'file')

            img = imread([img_path, '/', imname '.jpg']);
            % load annotation -- anno
            load([anno_path, imname]);

            [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);

            %% Collect information about instances in the current image
            instances.id = unique(inst_mask(cls_mask==target_cls));
            instances.size_of = zeros(1,numel(instances.id));
            instances.has_eyes = zeros(1,numel(instances.id));
            instances.distance = zeros(1,numel(instances.id)); %distance between eyes

            for jj=1:numel(instances.id)
                instances.size_of(jj) = nnz(inst_mask==instances.id(jj));
                unique_parts = unique(part_mask(inst_mask==instances.id(jj)));
                if nnz(unique_parts==2)>0 && nnz(unique_parts==3)>0 
                    instances.has_eyes(jj) = 1;
                    cur_inst_part_mask = part_mask;
                    cur_inst_part_mask(inst_mask~=instances.id(jj)) = 0;
                    [eye1y eye1x] = find(cur_inst_part_mask==2);
                    eye1y = mean(eye1y);
                    eye1x = mean(eye1x);
                    [eye2y eye2x] = find(cur_inst_part_mask==3);
                    eye2y = mean(eye2y);
                    eye2x = mean(eye2x);
                    dist = sqrt((eye1y-eye2y)^2 + (eye1x-eye2x)^2);
                    instances.distance(jj) = dist;
                end
            end
            
            if idx==413
                keyboard;
            end
            
            %% Filter based on size, eye visibility, etc.
            for jj=1:numel(instances.id)
                if (instances.size_of(jj)>7000) && instances.has_eyes(jj)==1
                    %% get bbox
                    curinst = inst_mask==instances.id(jj);
                    [yy xx] = find(curinst);
                    minxx = min(xx); maxxx = max(xx);
                    minyy = min(yy); maxyy = max(yy);
                    width = maxxx-minxx+1; height = maxyy-minyy+1;
                    center_x = round(minxx+width/2); center_y = round(minyy+height/2);
                    % border of at least 20%
                    width = round(width*1.2); height = round(height*1.2);
                    minxx = max(1,round(center_x-width/2));
                    minyy = max(1,round(center_y-height/2));
                    bbox = [minxx, minyy, width, height];

                    cur_img = img; 
                    cur_img = imcrop(cur_img, bbox);

                    cur_inst_mask = zeros(size(inst_mask)); cur_inst_mask(inst_mask==instances.id(jj)) = 1;
                    cur_inst_mask = imcrop(cur_inst_mask, bbox);

                    cur_part_mask = part_mask; cur_part_mask(inst_mask~=instances.id(jj)) = 0;
                    cur_part_mask = imcrop(cur_part_mask, bbox);

                    % scale
                    scaleval = 16/instances.distance(jj); %magic number
                    cur_part_mask = imresize(cur_part_mask, scaleval, 'nearest'); % 'nearest' for sharper edges
                    cur_inst_mask = imresize(cur_inst_mask, scaleval, 'nearest');
                    cur_img = imresize(cur_img, scaleval, 'bicubic');

                    % check if eyes are still visible after rescaling
                    cur_inst_part_mask = cur_part_mask;
                    cur_inst_part_mask(cur_inst_mask~=instances.id(jj)) = 0;
                    [eye1y eye1x] = find(cur_inst_part_mask==2);
                    eye1y = mean(eye1y);
                    eye1x = mean(eye1x);
                    [eye2y eye2x] = find(cur_inst_part_mask==3);
                    eye2y = mean(eye2y);
                    eye2x = mean(eye2x);
                    if ~isnan([eye1y eye1x eye2y eye2x])
                        has_eyes = 1;
                    else
                        has_eyes = 0;
                    end

                    idxname = num2str(idx, '%.4d');

                    % final image has to be at least 250x250. if smaller, then the part mask is uninformative
                    if (size(cur_img,1)<250 && size(cur_img,2)<250 && has_eyes) 

                        imwrite(cur_img,       [imgsdir idxname '.png']);
                        imwrite(cur_inst_mask, [instmaskdir idxname '.png']);
                        imwrite(cur_part_mask, [partsdir idxname '.png']);

                        idx = idx+1;                       
                    end
                end
            end

            if (0)
                % display annotation
                subplot(2,2,1); imshow(img); title('Image');
                subplot(2,2,2); imshow(cls_mask, cmap); title('Class Mask');
                subplot(2,2,3); imshow(inst_mask, cmap); title('Instance Mask');
                subplot(2,2,4); imshow(part_mask, cmap); title('Part Mask');
                drawnow;
            end
        end
    end
end

function get_context_from_pascal_voc(imgsdir, instmaskdir, partmaskdir, filtsize, filename)

    imgsnames = dir([imgsdir '*.png']);
    sizes = cell(1,numel(imgsnames));
    for img_idx=1:numel(imgsnames)
        idxname = num2str(img_idx, '%.4d');
        img = imread([imgsdir idxname '.png']);
        sizes{img_idx} = size(img);
    end
    img_size = max(cell2mat(sizes'));
    img_size = img_size(1:2);
    
    imgN = numel(imgsnames);
    Xs = cell(1,imgN);
    masks = cell(1,imgN);
    context = cell(1,imgN);

    filts = makeLMfilters(filtsize);
    targetparts = 1:17;
    partsN = length(targetparts);
    
    for img_idx=1:imgN
        idxname = num2str(img_idx, '%.4d');
        timg = double(imread([imgsdir idxname '.png']))/255;
        img = zeros([img_size 3]);
        img(1:size(timg,1), 1:size(timg,2),:) = timg;
        
        tmask = imread([instmaskdir idxname '.png']);
        curmask2d = zeros(img_size);
        curmask2d(1:size(tmask,1), 1:size(tmask,2)) = tmask;
        masks{img_idx} = logical(curmask2d(:));
        
        tpart_mask = imread([partmaskdir idxname '.png']);
        part_mask = zeros(img_size);
        part_mask(1:size(tpart_mask,1), 1:size(tpart_mask,2)) = tpart_mask;
        
        filtresp = zeros(size(masks{img_idx}(:),1), (1+partsN)*size(filts,3));
        bcurmask2d = zeros(size(curmask2d)+2*filtsize);
        bcurmask2d(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2)) = curmask2d;
        for j=1:size(filts,3)
            tmp = filter2(filts(:,:,j), bcurmask2d, 'same');
            tmp = tmp(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2));
            filtresp(:,j) = tmp(:);
        end
        for i=1:partsN
            bcurmask2d = zeros(size(curmask2d)+2*filtsize);
            bcurmask2d(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2)) = part_mask==i;
            for j=1:size(filts,3)
                tmp = filter2(filts(:,:,j), bcurmask2d, 'same');
                tmp = tmp(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2));
                filtresp(:,i*size(filts,3)+j) = tmp(:);
            end
        end

        curcontext = filtresp(masks{img_idx}, :);

        [yy1 xx1] = find(part_mask==2); [yy2 xx2] = find(part_mask==3);
        yy1 = mean(yy1); xx1 = mean(xx1);
        yy2 = mean(yy2); xx2 = mean(xx2);
        xx1 = (xx1+xx2)/2; yy1 = (yy1+yy2)/2;
        mideye_img= zeros(size(curmask2d));
        mideye_img(round(yy1), round(xx1)) = 1;
        mideye_img = bwdist(mideye_img);
        curcontext = [curcontext mideye_img(masks{img_idx})];
        
        context{img_idx} = single(curcontext);

        img = reshape(img, [length(img(:))/3 3]);
        Xs{img_idx} = img(masks{img_idx},:);
        disp([num2str(img_idx) '/' num2str(imgN)]);
    end
    
    save(filename, '-v7.3', 'Xs', 'context', 'masks', 'img_size');
end

function get_context(data, filtsize, targetparts, filename)
    img_size = max(cell2mat(cellfun(@size, data.imgs, 'UniformOutput', false)'));
    img_size = img_size(1:2);
    
    imgN = length(data.imgs);
    Xs = cell(1,imgN);
    masks = cell(1,imgN);
    context = cell(1,imgN);
    
    filts = makeLMfilters(filtsize);
    partsN = length(targetparts);
    
    for img_idx=1:imgN
        h = img_size(1);
        w = img_size(2);
        curmask2d = zeros(img_size);
        curmask2d(1:size(data.masks{img_idx},1), 1:size(data.masks{img_idx},2)) = data.masks{img_idx};
        masks{img_idx} = logical(curmask2d(:));
        
        [xx yy] = meshgrid(1:w, 1:h);
        xx = (xx(:)-1)/(w-1);
        yy = (yy(:)-1)/(h-1);
        
        coors = [xx yy];
        context{img_idx} = coors(masks{img_idx},:);
         
        filtresp = zeros(size(masks{img_idx}(:),1), (1+partsN)*size(filts,3));
        bcurmask2d = zeros(size(curmask2d)+2*filtsize);
        bcurmask2d(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2)) = double(curmask2d);
        for j=1:size(filts,3)
            tmp = filter2(filts(:,:,j), bcurmask2d, 'same');
            tmp = tmp(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2));
            filtresp(:,j) = tmp(:);
        end
        
        for i=targetparts
            bcurmask2d = zeros(size(curmask2d)+2*filtsize);
            bcurmask2d(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2)) = double(data.labels{img_idx}(:,:,i));
            for j=1:size(filts,3)
                tmp = filter2(filts(:,:,j), bcurmask2d, 'same');
                tmp = tmp(filtsize+1:filtsize+size(curmask2d,1), filtsize+1:filtsize+size(curmask2d,2));
                filtresp(:,i*size(filts,3)+j) = tmp(:);
            end
        end
        
        context{img_idx} = single([context{img_idx} filtresp(masks{img_idx}, :)]);
       
        curcolors = zeros(nnz(masks{img_idx}), size(data.imgs{img_idx},3));
        for ch=1:size(data.imgs{img_idx},3)
            tmp = data.imgs{img_idx};
            tmp = tmp(:,:,ch);
            curcolors(:,ch) = tmp(masks{img_idx});
        end
        
        Xs{img_idx} = curcolors;
        disp([num2str(img_idx) '/' num2str(imgN)]);
    end
    
    save(filename, '-v7.3', 'Xs', 'context', 'masks', 'img_size');
end

function scale_images(data, scalefactor, filename)
    imgN = numel(data.imgs);
    
    masks = cell(1,imgN);
    imgs = cell(1,imgN);
    labels = cell(1,imgN);
    
    for img_idx = 1:imgN
        img = imresize(data.imgs{img_idx}, scalefactor, 'bicubic');
        mask = imresize(double(data.masks{img_idx}), scalefactor, 'bicubic');
        curlabels = imresize(double(data.labels{img_idx}), scalefactor, 'bicubic');
        
        lmask = mask>0.4;
        
        img = img./repmat(mask, [ 1 1 size(img,3)]);
        img(~repmat(lmask, [ 1 1 size(img,3)])) = 0;
        curlabels = curlabels./repmat(mask, [ 1 1 size(curlabels,3)]);
        curlabels(~repmat(lmask, [ 1 1 size(curlabels,3)])) = 0;
        
        imgs{img_idx} = img;
        masks{img_idx} = logical(lmask);
        labels{img_idx} = curlabels;
        disp([num2str(img_idx) '/' num2str(imgN)]);
    end
    
    Parts = data.Parts;
    save(filename, '-v7.3', 'imgs', 'labels', 'masks', 'Parts');
end

