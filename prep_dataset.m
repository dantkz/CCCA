function prep_dataset()
    prep_dataset_horses();
    prep_dataset_elephants();
end

function prep_dataset_horses()
    datafile = 'horse_data/horses_fullres.mat';
    if exist(datafile, 'file')==2
        data = load(datafile);
        scalefactor = 0.25;
        scaled_filename = ['horse_data/horses_' num2str(scalefactor) '.mat'];
        scale_images(data, scalefactor, scaled_filename);

        data = load(scaled_filename); 
        filtsize = 45;
        targetparts = [1 2 3 4 5 6 7]; %yes, we skip the tail
        filename = 'horse_data/horse_data_rgb';
        get_context_1(data, filtsize, targetparts, filename);
    else
        disp([datafile ' not found.']);
    end
end

function prep_dataset_elephants()
    datafile = 'elephant_data/elephants_fullres.mat';
    if exist(datafile, 'file')==2
        data = load(datafile);
        scalefactor = 0.16;
        scaled_filename = ['elephant_data/elephants_' num2str(scalefactor) '.mat'];
        scale_images(data, scalefactor, scaled_filename);

        data = load(scaled_filename); 
        filtsize = 25;
        targetparts = [1 2 3 4 5 6 7]; 
        filename = 'elephant_data/elephant_data_rgb';
        get_context_1(data, filtsize, targetparts, filename);
    else
        disp([datafile ' not found.']);
    end
end

function get_context_1(data, filtsize, targetparts, filename)
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
    
    save(filename, 'Xs', 'context', 'masks', 'img_size');

end

function scale_images(data, scalefactor, filename)
    Parts = data.Parts;
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
    
    save(filename, '-v7.3', 'imgs', 'labels', 'masks', 'Parts');
end
