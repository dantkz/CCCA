function [ res ] = get_visualization(imgvals, m, img_size, capvals, isjet, bgcolor, varargin)
    if nargin<5
        isjet = true;
    end

    if nargin<6
        bgcolor = 0;
    end
    
    do_greyscale = 1;
    colR = eye(3); colt = zeros(3,1);
    if length(varargin)==2
        if ~isempty(varargin{1}) && ~isempty(varargin{2})
            do_greyscale = 0;
            colR = varargin{1};
            colt = varargin{2};
        end
    end

    if size(imgvals, 2)==3
        do_greyscale = 0;
        imgvals = imgvals(:);
    end
    
    if do_greyscale
        res = get_visualization_from_greyscale(imgvals, m, img_size, capvals, isjet, bgcolor);    
    else
        chN = length(imgvals)/nnz(m);
        nimgvals = reshape(imgvals, [nnz(m) chN]);
        nimgvals = bsxfun(@plus, (colR*(nimgvals'))', colt');
        res = zeros([img_size size(nimgvals,2)]);
        for i=1:size(nimgvals,2)
            res(:,:,i) = get_visualization_from_greyscale(nimgvals(:,i), m, img_size, capvals, isjet, bgcolor);
        end
    end
end

function [ res ] = get_visualization_from_greyscale(imgvals, m, img_size, capvals, isjet, bgcolor)
%GET_VISUALIZATION Summary of this function goes here
%   Detailed explanation goes here
    if length(img_size)==1
        img_size = [img_size img_size];
    end
    
    res = zeros(img_size);
    mask = zeros(img_size);
    mask(m) = 1;
    mask = logical(mask);
    res(m) = imgvals;
    
    if size(capvals,1) == 1 && size(capvals,2)==1 && capvals(1)==0
        res = res-min(res(:));
        res = res./max(res(:));
    elseif size(capvals,2) == 2
        res = res-capvals(1);
        res = res./(capvals(2)-capvals(1));
        res(res<0) = 0;
        res(res>1) = 1;
    else
        %
    end
    
    res(~mask) = bgcolor;
    %res = imresize(res, 2*[img_size(1) img_size(2)], 'nearest');
    if (isjet)
        res = ImGray2Pseudocolor(res, 'Jet', 256);
    end
end
