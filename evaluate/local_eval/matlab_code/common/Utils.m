% This file mainly adopted from 3DFeatNet
% (https://github.com/yewzijian/3DFeatNet)
classdef Utils
methods(Static)

    function data = loadPointCloud(fname, m,maxnumpts)
        % Read point cloud .bin files
        %
        % Arguments
        %   m Number of dimensions for each point (Default:4)
        %
        % Returns
        %   data  [nxm] matrix representing the points. The first 3 columns 
        %         usually denote the XYZ position;
        %   m     Number of columns
        %   
        
        if nargin == 1
            m = 6;
            maxnumpts = -1;
        end
        if nargin == 2
            maxnumpts = -1;
        end
        
        

        finfo = dir(fname);
        fsize = finfo.bytes;
        fid = fopen(fname);

        numPts = fsize / (4*m);
        data = fread(fid, [m, numPts], 'float')';
        if maxnumpts > 1
            s = RandStream('mt19937ar','Seed',0);
            maxnumpts = min(numPts,maxnumpts);
            used = randperm(numPts);
            used = used(1:maxnumpts);
            data = data(1:maxnumpts,:);
            
        end
        fclose(fid);
        
    end
    function data = loadKeypointIndices(fname)
        % Added by Juan
        if nargin == 1
            maxnumpts = -1;
        end
        if endsWith(fname, '.bin')
            finfo = dir(fname);
            fsize = finfo.bytes;
            fid = fopen(fname);

            numPts = fsize / (4);
            data = fread(fid, [1, numPts], 'int32')';
            fclose(fid);
        else
            data = csvread(fname);
        end


    end
    function ind = load_indices_by_order(attention, num)
        % Added by Juan
        [~,idx] = sort(attention, 'descend');
        ind = idx(1:num);
        
    end
    function ind = load_indices_by_thresh(attention, thresh)
        % Added by Juan
        
        ind = find(attention > thresh);
        
    end
    
    function data = loadPointCloud_instance(fname, m, scale)
        % Read point cloud .bin files
        %
        % Arguments
        %   m Number of dimensions for each point (Default:4)
        %
        % Returns
        %   data  [nxm] matrix representing the points. The first 3 columns 
        %         usually denote the XYZ position;
        %   m     Number of columns
        %   
        
        if nargin == 1
            m = 6;
            scale = 1.0;
        end
        if nargin == 2
            scale = 1.0;
        end
        data = Utils.loadPointCloud(fname, m);
        data = pointCloud(data(:,1:3).*scale);
    end
    function data = loadPointCloud_list(fnamelist, m, scale)
        % Read point cloud .bin files
        %
        % Arguments
        %   m Number of dimensions for each point (Default:4)
        %
        % Returns
        %   data  [nxm] matrix representing the points. The first 3 columns 
        %         usually denote the XYZ position;
        %   m     Number of columns
        %   
        
        if nargin == 1
            m = 6;
            scale = 1.0;
        end
        if nargin == 2
            scale = 1.0;
        end
        data = [];
        for i = 1:length(fnamelist)
            name = fnamelist{i};
            pc = Utils.loadPointCloud(name, m);
            data = [data;pc];
        end
        data = pointCloud(data(:,1:3).*scale);
    end
    
    function savePointCloud(cloud, fname)
        xyz = cloud.Location;
        intensity = cloud.Intensity;
        
        if isempty(intensity)
            intensity = zeros(size(xyz,1), 1);
        end
        
        xyzi = horzcat(xyz, intensity);
        
        fid = fopen(fname, 'w');
        fwrite(fid, xyzi', 'float');
        fclose(fid);
    end
    
    function savetobin(cloud, fname)
        dim = size(cloud,2)
        
        fid = fopen(fname, 'w');
        fwrite(fid, cloud', 'float');
        fclose(fid);
    end
    
    function [split_fnames, split_xyz] = splitTrainTest(filenames, xyz)
        
        TRAIN_PROPORTION = 0.9;
        num_train = floor(size(filenames, 1) * TRAIN_PROPORTION);
        
        split_xyz{1} = xyz(1:num_train, :);
        split_xyz{2} = xyz(num_train+1:end, :);
        
        split_fnames{1} = filenames(1:num_train);
        split_fnames{2} = filenames(num_train+1:end);
        
    end
    
    function data = load_descriptors(fname, m)
                
        % Arguments
        %   m  Number of dimensions for each feature. Should be 3+d, for the
        %      xyz coordinates and the d-dim feature
        
        if nargin == 1
            m = 131;
        end

        finfo = dir(fname);
        fsize = finfo.bytes;
        fid = fopen(fname);

        numPts = fsize / (4*m);
        
        assert(numPts == round(numPts));
        data = fread(fid, [m, numPts], 'float')';
        fclose(fid);
        
    end
    
    function [xyz_feature_dim, augmentations, data] = load_descriptor_metadata(fname)
        
        augmentations = [];
        
        f = fopen(fname);
        
        first_line = fgetl(f);
        tokens = strsplit(first_line, '\t');
        xyz_feature_dim = str2double(tokens(1:2));
        if length(tokens) >= 3
            augmentations = tokens(3);
        end
        
        iFrame = 1;
        while true
            line = fgetl(f);
            if line == -1
                break
            end
            
            tokens = strsplit(line, '\t');

            d(iFrame).Id = str2double(tokens{1});
            d(iFrame).Filename = tokens{2};
            d(iFrame).Label = 0;
            d(iFrame).q = [1, 0, 0, 0];
            d(iFrame).t = [0, 0, 0];
            d(iFrame).s = [1];

            if length(tokens) >= 3
                d(iFrame).Label = str2double(tokens{3});
            end
            qts = str2double(tokens(4:11));
            if length(tokens) >= 4
                d(iFrame).q = qts(1:4);
                d(iFrame).t = qts(5:7);
                d(iFrame).s = qts(8);
            end
            iFrame = iFrame + 1;
        end
        
        data = d;
    end
    
    function pcshow_mask(pts, mask)
        clf, hold on
        cmap = colormap('lines');
        pcshow(pts(mask, :), cmap(1,:))
        pcshow(pts(~mask, :), cmap(2,:))
    end
    
    function transformed = apply_transform(xyz, transform)
        
        xyz = xyz(:, 1:3);
        xyz(:,4) = 1;
        transformed = xyz * transform';
        
        transformed = transformed(:,1:3);
    end
    
    function ax = pcshow_multiple(cloud_all, transforms)        
        
        cla, hold on
        cmap = lines;
        for i = 1 : length(cloud_all)
            if isa(cloud_all{i}, 'pointCloud')
                cloud_all{i} = cloud_all{i}.Location;
            end
            if nargin >= 2
                cloud_all{i} = Utils.apply_transform(cloud_all{i}(:,1:3), transforms{i});
            end
            
            ax = pcshow(cloud_all{i}(:,1:3), cmap(i,:), 'MarkerSize', 10);
        end
        
    end
    
    function ax = pcshow_keypoints(model, xyz)        
        
        cla, hold on
        cmap = lines;
        cmap = lines;
        ax = pcshow(model(:,1:3), cmap(1,:),'MarkerSize', 5); hold on
        
        % Draws all keypoints
        scatter3(xyz(:,1), xyz(:,2), xyz(:,3), 200, 'r.')
        
    end
    
    function ax = pcshow_matches(model1, model2, xyz1, xyz2, matches12, varargin)
    % Plots matches between two point clouds
    %
    % Arguments
    %
    %   model1, model2: N x 3 matrix for the 2 point clouds
    %
    %   xyz1, xyz2: m x 3 matrix containing the xyz of keypoints
    %
    %   matches12: n x 2 matrix containing the putative matches,
    %              each row (idx1, idx2) indicates that
    %              xyz1(matches12(:,1), :) in model1 is matched to
    %              xyz2(matches12(:,2), :) in model2
    %
    % Keyword arguments:
    %   'inlierIdx': Indicates the rows of matches12 which are
    %                 inliers. If provided, will draw inliers and outliers
    %                 in green and red respectively. At most one of
    %                 'inlier_idx' or 'k' should be provided.
    %   'k': Limit on number of matches to draw to avoid cluttering the 
    %        figure
    %
              
        p = inputParser;
        validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
        addParameter(p, 'inlierIdx', []);
        addParameter(p,'k', -1);
        parse(p, varargin{:});
        cla, hold on
        inlierIdx = p.Results.inlierIdx;
        k = p.Results.k;
        
        if k <= 0 || k > size(matches12, 1)
            k = size(matches12, 1);
        end
        
        % Compute the spread of the two models, so that the gap between the
        % two models can be adjusted appropriately
        spread = 1 * std(model1(:,1));
        T1 = -max(model1(:,1)) - spread;
        T2 = -min(model2(:,1)) + spread;
        
        spread = 1 * std(model1(:,1));
        T1 = (-max(model1(:,1)) - spread)*0.8;
        T2 = (-min(model2(:,1)) + spread)*0.8;
        
        model1(:,1) = model1(:,1) + T1;
        model2(:,1) = model2(:,1) + T2;
        xyz1(:,1) = xyz1(:,1) + T1;
        xyz2(:,1) = xyz2(:,1) + T2;
        
        joint = [model1; model2];
        
        cmap = lines;
        ax = pcshow(model1(:,1:3), cmap(1,:),'MarkerSize', 10); hold on
        ax = pcshow(model2(:,1:3), cmap(2,:),'MarkerSize', 10); hold on
      
        % Draws all keypoints
        matchcolor = [0.1660 0.6740 0.1];
        scatter3(xyz1(:,1), xyz1(:,2), xyz1(:,3), 5,matchcolor)
        scatter3(xyz2(:,1), xyz2(:,2), xyz2(:,3), 5, matchcolor)
        
        % Draw lines between matches
        if isempty(inlierIdx)
            
            matches12 = matches12(randperm(size(matches12,1), k),:);
            
            xyz1_aligned = xyz1(matches12(:, 1), :);
            xyz2_aligned = xyz2(matches12(:, 2), :);
            
            for i = 1 : size(xyz1_aligned, 1)
                plot3([xyz1_aligned(i,1), xyz2_aligned(i,1)], ...
                     [xyz1_aligned(i,2), xyz2_aligned(i,2)], ...
                     [xyz1_aligned(i,3), xyz2_aligned(i,3)],'Color',matchcolor);
            end            
            
        else
            
            correct = false(size(matches12, 1), 1);
            correct(inlierIdx) = true;
            
            xyz1_aligned = xyz1(matches12(:, 1), :);
            xyz2_aligned = xyz2(matches12(:, 2), :);
            
            proportionCorrect = mean(correct);
            numCorrectToPlot = ceil(proportionCorrect*k);
            numWrongToPlot = k - numCorrectToPlot;
           
            % Draw lines, first draw inliers then outliers
            X = [xyz1_aligned(correct, 1)'; xyz2_aligned(correct, 1)'];
            X(3,:) = nan;
            Y = [xyz1_aligned(correct, 2)'; xyz2_aligned(correct, 2)'];
            Y(3,:) = nan;
            Z = [xyz1_aligned(correct, 3)'; xyz2_aligned(correct, 3)'];
            Z(3,:) = nan;
            sampled = randsample(size(X, 2), numCorrectToPlot);
            X = X(:, sampled); Y = Y(:, sampled); Z = Z(:, sampled);
            X = X(:); Y = Y(:); Z = Z(:);            
            plot3(X, Y, Z, 'g-')
            
            X = [xyz1_aligned(~correct, 1)'; xyz2_aligned(~correct, 1)'];
            X(3,:) = nan;
            Y = [xyz1_aligned(~correct, 2)'; xyz2_aligned(~correct, 2)'];
            Y(3,:) = nan;
            Z = [xyz1_aligned(~correct, 3)'; xyz2_aligned(~correct, 3)'];
            Z(3,:) = nan;
            sampled = randsample(size(X, 2), numWrongToPlot);
            X = X(:, sampled); Y = Y(:, sampled); Z = Z(:, sampled);
            X = X(:); Y = Y(:); Z = Z(:);
            plot3(X, Y, Z, 'r-');            
            
        end
        
        %xlabel('x'), ylabel('y'), zlabel('z')


    end
        
    
    function h = pcshow_pair(pts_1, pts_2, ind_1, ind_2)
        % pcshow in 2 separate, linked figures
        % Optionally take in indices of keypoints to plot
        
        plot_keypoints = nargin >= 4;
        
        figure, clf, hold on
        set(gcf,'position',[100 800 800 600])
        ax1 = pcshow(pts_1);

        if plot_keypoints
            plot3(pts_1(ind_1, 1), pts_1(ind_1, 2), pts_1(ind_1, 3), 'r*')
        end
        figure, clf, hold on
        set(gcf,'position',[900 800 800 600])
        ax2 = pcshow(pts_2);

        if plot_keypoints
            plot3(pts_2(ind_2, 1), pts_2(ind_2, 2), pts_2(ind_2, 3), 'r*')
        end
        h = linkprop([ax1,ax2],{'CameraPosition','CameraUpVector', 'CameraViewAngle', 'CameraTarget'}); 
        
    end
    
    function rgb = float2rgb(a, cm, range)
    %FLOAT2RGB Convert float to RGB.
    
    if nargin == 2
        range = [min(a) max(a)];
    end
    a = min(max((a - range(1)) / (range(2) - range(1)), 0), 1);
    

    cm_matrix = colormap(cm);
    ind = round(a * (size(cm_matrix, 1)-1)) + 1;
    rgb = cm_matrix(ind, :);
    end
    
    function labelXYZ()
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    end
    
    function transformed = transform(data, q, t, s)
        
        data(:,4) = 1;
        
        T = [s * quat2rotm(q); t];
        
        transformed = data * T;
        
    end
    
    function [delta_t, delta_deg] = compareTransform_v1(A, B)
        delta_t = norm(A(1:3, 4) - B(1:3, 4));
        delta_R = A(1:3, 1:3)' * B(1:3, 1:3);
        eul = rotm2eul(delta_R);
        delta_deg = sum(abs(eul)) * 180 / pi;
    end
    function [delta_t, delta_deg] = compareTransform(A, B)
        delta_t = norm(A(1:3, 4) - B(1:3, 4));
        delta_R = A(1:3, 1:3)' * B(1:3, 1:3);
       
        [rx, ry, rz] = GetEulerAngles(delta_R);
        delta_deg = sum(abs([rx, ry, rz])) * 180 / pi;
    end
    function pcshow_pcxyz(pts)
        figure, clf, hold on
        cmap = colormap('lines');
        if size(pts,2)==3
            pcshow(pts, cmap(1,:))
        elseif size(pts,1)==3
            pcshow(pts', cmap(1,:))
        end
    end
    
end
    
end