% Copyright (C) 2020 Juan Du (Technical University of Munich)
% For more information see <https://vision.in.tum.de/research/vslam/dh3d>
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.


% Visualizes matches from oescriptors
%
% i.e. no augmentations, etc
clear;
addpath('./external');
addpath('./common');
FEATURE_DIM = 128;
DRAW_ALL_PUTATIVE = false;  % If true, will draw all inlier/outlier matches
MAX_MATCHES = 1000; % Maximum number of inlier+outlier matches to draw
isPrintWrong = false;



VISUALIZE_REG=true;
VISUAL_WRONG=false;

%% configurations
% when evaluating our local descriptor with other 3D detectors, we extract and save the
% dense feature map for each point cloud and select the nearest point and its corresponding feature for
% each given keypoint.
LOAD_KPTS = false; 
Kpts_Folder = ''; 
Kpts_Name = '';
KPTS_FOLDER = fullfile(Kpts_Folder, Kpts_Name);


% set the path to the generated descriptors
Desc_Folder = ''; 
DESC_FOLDER = fullfile(Desc_Folder, desc_name);
desc_name = 'DH3D';
ext_name = '_nms_res.bin';
with_att = 1;

% set the path to the testing set
PC_FOLDER = './data/oxford_test_local';
test_dataset = read_txt_oxford(fullfile(PC_FOLDER, 'oxford_test_local_gt.txt'));

% set the path to save the evaluation results
Out_Folder = './eval_res/'; 
result_save_file = fullfile(Out_Folder, sprintf('%s_%s.txt', desc_name, Kpts_Name));
fid = fopen(result_save_file,'w');

    

wrong_counter = 0;
inlier_ratio_array = [];
trial_count_array = [];
delta_t_array = [];
delta_deg_array = [];
total_num = size(test_dataset,1);



%% test on each pair
 for i = 1:1:total_num
        if mod(i,100) == 0
           fprintf('%d/%d, ', i, total_num);
        end
        
        anc_idx = test_dataset(i, 1);
        pos_idx = test_dataset(i, 2);
        t_gt = test_dataset(i, 3:5);
        q_gt = test_dataset(i, 6:9);
        T_gt = [quat2rotm(q_gt), t_gt'];
        
               
        % load anc pc
        anc_pc_np = Utils.loadPointCloud(fullfile(PC_FOLDER, sprintf('%d.bin', anc_idx)), 6); 
        % load anc desc
        anc_xyz_desc = Utils.load_descriptors(fullfile(DESC_FOLDER, sprintf('%d%s', anc_idx, ext_name)), sum(FEATURE_DIM+3+with_att));

        % load kp
        if LOAD_KPTS
            anc_kpts_load = Utils.loadPointCloud(fullfile(KPTS_FOLDER, sprintf('%d_xyz.bin', anc_idx)), 3); 
            [dist1, anc_desc_ind] = pdist2(anc_xyz_desc(:,1:3), anc_kpts_load, 'euclidean', 'smallest', 1);
            anc_xyz_desc = anc_xyz_desc(anc_desc_ind,:);
        end
            
      
        % load pos pc
        pos_pc_np = Utils.loadPointCloud(fullfile(PC_FOLDER, sprintf('%d.bin', pos_idx)), 6);  
        % load pos desc
        pos_xyz_desc = Utils.load_descriptors(fullfile(DESC_FOLDER, sprintf('%d%s', pos_idx, ext_name)), sum(FEATURE_DIM+3+with_att));

        % load pos kpts
        if LOAD_KPTS
            pos_kpts_load = Utils.loadPointCloud(fullfile(KPTS_FOLDER, sprintf('%d_xyz.bin', pos_idx)), 3); 
            [dist2, pos_desc_ind] = pdist2(pos_xyz_desc(:,1:3),pos_kpts_load, 'euclidean', 'smallest', 1);
            pos_xyz_desc = pos_xyz_desc(pos_desc_ind,:);
        end

        
        anc_kpts = anc_xyz_desc(:, 1:3);
        pos_kpts = pos_xyz_desc(:, 1:3);
        anc_desc = anc_xyz_desc(:, 4:FEATURE_DIM+3);
        pos_desc = pos_xyz_desc(:, 4:FEATURE_DIM+3);
                

            
        % match
        [dist, matches12] = pdist2(pos_desc, anc_desc, 'euclidean', 'smallest', 1);
        matches12 = [1:length(matches12); matches12]';
        cloud1_pts = anc_kpts(matches12(:,1), :);
        cloud2_pts = pos_kpts(matches12(:,2), :);
            

            
        [estimateRt, inlierIdx, trialCount] = ransacfitRt([cloud1_pts'; cloud2_pts'], 1.0, false);
           
        fprintf(fid, '%d, ', anc_idx);
        fprintf(fid, '%d, ', pos_idx);
        fprintf(fid, '%.3f,  %i, ', ...
                 length(inlierIdx)/size(matches12, 1), trialCount);
        for ii = 1:size(estimateRt,1)
            fprintf(fid,'%.6f %.6f %.6f %.6f',estimateRt(ii,:));
        end 
        fprintf(fid,'\n');

        try
            [delta_t, delta_deg] = Utils.compareTransform(T_gt, estimateRt);
        catch
            delta_t = 3;
            delta_deg = 6;
        end
        
        fail = false;
        if delta_t > 2 || delta_deg > 5
            fail = true;
            wrong_counter = wrong_counter + 1;
            

            if isPrintWrong
                fprintf('%d, ', anc_idx);
                fprintf('%d, ', pos_idx);

                fprintf('%f, %f, ', delta_t, delta_deg);
                fprintf('%.3f,  %i ', ...
                        length(inlierIdx)/size(matches12, 1), trialCount);
                fprintf('\n');
            end


        else
            % record the inlier/outlier ratio
            inlier_ratio_array = [inlier_ratio_array, length(inlierIdx)/size(matches12, 1)];
            trial_count_array = [trial_count_array, trialCount];
            delta_t_array = [delta_t_array, delta_t];
            delta_deg_array = [delta_deg_array, delta_deg];
        end



        if (VISUALIZE_REG || (fail && VISUAL_WRONG))
            figure(1); clf
            if DRAW_ALL_PUTATIVE
                Utils.pcshow_matches(anc_pc_np, pos_pc_np, ...
                     anc_kpts,pos_kpts, ...
                     matches12, 'inlierIdx', inlierIdx, 'k', MAX_MATCHES);

            else
                Utils.pcshow_matches(anc_pc_np, pos_pc_np, ...
                                     anc_kpts, pos_kpts, ...
                                     matches12(inlierIdx, :), 'k', MAX_MATCHES);
            end
            title('Matches')

            % Show alignment
            figure(2); clf
            Utils.pcshow_multiple({anc_pc_np, pos_pc_np}, {eye(4), estimateRt});
            title('Alignment')



            % Show alignment
            figure(3); clf
            Utils.pcshow_multiple({anc_pc_np, pos_pc_np}, {eye(4), T_gt});
            title('GroundTruth')

        end

 end
fprintf(fid, '\nwrong counter: %d\n%.4f\n%.4f\n%.4f\n', wrong_counter, 100*( 1- wrong_counter/total_num), mean(inlier_ratio_array), mean(trial_count_array));

fprintf(fid, '%.4f +- %.4f\n%.4f +- %.4f\n', mean(delta_t_array), std(delta_t_array), mean(delta_deg_array), std(delta_deg_array));

fclose(fid);

