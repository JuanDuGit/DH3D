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


clear;
addpath('./external');
addpath('./common');
FEATURE_DIM = 128;
DRAW_ALL_PUTATIVE = false;  % If true, will draw all inlier/outlier matches
MAX_MATCHES = 1000; % Maximum number of inlier+outlier matches to draw
isPrintWrong = true;

ext_name = '_nms_res.bin';
DATA_FOLDER = '../demo_data';
RES_FOLDER = '../demo_data/res_local';


VISUALIZE_REG=true;
VISUAL_WRONG=false;


t_gt = [0.1374   -0.3046   -0.0592];
q_gt = [0.9892   -0.0026    0.0257    0.1444];
anc_idx = '642';
pos_idx = '268';



T_gt = [quat2rotm(q_gt), t_gt'];
        
               
% load anc pc
anc_pc_np = Utils.loadPointCloud(fullfile(DATA_FOLDER, sprintf('%s.bin', anc_idx)), 3); 
% load anc desc
anc_xyz_desc = Utils.load_descriptors(fullfile(RES_FOLDER, sprintf('%s%s', anc_idx, ext_name)) , sum(FEATURE_DIM+4));





% load pos pc
pos_pc_np = Utils.loadPointCloud(fullfile(DATA_FOLDER, sprintf('%s.bin', pos_idx)), 3);  
% load pos desc
pos_xyz_desc = Utils.load_descriptors(fullfile(RES_FOLDER, sprintf('%s%s', pos_idx, ext_name)), sum(FEATURE_DIM+4));



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


try
    [delta_t, delta_deg] = Utils.compareTransform(T_gt, estimateRt);
catch
    delta_t = 3;
    delta_deg = 6;
end

fail = false;
if delta_t > 2 || delta_deg > 5
    fail = true;

    if isPrintWrong
        fprintf('%s, ', anc_idx);
        fprintf('%s, ', pos_idx);

        fprintf('%f, %f, ', delta_t, delta_deg);
        fprintf('%.3f,  %i ', ...
                length(inlierIdx)/size(matches12, 1), trialCount);
        fprintf('\n');
    end

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