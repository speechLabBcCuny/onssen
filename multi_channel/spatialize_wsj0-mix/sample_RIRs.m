%
% Sample info to generate RIRs to spatialize wsj0-mix dataset
%
% The data generation process is as follows:
%
%for a mixture and its s1 and s2:
%       draw room length from [5m, 10m], room width from [5m, 10m], and room height from [3m, 4m];
%       draw T60 from [0.2, 0.6]s;
%       place the array center at l/2+n_1, w/2+n_2, with the height drawn from [1m, 2m], and n_1 and n_2 drawn from [-0.2m, 0.2m];
%       place K=4 omni-directional microphones on or inside a sphere with radius drawn from [7.5cm, 12,5cm]. The first two mics are on the sphere (symemtric according to its origin) and the rest two inside the sphere. The distance between two microphones has to be at least 5cm. 
%       draw speaker i location (xi, yi, zi) such that l/2+n_1-1.5<=xi<=l/2+n_1+1.5, w/2+n_2-1.5<=yi<=w/2+n_2+1.5, 1.5<=zi<=2, enforcing that each source/arraycenter distance is at least 0.5m and each source/source distance is at least 1m
%       (NOTE that in the test set, we generated four additional microphones for the evaluation of beamforming. The additonal four microphones are randomly placed inside the sphere)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2017-2018 Mitsubishi Electric Research Labs 
%                     (Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sample_RIRs()

    MAX_NUM_RIRS = 20000+5000+3000;
    
    mtype           = 'omnidirectional';% Type of microphone
    order           = -1;               % -1 equals maximum reflection order!
    dim             = 3;                % Room dimension
    hp_filter       = 1;                % Enable high-pass filter
    orientation     = [0, 0];           % Microphone orientation (rad), doesn't matter for omnidirectional microphones

    room_l_range    = [5, 10]; %(m)
    room_w_range    = [5, 10]; %(m)
    room_h_range    = [3, 4];  %(m)

    array_z_range   = [1, 2]; %(m), array height
    array_r_range   = [0.075, 0.125]; %(m), array radius
    array_xy_jittering = 0.2; %(m)

    T60_range       = [0.2, 0.6]; %s

    M               = 4; %number of microphones. NOTE that initially only four mics were simulated, but later we decided to add four more on the test data only
    S               = 3; %number of sources

    min_spk_spk_dist    = 1.0;
    min_spk_center_dist = 0.5;
    min_mic_mic_dist    = 0.05;

    INFO = struct('mic_pos',[],'spk_pos',[],'room_dimension',[],'T60',repmat({0},1,MAX_NUM_RIRS),'mtype',mtype,'order',order,'dim',dim,'orientation',orientation,'hp_filter',hp_filter);
    
    MAX_NUM_TRIAL = 1000;
    %generate slightly more as some cases may fail
    for kk = 1 : MAX_NUM_RIRS

        fprintf('processing %d/%d, ', kk, MAX_NUM_RIRS);

        seed = kk*7;

        rng(seed,'twister'); %even with parfor, the results are replicable

        room_l  = draw(room_l_range);
        room_w  = draw(room_w_range);
        room_h  = draw(room_h_range);

        room_dimension = [room_l, room_w, room_h]; % Room dimensions

        % mic array center position
        array_x = room_l/2 + (rand*2-1)*array_xy_jittering; % array_x is +-array_xy_jittering near the center
        array_y = room_w/2 + (rand*2-1)*array_xy_jittering; % array_y is +-array_xy_jittering near the center
        array_z = draw(array_z_range);
        array_center = [array_x, array_y, array_z];

        % mic array radius
        array_r = draw(array_r_range);

        % Receiver/mic positions - two on a sphere (symmetric with respect to the origin), and the other two inside the sphere
        alpha_      = draw([0,360])/360*2*pi;
        beta_       = draw([0,360])/360*2*pi;
        mic_pos_1   = array_r*[cos(alpha_)*sin(beta_), sin(alpha_)*sin(beta_), cos(beta_)];
        mic_pos_2   = -mic_pos_1; %the oposite position
        success     = false;
        num_trials  = 0;
        while success == false && num_trials < MAX_NUM_TRIAL
            mic_pos = [mic_pos_1; mic_pos_2]; %the first two mics are on the sphere
            for mm = 1 : M -2
                [tmpx,tmpy,tmpz] = draw_a_point_inside_a_ball(array_r);
                mic_pos = [mic_pos; tmpx, tmpy, tmpz];
            end
            success = check_mic_pos(mic_pos, array_r, min_mic_mic_dist);
            if success == false
                fprintf('#');
                num_trials = num_trials + 1;
            end
        end
        if num_trials == MAX_NUM_TRIAL
            fprintf('maximum number of trials reached on kk=%d, array radius=%.3f. skiping\n', kk, array_r);
            continue;
        end
        mic_pos     = bsxfun(@plus, mic_pos, array_center);

        % Speaker positions
        
        spk_x_range = [array_center(1)-1.5, array_center(1)+1.5];
        spk_y_range = [array_center(2)-1.5, array_center(2)+1.5];
        spk_z_range = [1.5, 2];

        success = false;
        num_trials = 0;
        while success == false && num_trials < MAX_NUM_TRIAL
            spk_pos = [];
            for ss = 1 : S
                spk_pos = [spk_pos; draw(spk_x_range), draw(spk_y_range), draw(spk_z_range)];
            end
            success = check_spks_pos(spk_pos, array_center, min_spk_spk_dist, min_spk_center_dist);
            if success == false
                fprintf('.');
                num_trials = num_trials + 1;
            end
        end
        if num_trials == MAX_NUM_TRIAL
            fprintf('maximum number of trials reached on kk=%d. skiping\n', kk);
            continue;
        end

        % for reverb
        T60             = draw(T60_range);  % Reverberation time (s)
        fprintf('T60=%f\n', T60);

        %
        % generate 4 more mics randomly inside the sphere. 
        % note: this was generated only for testing in the ICASSP 2018 paper.
        new_mic_pos = [];
        for mm = 1 : 4
            [tmpx,tmpy,tmpz] = draw_a_point_inside_a_ball(array_r);
            new_mic_pos = [new_mic_pos; tmpx,tmpy,tmpz];
        end
        new_mic_pos = bsxfun(@plus, new_mic_pos, array_center);
        mic_pos = [mic_pos ; new_mic_pos];

        INFO(kk).room_dimension = room_dimension;
        INFO(kk).mic_pos = mic_pos;
        INFO(kk).spk_pos = spk_pos;
        INFO(kk).T60 = T60;


    end

    save('rir_info.mat','INFO');
end

function d = dist(a, b)
    d = sqrt(sum((a-b).^2));
end

function [x,y,z] = draw_a_point_inside_a_ball(r)
    x = draw([-r,r]);
    r = sqrt(r*r-x*x);
    y = draw([-r,r]);
    r = sqrt(r*r-y*y);
    z = draw([-r,r]);
end

function [success] = check_mic_pos(mic_pos, array_r, min_mic_mic_dist)
    %
    % check whether mic locations are good
    %
    success = false;
    for i = 1 : size(mic_pos,1)
        if i >= 3 && dist(mic_pos(i,:),[0,0,0]) > array_r
            return;
        end
        for j = i + 1 : size(mic_pos,1)
            if dist(mic_pos(i,:), mic_pos(j,:)) < min_mic_mic_dist
                return;
            end
        end
    end
    success = true;
end

function [success] = check_spks_pos(spk_pos, array_center, min_spk_spk_dist, min_spk_center_dist)
    %
    % check whether speaker locations are good
    %
    success = false;
    for i = 1 : size(spk_pos,1)
        for j = i + 1 : size(spk_pos,1)
            if dist(spk_pos(i,:), spk_pos(j,:)) < min_spk_spk_dist
                return;
            end
        end
        if dist(spk_pos(i,:),array_center) < min_spk_center_dist
            return;
        end
    end
    success = true;
end

function num = draw(range)
    if range(1) > range(2)
        error('error\n');
    end
    num = (range(2)-range(1))*rand+range(1);
end
