% 生成回波数据，生成HRRP数据
clc;clear;close all;
% config
config;

for i_set = SETS
    if ~exist(ECHO_DATA_PATH+i_set,'dir')
        mkdir(char(ECHO_DATA_PATH+i_set))
    end
    if ~exist(i_set,'dir')
        mkdir(char(i_set))
    end
end

rcs_list = dir(char(RCS_DATA_PATH));

for i_rcs = 1:length(rcs_list)
    %% config model parameters
    if endsWith(rcs_list(i_rcs).name,'.mat')
        rcs_model=rcs_list(i_rcs).name;
        model_name = split(rcs_model,'.');
        model_name = char(model_name(1));
        if startsWith(model_name,'W')
            if model_name=="Warhead2-5"
                bias.flip_theta = 0;
                bias.translation_z = 0;
                bias.complete_phi = 0;
            else
                bias.flip_theta = 1;
                bias.translation_z = 1;
                bias.complete_phi = 0;
            end
        else
            bias.flip_theta = 0;
            bias.translation_z = 0;
            bias.complete_phi = 1;
        end
        rcsdata=load(RCS_DATA_PATH+string(rcs_model));
        RCS=rcsdata.RCSData;
        disp([rcs_model,' has been load!'])
        
        %% config video
        for i_set = SETS
            video_list = dir(char(RD_DATA_PATH+i_set));
            for i_video = 1:length(video_list)
                if startsWith(video_list(i_video).name,model_name)
                    videodata = load(RD_DATA_PATH+i_set+string(video_list(i_video).name));
                    snr = split(video_list(i_video).name,'_');
                    snr = char(snr(2));
                    snr = str2double(snr(1:2));
                    video_match = videodata.data;
                    videodata.data = [];
                    para = videodata;
                    if model_name(1)=='W'
                        [echo,Mn] = get_precession_echo(para,RCS,bias,snr);
                        label = 1; 
                    else
                        [echo,Mn] = get_roll_echo(para,RCS,bias,snr);
                        label = 0; 
                    end
                    HRRPmap = get_HRRPmap_from_echo(echo,Mn);
                    HRRP_bj = max(video_match,[],2);
                    figure;imagesc(HRRPmap);
                    figure;imagesc(reshape(HRRP_bj,29,256));
                    save(ECHO_DATA_PATH+i_set+string(video_list(i_video).name),'echo','Mn','para');
                    save(i_set+string(video_list(i_video).name),'HRRPmap','label');
                end
            end
        end
        
        
    end
end


% precession_angle=10/180*pi;
% init_coneaxis_azi=0/180*pi;
% spin_frequency=1*2*pi;
% precession_frequency=1*2*pi;
% los_ele=30/180*pi;
% los_azi=270/180*pi;
% init_conespin_phi=0;






