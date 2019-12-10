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
            elseif startsWith(model_name,'Warhead1')
                bias.flip_theta = 1;
                bias.translation_z = 0;
                bias.complete_phi = 2;
            else
                bias.flip_theta = 1;
                bias.translation_z = 1;
                bias.complete_phi = 0;
            end
        else
            continue;
            bias.flip_theta = 0;
            bias.translation_z = 0;
            bias.complete_phi = 1;
        end
        rcsdata=load(RCS_DATA_PATH+string(rcs_model));
        RCS=rcsdata.RCSData;
        disp([rcs_model,' has been load!'])
        
        %% config video
        for i_set = SETS
            video_list = dir(char(ECHO_DATA_PATH+i_set));
            for i_video = 1:length(video_list)
                if startsWith(video_list(i_video).name,model_name) && ~exist(i_set+string(video_list(i_video).name),'file')
                    videodata = load(ECHO_DATA_PATH+i_set+string(video_list(i_video).name));
                    snr = split(video_list(i_video).name,'_');
                    snr = char(snr(2));
                    snr = str2double(snr(1:2));
                    para = videodata.para;
                    if model_name(1)=='W'
                        echo = get_precession_echo(para,RCS,bias,snr);
                        if model_name(8)=='1'
                            label = 2;
                        else
                            label = 1;
                        end
                    else
                        echo = get_roll_echo(para,RCS,bias,snr);
                        label = 0; 
                    end
                    
                    save(i_set+string(video_list(i_video).name),'echo','para');
                    disp(i_set+string(video_list(i_video).name))
                end
            end
        end
        
        
    end
end








