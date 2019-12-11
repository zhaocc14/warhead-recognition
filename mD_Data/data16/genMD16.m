% 生成回波数据，生成HRRP数据
clc;clear;close all;
% config
config;

for i_set = SETS
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
        if startsWith(model_name,'Warhead1-')
            
%             bias.flip_theta = 1;
%             bias.translation_z = 0;
%             bias.complete_phi = 2;
 
            rcsdata=load(RCS_DATA_PATH+string(rcs_model));
            RCS=rcsdata.RCSData;
            disp([rcs_model,' has been load!'])
            
            %% gen mD data
            for i_set = SETS
                echo_list = dir(char(ECHO_DATA_PATH+i_set));
                for i_video = 1:length(echo_list)
                    if startsWith(echo_list(i_video).name,model_name) && ~exist(i_set+string(echo_list(i_video).name),'file')
                        echodata = load(ECHO_DATA_PATH+i_set+string(echo_list(i_video).name));
                        echo = echodata.echo;
                        echo = echo(:,64);
                        mDmap = get_mDmap_from_echo(echo);
                        para = echodata.para;
                        label = 2;
                  
                        save(i_set+"mD"+echo_list(i_video).name,'mDmap','para','label');  
                        disp(i_set+"mD"+echo_list(i_video).name)
                    end
                    
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






