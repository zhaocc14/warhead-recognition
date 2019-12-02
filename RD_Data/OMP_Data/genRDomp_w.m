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
    if endsWith(rcs_list(i_rcs).name,'.mat') && startsWith(rcs_list(i_rcs).name,'W')
        rcs_model=rcs_list(i_rcs).name;
        model_name = split(rcs_model,'.');
        model_name = char(model_name(1));
        
        if model_name=="Warhead2-5"
            bias.flip_theta = 0;
            bias.translation_z = 0;
            bias.complete_phi = 0;
        else
            bias.flip_theta = 1;
            bias.translation_z = 1;
            bias.complete_phi = 0;
        end

        
        rcsdata=load(RCS_DATA_PATH+string(rcs_model));
        RCS=rcsdata.RCSData;
        disp([rcs_model,' has been load!'])
        
        %% config video
        for i_set = SETS
            echo_list = dir(char(ECHO_DATA_PATH+i_set));
            for i_echo = 1:length(echo_list)
                if startsWith(echo_list(i_echo).name,model_name) && ~exist(i_set+string(echo_list(i_echo).name),'file')
                    echodata = load(ECHO_DATA_PATH+i_set+string(echo_list(i_echo).name));
                    
                    OMPmap = get_OMPmap_from_echo(echodata.echo,echodata.Mn);
                    label = echodata.para.label;

                    save(i_set+string(echo_list(i_echo).name),'OMPmap','label');
                    disp(i_set+string(echo_list(i_echo).name))
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






