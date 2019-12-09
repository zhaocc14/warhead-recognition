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
        if startsWith(model_name,'Warhead1-')
            
            bias.flip_theta = 1;
            bias.translation_z = 0;
            bias.complete_phi = 2;
 
            rcsdata=load(RCS_DATA_PATH+string(rcs_model));
            RCS=rcsdata.RCSData;
            disp([rcs_model,' has been load!'])
            
            %% gen echo data
            for i_sample = 0:(SampleNum-1)
                for snr = [20,25,30]
                    para.precession_angle = rand()*15/180*pi;
                    para.precession_frequency = 0.3+rand()*2;
                    para.spin_frequency = 0.3+rand()*2;
                    para.los_ele = rand()*60/180*pi;
                    para.los_azi = rand()*360/180*pi;
                    para.init_coneaxis_azi = rand()*360/180*pi;
                    para.init_conespin_phi = rand()*360/180*pi;
                    para.label = '2';
                    
                    [echo,Mn] = get_precession_echo(para,RCS,bias,snr);
                        
                    if startsWith(model_name,'Warhead1-1') || startsWith(model_name,'Warhead1-2') 
                        i_set = SETS(1);
                    else
                        if i_sample<SampleNum * 0.6
                            i_set = SETS(2);
                        else
                            i_set = SETS(3);
                        end
                    end
                    
                    save(ECHO_DATA_PATH+i_set+model_name+"_"+num2str(snr)+"dB"+num2str(i_sample)+".mat",'echo','Mn','para');
                    
                    disp(i_set+model_name+"_"+num2str(snr)+"dB"+num2str(i_sample))
                    
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






