function [echo,Mn]=get_roll_echo(parameters,RCS,bias,snr)
% Get echo within target doing precession
% Input :
%       parameters of precession posture and radar sight
%       RCS : RCS data
%       bias : flip,translation and complete
%       snr : snr
% Output:
%       echo
%       Mn : frequncey hopping code
config;
Mn=randi(M,1,Coh_pulse_num*Tsim);

init_rotationaxis_ele = parameters.init_rotationaxis_ele;
roll_frequency = parameters.roll_frequency;
los_ele = parameters.los_ele;
los_azi = parameters.los_azi;
init_rotationaxis_azi = parameters.init_rotationaxis_azi;
        

LOS=[sin(los_ele)*cos(los_azi);sin(los_ele)*sin(los_azi);cos(los_ele)];

flip_theta = bias.flip_theta;
translation_z = bias.translation_z;
complete_phi = bias.complete_phi;

sk=zeros(Coh_pulse_num*Tsim,1);
for n=1:Coh_pulse_num*Tsim
    tjump=n/PRF;
    rotationaxis=[sin(init_rotationaxis_ele)*cos(init_rotationaxis_azi),...
                                     sin(init_rotationaxis_ele)*sin(init_rotationaxis_azi),...
                                     cos(init_rotationaxis_ele)];
    
    es=[0,-rotationaxis(3),rotationaxis(2);rotationaxis(3),0,-rotationaxis(1);-rotationaxis(2),rotationaxis(1),0];
    Ts=eye(3)+es*sin(2*pi*roll_frequency*tjump)+es*es*(1-cos(2*pi*roll_frequency*tjump));

    los=Ts^-1*LOS;
    [phi,theta,~]=cart2sph(los(1),los(2),los(3));
%     phi = phi+init_conespin_phi;
    
    if flip_theta==1
        theta = round(1800-(pi/2-theta)/pi*1800);
    else
        theta =  round((pi/2-theta)/pi*1800);
    end
    
    if complete_phi == 0
        phi = mod(round(phi/pi*1800),900)+1;
        if phi>=451
            phi = 901-phi;
        end
    else
        phi = mod(round(phi/pi*1800),3600)+1;
    end
    
    disp([theta,phi]);
    
    AmpRes=RCS(:,theta,phi);
    
    if translation_z==1
        sk(n)=AmpRes(Mn(n))*exp(-1j*2*pi*(f0+Mn(n)*df)*2/c);
    else
        sk(n)=AmpRes(Mn(n));
    end
    
end

Psignal = (sk'*sk)/length(sk);
Pnoise = Psignal/(10^(snr/10));
sigma = sqrt(Pnoise/2);
echo = sk + sigma*randn(length(sk),1)+1j*sigma*randn(length(sk),1);
