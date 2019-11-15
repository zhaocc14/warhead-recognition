function [echo,Mn]=get_precession_echo(parameter,RCS,bias,snr)
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

precession_angle = parameter.precession_angle;
init_coneaxis_azi = parameter.init_coneaxis_azi;
spin_frequency = parameter.spin_frequency * 2 * pi;
precession_frequency = parameter.precession_frequency * 2 * pi;
los_ele = parameter.los_ele;
los_azi = parameter.los_azi;
init_conespin_phi = parameter.init_conespin_phi;
LOS=[sin(los_ele)*cos(los_azi);sin(los_ele)*sin(los_azi);cos(los_ele)];

flip_theta = bias.flip_theta;
translation_z = bias.translation_z;
complete_phi = bias.complete_phi;

sk=zeros(Coh_pulse_num*Tsim,1);
for n=1:Coh_pulse_num*Tsim
    tjump=n/PRF;
 
    Rinit=[[cos(precession_angle)*cos(init_coneaxis_azi+pi);cos(precession_angle)*sin(init_coneaxis_azi+pi);sin(precession_angle)],...
        [cos(init_coneaxis_azi-pi/2);sin(init_coneaxis_azi-pi/2);0],...
        [sin(precession_angle)*cos(init_coneaxis_azi);sin(precession_angle)*sin(init_coneaxis_azi);cos(precession_angle)]];
    ep=[0,-1,0;1,0,0;0,0,0];
    Tp=eye(3)+ep*sin(precession_frequency*tjump)+ep*ep*(1-cos(precession_frequency*tjump));
    oz=Tp*Rinit*[0;0;1];
    
    es=[0,-oz(3),oz(2);oz(3),0,-oz(1);-oz(2),oz(1),0];
    Ts=eye(3)+es*sin(spin_frequency*tjump)+es*es*(1-cos(spin_frequency*tjump));

    A=Ts*Tp*Rinit;
    los=A^-1*LOS;
    [phi,theta,~]=cart2sph(los(1),los(2),los(3));
    phi = phi+init_conespin_phi;
    
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
