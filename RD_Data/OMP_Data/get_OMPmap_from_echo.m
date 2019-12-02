function OMPmap = get_OMPmap_from_echo(echo,Mn)

config;

r=linspace(-c/4/B,c/4/B-c/2/256/B,Coh_pulse_num);
k=5;
posB=1;
posE=posB+Coh_pulse_num-1;
it=1;
v=linspace(-20,20,Coh_pulse_num);

RSP=zeros(29,256,256)+1j*zeros(29,256,256);
while(posE<=Coh_pulse_num*Tsim)
    sm=echo(posB:posE);
    Mnsub=Mn(posB:posE).';
    D=zeros(Coh_pulse_num,length(v)*length(r));
    for ir=1:length(r)
        for iv=1:length(v)
            D(:,(ir-1)*length(v)+iv)=exp(-1j*4*pi/c*(f0+Mnsub*df).*(r(ir)+v(iv)*(1:Coh_pulse_num).'*Tr));
        end
    end   
    x=omp(k, D, sm);
    P=reshape(x,length(v),length(r));
    RSP(it,:,:)=P;
    
    it=it+1;
    posB=posB+sw;
    posE=posB+Coh_pulse_num-1;
end
% RSP=(RSP/max(max(RSP)));
OMPmap = RSP;
