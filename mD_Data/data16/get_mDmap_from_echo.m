function mDmap = get_mDmap_from_echo(echo)
config;
index = 1;
mDmap = zeros(29,256);
for i = 1:29
    sk = echo(index:index+Coh_pulse_num-1);
    sk = fftshift(fft(sk,Coh_pulse_num));
    mDmap(i,:) = abs(sk);
    index = index+sw;
end