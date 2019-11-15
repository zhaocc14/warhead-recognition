% config
%% dir path
RCS_DATA_PATH = "..\RCSData\";
RD_DATA_PATH = "..\RD_Data\data16\";
ECHO_DATA_PATH = "..\ECHO_Data\";
SETS = ["evaluate\","train\","val\"];
TASK = ["precession","roll"];

%% constant
c=3e8;
f0=9e9;
df=16e6;
PRF=10e3;
Tr=1/PRF;
Fs=50e6;
Tsim=8;
Coh_pulse_num=256;
M=64;
sw=64;
B=16e6;