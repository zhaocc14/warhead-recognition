% load('Debris-d1-018_30dB.mat')
% load('data16\evaluate\Warhead2-10_20dB12.mat')

clc;clear;
path = 'data16\\train';
d = dir(path);
while 1
    i = randi(length(d)-2);
    load([path,'\\',d(i+2).name]);
%     data = data.data;
    data = max(data,[],2);
    data=reshape(data,29,256);
    data=data/max(max(data))
%     for p=1:13
%         data(p,:)=data(p,:)/max(data(p,:));
%     end
    imagesc(data(1:13,:));
    title(d(i+2).name)
    
end

% [l,h,w]=size(data);
% for n=1:1
% for i=1:l
%     frame=reshape(data(i,:,:),h,w);
%     frame=frame/max(max(frame));
%     frame=frame.^5;
%     imagesc(frame);
%     title(num2str(n))
%     f(i)=getframe;
% end
% pause(0.5)
% end