clc;clear;
path = 'train';
d = dir(path);
while 1
    i = randi(length(d)-2);
    data = load([path,'\\',d(i+2).name]);
    data = data.data;
    imagesc(data(1:13,:));
    title(d(i+2).name)
    
end