clc;clear;close all;
dirs = ["train/","val/","evaluate/"];
FILEPATH = "../../mD_Data/OMP_Data/";
if ~exist(FILEPATH,'dir')
    mkdir(FILEPATH)
end
for id = dirs
    if ~exist(FILEPATH+id,'dir')
        mkdir(FILEPATH+id)
    end
end

for id = dirs
    files = dir(id);
    for i = 1:length(files)
        if exist(FILEPATH+id+files(i).name,'file')
            continue;
        end
        if endsWith(string(files(i).name),".mat")
            data = load(id+string(files(i).name));
            mDmap = max(abs(data.OMPmap),[],3);
            label = data.label;
            save(FILEPATH+id+files(i).name,'mDmap','label');
            disp(FILEPATH+id+files(i).name)
        end
    end
end