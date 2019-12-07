clc;clear;close all;
dirs = ["train/","val/","evaluate/"];
FILEPATH = "../../HRRP_Data/";
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
            HRRPmap = max(abs(data.OMPmap),[],2);
            HRRPmap = squeeze(HRRPmap);
            label = data.label;
            save(FILEPATH+id+files(i).name,'HRRPmap','label');
            disp(FILEPATH+id+files(i).name)
        end
    end
end