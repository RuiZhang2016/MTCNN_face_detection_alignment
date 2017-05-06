% STARTUP sets helpful paths
home = '../../';
addpath(strcat(home,'./external/caffe/matlab/+caffe/private'));
caffePath = fullfile(home, 'external', 'caffe', 'matlab');
pdollarToolboxPath=fullfile(home, 'toolbox');
dataPath=fullfile(home,'data');

% Add paths
addpath(genpath(caffePath));
addpath(genpath(pdollarToolboxPath));