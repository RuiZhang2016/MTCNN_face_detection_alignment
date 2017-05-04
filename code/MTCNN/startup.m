% STARTUP sets helpful paths
home = '../../';
caffePath = fullfile(home, 'external', 'caffe', 'matlab');
caffeModelPath='./model';
pdollarToolboxPath=fullfile(home, 'toolbox');
dataPath=fullfile(home,'data');

% Add paths
addpath(genpath(caffePath));
addpath(genpath(pdollarToolboxPath));