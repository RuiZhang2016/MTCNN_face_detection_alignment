% Add paths
run startup.m

% Run caffe on cpu
caffe.set_mode_cpu();

% Load trained models
[PNet,RNet,ONet,LNet] = loadModels();

% Minimum size of face
minsize=50;

% Three steps's threshold
threshold=[0.5 0.8 0.7];

% Scale factor
factor=0.65;

% Read images
testFiles = dir(dataPath);
testFiles(1) = []; % Remove first two files
testFiles(1) = [];
faces = cell(0);

for ii=3:length(testFiles)
    ii
	img=imread(strcat(testFiles(ii).folder,'/',testFiles(ii).name));

    tic
    [boundingboxes points]=detectFace(img,minsize,PNet,RNet,ONet,LNet,threshold,false,factor);
	toc
    figure,imshow(img)
    plotResult(img,boundingboxes, points);
 	pause
end
