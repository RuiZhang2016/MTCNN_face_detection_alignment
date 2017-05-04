% add paths
run startup.m

%run caffe on cpu
caffe.set_mode_cpu();

%load caffe models
prototxtDir =strcat(caffeModelPath,'/det1.prototxt');
modelDir = strcat(caffeModelPath,'/det1.caffemodel');
PNet=caffe.Net(prototxtDir,modelDir,'test');
prototxtDir = strcat(caffeModelPath,'/det2.prototxt');
modelDir = strcat(caffeModelPath,'/det2.caffemodel');
RNet=caffe.Net(prototxtDir,modelDir,'test');	
prototxtDir = strcat(caffeModelPath,'/det3.prototxt');
modelDir = strcat(caffeModelPath,'/det3.caffemodel');
ONet=caffe.Net(prototxtDir,modelDir,'test');
prototxtDir =  strcat(caffeModelPath,'/det4.prototxt');
modelDir =  strcat(caffeModelPath,'/det4.caffemodel');
LNet=caffe.Net(prototxtDir,modelDir,'test');

%minimum size of face
minsize=10;

%three steps's threshold
threshold=[0.6 0.7 0.7];

%scale factor
factor=0.709;

% read images
testFiles = dir(dataPath);
testFiles(1) = []; % Remove first two files
testFiles(1) = [];

for ii=1:length(testFiles)
    ii
	img=imread(strcat(testFiles(ii).folder,'/',testFiles(ii).name));
    img = imresize(img,[200,200]);
	%we recommend you to set minsize as x * short side
    tic
    [boudingboxes points]=DetectFace(img,minsize,PNet,RNet,ONet,LNet,...
    threshold,false,factor);
	toc
    faces{i,1}={boudingboxes};
	faces{i,2}={points'};
	%show detection result
	numbox=size(boudingboxes,1);
	figure,imshow(img)
	hold on; 
	for j=1:numbox
		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
		r=rectangle('Position',[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
    end
    hold off; 
 	pause
end
% save result box landmark