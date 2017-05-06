function [ output_args ] = plotResult( img, boxes, points )
%PLOTRESULT shows detection results
%   img: a matrix containing the detected image, in a size of [m,n,3];
%   boxes: bounding box returned by MTCNN
%   points: feature locations returned by MTCNN

nbox=size(boxes,1);
figure,imshow(img);
hold on;
for ii=1:nbox
    plot(points(1:5,ii),points(6:10,ii),'r.','MarkerSize',6);
    rectangle('Position',[boxes(ii,1:2) boxes(ii,3:4)-boxes(ii,1:2)],...
    'Edgecolor','r','LineWidth',2);
end
hold off;
end

