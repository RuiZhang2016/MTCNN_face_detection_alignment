function [totalBoxes points] = DetectFace(img,minsize,PNet,RNet,ONet,LNet,threshold,fastresize,factor)
	%im: input image
	%minsize: minimum of faces' size
	%pnet, rnet, onet: caffemodel
	%threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
	%fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
	factorCount=0;
	totalBoxes=[];
	points=[];
	[h,w]=size(img);
	minl=min([w h]);
    
    %img=single(img);
    
	if fastresize
		imData=(single(img)-127.5)*0.0078125;
    end
    m=12/minsize;
	minl=minl*m;
	%creat scale pyramid
    scales=[];
	while (minl>=12)
		scales=[scales m*factor^(factorCount)];
		minl=minl*factor;
		factorCount=factorCount+1;
	end
	%first stage
	for j = 1:size(scales,2)
		scale=scales(j);
		hs=ceil(h*scale);
		ws=ceil(w*scale);
		if fastresize
			imData=imResample(imData,[hs ws],'bilinear');
		else 
			imData=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;
        end
        PNet.blobs('data').reshape([hs ws 3 1]);
		out=PNet.forward({imData});
		boxes=generateBoundingBox(out{2}(:,:,2),out{1},scale,threshold(1));
		%inter-scale nms
		pick=nms(boxes,0.5,'Union');
		boxes=boxes(pick,:);
		if ~isempty(boxes)
			totalBoxes=[totalBoxes;boxes];
		end
	end
	numbox=size(totalBoxes,1);
	if ~isempty(totalBoxes)
		pick=nms(totalBoxes,0.7,'Union');
		totalBoxes=totalBoxes(pick,:);
		bbw=totalBoxes(:,3)-totalBoxes(:,1);
		bbh=totalBoxes(:,4)-totalBoxes(:,2);
		totalBoxes=[totalBoxes(:,1)+totalBoxes(:,6).*bbw totalBoxes(:,2)+totalBoxes(:,7).*bbh totalBoxes(:,3)+totalBoxes(:,8).*bbw totalBoxes(:,4)+totalBoxes(:,9).*bbh totalBoxes(:,5)];	
		totalBoxes=rerec(totalBoxes);
		totalBoxes(:,1:4)=fix(totalBoxes(:,1:4));
		[dy edy dx edx y ey x ex tmpw tmph]=pad(totalBoxes,w,h);
	end
	numbox=size(totalBoxes,1);
	if numbox>0
		%second stage
 		tempimg=zeros(24,24,3,numbox);
		for k=1:numbox
			tmp=zeros(tmph(k),tmpw(k),3);
			tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
			tempimg(:,:,:,k)=imResample(tmp,[24 24],'bilinear');
		end
        tempimg=(tempimg-127.5)*0.0078125;
		RNet.blobs('data').reshape([24 24 3 numbox]);
		out=RNet.forward({tempimg});
		score=squeeze(out{2}(2,:));
		pass=find(score>threshold(2));
		totalBoxes=[totalBoxes(pass,1:4) score(pass)'];
		mv=out{1}(:,pass);
		if size(totalBoxes,1)>0		
			pick=nms(totalBoxes,0.7,'Union');
			totalBoxes=totalBoxes(pick,:);     
            totalBoxes=bbreg(totalBoxes,mv(:,pick)');	
            totalBoxes=rerec(totalBoxes);
		end
		numbox=size(totalBoxes,1);
		if numbox>0
			%third stage
			totalBoxes=fix(totalBoxes);
			[dy edy dx edx y ey x ex tmpw tmph]=pad(totalBoxes,w,h);
            tempimg=zeros(48,48,3,numbox);
			for k=1:numbox
				tmp=zeros(tmph(k),tmpw(k),3);
				tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
				tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
			end
			tempimg=(tempimg-127.5)*0.0078125;
			ONet.blobs('data').reshape([48 48 3 numbox]);
			out=ONet.forward({tempimg});
			score=squeeze(out{3}(2,:));
			points=out{2};
			pass=find(score>threshold(3));
			points=points(:,pass);
			totalBoxes=[totalBoxes(pass,1:4) score(pass)'];
			mv=out{1}(:,pass);
			bbw=totalBoxes(:,3)-totalBoxes(:,1)+1;
            bbh=totalBoxes(:,4)-totalBoxes(:,2)+1;
            points(1:5,:)=repmat(bbw',[5 1]).*points(1:5,:)+repmat(totalBoxes(:,1)',[5 1])-1;
            points(6:10,:)=repmat(bbh',[5 1]).*points(6:10,:)+repmat(totalBoxes(:,2)',[5 1])-1;
			if size(totalBoxes,1)>0				
				totalBoxes=bbreg(totalBoxes,mv(:,:)');	
                pick=nms(totalBoxes,0.7,'Min');
				totalBoxes=totalBoxes(pick,:);  				
                points=points(:,pick);
			end
		end
		numbox=size(totalBoxes,1);
		%extended stage
		if numbox>0 
			tempimg=zeros(24,24,15,numbox);
			patchw=max([totalBoxes(:,3)-totalBoxes(:,1)+1 totalBoxes(:,4)-totalBoxes(:,2)+1]');
			patchw=fix(0.25*patchw);	
			tmp=find(mod(patchw,2)==1);
			patchw(tmp)=patchw(tmp)+1;
			pointx=ones(numbox,5);
			pointy=ones(numbox,5);
			for k=1:5
				tmp=[points(k,:);points(k+5,:)];
				x=fix(tmp(1,:)-0.5*patchw);
				y=fix(tmp(2,:)-0.5*patchw);
				[dy edy dx edx y ey x ex tmpw tmph]=pad([x' y' x'+patchw' y'+patchw'],w,h);
				for j=1:numbox
					tmpim=zeros(tmpw(j),tmpw(j),3);
					tmpim(dy(j):edy(j),dx(j):edx(j),:)=img(y(j):ey(j),x(j):ex(j),:);
					tempimg(:,:,(k-1)*3+1:(k-1)*3+3,j)=imResample(tmpim,[24 24],'bilinear');
				end
			end
			LNet.blobs('data').reshape([24 24 15 numbox]);
			tempimg=(tempimg-127.5)*0.0078125;
			out=LNet.forward({tempimg});
			score=squeeze(out{3}(2,:));
			for k=1:5
				tmp=[points(k,:);points(k+5,:)];
				%do not make a large movement
				temp=find(abs(out{k}(1,:)-0.5)>0.35);
				if ~isempty(temp)
					l=length(temp);
					out{k}(:,temp)=ones(2,l)*0.5;
				end
				temp=find(abs(out{k}(2,:)-0.5)>0.35);  
				if ~isempty(temp)
					l=length(temp);
					out{k}(:,temp)=ones(2,l)*0.5;
				end
				pointx(:,k)=(tmp(1,:)-0.5*patchw+out{k}(1,:).*patchw)';
				pointy(:,k)=(tmp(2,:)-0.5*patchw+out{k}(2,:).*patchw)';
			end
			for j=1:numbox
				points(:,j)=[pointx(j,:)';pointy(j,:)'];
			end
		end
    end 	
end

