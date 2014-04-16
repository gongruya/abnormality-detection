i = 18;
obj = VideoReader(['../Avenue Dataset/testing_videos/', sprintf('%.2d', 10),'.avi']);
numFrames = get(obj, 'NumberOfFrames');
for ii = 1 : numFrames
    curFrame = imresize(rgb2gray(im2double(read(obj, ii))), [120, 160]);
    imshow(curFrame);
    getframe;
end