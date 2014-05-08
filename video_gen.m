function video_gen(video_name, save_name)
    obj = VideoReader(video_name);
    tot = get(obj, 'NumberOfFrames');
    Video_Output = [];
    for ii = 1 : tot
        curFrame = rgb2gray(imresize(read(obj, ii), [90, 160]));
        Video_Output(:, :, ii) = curFrame;
        fprintf('%d/%d ', ii, tot);
    end
    save([save_name], 'Video_Output');
    disp(['The matlab data of video ', video_name, ' is saved.']);
%     for ii = 1: size(output, 4)
%         imshow(output(:, :, :, ii));
%         pause(1/100);
%     end
end