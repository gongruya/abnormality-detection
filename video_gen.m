function video_gen(video_name, save_name)
    obj = VideoReader(video_name);
    tot = get(obj, 'NumberOfFrames');
    output = [];
    for ii = 1 : tot
        curFrame = imresize(im2double(read(obj, ii)), [120, 160]);
        output(:, :, :, ii) = curFrame;
        fprintf('%d/%d ', ii, tot);
    end
    save([save_name], 'output');
    disp(['The matlab data of video ', video_name, ' is saved.']);
%     for ii = 1: size(output, 4)
%         imshow(output(:, :, :, ii));
%         pause(1/100);
%     end
end