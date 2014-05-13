function video_gen(video_name, save_name)
    obj = VideoReader(video_name);
    tot = get(obj, 'NumberOfFrames');
    Video_Output = zeros(120, 160, 3, tot);
    for ii = 1 : tot
        curFrame = imresize(read(obj, ii), [120, 160]);
        Video_Output(:, :, :, ii) = curFrame;
        fprintf('%d/%d ', ii, tot);
        if mod(ii, 1000) == 0 
            fprintf('\n');
        end
    end
    save([save_name], 'Video_Output');
    disp(['The matlab data of video ', video_name, ' is saved.']);
%     for ii = 1: size(output, 4)
%         imshow(output(:, :, :, ii));
%         pause(1/100);
%     end
end