for ii = 1: size(Video_Output, 4)
         imshow(Video_Output(:, :, :, ii)/255);
         pause(1/100);
end