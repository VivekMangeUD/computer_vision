function [final_image] = reduce_image(input_image) 
% % Verify why is image not loading with same size

% Reduce Image size by half
scale = [0.5 0.5]; % Change value to adjust the reduction
original_size = size(input_image); 
reduced_size = max(floor(scale.*original_size(1:2)),1); 

rows = min(round(((1:reduced_size(1))-0.5)./scale(1)+0.5),original_size(1)); 
cols = min(round(((1:reduced_size(2))-0.5)./scale(2)+0.5),original_size(2)); 

final_image = input_image(rows,cols,:); 
% imwrite(final_image,'reduced_image.png');

% Prints
% imshow(input_image); % Displat OG image
% size(input_image) % display OG image size for verification
% figure(2);
% imshow(final_image); % final image
% size(final_image) % final image size

end