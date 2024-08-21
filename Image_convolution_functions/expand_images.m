
function [output_image] = expand_images(input_image) 

scale = [2 2]; % scale ro expand
original_size = size(input_image);  
expand_size = max(floor(scale.*original_size(1:2)),1); 

rows = min(round(((1:expand_size(1))-2)./scale(1)+2),original_size(1)); 
cols = min(round(((1:expand_size(2))-2)./scale(2)+2),original_size(2)); 

output_image = input_image(rows,cols,:); 
% imwrite(output_image,'lena expanded.png'); 

% figure(1);
% imshow(input_image);
% size(input_image) 
% figure(2);
% imshow(output_image); 
% size(output_image) 

end