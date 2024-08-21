% function [output_image] = custom_convolution(input_image,kernel)
% 
% % image_class = class(input_image);
% kernel_len = length(kernel);
% % Padding 0
% layers = floor(kernel_len/2);
% image_paddding = padarray(input_image, [layers, layers]);
% %Separating each channel of the image for convolution
% red_chan = image_paddding(:, :, 1); 
% green_chan = image_paddding(:, :, 2);
% blue_chan = image_paddding(:, :, 3);
% 
% new_red_chan = convolution(red_chan,kernel,layers,kernel_len);
% % new_red_chan = cast(new_red_chan,'like',input_image); %converting to uint8 form from double
% new_red_chan = uint8(new_red_chan);
% new_red_chan = new_red_chan(layers:end-layers,layers:end-layers);
% 
% 
% new_green_chan = convolution(green_chan,kernel,layers,kernel_len);
% % greenCompNew = cast(greenCompNew,'like',input_image); %converting to uint8 form from double
% new_green_chan = uint8(new_green_chan);
% new_green_chan = new_green_chan(layers:end-layers,layers:end-layers);
% 
% new_blue_chan = convolution(blue_chan,kernel,layers,kernel_len);
% % new_blue_chan = cast(new_blue_chan,'like',input_image); %converting to uint8 form from double
% new_blue_chan = uint8(new_blue_chan);
% new_blue_chan = new_blue_chan(layers:end-layers,layers:end-layers);
% % concate 3
% output_image = cat(3, new_red_chan, new_green_chan, new_blue_chan);
% 
% 
% %custom convolution function
%     function  final_image  = convolution(image,kernel,layers,kernel_len)
% 
% [input_image_rows, input_image_cols] = size(image);
% if size(kernel, 1) == 1 % 1D kernel
%     for final_column = layers + 1 : input_image_cols - layers
%         for final_row = layers + 1 : input_image_rows - layers
%             addition = 0;
%             for i = 1 : kernel_len
%                 image_col = final_column + i - layers - 1;
%                 addition = addition + double(image(final_row, image_col)) * kernel(i);
%             end
%             final_image(final_row, final_column) = addition;
%         end
%     end
% else
%     for final_column = layers + 1 : input_image_cols - layers
% 	    for final_row = layers + 1 : input_image_rows - layers
% 		    addition = 0;
% 		    for i = 1 : kernel_len
% 			    image_col = final_column + i - layers - 1;
% 			    for j = 1 : kernel_len
% 				    image_row = final_row + j - layers - 1;
% 				    addition = addition + double(image(image_row, image_col)) * kernel(j, i); 
% 			    end
%             end
% 		    final_image(final_row, final_column) = addition; 
% 	    end
%     end
% end
%     end
% end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [output_image] = custom_convolution(input_image, kernel)

kernel_len = length(kernel);
layers = floor(kernel_len/2);
image_padding = padarray(input_image, [layers, layers]);

red_chan = image_padding(:, :, 1);
green_chan = image_padding(:, :, 2);
blue_chan = image_padding(:, :, 3);

new_red_chan = convolution(red_chan, kernel, layers, kernel_len);
new_red_chan = uint8(new_red_chan);
new_red_chan = new_red_chan(layers+1:end-layers, layers+1:end-layers);

new_green_chan = convolution(green_chan, kernel, layers, kernel_len);
new_green_chan = uint8(new_green_chan);
new_green_chan = new_green_chan(layers+1:end-layers, layers+1:end-layers);

new_blue_chan = convolution(blue_chan, kernel, layers, kernel_len);
new_blue_chan = uint8(new_blue_chan);
new_blue_chan = new_blue_chan(layers+1:end-layers, layers+1:end-layers);

output_image = cat(3, new_red_chan, new_green_chan, new_blue_chan);

end

function final_image = convolution(image, kernel, layers, kernel_len)

[input_image_rows, input_image_cols] = size(image);
final_image = zeros(input_image_rows, input_image_cols);

if size(kernel, 1) == 1 % 1D kernel
    for final_column = layers + 1 : input_image_cols - layers
        for final_row = layers + 1 : input_image_rows - layers
            addition = 0;
            for i = 1 : kernel_len
                image_col = final_column + i - layers - 1;
                addition = addition + double(image(final_row, image_col)) * kernel(i);
            end
            final_image(final_row, final_column) = addition;
        end
    end
elseif size(kernel, 2) == 1 % Handle 1D kernel in the vertical direction
    for final_column = layers + 1 : input_image_cols - layers
        for final_row = layers + 1 : input_image_rows - layers
            addition = 0;
            for i = 1 : kernel_len
                image_row = final_row + i - layers - 1;
                addition = addition + double(image(image_row, final_column)) * kernel(i);
            end
            final_image(final_row, final_column) = addition;
        end
    end
else % 2D kernel
    for final_column = layers + 1 : input_image_cols - layers
        for final_row = layers + 1 : input_image_rows - layers
            addition = 0;
            for i = 1 : kernel_len
                image_col = final_column + i - layers - 1;
                for j = 1 : kernel_len
                    image_row = final_row + j - layers - 1;
                    addition = addition + double(image(image_row, image_col)) * kernel(j, i);
                end
            end
            final_image(final_row, final_column) = addition;
        end
    end
end

end
