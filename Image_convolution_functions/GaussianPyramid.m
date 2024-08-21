
function gaussian_pyramid = GaussianPyramid(input_image,n)

kernel = 1/16.*[1 2 1 ; 2 4 2 ; 1 2 1]; 
gaussian_pyramid = cell(1,n);

for i = 1:n+1
    if i == 1
        convolue_image = custom_convolution(input_image,kernel); % apply guassain filter and no reduction
        gaussian_image(i).img = convolue_image;
    else 
        image = custom_convolution(gaussian_image(i-1).img,kernel); % apply filter and then reduce
        reduced_image = reduce_image(image);
        gaussian_image(i).img = reduced_image; 
    	gaussian_pyramid{i-1} = gaussian_image(i).img;
    end
    % figure(i)
    % imshow(gaussian_image(i).img)
end
end
