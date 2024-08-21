close all;
clear all;

image = imread('pr1\CISC642 - PR1\CISC642 - PR1\Template\PartB\lena.png');

% Display the image
% imshow(image);

% sobel_horizontal = [-1, -2, -1;
%                      0,  0,  0;
%                      1,  2,  1];

% %%% Convolution
% kernel = 1/16.*[1 2 1 ; 2 4 2 ; 1 2 1]; 
% convolued_image = custom_convolution(image, kernel);
% figure
% imshow(convolued_image)
% title('convolued_image');
% imwrite(convolued_image,"pr1\submission\PartB\output_images\convolued_image.png")
% 
% %%%% REMEMBER TO APPLY GUASSIAN FILTER BEFORE REDUCING
% % %%% 1D kernel and reduce image
% sigma = 1.0;     % Adjust this value for the desired blurriness
% kernelSize = 5;
% % Generate the 1D Gaussian kernel
% [gaussian_1d_kernel, verical_kernel] = createGaussianKernel(sigma, kernelSize);
% % disp(gaussian_1d_kernel);
% % disp(verical_kernel);
% vertical_convo = custom_convolution(image, verical_kernel);
% horizontal_convo = custom_convolution(vertical_convo, gaussian_1d_kernel);
% reduced_1d_gaussian = reduce_image(horizontal_convo);
% figure
% imshow(reduced_1d_gaussian);
% title('reduced_1d_gaussian');
% imwrite(reduced_1d_gaussian,"pr1\submission\PartB\output_images\reduced_image.png")


% kernel = 1/16.*[1 2 1 ; 2 4 2 ; 1 2 1]; 
% kernel_g = createGaussianKernel(1.0,5); 
% convolued_image = custom_convolution(image, kernel_g);
% reduced_image = reduce_image(image);
% imshow(reduced_image)
% imwrite(reduced_image,'reduced_image.png');
% imwrite(reduced_image,"pr1\submission\PartB\output_images\reduced_image.png")

% % expand
% expanded_image = expand_images(image);
% figure
% imshow(expanded_image)
% title('expanded_image')
% imwrite(expanded_image,"pr1\submission\PartB\output_images\expanded_image.png")

% Gaussian pyramid
gaussianPyramid = GaussianPyramid(image,3);
for i = 1:length(gaussianPyramid)
    figure
    imshow(gaussianPyramid{i})
    title('gaussian images', i)
    outputDir = 'pr1\submission\PartB\output_images\Gaussian_level_';
    fileName = sprintf('%s%d.png', outputDir, i);
    imwrite(gaussianPyramid{i}, fileName)
end

% % Laplase pyramid
% laplacePyramid = LaplacianPyramids(image,3);
% for i = 1:length(laplacePyramid)
%     figure
%     imshow(laplacePyramid{i})
%     title('Laplase images', i)
%     outputDir = 'pr1\submission\PartB\output_images\Laplace_level_';
%     fileName = sprintf('%s%d.png', outputDir, i);
%     imwrite(laplacePyramid{i}, fileName)
% end

% Reconstruct

R = Rconstruct(image,3);
figure
imshow(R(1).img)
title('Reconstruct image')
s = R(1).img;
imwrite(s, 'Reconstructed Image')

error = image - R(1).img;
figure
imshow(error)
title('error image')
imwrite(error, "Error Difference")

%%%% get points from image
% % imshow('lena.png')
% % [x,y] = getpts
% % image1 = imread("happy_smile.jpg");
% % image2 = imread('sad_smile.jpg');
% % laplacePyramid1 = LaplacianPyramids(image1,3);
% % laplacePyramid2 = LaplacianPyramids(image2,3);
% % 
% % Display the left image on the left side of the figure
% % subplot(1, 2, 1);
% % imshow(image1);
% % title('Left Image');
% % 
% % Display the right image on the right side of the figure
% % subplot(1, 2, 2);
% % imshow(image2);
% % title('Right Image');
% % 
% % Implement a mouse callback function to capture user input for blend boundaries
% % Use ginput to collect mouse clicks from the user
% % Process the user input to define blend boundaries and perform mosaicking
% % You'll need to define this callback function yourself
% % 
% % Example:
% % [x, y] = ginput(2); % Capture two points for the blend boundary


