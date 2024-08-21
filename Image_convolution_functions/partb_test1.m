close;
clear;
image = imread('pr1\CISC642 - PR1\CISC642 - PR1\Template\PartB\lena.png');

% Display the image
% imshow(image);

% sobel_horizontal = [-1, -2, -1;
%                      0,  0,  0;
%                      1,  2,  1];

% kernel = 1/16.*[1 2 1 ; 2 4 2 ; 1 2 1]; 
% convolued_image = custom_convolution(image, kernel);
% imshow(convolued_image)
% imwrite(convolued_image,"pr1\submission\PartB\output_images\convolued_image.png")

%%%% REMEMBER TO APPLY GUASSIAN FILTER BEFORE REDUCING
% kernel = 1/16.*[1 2 1 ; 2 4 2 ; 1 2 1]; 
% convolued_image = custom_convolution(image, kernel);
% reduced_image = reduce_image(image);
% imshow(reduced_image)
% imwrite(reduced_image,'reduced_image.png');
% imwrite(reduced_image,"pr1\submission\PartB\output_images\reduced_image.png")

% expanded_image = expand_images(image);
% imshow(expanded_image)
% imwrite(expanded_image,"pr1\submission\PartB\output_images\expanded_image.png")
% gaussianPyramid = GaussianPyramid(image,3);
% laplacePyramid = LaplacianPyramids(image,3);
% R = Rconstruct(image,3);
% figure(1)
% imshow(R(1).img)
% error = image - R(1).img;
% figure(2)
% imshow(error)


%%%% get points from image
% imshow('lena.png')
% [x,y] = getpts
image1 = imread("happy_smile.jpg");
image2 = imread('sad_smile.jpg');
% laplacePyramid1 = LaplacianPyramids(image1,3);
% laplacePyramid2 = LaplacianPyramids(image2,3);

% Display the left image on the left side of the figure
subplot(1, 2, 1);
imshow(image1);
title('Left Image');

% Display the right image on the right side of the figure
subplot(1, 2, 2);
imshow(image2);
title('Right Image');

% Implement a mouse callback function to capture user input for blend boundaries
% Use ginput to collect mouse clicks from the user
% Process the user input to define blend boundaries and perform mosaicking
% You'll need to define this callback function yourself

% Example:
[x, y] = ginput(2); % Capture two points for the blend boundary


