function laplacePyramid = LaplacianPyramids(input_image,n)

laplacePyramid = cell(1,n);
gaussianPyramid = cell(1,n+1);

gaussianPyramid = GaussianPyramid(input_image,n);

% laplacePyramid{1} = gaussianPyramid[1] - expand_images(gaussianPyramid[2])
% laplacePyramid{2} = gaussianPyramid[2] - expand_images(gaussianPyramid[3])
% laplacePyramid{3} = gaussianPyramid[3]

for i = 1:n
    if i == n
        laplacePyramid{i} = gaussianPyramid{i};
        laplace_image = laplacePyramid{i};
    else
        laplacePyramid{i} = gaussianPyramid{i} - expand_images(gaussianPyramid{i+1});
        laplace_image = laplacePyramid{i};
    end
    % figure(i)
    % imshow(laplace_image)
end
end