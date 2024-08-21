function [gaussianKernel, vertical_k] = createGaussianKernel(sigma, kernelSize)
    % Calculate the center of the kernel
    center = (kernelSize - 1) / 2;
    
    % Initialize an empty kernel
    gaussianKernel = zeros(1, kernelSize);
    
    % Calculate the Gaussian values for each position in the kernel
    for x = 1:kernelSize
        gaussianKernel(x) = (1 / (sqrt(2 * pi) * sigma)) * exp(-((x - 1 - center)^2) / (2 * sigma^2));
    end
    
    % Normalize the kernel so that it sums to 1
    gaussianKernel = gaussianKernel / sum(gaussianKernel);
    vertical_k = gaussianKernel';
end
