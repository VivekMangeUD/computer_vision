function reconstruct = Rconstruct(I,n)
reconstruct = cell(1,n);
% n = 3;
K = 1/16.*[1 2 1 ; 2 4 2 ; 1 2 1]; 

% gaussianPyramid = GaussianPyramid(image,3);
% I = image;
for i = 1:n
    if i == 1
        con_image = custom_convolution(I,K);
        recon_image(i).img = con_image;      
    else 
        con_image = custom_convolution(recon_image(i-1).img,K);
        reduced_img = reduce_image(con_image);
        recon_image(i).img = reduced_img;        
    end
end
for j = n-1:1       
    recon_image(j).img = reduced_img;
    reduced_img = custom_convolution(recon_image(j).img,K);
    after_exapn = expand_images(reduced_img);
    recon_image(j).img = after_exapn;
    recon_image(j-1).img = recon_image(j-1).img + laplacePyramid(recon_image(j).img,n);     
end
% figure(1);
% imshow(recon_image(1).img);
reconstruct = recon_image;
end
