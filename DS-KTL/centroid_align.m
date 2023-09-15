function [Cn,Xn]=centroid_align(x,str)

    tmp_cov=zeros(size(x,1),size(x,1),size(x,3));
    for i=1:size(x,3)
        tmp_cov(:,:,i)=cov(x(:,:,i)');
    end

    C = mean_covariances(tmp_cov,str);
    P = C^(-1/2);
    
    Cn=zeros(size(x,1),size(x,1),size(x,3));
    for j=1:size(x,3)
        Cn(:,:,j)=P*squeeze(tmp_cov(:,:,j))*P;
    end

    Xn=zeros(size(x));
    for i=1:size(x,3)
        Xn(:,:,i)=P*x(:,:,i);
    end
end
