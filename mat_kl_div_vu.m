function div=mat_kl_div_vu(x,y)
%*******************************************************************
%* Compute the KL divergence between two vector x(i,:) and y(i,:)
%* INPUT
%* x: vector [N x D]
%* y: vector [N x D]
% or 
%* x: [1 x D]
%* y: [N x D]
%* OUTPUT
%* div: KL divergence between x and y: [N x 1]
%*******************************************************************
[N,D]=size(y);
regularizer=0.1/D;

y=y+regularizer;
% normalize each row sum to 1
y=bsxfun(@times,y,1./sum(y,2));

if (isequal(size(x),size(y)))
    x=x+regularizer;
    x=bsxfun(@times,x,1./sum(x,2));
  
    div = sum(x.*(log(x)-log(y)),2);
else
    x=x+regularizer;
    % normalize each row sum to 1
    x=bsxfun(@times,x,1./sum(x,2));
    
    x_mat=ones(N,1)*x;
    div = sum(x_mat.*(log(x_mat)-log(y)),2);
 
end
end