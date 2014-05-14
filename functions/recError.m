function  Re = recError(X, R, ThrTest)
   K = length(R);
   reSet = 1 : size(X,2);
   Re = ones(1,size(X,2));
   for ii = 1 : K 
       Re(reSet) = norm(R(ii).val*X(:,reSet))^2; 
       idx = find(Re < ThrTest);
       %disp([ num2str(ii),' is ',num2str(length(reSet))]);
       reSet = setdiff(reSet,idx);
   end
    
end