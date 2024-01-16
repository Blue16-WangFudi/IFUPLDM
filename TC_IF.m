function s = TC_IF(Data,Label,Kernel)
%策略一
   N_Samples = length(Label);
   s = zeros(N_Samples, 1);
   Data_Pos = Data(Label==1, :);
   N_Pos = sum(Label==1);
   e_Pos = ones(N_Pos, 1);
   
   Data_Neg = Data(Label==-1, :); 
   N_Neg = sum(Label==-1);
   e_Neg = ones(N_Neg, 1);
   
   P_Ker_P = Function_Kernel(Data_Pos, Data_Pos, Kernel);
   P_Ker_N = Function_Kernel(Data_Pos, Data_Neg, Kernel);
   N_Ker_N = Function_Kernel(Data_Neg, Data_Neg, Kernel);
   
   %正类
   P_P = sqrt(diag(P_Ker_P)-2*P_Ker_P*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Pos/(N_Pos^2));   % p_i（正）与正中心之间的距离
   r_Pos = max(P_P); %正样本到正类的最大距离
   P_N = sqrt(diag(P_Ker_P)-2*P_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Pos/(N_Neg^2));   % p_i（正）与负中心之间的距离
   %负类
   N_N = sqrt(diag(N_Ker_N)-2*N_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Neg/(N_Neg^2));   % n_i（负）与负中心之间的距离
   r_Neg = max(N_N); %负样本到负类的最大距离
   N_P = sqrt(diag(N_Ker_N)-2*P_Ker_N'*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Neg/(N_Pos^2));  % n_i（负）与正中心之间的距离
    
   d_p = P_P-P_N; %正样本，到正类的距离减去负类的距离
   d_n = N_N-N_P; %负样本，到负类的距离减去正类的距离
   d_pn = (e_Pos'*P_Ker_P*e_Pos)/(N_Pos^2) + (e_Neg'*N_Ker_N*e_Neg)/(N_Neg^2) - sum(P_Ker_N(:))/(N_Pos * N_Neg); %正负类中心的距离
   %v
   Nmem1 = (d_p - min(d_p))./((d_pn - min(d_p)));
   Nmem2 = (d_n - min(d_n))./((d_pn - min(d_n)));
     
   %u
   Mem1 = (1-P_P/(r_Pos+1e-7));   
   Mem2 = (1- N_N/(r_Neg+1e-7));
   
%策略二
   P_Samples = size(Data_Pos,1);
   Data = [Data_Pos;Data_Neg];
   A_Ker_A = Function_Kernel(Data, Data, Kernel);

   a_a = sqrt(diag(A_Ker_A)*ones(1,N_Samples)-2*A_Ker_A+ones(N_Samples,1)*diag(A_Ker_A)');%所有样本点相互之间的距离（前面m行都是正类，后面n行是负类）
   alpha = 0.1;%调整参数
   logic = a_a>0&a_a<alpha;
   rho = ones(N_Samples,1);
   rho(1:P_Samples,1) = sum(logic(1:P_Samples,P_Samples+1:end),2)./(sum(logic(1:P_Samples,:),2)+1e-7);
   rho(P_Samples+1:end,1) = sum(logic(P_Samples+1:end,1:P_Samples),2)./(sum(logic(P_Samples+1:end,:),2)+1e-7);
   
   Mem12 = Mem1;
   Nmem12= (1-Mem12).*rho(1:P_Samples);  
   Mem22 = Mem2;
   Nmem22= (1-Mem22).*rho(P_Samples+1:end);

   Nmem1 = max(Nmem1,Nmem12);
   Nmem2 = max(Nmem2,Nmem22);
   s1=sqrt((Mem1.^2+(1-Nmem1).^2)./2);
   s2=sqrt((Mem2.^2+(1-Nmem2).^2)./2);
   s(Label==1) = s1; 
   s(Label==-1) = s2; 
   
end

