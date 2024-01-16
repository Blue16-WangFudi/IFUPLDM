function s = IF(Data, Label, Kernel)


   A_Samples = length(Label);
   s = zeros(A_Samples, 1);
 % Abstract the positive and negative data  ��ȡ��������
   Data_Pos = Data(Label==1, :);
   P_Samples = size(Data_Pos,1);
   N_Pos = sum(Label==1);
   e_Pos = ones(N_Pos, 1);
   
   Data_Neg = Data(Label==-1, :); 
   N_Samples = size(Data_Neg,1);
   N_Neg = sum(Label==-1);
   e_Neg = ones(N_Neg, 1);
   Data = [Data_Pos;Data_Neg];
 % Processing
   A_Ker_A = Function_Kernel(Data, Data, Kernel);
   P_Ker_P = Function_Kernel(Data_Pos, Data_Pos, Kernel);
   P_Ker_N = Function_Kernel(Data_Pos, Data_Neg, Kernel);
   N_Ker_N = Function_Kernel(Data_Neg, Data_Neg, Kernel);
   
%    a_b =diag(A_Ker_A)*ones(1,N_Samples)-2*A_Ker_A+ones(N_Samples,1)*diag(A_Ker_A)';
   a_a = sqrt(diag(A_Ker_A)*ones(1,A_Samples)-2*A_Ker_A+ones(A_Samples,1)*diag(A_Ker_A)');%�����������໥֮��ľ��루ǰ��m�ж������࣬����n���Ǹ��ࣩ
   alpha = 0.1;%��������
   logic = a_a>0&a_a<alpha;
   rho = ones(A_Samples,1);
   rho(1:P_Samples,1) = sum(logic(1:P_Samples,P_Samples+1:end),2)./(sum(logic(1:P_Samples,:),2)+1e-7);
   rho(P_Samples+1:end,1) = sum(logic(P_Samples+1:end,1:P_Samples),2)./(sum(logic(P_Samples+1:end,:),2)+1e-7);
   
   
   
   P_P = sqrt(diag(P_Ker_P)-2*P_Ker_P*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Pos/(N_Pos^2));   % p_i��������������֮��ľ���
   r_s = max(P_P);
   delta_Pos = 0.1*r_s;
   P_N = sqrt(diag(P_Ker_P)-2*P_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Pos/(N_Neg^2));   % p_i�������븺����֮��ľ���
   
   N_N = sqrt(diag(N_Ker_N)-2*N_Ker_N*e_Neg/N_Neg+(e_Neg'*N_Ker_N*e_Neg)*e_Neg/(N_Neg^2));   % n_i�������븺����֮��ľ���
   s_s = max(N_N);
   delta_Neg = 0.1*s_s;
   N_P = sqrt(diag(N_Ker_N)-2*P_Ker_N'*e_Pos/N_Pos+(e_Pos'*P_Ker_P*e_Pos)*e_Neg/(N_Pos^2));  % n_i��������������֮��ľ���
      

   
   Mem1=e_Pos-P_P/(r_s+10e-7);
   Nmem1= (1-Mem1).*rho(1:P_Samples);
   for i =1: P_Samples
       if(rho(i)==0)
           s(i) = Mem1(i);
       end
       if(Nmem1(i)>=Mem1(i))
           s(i) = 0;  
       end
       if(Nmem1(i)<Mem1(i)) && rho(i)~=0
           s(i)=(1-Nmem1(i))/(2-Mem1(i)-Nmem1(i));
       end
   
   end
   

    Mem2=e_Neg-N_N/(s_s+10e-7);
    Nmem2= (1-Mem2).*rho(P_Samples+1:end);
    for i =P_Samples+1:A_Samples
       if(rho(i)==0)
           s(i) = Mem2(i-P_Samples);
       end
       if(Nmem2(i-P_Samples)>=Mem2(i-P_Samples))
           s(i) = 0;  
       end
       if(Nmem2(i-P_Samples)<Mem2(i-P_Samples))
           s(i)=(1-Nmem2(i-P_Samples))/(2-Mem2(i-P_Samples)-Nmem2(i-P_Samples));
       end
   
   end
   S.s1 = s(1:P_Samples);
   S.s2 = s(P_Samples+1:end);
   
   s(Label==1) = S.s1; 
   s(Label==-1) = S.s2; 
   
   
   
   
   



end

