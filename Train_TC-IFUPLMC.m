function Outs_Train = Train_UPLDM(Samples_Train ,Labels_Train, lambda1, lambda2, tau, C, Kernel)
    
    m = size(Samples_Train,1);
    e = ones(m, 1);
    K = Function_Kernel(Samples_Train, Samples_Train, Kernel);
    CR = 1e-7;
    Q = 4*lambda1*(m*K'*K - K*Labels_Train*(Labels_Train'*K')+4*lambda1*(m^2)*K)/(m^2)+ CR*eye(m);
    A = diag(Labels_Train)*K*inv(Q)*K*diag(Labels_Train);
    if length(C) == 1
        C = C*ones(1,m);
    end
    eta0= zeros(m, 1);
    options = optimoptions('quadprog', 'Display', 'off');
    H_quad = A;
    f_quad = lambda2*A*e/m-e; 
    lb = sign(-tau)*C.*abs(tau);
    ub = C;
    
    eta = quadprog(H_quad, f_quad, [], [], [], [], lb, ub, eta0, options);
    deta = inv(Q)*K*diag(Labels_Train)*(lambda2*e/m+eta);
    
    Outs_Train.deta = deta;
    Outs_Train.Samples = Samples_Train;
    Outs_Train.Labels = Labels_Train;
    Outs_Train.Ker = Kernel;


end
