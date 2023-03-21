% [start] Neural Network OM solve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Parameters:
    %% Parameters for dataset generation:
        % num_target : set of digits to be identified.
        %    tr_freq : frequency of the digits target in the data set.
        %    tr_seed : seed for the training set random generation.
        %       tr_p : size of the training set.
        %    te_seed : seed for the test set random generation.
        %       te_q : size of the test set.
    %% Parameters for optimization:
%         la : coefficient lambda of the decay factor.
%       epsG : optimality tolerance.
%       kmax : maximum number of iterations.
%        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
%     ialmax :  formula for the maximum step lenght (1 or 2).
%    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
%      epsal : minimum progress in alpha, algorithm up_BLSNW32
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg_al0 : \alpha^{SG}_0.
%      sg_be : \beta^{SG}.
%      sg_ga : \gamma^{SG}.
%    sg_emax : e^{SG_{max}.
%   sg_ebest : e^{SG}_{best}.
%    sg_seed : seed for the first random permutation of the SG.
%        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
%        irc : re-starting condition for the CGM (useless in this project).
%         nu : parameter of the RC2 for the CGM  (useless in this project).
    %% Output parameters:
%
%        Xtr : X^{TR}.
%        ytr : y^{TR}.
%         wo : w^*.
%         fo : {\tilde L}^*.
%     tr_acc : Accuracy^{TR}.
%        Xte : X^{TE}.
%        yte : y^{TE}.
%     te_acc : Accuracy^{TE}.
%      niter : total number of iterations.
%        tex : total running time (see "tic" "toc" Matlab commands).
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
    %% Generate the training and test datasets
    [Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    [Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, 0.0);
    %% Part 1: recognition of some specific target with GM and QNM
    w = zeros(size(Xtr,1),1);                                               %% como defines w?? unos?? ceros --> 100%
    sig = @(X) 1./(1 + exp(-X)); y = @(X,w) sig(w'*sig(X));
    L = @(w) (norm(y(Xtr,w) - ytr)^2)/size(ytr,2) + (la*norm(w)^2)/2;
    gL = @(w) (2*sig(Xtr)*((y(Xtr,w) - ytr).*y(Xtr,w).*(1 - y(Xtr,w)))')/size(ytr,2) + la*w;
    
    if isd == 1 || isd == 3
        [wk,dk,Lk,gLk,alk,iWk,betak,Hk,niter] = uo_nn_solve_opt(w,L,gL,epsG,kmax,ialmax,kmaxBLS,epsal,c1,c2,isd,icg,irc,nu);
    end
    %% Part 2: stochastic gradient method (SGM)
    if isd == 7
        [wk,Lk,niter] = uo_nn_solve_sgm(w,L,gL,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed);
    end
    %% accuracy and output variables
    wo=wk(:,end); fo=Lk(:,end);
    
    tr_v = []; te_v = [];
    for j = 1:tr_p tr_v = [tr_v, round(y(Xtr(1:end,j),wo))]; end
    for j = 1:te_q te_v = [te_v, round(y(Xte(1:end,j),wo))]; end
    tr_acc = (100/tr_p) * sum(tr_v == ytr);
    te_acc = (100/te_q) * sum(te_v == yte);
    tex=0;
end
% [end] Neural Network OM solve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% [start] FDM Alg. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [wk,dk,Lk,gLk,alk,ioutk,betak,Hk,k] = uo_nn_solve_opt(w,L,gL,epsG,kmax,ialmax,kmaxBLS,epsal,c1,c2,isd,icg,irc,nu)
    k = 1; n = size(w,1); DC = 1;
    wk = [w]; dk = []; Lk = [L(w)]; gLk = [gL(w)]; alk = []; ioutk = [];
    betak = []; Hk = [];
    while norm(gL(w)) > epsG && k < kmax && DC
        [d, beta, H] = uo_nn_descent_direction(w,wk,dk,Hk,gL,isd,icg,irc,nu,k,n);
        dk = [dk, d]; betak = [betak, beta]; Hk(:,:,k) = H;

        if k > 1 ialmax = alk(:,k-1)*(gLk(:,k-1)'*dk(:,k-1))/(gL(w)'*d); end  %% probar para min tiempo
%         if k > 1 ialmax = 2*(L(w)-Lk(:,k-1))/(gL(w)'*d); end
        [alpha,iout] = uo_BLSNW32(L,gL,w,d,ialmax,c1,c2,kmaxBLS,epsal);
        alk = [alk, alpha]; ioutk = [ioutk, iout];
        
        if gL(w)'*d >= 0; DC = 0; end
        w = w + alpha*d;
        wk = [wk, w]; Lk = [Lk, L(w)]; gLk = [gLk, gL(w)];
        k = k+1;
    end
end
% [end] FDM Alg. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% [start] Stochastic Gradient Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [wk,Lk,niter] = uo_nn_solve_sgm(w,L,gL,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed)
    p = size(Xtr, 2);
    m = abs(sg_ga*p); ke = ceil(p/m); kmax = sg_emax*ke; e = 0; s = 0; Lbest = Inf; k = 0;
    while e <= sg_emax && s < sg_ebest
        P = Xtr(:,randsample(1:p,p));
        for i = 0:ceil(p/m-1)
            S = 
        end
    end
end
% [end] Stochastic Gradient Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%