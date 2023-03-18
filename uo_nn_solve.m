% [start] Neural Network OM solve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input Parameters:
    %% Parameters for dataset generation:
        % num_target: set of digits to be identified.
        % tr_freq: frequency of the digits target in the training dataset.
        % tr_p, te_q: number of samples for the training/test dataset.
        % tr_seed, te_seed: seed for the training/test random numbers generator.
    %% Parameters for optimization:
        % la: L2 regularization.
        % epsG, kmax: Stopping criteria.
        % ils:
        % ialmax, kmaxBLS, epsal: , maximum number of iterations of the BLS algorithm, minimum variation between two consecutive reductions of al.
        % c1,c2: Wolfe condition parameters.
        % isd, icg, irc, nu: search direction, conjugate gradient variant, restart condition.
        % sg_al0, sg_be, sg_ga:
        % sg_emax, sg_ebest, sg_seed:
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
    %% Generate the training and test datasets
    [Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
    [Xte,yte] = uo_nn_dataset(te_seed, te_q, num_target, 0.0);
    %% Part 1: recognition of some specific target with GM and QNM
    w = zeros(size(Xtr,1),1);                                               %% como defines w? porq con uno no cambia??
    sig = @(X) 1./(1 + exp(-X)); y = @(X,w) sig(w'*sig(X));
    L = @(w) (norm(y(Xtr,w) - ytr)^2)/size(ytr,2) + (la*norm(w)^2)/2;
    gL = @(w) (2*sig(Xtr)*((y(Xtr,w) - ytr).*y(Xtr,w).*(1 - y(Xtr,w)))')/size(ytr,2) + la*w;
    [wk,dk,Lk,gLk,alk,iWk,betak,Hk,niter] = uo_nn_solve_opt(w,L,gL,epsG,kmax,ialmax,kmaxBLS,epsal,c1,c2,isd,icg,irc,nu);

    %% Part 2: stochastic gradient method (SGM)

    %% Part 3: comparation of the performance

    %% Output parameters
    wo=wk(:,end);fo=Lk(:,end);
    tr_acc=100/tr_p;
    te_acc=100/te_q;
    tex=0;
end
% [end] Neural Network OM solve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% [start] Alg. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [wk,dk,Lk,gLk,alk,ioutk,betak,Hk,k] = uo_nn_solve_opt(w,L,gL,epsG,kmax,ialmax,kmaxBLS,epsal,c1,c2,isd,icg,irc,nu)
    k = 1; n = size(w,1); DC = 1;
    wk = [w]; dk = []; Lk = [L(w)]; gLk = [gL(w)]; alk = []; ioutk = [];
    betak = []; Hk = [];
    while norm(gL(w)) > epsG && k < kmax && DC
        [d, beta, H] = uo_nn_descent_direction(w,wk,dk,Hk,gL,isd,icg,irc,nu,k,n);
        dk = [dk, d]; betak = [betak, beta]; Hk(:,:,k) = H;

        if k > 1 ialmax = alk(:,k-1)*(gLk(:,k-1)'*dk(:,k-1))/(gL(w)'*d); end  %% alk(k-1)  al1 o al2?
        [alpha,iout] = uo_BLSNW32(L,gL,w,d,ialmax,c1,c2,kmaxBLS,epsal);
        alk = [alk, alpha]; ioutk = [ioutk, iout];
        
        if gL(w)'*d >= 0; DC = 0; end
        w = w + alpha*d;
        wk = [wk, w]; Lk = [Lk, L(w)]; gLk = [gLk, gL(w)];
        k = k+1;
    end
end
% [end] Alg. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









%