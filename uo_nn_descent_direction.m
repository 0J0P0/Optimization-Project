% [start] Descent direction FDM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d, beta, H] = uo_nn_descent_direction(w,wk,dk,Hk,gL,isd,icg,irc,nu,k,n)
    beta = 0; H = eye(n);
    %% Gradient Method
    if isd == 1
        d = -gL(w);
    elseif isd == 3
    %% Quasi-Newton Method
        if k > 1
            s = w - wk(:,k-1); y = gL(w) - gL(wk(:,k-1)); rho = 1/(y'*s);
            H = (eye(n) - rho*s*y')*Hk(:,:,k-1)*(eye(n) - rho*y*s') + rho*(s*s');
        end
        d = -H*gL(w);
    end
end
% [end] Descent direction FDM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%