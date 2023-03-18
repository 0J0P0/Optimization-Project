% [start] Descent direction FDM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [d, beta, H] = uo_nn_descent_direction(w,wk,dk,Hk,gL,isd,icg,irc,nu,k,n)
    beta = 0; H = eye(n);
    if isd == 1  % Gradient Method
        d = -gL(w);
    elseif isd == 2  % Conjugate Gradient Method
        if k > 1
            beta = uo_cgm_beta(w,wk,gL,icg,irc,nu,k,n);
            d = -gL(w) + beta*dk(:,k-1);
        else
            d = -gL(w);
        end
    elseif isd == 3  % Quasi-Newton Method
        if k > 1
            s = w - wk(:,k-1); y = gL(w) - gL(wk(:,k-1)); rho = 1/(y'*s);
            H = (eye(n) - rho*s*y')*Hk(:,:,k-1)*(eye(n) - rho*y*s') + rho*(s*s');
        end
        d = -H*gL(w);
    end
end
% [end] Descent direction FDM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% [start] CGM beta %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function beta = uo_cgm_beta(w,wk,gL,icg,irc,nu,k,n)
    beta = 0;
    restart = false;
    if irc == 1 && mod(k-1,n) == 0  %% RC1
        restart = true;
    elseif irc == 2 && abs(gL(w)'*gL(wk(:,k-1)))/(norm(gL(w))^2) >= nu  %% RC2
        restart = true;
    end
    if ~restart
        if icg == 1  % Fletcher−Reeves
            beta = (gL(w)'*gL(w))/(norm(gL(wk(:,k-1)))^2);
        elseif icg == 2  % Polak−Ribière+
            beta = max(0, (gL(w)'*(gL(w)-gL(wk(:,k-1))))/(norm(gL(wk(:,k-1)))^2));
        end
    end
end
% [end] CGM beta %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%