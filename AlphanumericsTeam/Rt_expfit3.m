function [Rt, A, Lambda, ExpFit] = Rt_expfit3(NewCases, wlen, time_unit)
% Estimates the parameters of an exponential fit using nonlinear least
% squares over the past wlen number of new cases acquired over constant
% time intervals time_unit
%
% Reza Sameni
% Dec 2020
% Email: reza.sameni@gmail.com

NewCases = NewCases(:)';
L = length(NewCases); % The input signal length
r = zeros(1, L); % The growth rate (scalar)
A = zeros(1, L); % The amplitude of the exp fit
n = -wlen:-1; % time sequence of the last wlen samples
options = optimset('TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 250);
for mm = wlen : L
    segment = NewCases(mm - wlen + 1: mm); % a segment of wlen samples
    if(length(find(segment ~= 0)) < ceil(wlen))
        A(mm) = NewCases(mm);
        r(mm) = 0;
    else
        InitParams = [NewCases(mm) 0];
        EstParams = nlinfit(n/time_unit, segment, @ExpModel, InitParams, options);
        %     EstParams = lsqcurvefit(@ExpModel, InitParams, n/time_unit, segment);
        A(mm) = EstParams(1);
        r(mm) = EstParams(2);
    end
end

Rt = exp(r); % The reproduction rate
ExpFit = A .* Rt; % The exponential fit
Lambda = r/time_unit; % The reproduction eigenvalue (inverse time unit)

end

function y = ExpModel(params, t)
A = params(1);
lambda = params(2);
y = A*exp(lambda*t);
end
