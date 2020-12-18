function [Rt, Lambda, RtSmoothed, LambdaSmoothed] = Rt_expfit2(NewCases, wlen, generation_period, time_unit)
% Estimates the parameters of an exponential fit.
%
% Reza Sameni
% Dec 2020
% Email: reza.sameni@gmail.com

NewCases = NewCases(:)';
Lambda = [zeros(1, generation_period) , log(NewCases(1 + generation_period : end)./NewCases(1 : end - generation_period))]/generation_period; % The reproduction eigenvalue (inverse time unit)
LambdaSmoothed = filter(ones(1, wlen), wlen, Lambda);

Rt = exp(Lambda * time_unit); % The reproduction rate
RtSmoothed = exp(LambdaSmoothed * time_unit); % The smoothed reproduction rate


