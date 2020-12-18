close all
clear
clc

plot_figures = true;

% Read Oxford time-series data
all_data = importdata('./../../covid-policy-tracker/data/OxCGRT_latest.csv');
% % % all_ip_data = importdata('./../covid_xprize/validation/data/2020-09-30_historical_ip.csv');

ColumnHeaders = all_data.textdata(1, :); % Column titles
AllCountryNames = all_data.textdata(2:end, 1); % All country names
AllRegionNames = all_data.textdata(2:end, 3); % All region names

% Make GeoID code
% % % delim = cell(length(AllRegionNames), 1);
% % % st = ~strcmp('', AllRegionNames);
% % % delim(st) = {'_'};
delim = '';
GeoID = strcat(all_data.textdata(2:end, 2), delim, all_data.textdata(2:end, 4));

% Make a dictionary of country-regions
[CountryAndRegionList, IA, IC] = unique(GeoID, 'stable');

NumGeoLocations = length(CountryAndRegionList); % Number of country-region pairs

% Note: 'MOVINGAVERAGE-CAUSAL' is the contest standard and only evaluation algorithm
filter_type = 'MOVINGAVERAGE-CAUSAL'; % 'MOVINGAVERAGE-NONCAUSAL' or 'MOVINGAVERAGE-CAUSAL' ' or 'MOVINGMEDIAN' or 'TIKHONOV'; % The last two call functions from the OSET package (oset.ir)

% Different methods for calculating the reproduction rate
LtMethod1 = [];
LtMethod2 = [];
LtMethod3 = [];
LtMethod4 = [];
min_cases = 100;
for k = 204 : 210% 1: NumGeoLocations
    all_geoid_entry_indexes = find(string(GeoID) == CountryAndRegionList(k));
    all_geoid_data = all_data.data(all_geoid_entry_indexes + 1 , :);
    all_geoid_textdata = all_data.textdata(all_geoid_entry_indexes + 1 , :);
    
    dates_unsorted = all_geoid_data(:, 1);
    [dates, date_indexes] = sort(dates_unsorted, 'ascend');
    AllConfirmedCases = all_geoid_data(date_indexes, 33); % Confirmed cases
    start_date = find(AllConfirmedCases >= min_cases, 1, 'first');
    
    ConfirmedCases = all_geoid_data(date_indexes(start_date : end), 33); % Confirmed cases
    InterventionPlans = all_geoid_data(date_indexes(start_date : end), 2 : 32);
    InterventionPlans(:, [19, 20, 25, 26]) = []; % Remove economic and health IP in USD units
    no_ip_data_points = isnan(InterventionPlans);
    %     ip_indexes_available = find(~no_ip_data_points);
    %     ip_indexes_unavailable = find(no_ip_data_points);
    %     InterventionPlans(no_ip_data_points) = nan;%1e-4*randn(1, length(find(no_ip_data_points)));
    
    IPmn = nanmean(InterventionPlans, 1);
    IPstd = nanstd(InterventionPlans, [], 1); % max(InterventionPlans, [], 1); %
    InterventionPlans = (InterventionPlans - IPmn(ones(size(InterventionPlans, 1), 1), :))./IPstd(ones(size(InterventionPlans, 1), 1), :);
    InterventionPlans(isnan(InterventionPlans)) = 0;%min(InterventionPlans(:));%1e-3*randn(1, length(ip_indexes_unavailable));
    %     ConfirmedDeaths = all_geoid_data(date_indexes, 34); % Death cases
    
    NewCases = [diff(ConfirmedCases) ; nan]; % calculate the
    NewCases(NewCases < 0) = 0;
    
    % Replace nans with 0 and the last one with the previous number
    NewCasesFilled = NewCases;
    NewCasesFilled(isnan(NewCasesFilled)) = 0; %%% 0 or 1 <----
    if(isnan(NewCases(end))) % Just fill the last missing date (if any) with the previous day
        NewCasesFilled(end) = NewCasesFilled(end-1);
    end
    
    % Smooth the data using Tikhonov regularization
    switch filter_type
        case 'TIKHONOV'
            % DiffOrderOrFilterCoefs = [1 -2 1];
            DiffOrderOrFilterCoefs = 2;
            lambda = 25.0;
            NewCasesSmoothed = TikhonovRegularization(NewCasesFilled', DiffOrderOrFilterCoefs, lambda)';
        case 'MOVINGAVERAGE-CAUSAL'
            wlen = 7;
            NewCasesSmoothed = filter(ones(1, wlen), wlen, NewCasesFilled); % causal
        case 'MOVINGAVERAGE-NONCAUSAL'
            wlen = 7;
            NewCasesSmoothed = BaseLine1(NewCasesFilled', wlen, 'mn')'; % non-causal (zero-phase)
        case 'MOVINGMEDIAN'
            NewCasesSmoothed = BaseLine1(NewCasesFilled', 3, 'md')';
            NewCasesSmoothed = BaseLine1(NewCasesSmoothed', 7, 'mn')';
        otherwise
            error('Unknown filter type');
    end
    
    %     NewCasesSmoothedLog = log(NewCasesSmoothed);
    %
    %     wlen = 7;
    %     Amp = zeros(1, length(NewCasesSmoothedLog));
    %     Lambda1 = Amp;
    %     n = -wlen:-1;
    %     En = mean(n);
    %     En2 = mean(n.^2);
    %     Det = En2 - En^2;
    %     for mm = wlen : length(NewCasesSmoothedLog)
    %         segment = NewCasesSmoothedLog(mm - wlen + 1: mm)';
    %         Amp(mm) = (mean(segment)*En2 - mean(n .* segment)*En)/Det;
    %         Lambda1(mm) = (mean(n .* segment) - mean(segment)* En)/Det;
    %     end
    %     %     Amp(isnan(Amp)) = 0;
    %     %     Lambda1(isnan(Lambda1)) = 0;
    %
    %     LtMethod1 = cat(1, LtMethod1, Lambda1);
    %     PointWiseFit = exp(Amp + Lambda1);
    wlen = 3;
    [~, Amp1, Lambda1, PointWiseFit1] = Rt_expfit1(NewCasesSmoothed, wlen, 1);
    LtMethod1 = cat(1, LtMethod1, {Lambda1});
    
    % % %     lag = 1;
    % % % %     Rate = log(NewCasesSmoothed(1 + lag : end)./NewCasesSmoothed(1 : end - lag));
    % % %     Rate = [zeros(1, lag) ; log(NewCasesSmoothed(1 + lag : end)./NewCasesSmoothed(1 : end - lag))];
    % % %     %     Rate = log(NewCases(1 + lag : end)./NewCases(1 : end - lag));
    % % %
    % % %     avgdays = 7;
    % % %     R0period = 7;
    % % %     R = filter(ones(1, avgdays), avgdays, Rate);
    generation_period = 3;
    [~, Lambda2, RtSmoothed, Lambda2Smoothed] = Rt_expfit2(NewCasesSmoothed, wlen, generation_period, 1);
    
    [Rt3, Amp3, Lambda3, PointWiseFit3] = Rt_expfit3(NewCasesSmoothed, wlen, 1);
    
    LtMethod2 = cat(1, LtMethod2, {Lambda2});
    LtMethod3 = cat(1, LtMethod3, {Lambda2Smoothed});
    LtMethod4 = cat(1, LtMethod4, {Lambda3});
    
    %     Noise = NewCasesFilled - NewCasesSmoothed;
    %
    %     NoiseNormalized = Noise./NewCasesSmoothed;
    %     NoiseNormalized(isnan(NoiseNormalized)) = 0;
    %
    % % %     % To check vs the Johns Hopkins University dataset (DONE!)
    % % %     AllCasesFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv';
    % % %     AllDeathsFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv';
    % % %     AllRecoveredFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv';
    % % %     RegionList = {'US'};
    % % %     min_cases = 100; % min number of cases
    % % %     % load COBID-19 data (https://github.com/CSSEGISandData/COVID-19.git)
    % % %     [TotalCases, Infected, Recovered, Deceased, FirstCaseDateIndex, MinCaseDateIndex, NumDays] = ReadCOVID19Data(AllCasesFname, AllDeathsFname, AllRecoveredFname, RegionList, min_cases);
    
    % Noise & Ones augmented (ones compensates for DC and noise for randomness)
    %     InterventionPlansAug = [InterventionPlans, randn(size(InterventionPlans, 1), 1), ones(size(InterventionPlans, 1), 1)];
    
    % Noise, Ones and IP Ramp augmented (ones compensates for DC and noise for randomness)
    %         InterventionPlansAug = [InterventionPlans, cumsum(InterventionPlans, 1), randn(size(InterventionPlans, 1), 1), ones(size(InterventionPlans, 1), 1)];
    
    % Noise, Ones, IP lagged and IP Ramp augmented (ones compensates for DC and noise for randomness)
    lag = 3;
    InterventionPlansLagged = [zeros(lag, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag, :)];
    InterventionPlansAug = [InterventionPlans, InterventionPlansLagged, cumsum(InterventionPlans, 1), cumsum(InterventionPlansLagged, 1), randn(size(InterventionPlans, 1), 1), ones(size(InterventionPlans, 1), 1)];
    
    % Noise, Ones and IP lagged (ones compensates for DC and noise for randomness)
    %     InterventionPlansAug = [InterventionPlans, ones(size(InterventionPlans, 1), 1), randn(size(InterventionPlans, 1), 1)];
    
    train_samples = size(InterventionPlansAug, 1) - 0;%round(1.0*size(InterventionPlansAug, 1));
    IPtoRateMap = InterventionPlansAug(1 : train_samples, :)\Lambda2Smoothed(1 : train_samples)';
    
    %         LambdaHat = InterventionPlansAug*(InterventionPlansAug\Lambda2');
    LambdaHat = (InterventionPlansAug*IPtoRateMap)';
    %         lambda_threshold = 0.2;
    %         LambdaHat(LambdaHat > lambda_threshold) = lambda_threshold;
    %         LambdaHat(LambdaHat < -lambda_threshold) = -lambda_threshold;
    % % %         LambdaHat(train_samples + 1 : end) = 0.99 * LambdaHat(train_samples + 1 : end);
    
    % Noise augmented
    %         InterventionPlansAug = [InterventionPlans randn(size(InterventionPlans, 1), 1)];
    %         LambdaHat = InterventionPlansAug*(InterventionPlansAug\Lambda2');
    
    % Sign-based
    %         InterventionPlansAug = [InterventionPlans randn(size(InterventionPlans, 1), 1)];
    %         LambdaHat = InterventionPlansAug*(InterventionPlansAug\sign(Lambda2'));
    
    % Build an estimate of the new cases
    CumLambdaHat = cumsum(LambdaHat);
    NewCasesEstimate = 2*NewCasesSmoothed(1)*exp(CumLambdaHat);
    
    e = Lambda2 - LambdaHat;
    
    if(plot_figures)
        %     dn = datenum(string(dates),'yyyymmdd');
        dn = 1 : length(NewCasesSmoothed);
        ind_pos = find(Lambda2 >= 0);
        ind_neg = find(Lambda2 < 0);
        
        lgn = [];
        figure
        hold on;
        %         plot(dn, InterventionPlans, 'color', 0.8*ones(1, 3)); lgn = cat(2, lgn, {'IP'});
        plot(dn, mean(InterventionPlans, 2), 'color', 0.4*ones(1, 3)); lgn = cat(2, lgn, {'Mean IP'});
        %         plot(dn, Lambda1, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda1'});
        plot(dn, Lambda2, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2'});
        plot(dn, Lambda2Smoothed, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2Smoothed'});
        %         plot(dn, Lambda3, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda3'});
        %         plot(dn , e, 'linewidth', 3); lgn = cat(2, lgn, {'Recon error'});
        plot(dn , LambdaHat, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHat'});
        legend(lgn);
        grid
        title(CountryAndRegionList(k), 'interpreter', 'none');
        %     datetick('x','mmm/dd', 'keepticks','keeplimits')
        axis 'tight'
        
        lgn = [];
        figure
        hold on
        %     plot(dn, ConfirmedCases); lgn = cat(2, lgn, {'ConfirmedCases'});
        %     plot(dn, ConfirmedDeaths); lgn = cat(2, lgn, {'ConfirmedDeaths'});
        %         plot(dn, NewCases); lgn = cat(2, lgn, {'NewCases'});
        plot(dn, NewCasesSmoothed, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesSmoothed'});
        plot(dn(ind_pos), NewCasesSmoothed(ind_pos)', 'bo'); lgn = cat(2, lgn, {'NewCasesSmoothed Rising'});
        plot(dn(ind_neg), NewCasesSmoothed(ind_neg)', 'ro'); lgn = cat(2, lgn, {'NewCasesSmoothed Falling'});
        plot(dn, PointWiseFit1); lgn = cat(2, lgn, {'PointWiseFit1'});
        plot(dn, PointWiseFit3); lgn = cat(2, lgn, {'PointWiseFit3'});
        plot(dn, NewCasesEstimate, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimate'});
        legend(lgn);
        grid
        title(CountryAndRegionList(k), 'interpreter', 'none');
        %     datetick('x','mmm/dd', 'keepticks','keeplimits')
        axis 'tight'
        
        % % %         lgn = [];
        % % %         figure
        % % %         hold on
        % % %         % % %                 plot(dn, Amp1); lgn = cat(2, lgn, {'Amp1'});
        % % %         %     plot(dn, Lambda1/nanstd(Lambda1)*nanstd(NewCasesSmoothed)); lgn = cat(2, lgn, {'Normalized Lambda1'});
        % % %         plot(dn, Lambda1); lgn = cat(2, lgn, {'Lambda1'});
        % % %         plot(dn, Lambda2); lgn = cat(2, lgn, {'Lambda2'});
        % % %         plot(dn, Lambda2Smoothed); lgn = cat(2, lgn, {'Lambda2Smoothed'});
        % % %         plot(dn, Lambda3); lgn = cat(2, lgn, {'Lambda3'});
        % % %         %     plot(dn, exp(Amp).*exp(Lambda1)); lgn = cat(2, lgn, {'Amp'});
        % % %         %     plot(dn, Lambda1); lgn = cat(2, lgn, {'Lambda1'});
        % % %         % %     plot(dn, NoiseNormalized); lgn = cat(2, lgn, {'NoiseNormalized'});
        % % %         % % %     plot([nan(1, 21) TotalCases]); lgn = cat(2, lgn, {'ConfirmedCases JHU'});
        % % %         legend(lgn);
        % % %         grid
        % % %         title(CountryAndRegionList(k), 'interpreter', 'none');
        % % %         %     datetick('x','mmm/dd', 'keepticks','keeplimits')
        % % %         axis 'tight'
        
    end
end

% figure
% hold on
% plot(real(LtMethod1'));
% plot(real(LtMethod1(1, :)'), 'k', 'linewidth', 3);
% plot(nanmean(real(LtMethod1), 1), 'r', 'linewidth', 3);
% grid
% title('US Average R0');