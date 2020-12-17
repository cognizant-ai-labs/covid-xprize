close all
clear
clc

% parameters
plot_figures = true; % plot per-region/country plots or not
min_cases = 100; % the minimum cases start date for processing each region/country
start_date = 20200101; % start date
end_date = 20201225; % end date
predict_ahead_num_days = 28; % number of days to predict ahead
Rt_wlen = 7; % Reproduction rate estimation window
Rt_generation_period = 3; % The generation period used for calculating the reproduction number
lambda_threshold = 10.25; % The threshold for the maximum absolute value of the reproduction rates exponent lambda
filter_type = 'MOVINGAVERAGE-CAUSAL'; % 'MOVINGAVERAGE-NONCAUSAL' or 'MOVINGAVERAGE-CAUSAL' ' or 'MOVINGMEDIAN' or 'TIKHONOV'; % The last two call functions from the OSET package (oset.ir). Note: 'MOVINGAVERAGE-CAUSAL' is the contest standard and only evaluation algorithm
% Tikhonov regularization params (if selected by filter_type):
% DiffOrderOrFilterCoefs = [1 -2 1]; % Smoothness filter coefs
DiffOrderOrFilterCoefs = 2; % Smoothness filter order
gamma = 25.0; % Tikhonov roughness penalty

data_file = './../../covid-policy-tracker/data/OxCGRT_latest.csv'; % The data-file cloned from: https://github.com/OxCGRT/covid-policy-tracker/tree/master/data
included_IP = {'C1_School closing',... % see full descriptions at: https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md
    'C1_Flag',...
    'C2_Workplace closing',...
    'C2_Flag',...
    'C3_Cancel public events',...
    'C3_Flag',...
    'C4_Restrictions on gatherings',...
    'C4_Flag',...
    'C5_Close public transport',...
    'C5_Flag',...
    'C6_Stay at home requirements',...
    'C6_Flag',...
    'C7_Restrictions on internal movement',...
    'C7_Flag',...
    'C8_International travel controls',...
    'E1_Income support',...
    'E1_Flag',...
    'E2_Debt/contract relief',... % 'E3_Fiscal measures', 'E4_International support',...
    'H1_Public information campaigns',...
    'H1_Flag',...
    'H2_Testing policy',...
    'H3_Contact tracing',... % 'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',...
    'H6_Facial Coverings',...
    'H6_Flag',... % {'H7_Vaccination policy', 'H7_Flag',...
    };

% Read Oxford time-series data
import_opt = detectImportOptions(data_file);
import_opt.VariableNamingRule = 'preserve';
% import_opt.VariableTypes(16) = {'double'}; % C5_Flag on column 16 is of type double (the format is incorrectly read as 'char')
import_opt.VariableTypes(6:end) = {'double'}; % All except the first five columns are 'double' (correct if not correctly identified)
all_data = readtable(data_file, import_opt);%'ReadVariableNames', true, 'VariableNamingRule', 'preserve', 'TextType', 'string', 'detectImportOptions', 'true');
ColumnHeaders = all_data.Properties.VariableNames; % Column titles
AllCountryNames = all_data.CountryName;%(:, ismember(ColumnHeaders,'CountryName')); % All country names
AllCountryCodes = all_data.CountryCode;%(:, ismember(ColumnHeaders,'CountryCode')); % All country codes
AllRegionNames = all_data.RegionName;%(:, ismember(ColumnHeaders,'RegionName')); % All region names
AllRegionCodes = all_data.RegionCode;%(:, ismember(ColumnHeaders,'RegionCode')); % All region codes

% Make GeoID code (A combination of region and country codes)
GeoID = strcat(string(AllCountryCodes), string(AllRegionCodes));

% Make a dictionary of country-regions
[CountryAndRegionList, IA, IC] = unique(GeoID, 'stable');

NumGeoLocations = length(CountryAndRegionList); % Number of country-region pairs

% FEATURE EXTRACTION (Different methods for calculating the reproduction rate)
for k = 219 : 221% 1: NumGeoLocations
    %     row_indexes = GeoID == CountryAndRegionList(k) & all_data.ConfirmedCases > min_cases;
    %     geoid_all_row_indexes = GeoID == CountryAndRegionList(k) & all_data.Date >= start_date & all_data.Date <= end_date;
    geoid_all_row_indexes = GeoID == CountryAndRegionList(k) & all_data.ConfirmedCases > min_cases & all_data.Date >= start_date & all_data.Date <= end_date;
    
    ConfirmedCases = all_data.ConfirmedCases(geoid_all_row_indexes); % Confirmed cases
    ConfirmedDeaths = all_data.ConfirmedDeaths(geoid_all_row_indexes); % Death cases
    InterventionPlans = all_data{geoid_all_row_indexes, included_IP}; % Region/Country intervention plans
    
    InterventionPlans(isnan(InterventionPlans)) = 0; % Replace N/A IP with no IP
    
    % Calculate the number of new cases
    NewCases = [nan; diff(ConfirmedCases)]; % calculate the new daily cases
    NewCases(NewCases < 0) = 0;
    
    % Replace nans with 0 and the last one with the previous number
    NewCasesFilled = NewCases;
    NewCasesFilled(isnan(NewCasesFilled)) = 0;
    if(isnan(NewCases(end))) % Just fill the last missing date (if any) with the previous day
        NewCasesFilled(end) = NewCasesFilled(end-1);
    end
    
    % Smooth the newcases time series
    switch filter_type
        case 'TIKHONOV' % Tikhonov regularization
            NewCasesSmoothed = TikhonovRegularization(NewCasesFilled', DiffOrderOrFilterCoefs, gamma)';
        case 'MOVINGAVERAGE-CAUSAL'
            NewCasesSmoothed = filter(ones(1, Rt_wlen), Rt_wlen, NewCasesFilled); % causal
        case 'MOVINGAVERAGE-NONCAUSAL'
            NewCasesSmoothed = BaseLine1(NewCasesFilled', Rt_wlen, 'mn')'; % non-causal (zero-phase)
        case 'MOVINGMEDIAN'
            NewCasesSmoothed = BaseLine1(NewCasesFilled', floor(Rt_wlen/2), 'md')';
            NewCasesSmoothed = BaseLine1(NewCasesSmoothed', Rt_wlen, 'mn')';
        otherwise
            error('Unknown filter type');
    end
    
    [~, Amp1, Lambda1, PointWiseFit1] = Rt_expfit1(NewCasesSmoothed, Rt_wlen, 1);
    
    [~, Lambda2, RtSmoothed, Lambda2Smoothed] = Rt_expfit2(NewCasesSmoothed, Rt_wlen, Rt_generation_period, 1);
    
    [Rt3, Amp3, Lambda3, PointWiseFit3] = Rt_expfit3(NewCasesSmoothed, Rt_wlen, 1);
    
    % Generate feature vectors based on intervention plans
    numTimeStepsTrain = size(InterventionPlans, 1) - predict_ahead_num_days;%floor(0.65*numel(y_data));
    numTimeStepsTest = size(InterventionPlans, 1) - numTimeStepsTrain;
    
    % Lagged intervention plans
    lag1 = 3;
    InterventionPlansLagged1 = [zeros(lag1, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag1, :)];
    lag2 = 5;
    InterventionPlansLagged2 = [zeros(lag2, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag2, :)];
    lag3 = 7;
    InterventionPlansLagged3 = [zeros(lag3, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag3, :)];
    
    % Regression/Classification Phase
    % Y Data
    y_data = Lambda2';
    %     y_data = Lambda2Smoothed';
    
    % Replace nans in y_data with latest non-nan values
    %         I_nans = find(isnan(y_data) | isinf(y_data));
    for jj = 2 : length(y_data)
        if(isnan(y_data(jj)) || isinf(y_data(jj)))
            y_data(jj) = y_data(jj -1);
        end
    end
    % Remove any remaining nans
    I_non_nans = ~isnan(y_data);
    y_data = y_data(I_non_nans);
    
    % y_data train and test sets
    y_data_train = y_data(1 : numTimeStepsTrain);
    y_data_test = y_data(numTimeStepsTrain + 1 : end);
    
    % Autoregressive model
    ar_order = 14;
    ar_learninghistory = 120;
    
    ar_train_segment = y_data_train(end - ar_learninghistory + 1 : end);
    ar_sys = ar(ar_train_segment, ar_order);
    A_ar = get(ar_sys, 'A');
    noisevar_ar = get(ar_sys, 'NoiseVariance');
    zi = filtic(sqrt(noisevar_ar), A_ar, y_data_train(end:-1:1));
    y_pred_ar = filter(sqrt(noisevar_ar), A_ar, randn(1, numTimeStepsTest), zi)';
    LambdaHatARX = [y_data_train ; y_pred_ar]';
    
    AllFeatures = [InterventionPlans, InterventionPlansLagged1, InterventionPlansLagged2, InterventionPlansLagged3, LambdaHatARX', ones(size(InterventionPlans, 1), 1)];
    %     AllFeatures = [InterventionPlans, InterventionPlansLagged1, InterventionPlansLagged2, InterventionPlansLagged3, randn(size(InterventionPlans, 1), 1), ones(size(InterventionPlans, 1), 1)];
    %         AllFeatures = [cumsum(InterventionPlansLagged1, 1), InterventionPlansLagged1, cumsum(InterventionPlansLagged3, 1), InterventionPlansLagged3, ones(size(InterventionPlans, 1), 1)];
    % Noise, Ones and IP lagged (ones compensates for DC and noise for randomness)
    %     AllFeatures = [InterventionPlans, ones(size(InterventionPlans, 1), 1), randn(size(InterventionPlans, 1), 1)];
    
    %     figure
    %     plot([y_data_train ; y_data_test])
    %     hold on
    %     plot([y_data_train ; y_pred_ar]);
    %     grid
    %     legend('real sequence', 'AR fit');
    
    % X Data
    x_data = AllFeatures;
    x_data = x_data(I_non_nans, :);
    
    % x_data train and test sets
    x_data_train = x_data(1 : numTimeStepsTrain, :);
    x_data_test = x_data(numTimeStepsTrain + 1 : end, :);
    
    % Method: Linear predictor
    % LS_regularization_factor
    IPtoRateMap = x_data_train\y_data_train;
    y_pred_lin = x_data_test*IPtoRateMap;
    LambdaHatLinear = [y_data_train ; y_pred_lin]';
    
    % Method: SVM
    Mdlsvm = fitrsvm(x_data_train, y_data_train);%, , 'KFold', 10);
    Mdlsvm.ConvergenceInfo.Converged
    y_pred_svm = predict(Mdlsvm, x_data_test);
    LambdaHatSVM = [y_data_train ; y_pred_svm]';
    
    % Method: SVM with gaussian kernel
    Mdlsvmgau = fitrsvm(x_data_train, y_data_train,'KernelFunction','gaussian', 'Standardize', true);%, 'Standardize', true, 'KFold', 10);
    Mdlsvmgau.ConvergenceInfo.Converged
    y_pred_svmgau = predict(Mdlsvmgau, x_data_test);
    LambdaHatSVMGAU = [y_data_train ; y_pred_svmgau]';
    
    % Method: LSTM
    numFeatures = size(x_data, 2);
    numResponses = 1;
    numHiddenUnits = 200;
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    options = trainingOptions('adam', ...
        'MaxEpochs',150, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    MdlLSTM = trainNetwork(x_data_train', y_data_train', layers, options);
    MdlLSTM = predictAndUpdateState(MdlLSTM, x_data_train');
    [MdlLSTM, y_pred_lstm] = predictAndUpdateState(MdlLSTM, x_data_test');
    LambdaHatLSTM = [y_data_train ; y_pred_lstm']';
    
    
    % Postprocess the estimates
    % Clip the max incline/decline
    counter = 1 : length(LambdaHatLinear);
    I_pos = LambdaHatLinear > lambda_threshold & counter > numTimeStepsTrain;
    I_neg = LambdaHatLinear < -lambda_threshold & counter > numTimeStepsTrain;
    LambdaHatLinear(I_pos) = lambda_threshold;
    LambdaHatLinear(I_neg) = -lambda_threshold;
    
    I_pos = LambdaHatSVM > lambda_threshold & counter > numTimeStepsTrain;
    I_neg = LambdaHatSVM < -lambda_threshold & counter > numTimeStepsTrain;
    LambdaHatSVM(I_pos) = lambda_threshold;
    LambdaHatSVM(I_neg) = -lambda_threshold;
    
    I_pos = LambdaHatSVMGAU > lambda_threshold & counter > numTimeStepsTrain;
    I_neg = LambdaHatSVMGAU < -lambda_threshold & counter > numTimeStepsTrain;
    LambdaHatSVMGAU(I_pos) = lambda_threshold;
    LambdaHatSVMGAU(I_neg) = -lambda_threshold;
    
    I_pos = LambdaHatLSTM > lambda_threshold & counter > numTimeStepsTrain;
    I_neg = LambdaHatLSTM < -lambda_threshold & counter > numTimeStepsTrain;
    LambdaHatLSTM(I_pos) = lambda_threshold;
    LambdaHatLSTM(I_neg) = -lambda_threshold;

    % Build an estimate of the new cases
    CumLambdaHatLinear = cumsum(LambdaHatLinear(counter > numTimeStepsTrain))';
    NewCasesEstimateLinear = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesSmoothed(numTimeStepsTrain)*exp(CumLambdaHatLinear)];
    
    CumLambdaHatSVM = cumsum(LambdaHatSVM(counter > numTimeStepsTrain))';
    NewCasesEstimateSVM = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesSmoothed(numTimeStepsTrain)*exp(CumLambdaHatSVM)];
    
    CumLambdaHatSVMGAU = cumsum(LambdaHatSVMGAU(counter > numTimeStepsTrain))';
    NewCasesEstimateSVMGAU = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesSmoothed(numTimeStepsTrain)*exp(CumLambdaHatSVMGAU)];
    
    CumLambdaHatLSTM = cumsum(LambdaHatLSTM(counter > numTimeStepsTrain))';
    NewCasesEstimateLSTM = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesSmoothed(numTimeStepsTrain)*exp(CumLambdaHatLSTM)];

    if(plot_figures)
        %     dn = datenum(string(geoid_dates_unsorted),'yyyymmdd');
        dn = 1 : length(NewCasesSmoothed);
        ind_pos = find(Lambda2 >= 0);
        ind_neg = find(Lambda2 < 0);
        
        lgn = [];
        figure
        hold on;
        %         plot(dn, InterventionPlans, 'color', 0.8*ones(1, 3)); lgn = cat(2, lgn, {'IP'});
        plot(dn, mean(InterventionPlans, 2), 'color', 0.4*ones(1, 3)); lgn = cat(2, lgn, {'Mean IP'});
        plot(dn, Lambda1, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda1'});
        plot(dn, Lambda2, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2'});
        plot(dn, Lambda2Smoothed, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2Smoothed'});
        plot(dn, Lambda3, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda3'});
        plot(dn , LambdaHatARX, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatARX'});
        plot(dn , LambdaHatLinear, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatLinear'});
        plot(dn , LambdaHatSVM, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatSVM'});
        plot(dn , LambdaHatSVMGAU, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatSVMGAU'});
        % % %         plot(dn , (LambdaHatLinear + LambdaHatSVM)/2, 'linewidth', 3); lgn = cat(2, lgn, {'Avg. LambdaHats'});
        plot(dn , LambdaHatLSTM, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatLSTM'});
        
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
        %         plot(dn, PointWiseFit1); lgn = cat(2, lgn, {'PointWiseFit1'});
        %         plot(dn, PointWiseFit3); lgn = cat(2, lgn, {'PointWiseFit3'});
        plot(dn, NewCasesEstimateLinear, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateLinear'});
        plot(dn, NewCasesEstimateSVM, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateSVM'});
        plot(dn, NewCasesEstimateSVMGAU, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateSVMGAU'});
        plot(dn, NewCasesEstimateLSTM, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateLSTM'});
        
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