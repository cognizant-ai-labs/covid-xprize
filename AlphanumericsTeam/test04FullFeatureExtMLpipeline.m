close all
clear
clc

% parameters
LINEAR = false; % perform LINEAR estimate or not
ARX = false; % perform ARX estimate or not
LSTM = false; % perform LSTM or not
LSTM_ON_NEW_CASES = false;
SVM = false; % perform SVM or not
plot_figures = true; % plot per-region/country plots or not

start_date_criterion = 'DATE_BASED'; %'MIN_CASE_BASED'; % 'MIN_CASE_BASED'/'DATE_BASED'/'DATA_OR_MIN_CASE_BASED'
min_cases = 100; % the minimum cases start date for processing each region/country
start_date = 20200101; % start date
end_date = 20201225; % end date

lambda_baseline_wlen = 14; % window length used for lambda baseline calculation
ar_order = 12; % Autoregressive model
ar_learninghistory = 120; % Autoregressive model estimation history look back
predict_ahead_num_days = 30; % number of days to predict ahead
Rt_wlen = 7; % Reproduction rate estimation window
Rt_generation_period = 3; % The generation period used for calculating the reproduction number
lambda_threshold = 0.06; % The threshold for the maximum absolute value of the reproduction rates exponent lambda
filter_type = 'MOVINGAVERAGE-CAUSAL'; % 'MOVINGAVERAGE-NONCAUSAL' or 'MOVINGAVERAGE-CAUSAL' ' or 'MOVINGMEDIAN' or 'TIKHONOV'; % The last two call functions from the OSET package (oset.ir). Note: 'MOVINGAVERAGE-CAUSAL' is the contest standard and only evaluation algorithm
% Tikhonov regularization params (if selected by filter_type):
% DiffOrderOrFilterCoefs = [1 -2 1]; % Smoothness filter coefs
DiffOrderOrFilterCoefs = 2; % Smoothness filter order
gamma = 25.0; % Tikhonov roughness penalty

% data_file = './../../covid-policy-tracker/data/OxCGRT_latest.csv'; % The data-file cloned from: https://github.com/OxCGRT/covid-policy-tracker/tree/master/data
data_file = './data/files/OxCGRT_latest_aug.csv'; % The data-file cloned from: https://github.com/OxCGRT/covid-policy-tracker/tree/master/data
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
    'C8_International travel controls',... %'E1_Income support', 'E1_Flag', 'E2_Debt/contract relief',... % 'E3_Fiscal measures', 'E4_International support',...
    'H1_Public information campaigns',...
    'H1_Flag',...
    'H2_Testing policy',...
    'H3_Contact tracing',... % 'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',...
    'H6_Facial Coverings',...
    'H6_Flag',... % {'H7_Vaccination policy', 'H7_Flag',...
    'Holidays'...
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
for k = 1 : 5 %122 : 125%225 %1 : NumGeoLocations
    k
    switch start_date_criterion
        case 'MIN_CASE_BASED'
            geoid_all_row_indexes = GeoID == CountryAndRegionList(k) & all_data.ConfirmedCases > min_cases & all_data.Date <= end_date;
        case 'DATE_BASED'
            geoid_all_row_indexes = GeoID == CountryAndRegionList(k) & all_data.Date >= start_date & all_data.Date <= end_date;
        case 'DATA_OR_MIN_CASE_BASED'
            geoid_all_row_indexes = GeoID == CountryAndRegionList(k) & all_data.ConfirmedCases > min_cases & all_data.Date >= start_date & all_data.Date <= end_date;
    end
    ConfirmedCases = all_data.ConfirmedCases(geoid_all_row_indexes); % Confirmed cases
    ConfirmedDeaths = all_data.ConfirmedDeaths(geoid_all_row_indexes); % Death cases
    
    Holidays = all_data.Holidays(geoid_all_row_indexes); % Country/Region holidays
    
    % Make long weekends
    Holidays(isnan(Holidays)) = 0;
    if(1)
        for mm = 2 : length(Holidays)-1
            if(Holidays(mm-1) ~= 0 && Holidays(mm+1) ~= 0)
                Holidays(mm-1) = -2; % Long holidays have reverse impact
                Holidays(mm) = -2;
                Holidays(mm+1) = -2;
            end
        end
        % Remove normal weekends
        % Holidays(Holidays == 1) = 0;
    end
    
    %     for mm = 2 : length(Holidays)-1
    %         if(Holidays(mm-1) ~= 0 && Holidays(mm+1) ~= 0)
    %             Holidays(mm) = 1;
    %         end
    %     end
    %
    %     % Make long weekends more risky
    %     for mm = 2 : length(Holidays)-1
    %         if(Holidays(mm-1) ~= 0 && Holidays(mm) ~= 0 && Holidays(mm+1) ~= 0)
    %             Holidays(mm-1) = 2;
    %             Holidays(mm) = 2;
    %             Holidays(mm+1) = 2;
    %         end
    %     end
    
    all_data.Holidays(geoid_all_row_indexes) = Holidays;
    
    InterventionPlans = all_data{geoid_all_row_indexes, included_IP}; % Region/Country intervention plans
    
    % Replace all N/A IP with previous IP
    for j = 1 : size(InterventionPlans, 2)
        for i = 2 : size(InterventionPlans, 1)
            if(isnan(InterventionPlans(i, j)) && not(isnan(InterventionPlans(i - 1, j))))
                InterventionPlans(i, j) = InterventionPlans(i - 1, j);
            end
        end
    end
    InterventionPlans(isnan(InterventionPlans)) = 0; % Replace any remaining N/A IP with no IP
    
    % Calculate the number of new cases
    NewCases = [nan; diff(ConfirmedCases)]; % calculate the new daily cases
    NewCases(NewCases < 0) = 0;
    
    if(length(NewCases) < 2)
        warning(strcat("Insufficient data for ", CountryAndRegionList(k), ". Skipping the country/region."));
        continue;
    end
    
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
%     Lambda2Baseline = BaseLine1(Lambda2, lambda_baseline_wlen, 'md');
Lambda2Baseline = BaseLine2(Lambda2, lambda_baseline_wlen, lambda_baseline_wlen, 'md');
    
    [Rt3, Amp3, Lambda3, PointWiseFit3] = Rt_expfit3(NewCasesSmoothed, Rt_wlen, 1);
    
    % Generate feature vectors based on intervention plans
    numTimeStepsTrain = size(InterventionPlans, 1) - predict_ahead_num_days;%floor(0.65*numel(y_data));
    numTimeStepsTest = size(InterventionPlans, 1) - numTimeStepsTrain;
    
    % Lagged intervention plans
    lag1 = 3;
    InterventionPlansLagged1 = [zeros(lag1, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag1, :)];
    %     InterventionPlansLagged1 = filter(ones(1, lag1), lag1, InterventionPlans);
    lag2 = 5;
    InterventionPlansLagged2 = [zeros(lag2, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag2, :)];
    %     InterventionPlansLagged2 = filter(ones(1, lag2), lag2, InterventionPlans);
    lag3 = 7;
    InterventionPlansLagged3 = [zeros(lag3, size(InterventionPlans, 2)) ; InterventionPlans(1 : end - lag3, :)];
    %     InterventionPlansLagged3 = filter(ones(1, lag3), lag3, InterventionPlans);
    
    % Regression/Classification Phase
    % Y Data
    
    lambda_vector = Lambda1';
    %     lambda_vector = Lambda2';
    %     lambda_vector = Lambda2Smoothed';
    %     y_data = diff([lambda_vector(1) ; lambda_vector]);
    y_data = lambda_vector;
    
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
    
    y_data2 = NewCasesSmoothed;
    for jj = 2 : length(y_data2)
        if(isnan(y_data2(jj)) || isinf(y_data2(jj)))
            y_data2(jj) = y_data2(jj -1);
        end
    end
    
    % Remove any remaining nans
    I_non_nans2 = ~isnan(y_data2);
    y_data2 = y_data2(I_non_nans2);
    y_max = 10.0 * max(y_data2);
    y_data2 = y_data2/y_max;
    
    % y_data2 train and test sets
    y_data2_train = y_data2(1 : numTimeStepsTrain);
    y_data2_test = y_data2(numTimeStepsTrain + 1 : end);
    
    % Smooth the intervention plans
    %     smoothing_wlen = 7;
    %     InterventionPlans = filter(hamming(smoothing_wlen), 1, InterventionPlans);
    
    %     InterventionPlansZeroMean = InterventionPlans - ones(size(InterventionPlans, 1), 1)*mean(InterventionPlans);
    %     AllFeatures = [InterventionPlans cumsum(InterventionPlansZeroMean, 1) ones(size(InterventionPlans, 1), 1)];
    AllFeatures = [InterventionPlans];% InterventionPlansLagged1 InterventionPlansLagged2 InterventionPlansLagged3 ones(size(InterventionPlans, 1), 1) randn(size(InterventionPlans, 1), 1)];%, InterventionPlansLagged1, InterventionPlansLagged2, InterventionPlansLagged3, ones(size(InterventionPlans, 1), 1)];
    %     AllFeatures = [InterventionPlansLagged1, LambdaHatARX', ones(size(InterventionPlans, 1), 1)];
    %     AllFeatures = [InterventionPlans, InterventionPlansLagged1, InterventionPlansLagged2, InterventionPlansLagged3, LambdaHatARX', ones(size(InterventionPlans, 1), 1)];
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
    
    % Normalize X data
    x_mx = max(abs(x_data));
    x_mx(x_mx == 0) = 1;
    x_data = x_data ./ (ones(size(x_data, 1), 1) * x_mx);
    
    % x_data train and test sets
    x_data_train = x_data(1 : numTimeStepsTrain, :);
    x_data_test = x_data(numTimeStepsTrain + 1 : end, :);
    
    % Method: Autoregressive model
    if(ARX)
        ar_train_segment = y_data_train(end - ar_learninghistory + 1 : end);
        ar_sys = ar(ar_train_segment, ar_order);
        A_ar = get(ar_sys, 'A');
        noisevar_ar = get(ar_sys, 'NoiseVariance');
        zi = filtic(sqrt(noisevar_ar), A_ar, y_data_train(end:-1:1));
        y_pred_ar = filter(sqrt(noisevar_ar), A_ar, randn(1, numTimeStepsTest), zi)';
        LambdaHatARX = [y_data_train ; y_pred_ar]';
        %     LambdaHatARX = LambdaHatARX + lambda_vector(numTimeStepsTrain) - LambdaHatARX(numTimeStepsTrain); % correct offset
    end
    
    % Method: Linear predictor
    if(LINEAR)
        % LS_regularization_factor
        %     IPtoRateMap = x_data_train\y_data_train;
        beta = 1e-6;
        IPtoRateMap = (x_data_train'*x_data_train + beta*eye(size(x_data_train,2)))\(x_data_train'*y_data_train);
        y_pred_lin = x_data_test*IPtoRateMap;
        LambdaHatLinear = [y_data_train ; y_pred_lin]';
        %     LambdaHatLinear = LambdaHatLinear + lambda_vector(numTimeStepsTrain) - LambdaHatLinear(numTimeStepsTrain); % correct offset
    end
    
    % Method: SVM
    if(SVM)
        Mdlsvm = fitrsvm(x_data_train, y_data_train);%, , 'KFold', 10);
        %     Mdlsvm.ConvergenceInfo.Converged
        y_pred_svm = predict(Mdlsvm, x_data_test);
        LambdaHatSVM = [y_data_train ; y_pred_svm]';
        %     LambdaHatSVM = LambdaHatSVM + lambda_vector(numTimeStepsTrain) - LambdaHatSVM(numTimeStepsTrain); % correct offset
        
        % Method: SVM with gaussian kernel
        Mdlsvmgau = fitrsvm(x_data_train, y_data_train,'KernelFunction','gaussian');%, 'KernelScale','auto', 'Standardize', true);%, 'Standardize', true, 'KFold', 10);
        %     Mdlsvmgau.ConvergenceInfo.Converged
        y_pred_svmgau = predict(Mdlsvmgau, x_data_test);
        LambdaHatSVMGAU = [y_data_train ; y_pred_svmgau]';
        %     LambdaHatSVMGAU = LambdaHatSVMGAU + lambda_vector(numTimeStepsTrain) - LambdaHatSVMGAU(numTimeStepsTrain); % correct offset
    end
    % Method: Auto optimize parameters
    %     Mdlopt = fitrsvm(x_data_train, y_data_train,'OptimizeHyperparameters','auto',...
    %         'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    %         'expected-improvement-plus'));
    %     y_pred_svmopt = predict(Mdlopt, x_data_test);
    %     LambdaHatSVMOPT = [y_data_train ; y_pred_svmopt]';
    
    % Method: LSTM
    if(LSTM)
        numFeatures = size(x_data, 2) + 1; % The IP features plus previous time series
        numHiddenUnits = 50;
        numResponses = 1;
        layers = [ ...
            %             sequenceInputLayer(numFeatures)%, 'Normalization', 'zscore')
            sequenceInputLayer(numFeatures, 'Normalization', 'rescale-zero-one')
            %
            %             fullyConnectedLayer(numFeatures)
            lstmLayer(numFeatures)
            %             fullyConnectedLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %                         lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             fullyConnectedLayer(numFeatures)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            %             lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            %             MyTanhLayer(numResponses, 'Tanh', 0.06) %tanhLayer
            %             expLayer(numResponses, 'Exp')
            % reluLayer
            regressionLayer];
        options = trainingOptions('adam', ...
            'MaxEpochs',250, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.001, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',1, ...
            'Plots','none');%'training-progress');
        
        if(LSTM_ON_NEW_CASES)
            x_data_train_augmented = [x_data_train, [y_data2_train(1, :) ; y_data2_train(1 : end - 1, :)]];
            % % %         x_data_train_augmented = (x_data_train_augmented - ones(size(x_data_train_augmented, 1), 1)*nanmean(x_data_train_augmented));
            % % %         x_data_train_augmented = x_data_train_augmented./(ones(size(x_data_train_augmented, 1), 1)*nanstd(x_data_train_augmented));
            
            MdlLSTM = trainNetwork(x_data_train_augmented', y_data2_train', layers, options);
            y_pred_last = y_data2_train(end);%y_pred(end);
        else
            x_data_train_augmented = [x_data_train, [y_data_train(1, :) ; y_data_train(1 : end - 1, :)]];
            
            MdlLSTM = trainNetwork(x_data_train_augmented', y_data_train', layers, options);
            y_pred_last = y_data_train(end);%y_pred(end);
            
        end
        [MdlLSTM, y_pred] = predictAndUpdateState(MdlLSTM, x_data_train_augmented');
        
        y_pred_lstm = zeros(numTimeStepsTest, 1);
        for ii = 1 : numTimeStepsTest
            x_data_test_augmented = [x_data_test(ii, :), y_pred_last];
            [MdlLSTM, y_pred_last] = predictAndUpdateState(MdlLSTM, x_data_test_augmented','ExecutionEnvironment','cpu');
            
            if(LSTM_ON_NEW_CASES)
                y_pred_lstm(ii, :) = y_pred_last * y_max;
            else
                % Clip the max incline/decline
                if(y_pred_last > lambda_threshold)
                    y_pred_last = lambda_threshold;
                elseif(y_pred_last < -lambda_threshold)
                    y_pred_last = -lambda_threshold;
                end
                y_pred_lstm(ii, :) = y_pred_last;
            end
        end
        
        %         MdlLSTM = predictAndUpdateState(MdlLSTM, [x_data_train(2 : end, :) y_data_train(1 : end - 1, :)]');
        %
        %         [MdlLSTM, y_pred_lstm] = predictAndUpdateState(MdlLSTM, x_data_test');
        %         LambdaHatLSTM = [y_data_train ; y_pred_lstm]';
        LambdaHatLSTM = [y_pred' ; y_pred_lstm]';
        %         LambdaHatLSTM = LambdaHatLSTM + lambda_vector(numTimeStepsTrain) - LambdaHatLSTM(numTimeStepsTrain); % correct offset
    end
    
    % Postprocess the estimates
    % Clip the max incline/decline
    counter = 1 : length(NewCases);
    if(LINEAR)
        I_pos = LambdaHatLinear > lambda_threshold & counter > numTimeStepsTrain;
        I_neg = LambdaHatLinear < -lambda_threshold & counter > numTimeStepsTrain;
        LambdaHatLinear(I_pos) = lambda_threshold;
        LambdaHatLinear(I_neg) = -lambda_threshold;
    end
    
    if(ARX)
        I_pos = LambdaHatARX > lambda_threshold & counter > numTimeStepsTrain;
        I_neg = LambdaHatARX < -lambda_threshold & counter > numTimeStepsTrain;
        LambdaHatARX(I_pos) = lambda_threshold;
        LambdaHatARX(I_neg) = -lambda_threshold;
    end
    
    if(SVM)
        I_pos = LambdaHatSVM > lambda_threshold & counter > numTimeStepsTrain;
        I_neg = LambdaHatSVM < -lambda_threshold & counter > numTimeStepsTrain;
        LambdaHatSVM(I_pos) = lambda_threshold;
        LambdaHatSVM(I_neg) = -lambda_threshold;
        
        I_pos = LambdaHatSVMGAU > lambda_threshold & counter > numTimeStepsTrain;
        I_neg = LambdaHatSVMGAU < -lambda_threshold & counter > numTimeStepsTrain;
        LambdaHatSVMGAU(I_pos) = lambda_threshold;
        LambdaHatSVMGAU(I_neg) = -lambda_threshold;
    end
    %     I_pos = LambdaHatSVMOPT > lambda_threshold & counter > numTimeStepsTrain;
    %     I_neg = LambdaHatSVMOPT < -lambda_threshold & counter > numTimeStepsTrain;
    %     LambdaHatSVMOPT(I_pos) = lambda_threshold;
    %     LambdaHatSVMOPT(I_neg) = -lambda_threshold;
    
    if(LSTM && ~ LSTM_ON_NEW_CASES)
        I_pos = LambdaHatLSTM > lambda_threshold & counter > numTimeStepsTrain;
        I_neg = LambdaHatLSTM < -lambda_threshold & counter > numTimeStepsTrain;
        LambdaHatLSTM(I_pos) = lambda_threshold;
        LambdaHatLSTM(I_neg) = -lambda_threshold;
    end
    
    %     LambdaHatTotal = median([LambdaHatARX ; LambdaHatLinear ; LambdaHatSVM ; LambdaHatSVMGAU ; LambdaHatLSTM], 1);
    %     LambdaHatTotal = mean([LambdaHatARX ; LambdaHatLinear ; LambdaHatSVM ; LambdaHatSVMGAU], 1);
    
    % Build an estimate of the new cases
    NewCasesEstimateLastTrainValue = NewCasesSmoothed(numTimeStepsTrain);
    %     NewCasesEstimateLastTrainValue = Amp1(numTimeStepsTrain);
    
    if(LINEAR)
        CumLambdaHatLinear = cumsum(LambdaHatLinear(numTimeStepsTrain + 1 : end))';
        NewCasesEstimateLinear = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesEstimateLastTrainValue*exp(CumLambdaHatLinear)];
    end
    
    if(ARX)
        CumLambdaHatARX = cumsum(LambdaHatARX(numTimeStepsTrain + 1 : end))';
        NewCasesEstimateARX = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesEstimateLastTrainValue*exp(CumLambdaHatARX)];
    end
        
    if(SVM)
        CumLambdaHatSVM = cumsum(LambdaHatSVM(numTimeStepsTrain + 1 : end))';
        NewCasesEstimateSVM = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesEstimateLastTrainValue*exp(CumLambdaHatSVM)];
        
        CumLambdaHatSVMGAU = cumsum(LambdaHatSVMGAU(numTimeStepsTrain + 1 : end))';
        NewCasesEstimateSVMGAU = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesEstimateLastTrainValue*exp(CumLambdaHatSVMGAU)];
        %     CumLambdaHatSVMOPT = cumsum(LambdaHatSVMOPT(numTimeStepsTrain + 1 : end))';
        %     NewCasesEstimateSVMOPT = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesEstimateLastTrainValue*exp(CumLambdaHatSVMOPT)];
    end
    
    if(LSTM)
        if(LSTM_ON_NEW_CASES)
            NewCasesEstimateLSTM = [NewCasesSmoothed(1 : numTimeStepsTrain) ; LambdaHatLSTM(numTimeStepsTrain + 1 : end)'];
        else
            CumLambdaHatLSTM = cumsum(LambdaHatLSTM(numTimeStepsTrain + 1 : end))';
            NewCasesEstimateLSTM = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesEstimateLastTrainValue*exp(CumLambdaHatLSTM)];
        end
    end
    
    %     CumLambdaHatTotal = cumsum(LambdaHatTotal(numTimeStepsTrain + 1 : end))';
    %     NewCasesEstimateTotal = [NewCasesSmoothed(1 : numTimeStepsTrain) ; NewCasesSmoothed(numTimeStepsTrain)*exp(CumLambdaHatTotal)];
    % % %     NewCasesEstimateTotal = mean([NewCasesEstimateLinear NewCasesEstimateSVM NewCasesEstimateSVM NewCasesEstimateSVMGAU NewCasesEstimateSVMOPT], 2);
    
    if(plot_figures)
        %     dn = datenum(string(geoid_dates_unsorted),'yyyymmdd');
        dn = 1 : length(NewCasesSmoothed);
        ind_pos = find(Lambda2 >= 0);
        ind_neg = find(Lambda2 < 0);
        InterventionPlansAvg= nanmean(InterventionPlans, 2);
        
        if(1)
            lgn = [];
            figure
            hold on;
            %         plot(dn, InterventionPlans, 'color', 0.8*ones(1, 3)); lgn = cat(2, lgn, {'IP'});
            plot(dn, (InterventionPlansAvg - nanmean(InterventionPlansAvg))/nanstd(InterventionPlansAvg), 'color', 0.4*ones(1, 3)); lgn = cat(2, lgn, {'Mean IP'});
%             plot(dn, Lambda1, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda1'});
            plot(dn, Lambda2, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2'});
            plot(dn, Lambda2Baseline, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2Baseline'});
%             plot(dn, Lambda2Smoothed, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda2Smoothed'});
%             plot(dn, Lambda3, 'linewidth', 2); lgn = cat(2, lgn, {'Lambda3'});
            if(LINEAR)
                plot(dn , LambdaHatLinear, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatLinear'});
            end
            
            if(ARX)
                plot(dn , LambdaHatARX, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatARX'});
            end
            
            if(SVM)
                plot(dn , LambdaHatSVM, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatSVM'});
                plot(dn , LambdaHatSVMGAU, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatSVMGAU'});
                %         plot(dn , LambdaHatSVMOPT, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatSVMOPT'});
            end
            % % %         plot(dn , (LambdaHatLinear + LambdaHatSVM)/2, 'linewidth', 3); lgn = cat(2, lgn, {'Avg. LambdaHats'});
            if(LSTM)
                plot(dn , LambdaHatLSTM, 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatLSTM'});
            end
            % % %         plot(dn , LambdaHatTotal, '--', 'linewidth', 3); lgn = cat(2, lgn, {'LambdaHatTotal'});
            legend(lgn);
            grid
            title(CountryAndRegionList(k), 'interpreter', 'none');
            %     datetick('x','mmm/dd', 'keepticks','keeplimits')
            axis 'tight'
        end
        
        if(0)
            lgn = [];
            figure
            hold on
            %     plot(dn, ConfirmedCases); lgn = cat(2, lgn, {'ConfirmedCases'});
            %     plot(dn, ConfirmedDeaths); lgn = cat(2, lgn, {'ConfirmedDeaths'});
            %             plot(dn, NewCases); lgn = cat(2, lgn, {'NewCases'});
            %             plot(dn, nanstd(NewCases)*(InterventionPlansAvg - nanmean(InterventionPlansAvg))/nanstd(InterventionPlansAvg)); lgn = cat(2, lgn, {'Mean IP Scaled'});
            plot(dn, max(NewCases)*InterventionPlansAvg/max(InterventionPlansAvg)); lgn = cat(2, lgn, {'Mean IP Scaled'});
            plot(dn, NewCasesSmoothed, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesSmoothed'});
            plot(dn(ind_pos), NewCasesSmoothed(ind_pos)', 'bo'); lgn = cat(2, lgn, {'NewCasesSmoothed Rising'});
            plot(dn(ind_neg), NewCasesSmoothed(ind_neg)', 'ro'); lgn = cat(2, lgn, {'NewCasesSmoothed Falling'});
            %         plot(dn, PointWiseFit1); lgn = cat(2, lgn, {'PointWiseFit1'});
            %         plot(dn, PointWiseFit3); lgn = cat(2, lgn, {'PointWiseFit3'});
            if(LINEAR)
                plot(dn, NewCasesEstimateLinear, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateLinear'});
            end
            if(ARX)
                plot(dn, NewCasesEstimateARX, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateARX'});
            end
            if(SVM)
                plot(dn, NewCasesEstimateSVM, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateSVM'});
                plot(dn, NewCasesEstimateSVMGAU, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateSVMGAU'});
                %         plot(dn, NewCasesEstimateSVMOPT, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateSVMOPT'});
            end
            if(LSTM)
                plot(dn, NewCasesEstimateLSTM, 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateLSTM'});
            end
            % % %         plot(dn, NewCasesEstimateTotal, '--', 'Linewidth', 2); lgn = cat(2, lgn, {'NewCasesEstimateTotal'});
            %             plot(dn, Amp1, 'Linewidth', 2); lgn = cat(2, lgn, {'Amp1'});
            legend(lgn);
            grid
            title(CountryAndRegionList(k), 'interpreter', 'none');
            axis 'tight'
        end
    end
end