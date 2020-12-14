close all
clear
clc

% % % url = 'https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/OxCGRT_latest.csv';
% % % options = weboptions('RequestMethod','get','ArrayFormat','csv','ContentType','text');
% % % try
% % %     allData = webread(url,options);
% % % % % %     disp('CSV formatted data:');
% % % % % %     allData
% % % catch
% % %     disp('No information found.');
% % % end

% all_data = readtable('./../../covid-policy-tracker/data/OxCGRT_latest.csv', 'ReadVariableNames', true);
all_data = importdata('./../../covid-policy-tracker/data/OxCGRT_latest.csv');

% CountryName = unique(all_data.textdata(2:end, 1)); % make a dictionary of country names
% RegionName = unique(all_data.textdata(2:end, 3)); % make a dictionary of region names

ColumnHeaders = all_data.textdata(1, :); % Column titles
AllCountryNames = all_data.textdata(2:end, 1); % All country names
AllRegionNames = all_data.textdata(2:end, 3); % All region names

delim = '';
% % % delim = cell(length(AllRegionNames), 1);
% % % st = ~strcmp('', AllRegionNames);
% % % delim(st) = {'_'};
GeoID = strcat(all_data.textdata(2:end, 2), delim, all_data.textdata(2:end, 4));
[CountryAndRegionList, IA, IC] = unique(GeoID, 'stable');
NumGeoLocations = length(CountryAndRegionList);

filter_type = 'MOVINGAVERAGE'; % 'MOVINGAVERAGE' or 'MOVINGMEDIAN' or 'TIKHONOV'; % The last two call functions from the OSET package (oset.ir)
for k = 150 : 200%NumGeoLocations
    all_geoid_entry_indexes = find(string(GeoID) == CountryAndRegionList(k));
    all_geoid_data = all_data.data(all_geoid_entry_indexes + 1 , :);
    all_geoid_textdata = all_data.textdata(all_geoid_entry_indexes + 1 , :);
    
    dates_unsorted = all_geoid_data(:, 1);
    [dates, date_indexes] = sort(dates_unsorted, 'ascend');
    ConfirmedCases = all_geoid_data(date_indexes, 31);
    ConfirmedDeaths = all_geoid_data(date_indexes, 32);
    NewCases = [0; diff(ConfirmedCases)];
    
    % Replace nans with 0 and the last one with the previous number
    NewCasesFilled = NewCases;
    NewCasesFilled(isnan(NewCasesFilled)) = 0;
    if(isnan(NewCases(end)))
        NewCasesFilled(end) = NewCasesFilled(end-1);
    end
    
    %     NewCasesFilled = log(NewCasesFilled);
    
    % Smooth the data using Tikhonov regularization
    switch filter_type
        case 'TIKHONOV'
            % DiffOrderOrFilterCoefs = [1 -2 1];
            DiffOrderOrFilterCoefs = 2;
            lambda = 25.0;
            NewCasesSmoothed = TikhonovRegularization(NewCasesFilled', DiffOrderOrFilterCoefs, lambda)';
        case 'MOVINGAVERAGE'
            wlen = 7;
            NewCasesSmoothed = filter(ones(1, wlen), wlen, NewCasesFilled); % causal
            %             NewCasesSmoothed = BaseLine1(NewCasesFilled', wlen, 'mn')'; % non-causal
        case 'MOVINGMEDIAN'
            NewCasesSmoothed = BaseLine1(NewCasesFilled', 3, 'md')';
            NewCasesSmoothed = BaseLine1(NewCasesSmoothed', 7, 'mn')';
        otherwise
            error('Unknown filter type');
    end
    
    Noise = NewCasesFilled - NewCasesSmoothed;
    
    NoiseNormalized = Noise./NewCasesSmoothed;
    NoiseNormalized(isnan(NoiseNormalized)) = 0;
    
    % LSTM Forcasting
    if(false)
        start_time = 100;
        numTimeStepsTest = 21;
        %     numTimeStepsTrain = floor(0.9*numel(NoiseNormalized));
        %     dataTrain = NoiseNormalized(start_time : numTimeStepsTrain + 1)';
        %     dataTest = NoiseNormalized(numTimeStepsTrain + 1 : end)';
        dataTrain = NoiseNormalized(start_time : end - numTimeStepsTest)';
        dataTest = NoiseNormalized(end - numTimeStepsTest + 1 : end)';
        numTimeStepsTrain = length(dataTrain);
        
        mu = mean(dataTrain);
        sig = std(dataTrain);
        dataTrainStandardized = (dataTrain - mu) / sig;
        
        XTrain = dataTrainStandardized(1:end-1);
        YTrain = dataTrainStandardized(2:end);
        
        numFeatures = 1;
        numResponses = 1;
        numHiddenUnits = 100;
        
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
        options = trainingOptions('adam', ...
            'MaxEpochs',250, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',0, ...
            'Plots','training-progress');
        
        net = trainNetwork(XTrain,YTrain,layers,options);
        
        dataTestStandardized = (dataTest - mu) / sig;
        XTest = dataTestStandardized(1:end-1);
        
        net = predictAndUpdateState(net,XTrain);
        [net,YPred] = predictAndUpdateState(net,YTrain(end));
        
        numTimeStepsTest = numel(XTest);
        for i = 2:numTimeStepsTest
            [net, YPred(:,i)] = predictAndUpdateState(net, YPred(:,i-1), 'ExecutionEnvironment', 'cpu');
        end
        
        YPred = sig*YPred + mu;
        
        YTest = dataTest(2:end);
        rmse = sqrt(mean((YPred-YTest).^2))
        
        figure
        hold on
        idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
        plot((1: length([dataTrain dataTest])) + start_time - 1, [dataTrain dataTest],'g')
        plot((1: length(dataTrain(1:end-1))) + start_time - 1, dataTrain(1:end-1))
        plot(idx + start_time - 1,[NoiseNormalized(numTimeStepsTrain) YPred],'.-')
        hold off
        xlabel("Days")
        ylabel("Cases")
        title("Forecast")
        legend(["Observed" "Train" "Forecast"])
        grid
        
        figure
        subplot(2,1,1)
        plot(YTest)
        hold on
        plot(YPred,'.-')
        hold off
        legend(["Observed" "Forecast"])
        ylabel("Cases")
        title("Forecast")
        
        subplot(2,1,2)
        stem(YPred - YTest)
        xlabel("Month")
        ylabel("Error")
        title("RMSE = " + rmse)
    end
    
    % % %     % To check vs the Johns Hopkins University dataset (DONE!)
    % % %     AllCasesFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv';
    % % %     AllDeathsFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv';
    % % %     AllRecoveredFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv';
    % % %     RegionList = {'US'};
    % % %     min_cases = 100; % min number of cases
    % % %     % load COBID-19 data (https://github.com/CSSEGISandData/COVID-19.git)
    % % %     [TotalCases, Infected, Recovered, Deceased, FirstCaseDateIndex, MinCaseDateIndex, NumDays] = ReadCOVID19Data(AllCasesFname, AllDeathsFname, AllRecoveredFname, RegionList, min_cases);
    
    dn = datenum(string(dates),'yyyymmdd');
    lgn = {};
    figure
    hold on
    %     plot(dn, ConfirmedCases); lgn = cat(2, lgn, {'ConfirmedCases'});
    %     plot(dn, ConfirmedDeaths); lgn = cat(2, lgn, {'ConfirmedDeaths'});
    plot(dn, NewCases); lgn = cat(2, lgn, {'NewCases'});
    plot(dn, NewCasesSmoothed); lgn = cat(2, lgn, {'NewCasesSmoothed'});
    plot(dn, NoiseNormalized); lgn = cat(2, lgn, {'NoiseNormalized'});
    % % %     plot([nan(1, 21) TotalCases]); lgn = cat(2, lgn, {'ConfirmedCases JHU'});
    legend(lgn);
    grid
    title(CountryAndRegionList(k), 'interpreter', 'none');
    datetick('x','mmm/dd', 'keepticks','keeplimits')
    axis 'tight'
    
    
    
    
end