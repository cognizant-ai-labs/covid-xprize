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

RowHeaders = all_data.textdata(1, :); % Column titles
AllCountryNames = all_data.textdata(2:end, 1); % All country names
AllRegionNames = all_data.textdata(2:end, 3); % All region names

delim = '';
% % % delim = cell(length(AllRegionNames), 1);
% % % st = ~strcmp('', AllRegionNames);
% % % delim(st) = {'_'};
GeoID = strcat(all_data.textdata(2:end, 2), delim, all_data.textdata(2:end, 4));
[CountryAndRegionList, IA, IC] = unique(GeoID, 'stable');
NumGeoLocations = length(CountryAndRegionList);

for k = 204 : 204%NumGeoLocations
    all_geoid_entry_indexes = find(string(GeoID) == CountryAndRegionList(k));
    all_geoid_data = all_data.data(all_geoid_entry_indexes + 1 , :);
    all_geoid_textdata = all_data.textdata(all_geoid_entry_indexes + 1 , :);
    
    dates_unsorted = all_geoid_data(:, 1);
    [dates, date_indexes] = sort(dates_unsorted, 'ascend');
    ConfirmedCases = all_geoid_data(date_indexes, 31);
    ConfirmedDeaths = all_geoid_data(date_indexes, 32);
    NewCases = [0; diff(ConfirmedCases)];
    
    
    % % %     % To check vs the Johns Hopkins University dataset (DONE!)
    % % %     AllCasesFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv';
    % % %     AllDeathsFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv';
    % % %     AllRecoveredFname = './../../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv';
    % % %     RegionList = {'US'};
    % % %     min_cases = 100; % min number of cases
    % % %     % load COBID-19 data (https://github.com/CSSEGISandData/COVID-19.git)
    % % %     [TotalCases, Infected, Recovered, Deceased, FirstCaseDateIndex, MinCaseDateIndex, NumDays] = ReadCOVID19Data(AllCasesFname, AllDeathsFname, AllRecoveredFname, RegionList, min_cases);
    
    lgn = {};
    figure
    hold on
    plot(ConfirmedCases); lgn = cat(2, lgn, {'ConfirmedCases'});
    plot(ConfirmedDeaths); lgn = cat(2, lgn, {'ConfirmedDeaths'});
    plot(NewCases); lgn = cat(2, lgn, {'NewCases'});
    % % %     plot([nan(1, 21) TotalCases]); lgn = cat(2, lgn, {'ConfirmedCases JHU'});
    legend(lgn);
    grid
    title(CountryAndRegionList(k));
    
    
    
    
    
    
end