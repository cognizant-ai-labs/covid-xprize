function [TotalCases, Infected, Recovered, Deceased, FirstCaseDateIndex, MinCaseDateIndex, NumDays] = ReadCOVID19Data(confirmed_datafile, death_datafile, recovered_datafile, RegionList, min_cases)
AllCases = importdata(confirmed_datafile);
AllDeaths = importdata(death_datafile);
AllRecovered = importdata(recovered_datafile);
NumDays = size(AllCases.data, 2) - 2;
NumRegions = length(RegionList);

TotalCases = zeros(NumRegions, NumDays);
Infected = zeros(NumRegions, NumDays);
Recovered = zeros(NumRegions, NumDays);
Deceased = zeros(NumRegions, NumDays);
FirstCaseDateIndex = zeros(1, NumRegions);
MinCaseDateIndex = zeros(1, NumRegions);
for k = 1 : NumRegions
    CountryRowsAllCases = find(contains(AllCases.textdata(:, 2) , RegionList(k)));
    cases = sum(AllCases.data(CountryRowsAllCases - 1, 3:end), 1);
    
    CountryRowsAllDeaths = find(contains(AllDeaths.textdata(:, 2) , RegionList(k)));
    deaths = sum(AllDeaths.data(CountryRowsAllDeaths - 1, 3:end), 1);
    
    CountryRowsAllRecovered = find(contains(AllRecovered.textdata(:, 2) , RegionList(k)));
    recovered = sum(AllRecovered.data(CountryRowsAllRecovered - 1, 3:end), 1);
    
    FirstCaseDateIndex(k) = find(cases > 0, 1);
    MinCaseDateIndex(k) = find(cases >= min_cases, 1);
    
    TotalCases(k, :) = cases;
    Infected(k, :) = cases - deaths - recovered;
    Deceased(k, :) = deaths;
    Recovered(k, :) = recovered;
end