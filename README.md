# Capstone
Capstone for GA Immersive

This repo contains the scripts I utilized to build predictive models of default rates for three-year installment loans issued by Lending Club. If interested in the results, please go to the Powerpoint deck in the main folder.

If seeking to replicate the analysis, you will need to obtain the following raw data:

1) The entire loan book from Lending Club, downloadable here: https://www.lendingclub.com/info/download-data.action

2) Five year estimates of median individual income and proportion of people employed from the American Community Survey: 
http://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t

Given the size of the data (700+MB), I have chosen not to store it here.

Scripts:

The first two scripts, for cleaning and assembling the data, are in the Cleaning Folder.
The next script, marked with 3, is the EDA script where PCA and PCA results were obtained.
The fourth script, with models built with a Sequential Backward Feature Selection approach, are in the main folder, marked with 4 at the start of the file name.

As an aside, the gross total returns on the three-year loans seems quite low compared to the stated ROI Lending Club publishes. People replicating this analysis may want to dig into an appropriate way to measure this (I elected to take the difference between total payments received by Lending Club and the total funded amount as the gross total return, net of principal paid, which seems logical).
