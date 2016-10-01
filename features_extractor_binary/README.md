# Features Extractor Binary

## Commands
* FEATURE: To generate features
* CONFIG: To generate default config file

##  FEATURE params
* MFCC
* MS: Magnitude Spectrum
* ENERGY
* ZC: Zero Crossing 

## CONFIG params
* Currently not available

## VM arguments
* -Xmx1024M: In order to increase heap space for java to run the features extraction

## How to run the jar file:
* java [VM arguments] -jar main.jar [Commands] [Command params]

## Example:
* java -Xmx1024M -jar main.jar FEATURE MFCC MS ENERGY ZC
* java -jar main.jar CONFIG