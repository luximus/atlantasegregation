# atlantasegregation
Investigating segregation in Atlanta, GA with an agent-based model

---

This project uses data from the 2010 Decennial Census as well as the City of Atlanta Department of City Planning.

## Inputs
- The position of each census block in the city of Atlanta, Georgia
- The total number of households, occupied households, and vacant households
- The frequency of the race of the householder from the races that the U.S. Census Bureau defines within a block out of the population of occupied households
- The average proportion of people that move within the city each year (determined to be about 0.101)
- The minimum possible homogeneity for a household to be happy
- The maximum possible homogeneity for a household to be happy
- How much more likely an unhappy household is to move compared to a happy household (default is 10 times)

## Assumptions
- No one moves in or out of the city.
- People will not own multiple housing units, and anyone who owns a housing unit at the beginning of the simulation will also own a housing unit at the end of the simulation.
- All the inhabitants of a housing unit are the same race as the householder.
- Someone cares about the race of all neighbors located no more than 0.50 miles away.
- There are no factors besides the race of someone and their neighbors in the current day in determining the chance of someone moving, nor where they might move, including the fact that they moved recently, if applicable.
- There are no historical precedents.
- There are only two categories of happiness: happy and unhappy.
- All people might move for any reason at some point.
- The proportion of people that moved within Fulton County and lived in their current household for less than a year according to the 2010 American Community Survey is a good representation of the average proportion of people that move within the entire city of Atlanta.
- The proportion of people that move within the city each year is Poisson distributed—this also implies the assumptions listed here.
- The data from the 2010 Decennial Census are accurate.
- The minimum and maximum possible homogeneity and unhappiness weight are the same for everyone.

## Outputs
- A statistic for the new degree segregation of Atlanta by race, as well as a weighted average
- A map of Atlanta that shows the changes that have happened

## Usage
```shell
$ python3 atlantasegregation [-h] [-a AVERAGE_PROPORTION_OF_MOVEMENT_PER_YEAR] [-w UNHAPPINESS_WEIGHT] [-r RNG_SEED] [-n NUMBER_OF_ITERATIONS] minimum_homogeneity maximum_homogeneity

```
You must run the command from the root directory.