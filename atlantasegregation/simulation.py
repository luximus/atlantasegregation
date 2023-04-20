import os

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bars

from data import get_initial_state, RACES
from render import save_image


class Simulation:
    """A simulation of segregation over time in Atlanta, GA."""
    def __init__(self,
                 minimum_homogeneity: float,
                 maximum_homogeneity: float,
                 average_proportion_movement_per_year: float,
                 unhappiness_weight: float = 10.0,
                 rng_seed=None):
        """
        Initialize a Simulation with the state of Atlanta as recorded by the 2010 Decennial Census. This might take
        quite a bit of time if the initial state GeoJSON file has not been created.
        :param minimum_homogeneity: The minimum proportion of neighbors that need to be the same race for someone to be
        happy (required).
        :param maximum_homogeneity: The maximum proportion of neighbors that need to be the same race for someone to be
        happy (required).
        :param average_proportion_movement_per_year: The average proportion of the population that moves each year
        (required).
        :param unhappiness_weight: How much more likely an unhappy household will be selected by the random selector
        (defaults to 10.0).
        :param rng_seed: A seed for the random number generator. If None (default), then fresh, unpredictable entropy
        will be pulled from the operating system.
        """
        self.__minimum_homogeneity = minimum_homogeneity
        self.__maximum_homogeneity = maximum_homogeneity
        self.__unhappiness_weight = unhappiness_weight
        self.__max_household_sample_index = 0
        self.__max_vacant_household_sample_index = 0

        self.__state: gpd.GeoDataFrame = None  # type: ignore
        self.reset()
        self.average_movement_per_year = \
            int(average_proportion_movement_per_year * self.__state['Total'].sum())  # Getting actual average number of people that move each year

        self.rng = np.random.default_rng(rng_seed)

    def __calculate_starting_points(self):
        """
        Set up all the starting point logic that makes simple random selection possible.

        In order to make each happy household and each unhappy household have an equal chance of being selected, we have
        to number each race in each block with a "starting number." If any number from the starting number (inclusive)
        to the starting number in the next block (or race if at the end) (exclusive) is selected, one household of the
        selected race in the selected block will be moved.

        To give an example, here is a small sample dataset with the correct starting points:

            id   White   Black or African American   White starting point  Black or African American starting point
             0       5                           5                      0                                        34
             1       9                           1                      5                                        39
             2       1                           9                     14                                        49
             3       0                           0                     24                                        58
             4      10                          10                     24                                        58

        The maximum value chosen by the random number generator would be 68. This value is stored in
        `__max_household_sample_index`.

        The same happens with vacancies. The maximum value chosen by the random number generator for vacancies is stored
        in `__max_vacant_household_sample_index`.
        """
        homogeneity = self.homogeneity
        total_weighted: pd.DataFrame = pd.DataFrame(0, index=np.arange(len(self.__state)), columns=RACES)  # Will store the total weights of each block by race

        # Calculate the starting points for races in blocks.
        # This part is separate from the for loop because the logic changes a little for others
        race = RACES[0]
        total_weighted[race] = self.__state[race] * \
                               homogeneity[race] \
                                   .apply(lambda x: 1 if self.__minimum_homogeneity <= x <= self.__maximum_homogeneity else self.__unhappiness_weight)  # Apply the unhappiness weight if the people in a given household are unhappy
        total_weighted.loc[total_weighted[race].isna(), race] = 0  # This means there is no one of that race in the block
        self.__state.iloc[1:, self.__state.columns.get_loc(f'{race} starting point')] = total_weighted[race].cumsum().iloc[:-1]
        for index, race in enumerate(RACES[1:], start=1):
            race_cumulative_sum_index = self.__state.columns.get_loc(f'{race} starting point')
            total_weighted[race] = self.__state[race] * \
                                   homogeneity[race] \
                                       .apply(lambda x: 1 if self.__minimum_homogeneity <= x <= self.__maximum_homogeneity else self.__unhappiness_weight)
            total_weighted.loc[total_weighted[race].isna(), race] = 0

            # We need to add to the previous sum for this to work
            previous_sum = total_weighted.iloc[:, :index].sum().sum()
            self.__state.iloc[0, race_cumulative_sum_index] = previous_sum
            self.__state.iloc[1:, race_cumulative_sum_index] = previous_sum + total_weighted[race].cumsum().iloc[:-1]

        self.__max_household_sample_index = int(total_weighted.sum().sum())  # Setting the maximum index when the random number is chosen

        # Calculate the starting points for vacant housing.
        self.__state['Vacant starting point'] = [0] + self.__state['Vacant'].cumsum()[:-1].tolist()
        self.__max_vacant_household_sample_index = int(self.__state['Vacant'].sum())

    @property
    def number_of_households_by_race(self) -> pd.Series:
        """The total number of households, by race. Access the values by indexing with the name of the race."""
        return self.__state[RACES].sum()

    @property
    def homogeneity(self) -> pd.DataFrame:
        """
        The proportion of neighbors that are the same race for each block by race. If there are no members of that race
        in a block, then the value is NaN.
        """
        homogeneity: dict[str, pd.Series] = {}
        for race in RACES:
            homogeneity[race] = (self.__state[race] - 1 + self.__state[f'{race} neighbors']) / (  # - 1 because we exclude the person from which the homogeneity is being considered
                    self.__state['Occupied'] + self.__state['Occupied neighbors'])
            homogeneity[race].loc[self.__state[race] == 0] = np.nan  # Blocks without anyone of this race in them cannot be given a homogeneity score
        return pd.DataFrame(homogeneity)

    @property
    def segregation_by_race(self) -> pd.Series:
        """
        A metric for the amount of segregation by race. The value is calculated by taking the mean homogeneity across
        all blocks for a race, subtracting 0.25, and normalizing to a number from 0 to 1. If the value is less than 0,
        then it will be capped at 0.
        """
        segregation: pd.Series = (1 / 0.75) * (self.homogeneity.mean(axis=0) - 0.25)
        segregation.loc[segregation < 0] = 0
        return segregation

    @property
    def overall_segregation(self) -> float:
        """
        The overall segregation for the entire population. This is found by taking the average segregation by race,
        weighted by the number of households of that race.
        """
        return np.average(self.segregation_by_race, weights=self.number_of_households_by_race)

    @property
    def state(self) -> gpd.GeoDataFrame:
        """The current state of the simulation."""
        return self.__state.drop(labels=[f'{race} starting point' for race in RACES] + ['Vacant starting point'],  # The user shouldn't know about the starting point logic
                                 axis=1)

    @property
    def minimum_homogeneity(self) -> float:
        """The minimum proportion of neighbors that need to be the same race for someone to be happy."""
        return self.__minimum_homogeneity

    @minimum_homogeneity.setter
    def minimum_homogeneity(self, new_value: float) -> None:
        self.__minimum_homogeneity = new_value
        self.__calculate_starting_points()  # We need to recalculate the starting points because the happiness threshold has changed

    @property
    def maximum_homogeneity(self) -> float:
        """The maximum proportion of neighbors that need to be the same race for someone to be happy."""
        return self.__maximum_homogeneity

    @maximum_homogeneity.setter
    def maximum_homogeneity(self, new_value: float) -> None:
        self.__maximum_homogeneity = new_value
        self.__calculate_starting_points()  # We need to recalculate the starting points because the happiness threshold has changed

    @property
    def unhappiness_weight(self) -> float:
        """How much more likely an unhappy household will be selected by the random selector."""
        return self.__unhappiness_weight

    @unhappiness_weight.setter
    def unhappiness_weight(self, new_value: float) -> None:
        self.__unhappiness_weight = new_value
        self.__calculate_starting_points()  # We need to recalculate the starting points because the multiplier for unhappy households has changed

    def reset(self):
        """
        Reset the simulation to the initial state. This might take quite a bit of time if the initial state GeoJSON file
        has not been created.
        """
        self.__state = get_initial_state()
        self.__state = self.__state.assign(
            **{f'{race} starting point': [0 for _ in range(len(self.__state))] for race in RACES})
        self.__calculate_starting_points()

    def save_image(self, filepath: str | os.PathLike) -> None:
        """
        Save an image representing the state of the simulation to a file.
        :param filepath: The path to which the image will be saved.
        """
        save_image(self.__state, filepath)

    def move(self, index: int, race: str, target_index: int) -> None:
        """
        Move a single household of the given race from the initial index ot the target index.
        :param index: The index of the block in which the household that is moving is currently located.
        :param race: The race of the household that is moving.
        :param target_index: The index of the block in which the household that is moving will be located.
        """

        # Setting up a few indices so that we don't have to call a function with the same arguments twice
        race_index = self.__state.columns.get_loc(race)
        neighbors_index = self.__state.columns.get_loc('neighbors')
        occupied_index = self.__state.columns.get_loc('Occupied')
        race_neighbors_index = self.__state.columns.get_loc(f'{race} neighbors')
        occupied_neighbors_index = self.__state.columns.get_loc('Occupied neighbors')
        vacant_index = self.__state.columns.get_loc('Vacant')

        # Eliminating impossible scenarios
        if self.__state.iloc[index, race_index] <= 0:
            raise ValueError('there are no households of this race in the given block')

        if self.__state.iloc[target_index, vacant_index] <= 0:
            raise ValueError('the block at the target index has no vacancies')

        # Make the old household vacant and decrement the count of that race
        self.__state.iloc[index, race_index] -= 1
        self.__state.iloc[index, occupied_index] -= 1
        self.__state.iloc[index, vacant_index] += 1
        self.__state.iloc[self.__state.iloc[index, neighbors_index], race_neighbors_index] -= 1
        self.__state.iloc[self.__state.iloc[index, neighbors_index], occupied_neighbors_index] -= 1

        # Make the new household occupied and increment the count of that race
        self.__state.iloc[target_index, race_index] += 1
        self.__state.iloc[target_index, occupied_index] += 1
        self.__state.iloc[target_index, vacant_index] -= 1
        self.__state.iloc[self.__state.iloc[target_index, neighbors_index], race_neighbors_index] += 1
        self.__state.iloc[self.__state.iloc[target_index, neighbors_index], occupied_neighbors_index] += 1

        # Recalculate the starting points
        self.__calculate_starting_points()

    def step(self) -> int:
        """
        Step forward exactly one year. The number of households that move is determined by a Poisson distribution with a
        mean determined by the average proportion of people that move in this population. Unhappy households (that is,
        households where the homogeneity is outside the acceptable degree) are more likely to be chosen.

        :return: The number of people that moved. This value can be safely ignored.
        """

        # Determine the number of people that will be moving. We use a Poisson distribution because it describes events
        # that happen with a constant frequency within a given time frame (in this case, 1 year).
        number_of_people_moved = self.rng.poisson(self.average_movement_per_year)

        for _ in tqdm(range(self.rng.poisson(self.average_movement_per_year)), desc='Evaluating next time step'):
            # Choose a race in a block in which one family will be moving
            sample_index = self.rng.integers(0, self.__max_household_sample_index)
            index: int | None = None
            race: str | None = None
            for current_race in RACES:
                possible_blocks: pd.DataFrame = self.__state[self.__state[f'{current_race} starting point'] <= sample_index]
                if len(possible_blocks) == 0:
                    break
                index = len(possible_blocks) - 1
                race = current_race
            if index is None or race is None:
                raise RuntimeError('something impossible has happened')

            # Choose the family's destination
            sample_index = self.rng.integers(0, self.__max_vacant_household_sample_index)
            possible_target_blocks: pd.DataFrame = self.__state[self.__state[f'Vacant starting point'] <= sample_index]
            if len(possible_target_blocks) == 0:
                raise RuntimeError('something impossible has happened')
            target_index = len(possible_target_blocks) - 1

            self.move(index, race, target_index)

        return number_of_people_moved


if __name__ == '__main__':
    sim = Simulation(minimum_homogeneity=0.20, maximum_homogeneity=0.90, average_proportion_movement_per_year=.114 * (1 - .112), unhappiness_weight=100.0)
    print('Initial state:')
    print('\tSegregation:')
    print('\t', sim.segregation_by_race, sep='')
    # sim.save_image('simulation_imgs/img0')

    for year in range(1, 11):  # Do 10 years of simulation
        sim.step()
        print(f'After {year} years:')
        print('\tSegregation:')
        print('\t', sim.segregation_by_race, sep='')
        # sim.save_image(f'simulation_imgs/img{year}')
