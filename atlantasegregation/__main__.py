import argparse
import os
import sys

from simulation import Simulation


def main(minimum_homogeneity: float,
         maximum_homogeneity: float,
         average_proportion_of_movement_per_year: float = .114 * (1 - .112),
         unhappiness_weight: float = 10.0,
         rng_seed: int | None = None,
         number_of_iterations: int = 10) -> None:
    sim = Simulation(minimum_homogeneity=minimum_homogeneity,
                     maximum_homogeneity=maximum_homogeneity,
                     average_proportion_movement_per_year=average_proportion_of_movement_per_year,
                     unhappiness_weight=unhappiness_weight,
                     rng_seed=rng_seed)
    print('Initial state:')
    print('\tSegregation:')
    print('\t', sim.segregation_by_race, sep='')

    try:
        os.mkdir('./simulation_imgs')
    except FileExistsError:
        pass
    sim.save_image('./simulation_imgs/img0.png')

    for year in range(1, number_of_iterations + 1):
        sim.step()
        print(f'After {year} years:')
        print(f'The overall segregation is {sim.overall_segregation}:')
        print(sim.segregation_by_race)
        sim.save_image(f'./simulation_imgs/img{year}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='atlantasegregation',
                                     description='Run a simulation of segregation in Atlanta, GA.')
    parser.add_argument('minimum_homogeneity', type=float,
                        help='The minimum proportion of neighbors that need to be the same race for someone to be happy.')
    parser.add_argument('maximum_homogeneity', type=float,
                        help='The maximum proportion of neighbors that need to be the same race for someone to be happy.')
    parser.add_argument('-a', '--average-prop-mvt', dest='average_proportion_of_movement_per_year', type=float,
                        required=False, default=.114 * (1 - .112),
                        help='The average proportion of the population that moves each year.')
    parser.add_argument('-w', '--unhappiness-weight', dest='unhappiness_weight', type=float,
                        required=False, default=10.0,
                        help='How many times as likely an unhappy household will be selected by the random selector '
                             '(default = 10.0).')
    parser.add_argument('-r', '--rng-seed', dest='rng_seed', type=int,
                        required=False, default=None,
                        help='A seed for the random number generator.')
    parser.add_argument('-n', '--num-years', dest='number_of_iterations', type=int,
                        required=False, default=10,
                        help='The number of years (or iterations) for which the simulation will run (default = 10).')
    args = vars(parser.parse_args(sys.argv[1:]))
    main(**args)
