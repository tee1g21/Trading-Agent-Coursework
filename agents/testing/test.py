# performance testing

import datetime
import inspect
import itertools
import json
import os
import random
import time
import typing
from multiprocessing.pool import ThreadPool

from loguru import logger
from mable.cargo_bidding import TradingCompany
from mable.competition.generation import AuctionSimulationEngine
from mable.examples import environment, fleets
from mable.extensions.fuel_emissions import FuelSpecsBuilder
from mable.observers import EventObserver

import agents.testing.utils as tutils


class PerformanceTest:

    _resources = "./resources"
    _output_directory = "./out"

    def setup(
        self,
        trade_occurrence_frequency: int,
        trades_per_occurrence: int,
        num_auctions: int,
    ) -> None:
        if not os.path.exists(self._resources):
            raise ValueError(
                f"Cannot find Resources Directory relative to {os.path.abspath(os.path.curdir)}"
            )
        self._specifications_builder = environment.get_specification_builder(
            environment_files_path=self._resources,
            trade_occurrence_frequency=trade_occurrence_frequency,
            trades_per_occurrence=trades_per_occurrence,
            num_auctions=num_auctions,
            fixed_trades=None,
            use_only_precomputed_routes=True,
        )

    def add_company_random_fleet(
        self,
        company: type[TradingCompany],
        fleet_nums: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        my_fleet = fleets.mixed_fleet(
            num_suezmax=fleet_nums[0], num_aframax=fleet_nums[1], num_vlcc=fleet_nums[2]
        )
        self.add_company_with_fleet(company, my_fleet)

    def add_company_with_fleet(
        self, company: type[TradingCompany], fleet: list
    ) -> None:
        if not inspect.isclass(company):
            raise TypeError("company must be a class")
        if not issubclass(company, TradingCompany):
            raise TypeError("company must be TradingCompany")
        if self._specifications_builder is None:
            raise AssertionError("Must run Performancetest.setup first")

        logger.debug(f"Adding Company {{{company.__name__}, {fleet}}}")
        self._specifications_builder.add_company(
            company.Data(company, fleet, company.__name__)
        )

    def __init__(self):
        self.sim: AuctionSimulationEngine | None = None
        self._specifications_builder: FuelSpecsBuilder | None = None

    def set_output_directory(self, dir: str) -> None:
        self._output_directory = dir

    def get_output_directory(self) -> str:
        return self._output_directory

    def test(self) -> None:
        if not os.path.exists(self._output_directory):
            os.makedirs(self._output_directory)

        logger.debug("Generating Sim")
        self.sim = typing.cast(
            AuctionSimulationEngine,
            environment.generate_simulation(
                self._specifications_builder,
                show_detailed_auction_outcome=False,
                output_directory=self._output_directory,
                global_agent_timeout=60,
            ),
        )

        if self.sim:
            logger.debug("Running Sim")
            self.sim.run()
        else:
            logger.error("No sim to run!")

    def get_sim_results(self):
        if self.sim is None:
            return
        observers = self.sim.get_event_observers()
        for observer in observers:
            observer = typing.cast(EventObserver, observer)
            logger.debug(observer)


class TestingEnvironment:

    def __init__(self):
        self.all_test_results = []
        self._test_companies: list[type[TradingCompany]] = []
        self._test_fleet_combos: list[tuple[int, int, int]] = []
        self._trade_occurrence_frequency: int = 30
        self._trades_per_occurrence: int = 5
        self._num_auctions: int = 5

    def setup_companies(
        self, companies: list[type[TradingCompany]] | None, combination_size: int = 0
    ):
        """Setup the companies for the performance test

        Args:
            companies (list[type[TradingCompany]] | None): list of all company types to test.
            combination_size (int, optional): number of combinations to test. Defaults to 2.

        Raises:
            ValueError: if number of companies is smaller than the list of combination size
        """
        if companies:
            self._test_companies = companies
        if combination_size == 0:
            combination_size = len(self._test_companies)
        if len(self._test_companies) < combination_size:
            raise ValueError(
                "Combination size must be smaller than number of companies"
            )
        self._test_company_combination_size = combination_size

    def setup_random_fleets(self, max_suezmax=1, max_aframax=1, max_vlcc=1):
        """Generate all combinations of fleets up to numbers provided.

        Args:
            max_suezmax (int, optional): Defaults to 1.
            max_aframax (int, optional): Defaults to 1.
            max_vlcc (int, optional): Defaults to 1.
        """
        self._test_fleet_combos = []
        for suez_num in range(max_suezmax + 1):
            for afra_num in range(max_aframax + 1):
                for vlcc_num in range(max_vlcc + 1):
                    if (suez_num == 0) and (afra_num == 0) and (vlcc_num == 0):
                        continue
                    self._test_fleet_combos.append((suez_num, afra_num, vlcc_num))

    def reset_fleets(self):
        """Reset the Fleet numbers for manual modification."""
        self._test_fleet_combos = []

    def add_fleet_combo(self, suezmax: int, framax: int, vlcc: int):
        """Add a fleet combination manually."""
        self._test_fleet_combos.append((suezmax, framax, vlcc))

    def setup_environment(
        self,
        trade_occurrence_frequency: int,
        trades_per_occurrence: int,
        num_auctions: int,
    ):
        """Change Environment parameters.

        Args:
            trade_occurrence_frequency (int): _description_
            trades_per_occurrence (int): _description_
            num_auctions (int): _description_
        """
        self._trade_occurrence_frequency: int = trade_occurrence_frequency
        self._trades_per_occurrence: int = trades_per_occurrence
        self._num_auctions: int = num_auctions

    def run_tests(self, sample_size: int | None = 10, threads: int | None = 0):
        """Run all the tests given the simulation parameters

        Args:
            sample_size (int, optional): number of total sampels to use. None for all. Defaults to 10.
            threads (int | None, optional): number of threads if running multithreaded, otherwise None (Use 0 for num_proc). Defaults to 0.
        """
        logger.info("Running Test suite")
        logger.info("Params:")
        logger.info(f"Companies: {self._test_companies}")
        logger.info(f"Fleet numbers: {self._test_fleet_combos}")
        company_combos: list[tuple[type[TradingCompany], ...]] = list(
            itertools.combinations(
                self._test_companies, self._test_company_combination_size
            )
        )
        # logger.debug(company_combos)

        all_metrics = []

        for combo in company_combos:
            # Each set of tests has a combination of companies
            # Now we do all the fleets
            test_cases = []

            fleet_assignments = itertools.product(
                self._test_fleet_combos, repeat=len(combo)
            )

            for assignment in fleet_assignments:
                case = [(company, fleet) for company, fleet in zip(combo, assignment)]
                test_cases.append(case)

            # logger.debug(test_cases)

            if sample_size:
                testable_cases = random.sample(test_cases, sample_size)
            else:
                testable_cases = test_cases

            # run all the test cases
            all_metrics: list[tutils.MableMetrics] = []
            if threads:
                if threads == 0:
                    threads = None
                with ThreadPool(processes=threads) as pool:
                    all_metrics = pool.map(self.run_test, testable_cases)
            else:
                for case in testable_cases:
                    all_metrics.append(self.run_test(case))

            with open(
                f"out/test-{datetime.datetime.utcnow().timestamp()}.csv", "w"
            ) as f:
                f.write('"Company, fleet_size", ')
                for company in company_combos:
                    f.write(
                        f'"{company[0].__name__}, fuel_costs, auction_revenue, auction_losses, profits"\n'
                    )

                for metric in all_metrics:
                    string = metric.get_csv_string()
                    logger.info(string)
                    f.write(string)
                    f.write("\n")

    def run_test(
        self,
        companies_with_fleets: list[tuple[type[TradingCompany], tuple[int, int, int]]],
    ) -> tutils.MableMetrics:
        logger.info("Creating Performance Test")
        performance_test = PerformanceTest()
        logger.info("Setup Performance Test")
        performance_test.setup(
            self._trade_occurrence_frequency,
            self._trades_per_occurrence,
            self._num_auctions,
        )

        outdir = f"./out/test-{id(companies_with_fleets)}"
        performance_test.set_output_directory(outdir)

        logger.info("Adding Companies")
        for company in companies_with_fleets:
            performance_test.add_company_random_fleet(company[0], company[1])

        logger.info("Running Test")
        performance_test.test()
        logger.info("End Performance Test")

        metrics = self.get_test_results(outdir)
        if len(metrics) > 1:
            logger.warning("More than one output file for metrics: %s", metrics)
        if len(metrics) < 1:
            logger.error("Error collecting metrics from test")
        metrics[0].set_company_environments(companies_with_fleets)
        return metrics[0]

    @classmethod
    def get_test_results(cls, dir: str) -> list[tutils.MableMetrics]:
        metrics_list = []
        for file in os.listdir(dir):
            with open(os.path.join(dir, file), "r") as f:
                metrics_list.append(tutils.MableMetrics(json.load(f)))
        return metrics_list

    def print_results(self):
        pass
