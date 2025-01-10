from dataclasses import dataclass
from typing import List, Tuple, Dict

from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import Trade, Contract
from mable.simulation_space.universe import Location, OnJourney
from mable.transport_operation import Bid, Vessel, ScheduleProposal
from mable.transportation_scheduling import Schedule

"""
TODO:
known issues:

trade_cost_with_reloc seems to be underestimating costs somehow, I have no idea how
we don't bid on enough contracts, need to bias it to making more bids somehow
company pf prediction seems to jump around a fair but though stays in the right ballpark
"""

# -------------------[PRINT FORMATTERS]--------------------
def fmt_location(location: Location | OnJourney) -> str:
    if isinstance(location, OnJourney):
        return f"DST: ({location.destination.x}, {location.destination.y})"

    return f"{location.name}: ({location.x}, {location.y})"


def fmt_trade(trade: Trade) -> str:
    return f"[{fmt_location(trade.origin_port)}] => [{fmt_location(trade.destination_port)}]"


def fmt_vessel(vessel: Vessel) -> str:
    return f"[{vessel.name} @ {fmt_location(vessel.location)}]"


# ------------------------[STRUCTS]------------------------
@dataclass
class CompetitorData:
    cls: TradingCompany
    num_logged_trades: int
    profit_factor_history: List[float]

    def __init__(
            self,
            cls: TradingCompany
    ):
        self.cls = cls
        self.profit_factor_history = [1.5]


# ------------------------[COMPANY]------------------------
class CompanyWhatever(TradingCompany):
    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.competitors: Dict[str, CompetitorData] | None = None
        self.profit_factor: float = 1.4
        self.num_bids_prev: int = 0
        self.num_options_prev: int = 0
        self.future_trades: List[Trade] = []

    def init_competitors(self) -> None:
        self.competitors = {
            company.name: CompetitorData(company)
            for company in self.headquarters.get_companies()
            if company.name is not self.name
        }
        return

    def pre_inform(
            self,
            trades: List[Trade],
            time
    ) -> None:
        self.future_trades = trades
        return

    def inform(
            self,
            trades: List[Trade],
            *args, **kwargs
    ) -> List[Bid]:
        # some objects needed for initialisation are only available after __init__ for some reason :D
        # so call a second init function on the first inform
        if self.competitors is None:
            self.init_competitors()

        """
        auction started, propose bids
        """
        self.num_options_prev = len(trades)
        bids: List[Bid] = super().inform(trades, *args, **kwargs)
        self.num_bids_prev = len(bids)

        if len(bids) == 0:
            self.log("no bids")
        else:
            self.log("----- bids -----")
            for bid in bids:
                self.log(f"{fmt_trade(bid.trade)}: {bid.amount}")
            self.log("----------------")

        return bids

    def receive(
            self,
            contracts: List[Contract],
            auction_ledger: Dict[str, List[Contract]] | None = None,
            *args, **kwargs
    ) -> None:
        """
        on auction end
        """

        schedules = self.schedule(contracts)
        self.apply_schedules(schedules)

        for vessel in self.fleet:
            schedule = schedules[vessel].get_simple_schedule()
            trades_str = ','.join([f"[{x[0]}, {fmt_trade(x[1])}]" for x in schedule])
            if len(trades_str) == 0:
                trades_str = "None"
            self.log(f"\t{fmt_vessel(vessel)}: {trades_str}")

        if auction_ledger is not None:
            self.update_competitor_info(auction_ledger)

        self.update_profit_factor(len(contracts))
        self.log(f"profit factor: {self.profit_factor}")

    def propose_schedules(
            self,
            trades: List[Trade]
    ) -> ScheduleProposal:
        return self._propose_schedules(
            trades,
            self.predict_competitor_bids(trades),

            {vessel: vessel.schedule.copy() for vessel in self.fleet},
            {vessel: list() for vessel in self.fleet},
            dict()
        )

    def _propose_schedules(
            self,
            trades: List[Trade],
            competitor_bids: Dict[Trade, float],

            # accumulator args
            schedules: Dict[Vessel, Schedule],
            scheduled_trades: Dict[Vessel, List[Trade]],
            costs: Dict[Trade, float]
    ) -> ScheduleProposal:
        @dataclass
        class Data:
            trade: Trade
            vessel: Vessel
            schedule: Schedule
            cost: float
            profit: float

        # --------------------------[BASE CASE]---------------------------------
        # all trades have been selected for bids, return
        if len(trades) == 0:
            return self.make_bids(schedules, scheduled_trades, costs)

        # --------------------[PREDICT TRADE PROFITS]---------------------------

        # dict of all trades that are predicted to be profitable (we can underbid competition and still turn a profit)
        predicted_profits: List[Data] = list()
        # dict of all trades that we predict we will not win the bid for
        predicted_loss: List[Data] = list()
        for trade in trades:
            for vessel in self.fleet:
                cost = self.trade_cost_with_reloc(vessel, trade)
                cost += self.cost_to_closest_future_trade(vessel, trade) * 0.8

                if trade in competitor_bids:
                    profit = competitor_bids[trade] - cost
                else:
                    # this is kinda hacky but should never happen
                    profit = -cost * self.profit_factor

                new_schedule: Schedule = schedules[vessel].copy()
                new_schedule.add_transportation(trade)
                if not new_schedule.verify_schedule():
                    continue

                if profit > 0:
                    predicted_profits.append(Data(trade, vessel, new_schedule, cost, profit))
                else:
                    predicted_loss.append(Data(trade, vessel, new_schedule, cost, profit))

        # --------------------------[BASE CASE]---------------------------------
        if len(predicted_profits) == 0:
            # no more profitable trades to add

            # get all vessels without bids
            vessels_with_no_bids: List[Vessel] = [
                vessel
                for vessel, allocations
                in scheduled_trades.items()
                if len(allocations) == 0
            ]
            if len(vessels_with_no_bids) == 0:
                # all vessels have bids, exit
                return self.make_bids(schedules, scheduled_trades, costs)
            # some vessels have no bids, no point doing nothing with them
            predicted_profits = [
                data
                for data in predicted_loss
                if data.vessel in vessels_with_no_bids
            ]

            if len(predicted_profits) == 0:
                # there are no valid trades for remaining unallocated vessels, exit
                return self.make_bids(schedules, scheduled_trades, costs)

        # -------------------------[APPLY TRADE]--------------------------------
        # greedily apply the best trade
        max_profit = max(predicted_profits, key=lambda x: x.profit)

        # update accumulator args
        schedules[max_profit.vessel] = max_profit.schedule
        scheduled_trades[max_profit.vessel].append(max_profit.trade)
        costs[max_profit.trade] = max_profit.cost
        trades.remove(max_profit.trade)

        # recursively consider trades
        # n.b: this is a bit inefficient as only profit predictions for the vessel we just added a trade to will change
        # but this is quite a performance cost
        return self._propose_schedules(
            trades,
            competitor_bids,
            schedules,
            scheduled_trades,
            costs
        )

    ## Code to work out cost of nearest future trade
    def cost_to_closest_future_trade(
            self,
            vessel: VesselWithEngine,
            trade: Trade
    ) -> float:
        # if there are future trades find the closest one
        if self.future_trades:
            min_distance = min(
                min(
                    self.headquarters.get_network_distance(self.get_end_location(vessel), trade.origin_port)
                    for vessel
                    in self.fleet
                )
                for trade
                in self.future_trades
            )

            # caculate the cost to the closest future trade
            travel_time = vessel.get_travel_time(min_distance)
            cost = vessel.get_ballast_consumption(travel_time, vessel.speed)

            return cost
        else:
            print("No future trades")
            return 0

    def make_bids(
            self,
            schedules: Dict[Vessel, Schedule],
            scheduled_trades: Dict[Vessel, List[Trade]],
            costs: Dict[Trade, float]
    ):
        return ScheduleProposal(
            schedules,
            sum(scheduled_trades.values(), []),
            {
                trade: cost * self.profit_factor
                for trade, cost in costs.items()
            })

    def predict_competitor_bids(
            self,
            trades: List[Trade]
    ) -> Dict[Trade, float]:
        """
        for each trade, predict the lowest bid a competitor will make on it
        """
        return {
            trade: min([self.predict_bid(company_name, trade) for company_name in self.competitors.keys()], default=0)
            for trade in trades
        }

    def schedule(
            self,
            contracts: List[Contract]
    ) -> Dict[Vessel, Schedule]:
        """
        min cost scheduler
        this is used to schedule bids that we win in a way that minimises cost
        """
        return self._schedule(
            {vessel: vessel.schedule.copy() for vessel in self.fleet},
            contracts
        )

    def _schedule(
            self,
            schedules: Dict[Vessel, Schedule],
            contracts: List[Contract],
    ) -> Dict[Vessel, Schedule]:
        if len(contracts) == 0:
            return schedules

        current_contract = contracts[0]
        rest_contract = contracts[1:]

        # find the assignment to any vessel with the shortest completion time
        shortest: Tuple[Vessel, Schedule, float] | None = None
        for vessel, current_schedule in schedules.items():
            for schedule in CompanyWhatever.generate_options(current_contract.trade, current_schedule):
                completion_time = schedule.completion_time()
                if shortest is None or completion_time < shortest[2]:
                    shortest = (vessel, schedule, completion_time)

        if shortest is None:
            self.log(f"could not allocate contract: {fmt_trade(current_contract.trade)}")
        else:
            schedules[shortest[0]] = shortest[1]

        return self._schedule(schedules, rest_contract)

    @staticmethod
    def generate_options(
            trade: Trade,
            schedule: Schedule
    ) -> List[Schedule]:
        options: List[Schedule] = list()
        insertion_points: List[int] = schedule.get_insertion_points()[-8:]
        for idx_pick_up in range(len(insertion_points)):
            pick_up = insertion_points[idx_pick_up]
            insertion_points_after_pickup: List[int] = insertion_points[idx_pick_up:]
            for drop_off in insertion_points_after_pickup:
                schedule_copy = schedule.copy()
                schedule_copy.add_transportation(trade, pick_up, drop_off)
                if schedule_copy.verify_schedule():
                    options.append(schedule_copy)
        return options

# ----------------[UTILITY FUNCTIONS]-------------------
    @staticmethod
    def get_end_location(
            vessel: VesselWithEngine
    ):
        """
        get the location of a vessel after all scheduled actions
        """

        schedule: List[Tuple[str, Trade]] = vessel.schedule.get_simple_schedule()
        if len(schedule) == 0:
            location: Location | OnJourney = vessel.location
            if isinstance(location, OnJourney):
                return location.destination
            return location
        return schedule[-1][1].destination_port

    def trade_cost_with_reloc(
            self,
            vessel: VesselWithEngine,
            trade: Trade
    ):
        """
        calculate the cost of a trade, including the cost to get to the start location
        """
        return (self.trade_cost(vessel, trade)
                + self.p2p_cost(CompanyWhatever.get_end_location(vessel), trade.destination_port, vessel, False))

    def p2p_cost(
            self,
            start: Location,
            end: Location,
            vessel: VesselWithEngine,
            laden: bool
    ) -> float:
        """
        get the cost for a vessel to travel between two points
        """
        travel_distance: float = self.headquarters.get_network_distance(start, end)
        travel_time: float = vessel.get_travel_time(travel_distance)
        if laden:
            return vessel.get_laden_consumption(travel_time, vessel.speed)
        else:
            return vessel.get_ballast_consumption(travel_time, vessel.speed)

    def trade_cost(
            self,
            vessel: VesselWithEngine,
            trade: Trade
    ) -> float:
        """
        get the 'direct' costs of a trade
        loading/unloading/moving
        """
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_costs = vessel.get_loading_consumption(loading_time)
        unloading_costs = vessel.get_unloading_consumption(loading_time)
        travel_cost = self.p2p_cost(trade.origin_port, trade.destination_port, vessel, True)

        return loading_costs + unloading_costs + travel_cost

    # --------------------------[COMPETITOR INFO]---------------------------------
    def update_competitor_info(
            self,
            auction_ledger: Dict[str, List[Contract]]
    ):
        # Process each company's won contracts
        for company_name, won_contracts in auction_ledger.items():
            if company_name not in self.competitors:
                continue

            if len(won_contracts) == 0:
                continue

            company_fleet = self.competitors[company_name].cls.fleet

            profit_factor_sum = 0
            for contract in won_contracts:
                trade = contract.trade
                actual_payment = contract.payment

                # get the profit factor for this trade and add it to the sum
                best_cost = min(self.trade_cost_with_reloc(vessel, trade) for vessel in company_fleet)
                profit_factor_sum += actual_payment / best_cost

            # find the average profit factor for the last auction round
            # and add it to the history
            profit_factor_avg = CompanyWhatever.clamp(1.4, profit_factor_sum / len(won_contracts), 3)
            self.competitors[company_name].profit_factor_history.append(profit_factor_avg)

            self.log(f"new profit factor for: {company_name} -> {profit_factor_avg}")

    def weighted_profit_factor(
            self,
            company_name: str
    ) -> float:
        """
        gets the weighted average of all recorded profit factors for a company
        each auction has 1.5x the weight of the last
        """
        pfh = self.competitors[company_name].profit_factor_history
        weights = [1.1**i for i in range(len(pfh))]
        weighted_sum = sum(weight * value for weight, value in zip(weights, pfh))
        total_weight = sum(weights)
        return weighted_sum / total_weight

    def predict_payment(
            self,
            company_name: str,
            trade: Trade
    ) -> float:
        # Use the average profit factor
        average_profit_factor = self.weighted_profit_factor(company_name)

        # Predict cost for the trade
        best_cost = min(self.trade_cost_with_reloc(vessel, trade) for vessel in self.competitors[company_name].cls.fleet)

        # Predicted payment is based on the average profit factor
        predicted_payment = average_profit_factor * best_cost
        return predicted_payment

    def predict_bid(
            self,
            company_name,
            trade,
            multiplier=0.95
    ) -> float:
        predicted_payment = self.predict_payment(company_name, trade)

        # Assume bid is slightly below predicted payment
        predicted_bid = predicted_payment * multiplier  # Adjust multiplier based on observed behavior
        return predicted_bid

    # --------------[OUR PROFIT FACTOR]---------------
    def update_profit_factor(
            self,
            num_wins: int
    ) -> None:
        if self.num_bids_prev == 0 or self.num_options_prev == 0:
            return

        rate = num_wins / self.num_bids_prev
        tgt_rate = min(len(self.headquarters.get_companies()) / self.num_options_prev,
                       0.75)
        delta = tgt_rate - rate
        self.profit_factor -= delta

        self.profit_factor = self.clamp(1.4, self.profit_factor, 3)

        return

    @staticmethod
    def clamp(
            minvalue: float,
            value: float,
            maxvalue: float
    ) -> float:
        return max(minvalue, min(value, maxvalue))

    def log(
            self,
            text: str
    ) -> None:
        print(f"{self.name} # {text}")
