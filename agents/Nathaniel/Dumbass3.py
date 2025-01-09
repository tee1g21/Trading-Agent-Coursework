from dataclasses import dataclass
from typing import List, Tuple, Dict

from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import Trade, Contract, AuctionLedger
from mable.simulation_space.universe import Location, OnJourney
from mable.transport_operation import Bid, Vessel, ScheduleProposal
from mable.transportation_scheduling import Schedule


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
    bid_correction_factor: float
    num_logged_trades: int
    bids: List[Tuple[Trade, float]]

    def __init__(self):
        self.bid_correction_factor = 1.8
        self.num_logged_trades = 0
        self.bids = list()


# ------------------------[COMPANY]------------------------
class Dumbass3(TradingCompany):
    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.initialised = False

    def __init(self) -> None:
        self.competitors: Dict[TradingCompany, CompetitorData] = {
            company: CompetitorData()
            for company in self.headquarters.get_companies()
            if company.name is not self.name
        }
        self.profit_factor = 1.4

        self.initialised = True
        return

    def inform(
            self,
            trades: List[Trade],
            *args, **kwargs
    ) -> List[Bid]:
        # some objects needed for initialisation are only available after __init__ for some reason :D
        # so call a second init function on the first inform
        if not self.initialised:
            self.__init()

        """
        auction started, propose bids
        """

        bids: List[Bid] = super().inform(trades, *args, **kwargs)

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
            auction_ledger: AuctionLedger | None = None,
            *args, **kwargs
    ) -> None:
        """
        on auction end
        """
        # TODO: update competitor profit factors and out profit factor here

        schedules = self.schedule(contracts)
        self.apply_schedules(schedules)

        for vessel in self.fleet:
            schedule = schedules[vessel].get_simple_schedule()
            trades_str = ','.join([f"[{x[0]}, {fmt_trade(x[1])}]" for x in schedule])
            if len(trades_str) == 0:
                trades_str = "None"
            self.log(f"\t{fmt_vessel(vessel)}: {trades_str}")

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
                profit = competitor_bids[trade] - cost

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
            vessels_with_no_bids: List[Vessel] = list(
                map(lambda x: x[0],
                    filter(lambda x: len(x[1]) == 0,
                           scheduled_trades.items())
                    )
            )
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
            trade: self._predict_competitor_bids(trade)
            for trade in trades
        }

    def _predict_competitor_bids(
            self,
            trade: Trade
    ) -> float | None:
        """
        for a given trade, predict the lowest bid a competitor will make on it
        """
        all_predicted_bids: List[float] = list()

        for competitor in self.competitors.keys():
            for vessel in competitor.fleet:
                all_predicted_bids.append(self.predict_competitor_bid(vessel, competitor, trade))

        return min(all_predicted_bids)

    def predict_competitor_bid(
            self,
            vessel: VesselWithEngine,
            owner: TradingCompany,
            trade: Trade
    ) -> float:
        """
        predict how much a given competitor will bid on a given trade with a given vessel
        """
        return self.trade_cost_with_reloc(vessel, trade) * self.competitors[owner].bid_correction_factor

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
            for schedule in Dumbass3.generate_options(current_contract.trade, current_schedule):
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
                + self.p2p_cost(Dumbass3.get_end_location(vessel), trade.destination_port, vessel))

    def p2p_cost(
            self,
            start: Location,
            end: Location,
            vessel: VesselWithEngine
    ) -> float:
        """
        get the cost for a vessel to travel between two points
        """
        travel_distance: float = self.headquarters.get_network_distance(start, end)
        travel_time: float = vessel.get_travel_time(travel_distance)
        return vessel.get_laden_consumption(travel_time, vessel.speed)

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
        travel_cost = self.p2p_cost(trade.origin_port, trade.destination_port, vessel)

        return loading_costs + unloading_costs + travel_cost

    def log(
            self,
            text: str
    ):
        print(f"{self.name} # {text}")
