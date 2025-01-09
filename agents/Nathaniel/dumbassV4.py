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
    """
    contains
     1) An adaptive profit factor (updates after auctions).
     2) A simpler greedy approach for large auctions to avoid timeouts.
    """

    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.initialised = False
        # Tracks how many bids we placed in the *previous* round
        self.num_bids_last = 0
        # Base profit factor (cost multiplier) that adjusts over time:
        self.profit_factor = 1.4

    def __init(self) -> None:
        """
        A second init, called once 'headquarters' and other sim data are ready.
        We store competitor data and mark ourselves as initialized.
        """
        self.competitors: Dict[TradingCompany, CompetitorData] = {
            company: CompetitorData()
            for company in self.headquarters.get_companies()
            if company.name != self.name
        }
        # You can keep or adjust this starting profit factor:
        self.profit_factor = 1.4

        self.initialised = True

    def inform(
        self,
        trades: List[Trade],
        *args, 
        **kwargs
    ) -> List[Bid]:
        """
        Called by MABLE when a new auction (set of trades) starts.
        We do a late initialization here if needed, then let the parent inform.
        """
        if not self.initialised:
            self.__init()

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
        *args, 
        **kwargs
    ) -> None:
        """
        Called by MABLE at the end of an auction. We:
         1) Update schedules for contracts we won.
         2) Adjust profit_factor based on how many we won vs how many we bid.
        """
        # First, apply schedules for all newly won contracts
        schedules = self.schedule(contracts)
        self.apply_schedules(schedules)

        for vessel in self.fleet:
            schedule = schedules[vessel].get_simple_schedule()
            trades_str = ','.join([f"[{x[0]}, {fmt_trade(x[1])}]" for x in schedule])
            if len(trades_str) == 0:
                trades_str = "None"
            self.log(f"\t{fmt_vessel(vessel)}: {trades_str}")

        # --- Adaptive profit factor update ---
        #   If we actually placed bids last round (num_bids_last > 0),
        #   see how many we won. Then adjust.
        if self.num_bids_last != 0:
            #  e.g., if we placed 6 bids and won 2, ratio is 2/6 = 0.33
            #  Compare to threshold of 0.5; adapt factor up or down accordingly
            wl_th = 0.5
            wl_score = (len(contracts) / self.num_bids_last) - wl_th
            self.profit_factor += self.profit_factor * self.get_heat() * wl_score
            # Keep it above 1 to avoid suicidal bidding:
            if self.profit_factor < 1:
                self.profit_factor = 1

        self.log(f"Updated profit factor: {self.profit_factor}")

    def get_heat(self) -> float:
        """
        Example function to scale changes in profit factor over time:
        The later in the simulation (headquarters.current_time), the smaller the effect.
        """
        max_time = 4000
        min_heat = 0.1
        current_t = self.headquarters.current_time
        heat_ratio = (max_time - min(current_t, max_time)) / max_time
        return max(heat_ratio, min_heat)

    def propose_schedules(
        self,
        trades: List[Trade]
    ) -> ScheduleProposal:
        """
        MABLE calls this to ask: "How do you plan to schedule these trades,
        and how much will you bid for each one?"
        """
        # We'll track how many trades we're actually bidding on:
        # (i.e., the length of 'costs' we produce in make_bids).
        self.num_bids_last = 0

        # Prepare a copy of each vessel's schedule and an empty list for each vessel's assigned trades.
        schedules = {vessel: vessel.schedule.copy() for vessel in self.fleet}
        scheduled_trades = {vessel: list() for vessel in self.fleet}
        costs = dict()

        # If the auction is large (say > 8 trades), do a simpler approach to avoid recursion blow-up
        if len(trades) > 8:
            proposal = self._schedule_larger_auctions(trades, schedules, scheduled_trades, costs)
        else:
            # Otherwise, do the existing recursive approach
            proposal = self._propose_schedules(
                trades,
                self.predict_competitor_bids(trades),
                schedules,
                scheduled_trades,
                costs
            )

        # `proposal` is a ScheduleProposal. We can record how many bids we placed:
        self.num_bids_last = len(proposal.costs)
        return proposal

    def _schedule_larger_auctions(
        self,
        trades: List[Trade],
        schedules: Dict[Vessel, Schedule],
        scheduled_trades: Dict[Vessel, List[Trade]],
        costs: Dict[Trade, float]
    ) -> ScheduleProposal:
        """
        A simpler, non-recursive method for bigger auctions:
         1) Calculate profit for each (trade, vessel).
         2) Sort them by profit descending.
         3) Greedily assign if feasible.
        """
        # Gather all potential combos
        combos = []
        competitor_bids = self.predict_competitor_bids(trades)

        for trade in trades:
            for vessel in self.fleet:
                cost = self.trade_cost_with_reloc(vessel, trade)
                profit = competitor_bids[trade] - cost
                # If profitable and we can schedule it, store the combo
                temp_schedule = schedules[vessel].copy()
                temp_schedule.add_transportation(trade)
                if temp_schedule.verify_schedule() and profit > 0:
                    combos.append((trade, vessel, cost, profit))

        # Sort combos by profit descending
        combos.sort(key=lambda x: x[3], reverse=True)

        assigned_trades = set()
        for (trade, vessel, cost, profit) in combos:
            if trade not in assigned_trades:
                # Try to schedule
                new_sch = schedules[vessel].copy()
                new_sch.add_transportation(trade)
                if new_sch.verify_schedule():
                    schedules[vessel] = new_sch
                    scheduled_trades[vessel].append(trade)
                    costs[trade] = cost
                    assigned_trades.add(trade)

        return self.make_bids(schedules, scheduled_trades, costs)

    def _propose_schedules(
        self,
        trades: List[Trade],
        competitor_bids: Dict[Trade, float],
        schedules: Dict[Vessel, Schedule],
        scheduled_trades: Dict[Vessel, List[Trade]],
        costs: Dict[Trade, float]
    ) -> ScheduleProposal:
        """
        Original recursive approach for smaller auctions, with minimal changes.
        """
        @dataclass
        class Data:
            trade: Trade
            vessel: Vessel
            schedule: Schedule
            cost: float
            profit: float

        # Base case: no trades left
        if len(trades) == 0:
            return self.make_bids(schedules, scheduled_trades, costs)

        # Collect feasible profitable trades
        predicted_profits: List[Data] = []
        predicted_loss: List[Data] = []

        for trade in trades:
            for vessel in self.fleet:
                cost = self.trade_cost_with_reloc(vessel, trade)
                profit = competitor_bids[trade] - cost

                new_schedule = schedules[vessel].copy()
                new_schedule.add_transportation(trade)
                if not new_schedule.verify_schedule():
                    continue

                if profit > 0:
                    predicted_profits.append(Data(trade, vessel, new_schedule, cost, profit))
                else:
                    predicted_loss.append(Data(trade, vessel, new_schedule, cost, profit))

        # If no profitable trades remain
        if len(predicted_profits) == 0:
            # Assign leftover trades only if some vessels have no bids yet
            vessels_with_no_bids = [
                v for v, assigned in scheduled_trades.items() if len(assigned) == 0
            ]
            if len(vessels_with_no_bids) == 0:
                return self.make_bids(schedules, scheduled_trades, costs)

            # Possibly assign negative-profit trades to empty vessels
            # (This is your original logic.)
            predicted_profits = [
                d for d in predicted_loss if d.vessel in vessels_with_no_bids
            ]
            if len(predicted_profits) == 0:
                return self.make_bids(schedules, scheduled_trades, costs)

        # Greedily pick the best profit
        max_profit = max(predicted_profits, key=lambda x: x.profit)

        # Update accumulators
        schedules[max_profit.vessel] = max_profit.schedule
        scheduled_trades[max_profit.vessel].append(max_profit.trade)
        costs[max_profit.trade] = max_profit.cost
        trades.remove(max_profit.trade)

        # Recurse
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
    ) -> ScheduleProposal:
        """
        Final step: turn assigned trades & costs into a ScheduleProposal.
        We multiply each cost by 'self.profit_factor' to get the actual bid.
        """
        return ScheduleProposal(
            schedules,
            sum(scheduled_trades.values(), []),
            {
                trade: (cost * self.profit_factor)
                for trade, cost in costs.items()
            }
        )

    def predict_competitor_bids(
        self,
        trades: List[Trade]
    ) -> Dict[Trade, float]:
        """
        For each trade, guess the lowest competitor bid.
        """
        return {trade: self._predict_competitor_bids(trade) for trade in trades}

    def _predict_competitor_bids(
        self,
        trade: Trade
    ) -> float:
        all_predicted_bids: List[float] = []
        for competitor, cdata in self.competitors.items():
            for vessel in competitor.fleet:
                cost_guess = self.trade_cost_with_reloc(vessel, trade)
                # Multiply by competitor's correction factor
                all_predicted_bids.append(cost_guess * cdata.bid_correction_factor)

        if len(all_predicted_bids) == 0:
            return 999999.0
        return min(all_predicted_bids)

    def schedule(
        self,
        contracts: List[Contract]
    ) -> Dict[Vessel, Schedule]:
        """
        Min-cost scheduler for contracts we actually won.
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
        """
        Recursively allocate newly-won contracts by shortest completion time.
        """
        if len(contracts) == 0:
            return schedules

        current_contract = contracts[0]
        rest_contracts = contracts[1:]

        # find the assignment with the shortest completion time
        shortest = None
        for vessel, current_schedule in schedules.items():
            for schedule_option in Dumbass3.generate_options(current_contract.trade, current_schedule):
                completion_time = schedule_option.completion_time()
                if shortest is None or completion_time < shortest[2]:
                    shortest = (vessel, schedule_option, completion_time)

        if shortest is None:
            self.log(f"could not allocate contract: {fmt_trade(current_contract.trade)}")
        else:
            schedules[shortest[0]] = shortest[1]

        return self._schedule(schedules, rest_contracts)

    @staticmethod
    def generate_options(trade: Trade, schedule: Schedule) -> List[Schedule]:
        """
        For a given trade, try placing it in all feasible insertion points
        (the last 8 insertion points) to see which produce valid schedules.
        """
        options: List[Schedule] = []
        insertion_points = schedule.get_insertion_points()[-8:]
        for idx_pick_up in range(len(insertion_points)):
            pick_up = insertion_points[idx_pick_up]
            insertion_points_after_pickup = insertion_points[idx_pick_up:]
            for drop_off in insertion_points_after_pickup:
                schedule_copy = schedule.copy()
                schedule_copy.add_transportation(trade, pick_up, drop_off)
                if schedule_copy.verify_schedule():
                    options.append(schedule_copy)
        return options

    @staticmethod
    def get_end_location(vessel: VesselWithEngine):
        """
        Return the location of a vessel after its final scheduled action.
        """
        schedule: List[Tuple[str, Trade]] = vessel.schedule.get_simple_schedule()
        if len(schedule) == 0:
            location: Location | OnJourney = vessel.location
            if isinstance(location, OnJourney):
                return location.destination
            return location
        return schedule[-1][1].destination_port

    def trade_cost_with_reloc(self, vessel: VesselWithEngine, trade: Trade):
        """
        Cost = direct cost of the trade + relocation from vessel's end location 
        to the trade's destination (our original approach).
        """
        return (self.trade_cost(vessel, trade)
                + self.p2p_cost(Dumbass3.get_end_location(vessel), trade.destination_port, vessel))

    def p2p_cost(self, start: Location, end: Location, vessel: VesselWithEngine) -> float:
        """
        Fuel cost to travel from 'start' to 'end'.
        """
        travel_distance = self.headquarters.get_network_distance(start, end)
        travel_time = vessel.get_travel_time(travel_distance)
        return vessel.get_laden_consumption(travel_time, vessel.speed)

    def trade_cost(self, vessel: VesselWithEngine, trade: Trade) -> float:
        """
        Direct cost: loading + unloading + traveling from origin->destination.
        """
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_costs = vessel.get_loading_consumption(loading_time)
        unloading_costs = vessel.get_unloading_consumption(loading_time)
        travel_cost = self.p2p_cost(trade.origin_port, trade.destination_port, vessel)
        return loading_costs + unloading_costs + travel_cost

    def log(self, text: str):
        print(f"{self.name} # {text}")
