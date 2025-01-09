# company 12 

from typing import List, Dict
from mable.cargo_bidding import TradingCompany
from mable.shipping_market import Trade, Contract, AuctionLedger
from mable.transport_operation import Bid, Vessel, ScheduleProposal
from mable.transportation_scheduling import Schedule

class Company3(TradingCompany):
    """
    A simple shipping agent example that:
      - Computes competitor costs inline
      - Computes its own cost inline
      - Schedules trades if profitable (greedy approach)
    """

    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.initialised = False
        self.profit_factor = 1.1
        # We'll keep track of each competitor with a “bid factor”
        # but store them in a simple dictionary instead of a custom class.
        self.competitors: Dict[TradingCompany, float] = {}

    def pre_inform(self, *args, **kwargs):
        """
        Called before trades are announced for each round (if your MABLE version uses it).
        We won't do anything special here, but it's required by some templates.
        """
        pass

    def inform(self, trades: List[Trade], *args, **kwargs) -> List[Bid]:
        """
        Called when new Trades are available. We won't return real bids here;
        we'll produce them in `propose_schedules`. This method can log or update internal state.
        """
        # Initialize once we know the environment is set up
        if not self.initialised:
            # For each competitor, store a single “correction factor” for their bids
            all_companies = self.headquarters.get_companies()
            for c in all_companies:
                if c.name != self.name:
                    # Just a constant factor (e.g., 1.2) to estimate competitor’s cost
                    self.competitors[c] = 1.2
            self.initialised = True

        # Optionally let the parent class do something (usually returns an empty list)
        base_bids = super().inform(trades, *args, **kwargs)
        return base_bids

    def receive(
        self,
        contracts: List[Contract],
        auction_ledger: AuctionLedger | None = None,
        *args, 
        **kwargs
    ):
        """
        Called after an auction finishes. We could update competitor behavior here
        (e.g., adjust competitor factors if we see they won/lost), 
        but we'll just pass it to the parent.
        """
        return super().receive(contracts, auction_ledger, *args, **kwargs)

    def propose_schedules(self, trades: List[Trade]) -> ScheduleProposal:
        """
        The heart of the agent: produce a schedule (per vessel) and the bids for each trade.
        """
        # Prepare a (vessel -> copy of schedule) dictionary
        schedules = {v: v.schedule.copy() for v in self.fleet}
        # Keep track of which trades each vessel will handle
        vessel_trades = {v: [] for v in self.fleet}
        # Keep track of final bid amounts (trade -> amount)
        trade_bids = {}

        # 1) Compute an *estimated competitor bid* for each trade
        competitor_bids = {}
        for trade in trades:
            # We'll guess the minimal competitor cost among all competitor vessels
            # times that competitor's “bid_correction_factor”
            possible_bids = []
            for competitor, factor in self.competitors.items():
                for vessel in competitor.fleet:
                    # inline cost to do the trade:
                    #  (a) re-locate from vessel's last location to trade.origin_port
                    #  (b) loading/unloading
                    #  (c) travel from origin->destination
                    # We'll do a rough approximation because we cannot define a separate function.
                    distance_to_origin = self.headquarters.get_network_distance(
                        vessel.schedule.end_location if vessel.schedule else vessel.location,
                        trade.origin_port
                    )
                    travel_time_to_origin = vessel.get_travel_time(distance_to_origin)
                    cost_to_origin = vessel.get_laden_consumption(travel_time_to_origin, vessel.speed)

                    loading_t = vessel.get_loading_time(trade.cargo_type, trade.amount)
                    cost_loading = vessel.get_loading_consumption(loading_t)
                    cost_unloading = vessel.get_unloading_consumption(loading_t)

                    distance_main = self.headquarters.get_network_distance(
                        trade.origin_port, 
                        trade.destination_port
                    )
                    travel_time_main = vessel.get_travel_time(distance_main)
                    cost_main = vessel.get_laden_consumption(travel_time_main, vessel.speed)

                    total_competitor_cost = cost_to_origin + cost_loading + cost_unloading + cost_main
                    possible_bids.append(total_competitor_cost * factor)

            if len(possible_bids) == 0:
                # If no competitor data, assume a very large competitor bid
                competitor_bids[trade] = 999999.0
            else:
                competitor_bids[trade] = min(possible_bids)

        # 2) Now, greedily check if we can profit from each trade
        for trade in trades:
            best_profit = 0.0
            best_vessel = None
            best_cost = 0.0

            for vessel in self.fleet:
                # Current location is the end of that vessel's schedule if any
                current_loc = schedules[vessel].end_location

                # (a) cost to get to the trade’s origin
                dist_to_origin = self.headquarters.get_network_distance(current_loc, trade.origin_port)
                time_to_origin = vessel.get_travel_time(dist_to_origin)
                cost_to_origin = vessel.get_laden_consumption(time_to_origin, vessel.speed)

                # (b) loading/unloading
                load_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
                cost_load = vessel.get_loading_consumption(load_time)
                cost_unload = vessel.get_unloading_consumption(load_time)

                # (c) main travel cost from origin to destination
                dist_main = self.headquarters.get_network_distance(
                    trade.origin_port, 
                    trade.destination_port
                )
                time_main = vessel.get_travel_time(dist_main)
                cost_main = vessel.get_laden_consumption(time_main, vessel.speed)

                total_cost = cost_to_origin + cost_load + cost_unload + cost_main
                predicted_profit = competitor_bids[trade] - total_cost

                # Check if the vessel can feasibly add this trade to its schedule
                test_schedule = schedules[vessel].copy()
                test_schedule.add_transportation(trade)

                if test_schedule.verify_schedule() and predicted_profit > best_profit:
                    best_profit = predicted_profit
                    best_vessel = vessel
                    best_cost = total_cost

            # If we found a vessel with positive profit, assign the trade
            if best_vessel is not None and best_profit > 0:
                # Update that vessel’s schedule
                schedules[best_vessel].add_transportation(trade)
                vessel_trades[best_vessel].append(trade)

                # We’ll bid (our cost * profit_factor)
                bid_amount = best_cost * self.profit_factor
                trade_bids[trade] = bid_amount

        # 3) Combine all trades into a single list of “won” trades 
        #    (the ones we actually want to place a bid on)
        all_trades_we_bid_on = []
        for v in vessel_trades:
            all_trades_we_bid_on.extend(vessel_trades[v])

        # 4) Return a ScheduleProposal
        return ScheduleProposal(
            schedules, 
            all_trades_we_bid_on, 
            trade_bids
        )
