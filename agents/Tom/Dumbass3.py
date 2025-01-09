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
        self.bid_correction_factor = 1.2
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
        self.profit_factor = 1.1
        self.company_profit_factors = {}

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

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):

        # initialise default profit factors for missing companies
        for company in self.headquarters.get_companies():
           if company.name not in self.company_profit_factors:
               self.company_profit_factors[company.name] = [1.2]
               
        # Process each company's won contracts
        for company_name, won_contracts in auction_ledger.items():
            company_fleet = [c for c in self.headquarters.get_companies() if c.name == company_name].pop().fleet

            for contract in won_contracts:
                trade = contract.trade
                actual_payment = contract.payment

                # Step 1: Predict payment using the current profit factor
                predicted_payment = self.predict_payment(company_name, trade)

                # Step 2: Calculate and store the profit factor
                best_cost = float('inf')
                for vessel in company_fleet:
                    predicted_cost = self.predict_cost(vessel, trade)
                    if predicted_cost < best_cost:
                        best_cost = predicted_cost

                if best_cost > 0:
                    profit_factor = actual_payment / best_cost
                else:
                    profit_factor = float('inf')

                self.company_profit_factors[company_name].append(profit_factor)

                # Step 3: Adjust predictions dynamically
                if predicted_payment is not None:
                    self.adjust_predictions(company_name, actual_payment, predicted_payment)

                # Debugging
                print(f"Company: {company_name}")
                print(f"  Trade: {trade.origin_port.name} -> {trade.destination_port.name}")
                print(f"  Actual Payment: {actual_payment}, Predicted Payment: {predicted_payment}")
                print(f"  Adjustment Factor: {actual_payment / predicted_payment if predicted_payment > 0 else 'N/A'}")

    
        super().receive(contracts, auction_ledger, *args, **kwargs)

    
    def adjust_predictions(self, company_name, actual_payment, predicted_payment):
        if company_name not in self.company_profit_factors:
            self.company_profit_factors[company_name] = [1.2]  # Default value if not initialized

        # Calculate adjustment factor
        adjustment_factor = actual_payment / predicted_payment if predicted_payment > 0 else float('inf')

        # Update profit factor using a weighted moving average
        previous_factors = self.company_profit_factors[company_name]
        updated_factor = (sum(previous_factors) + adjustment_factor) / (len(previous_factors) + 1)
        
        self.company_profit_factors[company_name] = [updated_factor]  # Replace the list with the single updated value

        # Debug output
        print(f"Updated profit factor for {company_name}: {updated_factor}")

    
    def predict_payment(self, company_name, trade):
        if company_name not in self.company_profit_factors or not self.company_profit_factors[company_name]:
            return None  # No data available for prediction

        # Use the average profit factor
        average_profit_factor = sum(self.company_profit_factors[company_name]) / len(self.company_profit_factors[company_name])

        # Predict cost for the trade
        best_cost = float('inf')
        for vessel in self.fleet:
            predicted_cost = self.predict_cost(vessel, trade)
            if predicted_cost < best_cost:
                best_cost = predicted_cost

        # Predicted payment is based on the average profit factor
        predicted_payment = average_profit_factor * best_cost
        return predicted_payment
    
    def predict_bid(self, company_name, trade, multiplier=0.95):
        predicted_payment = self.predict_payment(company_name, trade)
        if predicted_payment is None:
            return None  # No data available for prediction

        # Assume bid is slightly below predicted payment
        predicted_bid = predicted_payment * multiplier  # Adjust multiplier based on observed behavior
        return predicted_bid

    def predict_cost(self, vessel, trade):
    
        # calculate loading and unloading costs
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_costs = vessel.get_loading_consumption(loading_time)
        unloading_costs = vessel.get_unloading_consumption(loading_time)
        
        # calculate travel costs
        travel_distance = self.headquarters.get_network_distance(
            trade.origin_port, trade.destination_port)
        travel_time = vessel.get_travel_time(travel_distance)
        travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
        
        # calculate total cost
        total_cost = loading_costs + unloading_costs + travel_cost  
        
        return total_cost

    def propose_schedules(
            self,
            trades: List[Trade]
    ) -> ScheduleProposal:
        return self._propose_schedules(
            trades,
            self.predict_competitor_bids(trades),

            {vessel: vessel.schedule for vessel in self.fleet},
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
        # --------------------------[BASE CASE]---------------------------------
        # all trades have been selected for bids, return
        if len(trades) == 0:
            return self.make_bids(schedules, scheduled_trades, costs)

        # --------------------[PREDICT TRADE PROFITS]---------------------------

        # (trade, vessel conducting trade) -> (vessel schedule with trade, cost,  profit)
        # dict of all trades that are predicted to be profitable (we can underbid competition and still turn a profit)
        predicted_profits: Dict[Tuple[Trade, Vessel], Tuple[Schedule, float, float]] = dict()
        # dict of all trades that we predict we will not win the bid for
        predicted_loss: Dict[Tuple[Trade, Vessel], Tuple[Schedule, float, float]] = dict()
        for trade in trades:
            for vessel in self.fleet:
                cost = self.trade_cost_with_reloc(vessel, trade)
                profit = competitor_bids[trade] - cost

                new_schedule: Schedule = schedules[vessel].copy()
                new_schedule.add_transportation(trade)
                if not new_schedule.verify_schedule():
                    continue

                if profit > 0:
                    predicted_profits[(trade, vessel)] = (new_schedule, cost, profit)
                else:
                    predicted_loss[(trade, vessel)] = (new_schedule, cost, profit)

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
            predicted_profits = {
                trade_option: trade_info
                for trade_option, trade_info in predicted_loss.items()
                if trade_option[1] in vessels_with_no_bids
            }

            if len(predicted_profits) == 0:
                # there are no valid trades for remaining unallocated vessels, exit
                return self.make_bids(schedules, scheduled_trades, costs)

        # -------------------------[APPLY TRADE]--------------------------------
        # greedily apply the best trade
        max_profit = max(list(predicted_profits.items()), key=lambda x: x[1][2])

        # update accumulator args
        # ...sorry about the tuple notation :D
        schedules[max_profit[0][1]] = max_profit[1][0]
        scheduled_trades[max_profit[0][1]].append(max_profit[0][0])
        costs[max_profit[0][0]] = max_profit[1][1]
        trades.remove(max_profit[0][0])

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

    # @staticmethod
    # def generate_options(
    #         trade: Trade,
    #         schedule: Schedule
    # ) -> List[Schedule]:
    #     options: List[Schedule] = list()
    #     insertion_points: List[int] = schedule.get_insertion_points()[-8:]
    #     for idx_pick_up in range(len(insertion_points)):
    #         pick_up = insertion_points[idx_pick_up]
    #         insertion_points_after_pickup: List[int] = insertion_points[idx_pick_up:]
    #         for drop_off in insertion_points_after_pickup:
    #             schedule_copy = schedule.copy()
    #             schedule_copy.add_transportation(trade, pick_up, drop_off)
    #             if schedule_copy.verify_schedule():
    #                 options.append(schedule_copy)
    #     return options

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
