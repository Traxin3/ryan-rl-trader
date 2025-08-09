import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from ..simulator.order import OrderType


class MarketImpactModel:
    """
    Realistic market impact and liquidity modeling for trading simulation.
    Implements slippage, bid-ask spreads, and liquidity-dependent execution.
    Now includes a synthetic LOB (top-N levels), participation-based nonlinear
    impact, and simple handling of limit order offsets and TIF semantics.
    """

    def __init__(
        self,
        base_spread_bps: float = 0.5,  # Base spread in basis points
        slippage_factor: float = 0.1,  # Slippage multiplier
        liquidity_impact: float = 0.05,  # Impact per volume unit
        spread_volatility_factor: float = 2.0,
        book_levels: int = 5,
        base_depth_units: float = 1000.0,
        depth_decay: float = 0.6,
        cancel_rate: float = 0.05,  # per step fraction canceled
        replace_rate: float = 0.04,  # per step fraction replenished
        impact_coefficient_bps: float = 20.0,
        impact_exponent: float = 0.75,
        adverse_selection_bps: float = 1.5,
        level_spacing_bps: float = 1.0,
        tick_size: float = 0.0001,
    ):
        self.base_spread_bps = base_spread_bps
        self.slippage_factor = slippage_factor
        self.liquidity_impact = liquidity_impact
        self.spread_volatility_factor = spread_volatility_factor

        self.book_levels = int(book_levels)
        self.base_depth_units = float(base_depth_units)
        self.depth_decay = float(depth_decay)
        self.cancel_rate = float(cancel_rate)
        self.replace_rate = float(replace_rate)
        self.impact_coefficient_bps = float(impact_coefficient_bps)
        self.impact_exponent = float(impact_exponent)
        self.adverse_selection_bps = float(adverse_selection_bps)
        self.level_spacing_bps = float(level_spacing_bps)
        self.tick_size = float(tick_size)

        self.session_spreads = {
            "asian": 1.2,  # Higher spreads during Asian session
            "london": 0.7,  # Tighter spreads during London
            "ny": 0.8,  # Moderate spreads during NY
            "overlap": 0.5,  # Tightest during overlaps
            "off_hours": 2.0,  # Widest during off hours
        }

        self.volatility_buckets = {
            "low": 0.8,  # < 0.5% hourly volatility
            "normal": 1.0,  # 0.5-1.5% hourly volatility
            "high": 1.5,  # 1.5-3% hourly volatility
            "extreme": 3.0,  # > 3% hourly volatility
        }

        self._lob: Dict[str, Dict[str, np.ndarray]] = {}

    def get_trading_session(self, dt: datetime) -> str:
        """Determine current trading session"""
        hour = dt.hour

        if 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 17:
            return "overlap"  # London-NY overlap
        elif 17 <= hour < 22:
            return "ny"
        elif 22 <= hour <= 23 or 0 <= hour < 8:
            return "asian"
        else:
            return "off_hours"

    def calculate_volatility_regime(
        self, price_data: np.ndarray, window: int = 24
    ) -> str:
        """Calculate current volatility regime label from recent prices"""
        if len(price_data) < 2:
            return "normal"

        returns = np.diff(np.log(price_data[-min(window, len(price_data)) :]))
        if len(returns) == 0:
            return "normal"

        hourly_vol = np.std(returns) * np.sqrt(24)  # Annualized hourly vol proxy

        if hourly_vol < 0.005:
            return "low"
        elif hourly_vol < 0.015:
            return "normal"
        elif hourly_vol < 0.03:
            return "high"
        else:
            return "extreme"

    def _estimate_vol_value(self, price_data: np.ndarray, window: int = 24) -> float:
        if len(price_data) <= 1:
            return 0.01
        returns = np.diff(np.log(price_data[-min(window, len(price_data)) :]))
        if len(returns) == 0:
            return 0.01
        return float(np.std(returns))

    def _quantize_price(self, price: float) -> float:
        """Quantize price to the nearest tick size."""
        ts = max(self.tick_size, 1e-12)
        return float(round(price / ts) * ts)

    def get_dynamic_spread(
        self,
        symbol: str,
        current_time: datetime,
        price_data: np.ndarray,
        current_price: float,
    ) -> Tuple[float, float]:
        """
        Calculate dynamic bid-ask spread based on market conditions
        Returns (bid_price, ask_price)
        """
        base_spread = self.base_spread_bps / 10000 * current_price

        session = self.get_trading_session(current_time)
        session_multiplier = self.session_spreads.get(session, 1.0)

        vol_regime = self.calculate_volatility_regime(price_data)
        vol_multiplier = self.volatility_buckets.get(vol_regime, 1.0)

        hour = current_time.hour
        minute = current_time.minute

        time_adjustment = 1.0
        session_transition_hours = [8, 13, 17, 22]  # Major session times
        for session_hour in session_transition_hours:
            time_diff = abs((hour * 60 + minute) - (session_hour * 60))
            if time_diff <= 30:  # 30 minutes window
                time_adjustment = max(time_adjustment, 1.5)

        total_spread = (
            base_spread * session_multiplier * vol_multiplier * time_adjustment
        )
        half_spread = total_spread / 2

        bid = current_price - half_spread
        ask = current_price + half_spread

        return bid, ask

    def _init_or_update_lob(
        self,
        symbol: str,
        current_time: datetime,
        price_data: np.ndarray,
        current_price: float,
    ) -> None:
        session = self.get_trading_session(current_time)
        liquidity_multiplier = {
            "london": 1.3,
            "ny": 1.2,
            "overlap": 1.5,
            "asian": 0.9,
            "off_hours": 0.6,
        }.get(session, 1.0)
        vol_value = self._estimate_vol_value(price_data)
        vol_scale = max(vol_value, 1e-4)
        bid, ask = self.get_dynamic_spread(
            symbol, current_time, price_data, current_price
        )
        mid = (bid + ask) / 2.0

        base_depth = self.base_depth_units * liquidity_multiplier / vol_scale
        level_sizes = np.array(
            [base_depth * (self.depth_decay**i) for i in range(self.book_levels)],
            dtype=float,
        )

        level_prices_ask = np.array(
            [
                ask * (1.0 + (i) * self.level_spacing_bps / 10000.0)
                for i in range(self.book_levels)
            ],
            dtype=float,
        )
        level_prices_bid = np.array(
            [
                bid * (1.0 - (i) * self.level_spacing_bps / 10000.0)
                for i in range(self.book_levels)
            ],
            dtype=float,
        )

        book = self._lob.get(symbol)
        if book is None:
            self._lob[symbol] = {
                "ask_prices": level_prices_ask.copy(),
                "ask_sizes": level_sizes.copy(),
                "bid_prices": level_prices_bid.copy(),
                "bid_sizes": level_sizes.copy(),
                "mid": mid,
            }
        else:
            book["ask_prices"] = 0.5 * book["ask_prices"] + 0.5 * level_prices_ask
            book["bid_prices"] = 0.5 * book["bid_prices"] + 0.5 * level_prices_bid
            book["ask_sizes"] = (1.0 - self.cancel_rate) * book["ask_sizes"]
            book["bid_sizes"] = (1.0 - self.cancel_rate) * book["bid_sizes"]
            book["ask_sizes"] += self.replace_rate * level_sizes
            book["bid_sizes"] += self.replace_rate * level_sizes
            book["ask_sizes"] = np.maximum(book["ask_sizes"], 1e-6)
            book["bid_sizes"] = np.maximum(book["bid_sizes"], 1e-6)
            book["mid"] = mid
            self._lob[symbol] = book

    def get_lob_metrics(
        self,
        symbol: str,
        current_time: datetime,
        price_data: np.ndarray,
        current_price: float,
    ) -> Dict[str, float]:
        """Return basic LOB metrics for features: total depth and imbalance."""
        self._init_or_update_lob(symbol, current_time, price_data, current_price)
        book = self._lob[symbol]
        total_bid = float(np.sum(book["bid_sizes"]))
        total_ask = float(np.sum(book["ask_sizes"]))
        depth_sum = total_bid + total_ask
        imbalance = 0.0
        if depth_sum > 0:
            imbalance = (total_bid - total_ask) / depth_sum
        return {
            "total_depth": depth_sum,
            "imbalance": imbalance,
            "mid": float(book["mid"]),
        }

    def calculate_slippage(
        self,
        order_type: OrderType,
        volume: float,
        current_price: float,
        volatility: float,
        liquidity_score: float = 1.0,
        participation_rate: Optional[float] = None,
        trend_sign: float = 0.0,
    ) -> float:
        """
        Calculate realistic slippage based on volume, volatility, and liquidity
        Returns the slippage amount (positive = adverse movement)
        """
        volume_impact = (
            self.liquidity_impact
            * np.log1p(max(volume, 0.0))
            / max(liquidity_score, 1e-6)
        )
        volatility_impact = max(volatility, 1e-6) * self.slippage_factor
        random_component = np.random.normal(0, volatility_impact * 0.5)

        if participation_rate is None:
            participation_rate = 0.0
        nonlinear_bps = self.impact_coefficient_bps * (
            max(participation_rate, 0.0) ** self.impact_exponent
        )
        nonlinear_impact = (nonlinear_bps / 10000.0) * current_price

        adverse_bps = (
            max(0.0, trend_sign * (1.0 if order_type == OrderType.Buy else -1.0))
            * self.adverse_selection_bps
        )
        adverse_impact = (adverse_bps / 10000.0) * current_price

        total_slippage = (
            volume_impact + volatility_impact + abs(random_component)
        ) * current_price
        total_slippage += nonlinear_impact + adverse_impact

        if order_type == OrderType.Buy:
            return total_slippage  # Buy at higher price
        else:
            return -total_slippage  # Sell at lower price

    def _estimate_market_notional(
        self, liquidity_score: float, vol_value: float
    ) -> float:
        return float(self.base_depth_units * liquidity_score / max(vol_value, 1e-4))

    def _trend_sign(self, price_data: np.ndarray, window: int = 8) -> float:
        if len(price_data) < 2:
            return 0.0
        r = np.diff(price_data[-min(window, len(price_data)) :])
        s = np.sign(np.sum(r))
        return float(s)

    def _walk_book(
        self, side: OrderType, volume: float, symbol: str
    ) -> Tuple[float, float, float]:
        """Walk the synthetic book to compute VWAP and fills. Returns (vwap, filled, remaining)."""
        book = self._lob[symbol]
        prices: np.ndarray = (
            book["ask_prices"] if side == OrderType.Buy else book["bid_prices"]
        )
        sizes: np.ndarray = (
            book["ask_sizes"] if side == OrderType.Buy else book["bid_sizes"]
        )
        remaining = max(volume, 0.0)
        cost = 0.0
        filled = 0.0
        for i in range(len(prices)):
            if remaining <= 1e-9:
                break
            take = min(remaining, sizes[i])
            filled += take
            cost += take * prices[i]
            sizes[i] -= take
            remaining -= take
        vwap = cost / max(filled, 1e-9)
        return float(vwap), float(filled), float(remaining)

    def get_execution_price(
        self,
        order_type: OrderType,
        volume: float,
        symbol: str,
        current_time: datetime,
        price_data: np.ndarray,
        current_price: float,
        market_order: bool = True,
        limit_offset_bps: Optional[float] = None,
        tif: str = "IOC",
    ) -> Tuple[float, Dict]:
        """
        Calculate realistic execution price including spreads, synthetic book and slippage.

        Returns:
            execution_price: Final execution price (VWAP if multiple fills)
            execution_info: Dict with breakdown of costs and microstructure fields
        """
        self._init_or_update_lob(symbol, current_time, price_data, current_price)
        book_metrics = self.get_lob_metrics(
            symbol, current_time, price_data, current_price
        )
        bid, ask = self.get_dynamic_spread(
            symbol, current_time, price_data, current_price
        )
        mid = (bid + ask) / 2.0

        vol_regime_label = self.calculate_volatility_regime(price_data)
        vol_value = self._estimate_vol_value(price_data)

        session = self.get_trading_session(current_time)
        liquidity_score = {
            "london": 1.2,
            "ny": 1.1,
            "overlap": 1.5,
            "asian": 0.8,
            "off_hours": 0.5,
        }.get(session, 1.0)

        est_market_notional = self._estimate_market_notional(liquidity_score, vol_value)
        participation_rate = float(
            min(1.0, max(0.0, volume / max(est_market_notional, 1e-6)))
        )
        trend_sign = self._trend_sign(price_data)

        filled = 0.0
        remaining = max(volume, 0.0)
        vwap_exec = current_price

        if market_order:
            vwap, filled, remaining = self._walk_book(order_type, volume, symbol)
            vwap_exec = vwap
        else:
            offset_bps = float(limit_offset_bps or 0.0)
            base_price = ask if order_type == OrderType.Buy else bid
            if order_type == OrderType.Buy:
                limit_unquant = base_price * (1.0 + offset_bps / 10000.0)
            else:
                limit_unquant = base_price * (1.0 - offset_bps / 10000.0)
            limit_price = self._quantize_price(limit_unquant)

            crosses = (
                (limit_price >= ask)
                if order_type == OrderType.Buy
                else (limit_price <= bid)
            )
            inside_spread = False
            queue_level_depth = 0.0
            inside_spread_distance_ticks = 0.0

            if crosses:
                vwap, filled, remaining = self._walk_book(order_type, volume, symbol)
                vwap_exec = vwap
            else:
                book = self._lob[symbol]
                if order_type == OrderType.Buy:
                    if limit_price >= bid:
                        inside_spread = True
                        queue_level_depth = 0.0
                        inside_spread_distance_ticks = max(
                            0.0, (limit_price - bid) / max(self.tick_size, 1e-12)
                        )
                    else:
                        bids = book["bid_prices"]
                        idx = int(
                            np.searchsorted(-bids, -limit_price, side="left")
                        )  # bids descending
                        idx = max(0, min(idx, len(bids) - 1))
                        queue_level_depth = float(book["bid_sizes"][idx])
                else:
                    if limit_price <= ask:
                        inside_spread = True
                        queue_level_depth = 0.0
                        inside_spread_distance_ticks = max(
                            0.0, (ask - limit_price) / max(self.tick_size, 1e-12)
                        )
                    else:
                        asks = book["ask_prices"]
                        idx = int(
                            np.searchsorted(asks, limit_price, side="left")
                        )  # asks ascending
                        idx = max(0, min(idx, len(asks) - 1))
                        queue_level_depth = float(book["ask_sizes"][idx])

                queue_outflow = (self.cancel_rate + self.replace_rate) * 0.5
                potential_fill = queue_outflow * volume
                tif_u = str(tif).upper()
                if tif_u == "FOK":
                    filled = volume if potential_fill >= volume else 0.0
                else:
                    filled = potential_fill
                remaining = max(0.0, volume - filled)
                vwap_exec = limit_price if filled > 0 else current_price

        fill_ratio = float(filled / max(volume, 1e-9))

        slippage = (
            self.calculate_slippage(
                order_type,
                volume=filled,
                current_price=mid,
                volatility=vol_value,
                liquidity_score=liquidity_score,
                participation_rate=participation_rate,
                trend_sign=trend_sign,
            )
            if filled > 0
            else 0.0
        )

        effective_spread_bps = float(
            2.0 * abs(vwap_exec - mid) / max(mid, 1e-9) * 10000.0
        )

        execution_price = (
            float(vwap_exec + (slippage if order_type == OrderType.Buy else -slippage))
            if filled > 0
            else float(current_price)
        )

        base_anchor = ask if order_type == OrderType.Buy else bid
        tick_delta = 0.0
        if not market_order:
            tick_delta = (self._quantize_price(limit_unquant) - base_anchor) / max(
                self.tick_size, 1e-12
            )
            if order_type == OrderType.Sell:
                tick_delta = -tick_delta

        execution_info = {
            "base_price": float(current_price),
            "bid": float(bid),
            "ask": float(ask),
            "mid": float(mid),
            "spread_bps": float(((ask - bid) / max(current_price, 1e-9)) * 10000.0),
            "slippage": float(slippage),
            "slippage_bps": float((slippage / max(current_price, 1e-9)) * 10000.0),
            "execution_price": float(execution_price),
            "vwap_price": float(vwap_exec),
            "effective_spread_bps": effective_spread_bps,
            "fill_ratio": fill_ratio,
            "remaining_volume": float(remaining),
            "participation_rate": participation_rate,
            "adverse_selection_bps": float(
                max(0.0, trend_sign * (1.0 if order_type == OrderType.Buy else -1.0))
                * self.adverse_selection_bps
            ),
            "session": session,
            "volatility_regime": vol_regime_label,
            "liquidity_score": liquidity_score,
            "total_cost_bps": float(
                ((execution_price - current_price) / max(current_price, 1e-9))
                * 10000.0
                * (1.0 if order_type == OrderType.Buy else -1.0)
            ),
            "book_total_depth": book_metrics["total_depth"],
            "book_imbalance": book_metrics["imbalance"],
            "tif": str(tif).upper(),
            "limit_offset_bps": (
                float(limit_offset_bps or 0.0) if not market_order else 0.0
            ),
            "limit_offset_ticks": float(tick_delta) if not market_order else 0.0,
            "limit_anchor": "ask" if order_type == OrderType.Buy else "bid",
            "tick_size": float(self.tick_size),
            "limit_price": float(limit_price) if not market_order else float("nan"),
            "crossed": bool(crosses) if not market_order else True,
            "inside_spread": bool(inside_spread) if not market_order else False,
            "inside_spread_distance_ticks": (
                float(inside_spread_distance_ticks) if not market_order else 0.0
            ),
            "queue_level_depth": float(queue_level_depth) if not market_order else 0.0,
            "queued_volume": (
                float(max(0.0, volume - filled)) if not market_order else 0.0
            ),
            "queue_outflow_rate": (
                float(queue_outflow)
                if (not market_order and "queue_outflow" in locals())
                else 0.0
            ),
            "cancel_rate": float(self.cancel_rate),
            "replace_rate": float(self.replace_rate),
        }

        return execution_price, execution_info

    def get_daily_liquidity_pattern(self, hour: int) -> float:
        """Get relative liquidity level by hour (24h pattern)"""
        liquidity_pattern = {
            0: 0.3,
            1: 0.2,
            2: 0.2,
            3: 0.3,
            4: 0.4,
            5: 0.5,  # Asian late/quiet
            6: 0.6,
            7: 0.7,
            8: 0.9,
            9: 1.0,
            10: 1.0,
            11: 1.0,  # London open
            12: 1.0,
            13: 1.2,
            14: 1.3,
            15: 1.4,
            16: 1.3,  # London-NY overlap
            17: 1.2,
            18: 1.0,
            19: 0.9,
            20: 0.8,
            21: 0.7,  # NY session
            22: 0.6,
            23: 0.4,  # Quiet period
        }
        return liquidity_pattern.get(hour, 0.5)
