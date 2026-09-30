import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from bittrade_data import utc, write_json
from bittrade_paper import (advance, book_levels, buy_fill, create_state, DEFAULTS,
                           epoch, fill_cost, notify, send_telegram, validate_state)

NOW = 1800000000  # Fixed fixture time, never used in production or sent to Telegram.


def fixture(now=NOW, probability=.8, bid=100., ask=100.1, symbol='btcjpy'):
    result = {'source': {'provider': 'bittrade', 'market_type': 'spot', 'symbol': symbol, 'timeframe': '5m'},
              'latest': {'candle': utc(int(now//300)*300-300), 'probability_positive_net_return': probability},
              'selected_confidence': .6, 'eligible_for_further_paper_evaluation': False,
              'gate_reasons': ['no_validation_candidate_passed'], 'horizon_candles': 6, 'sha256': 'fixture'}
    report = {'created_at': utc(now), 'results': {symbol+'_5m': result}}
    tickers = {'ts': now*1000, 'data': [{'symbol': symbol, 'bid': bid, 'ask': ask, 'vol': 20000000.}]}
    markets = [{'symbol': symbol, 'state': 'online', 'api-trading': 'enabled', 'amount-precision': 4,
                'sell-market-min-order-amt': .0001, 'sell-market-max-order-amt': 1000., 'min-order-value': 2.}]
    book = {'ts': now*1000, 'tick': {'asks': [[ask, 2000.]], 'bids': [[bid, 2000.]]}}
    return report, tickers, markets, lambda _: copy.deepcopy(book)


def step(state=None, now=NOW, **kwargs):
    if state is None:
        state = create_state(now)
    report, tickers, markets, book = fixture(now, **kwargs)
    return advance(state, report, tickers, markets, book, now)


class PortfolioTests(unittest.TestCase):
    def test_fractional_clock_survives_iso_rounding_and_reload(self):
        for fraction in (.1234567, .0000007, .9999997):
            with self.subTest(fraction=fraction):
                now = NOW + fraction
                state = step(now=now)
                validate_state(json.loads(json.dumps(state)), now)

    def test_live_book_entry_and_cash_reconciliation(self):
        state = step()
        p = state['positions']['btcjpy_5m']
        self.assertAlmostEqual(p['entry_price'], 100.1*1.0005)
        self.assertLessEqual(p['cost_jpy'], 10000)
        self.assertAlmostEqual(p['entry_fee_jpy'], p['quantity']*p['entry_price']*.001)
        self.assertAlmostEqual(state['cash_jpy'], 100000-p['cost_jpy'])
        self.assertLess(state['metrics']['equity_jpy'], 100000)
        validate_state(state, NOW)

    def test_exit_fills_at_observed_price_not_stop_price(self):
        state = step()
        p = state['positions']['btcjpy_5m']
        state = step(state, NOW+300, probability=.1, bid=95., ask=95.1)
        trade = state['closed_trades'][0]
        expected = p['quantity']*95.*.9995*.999-p['cost_jpy']
        self.assertEqual(trade['exit_reason'], 'stop_loss_observed')
        self.assertAlmostEqual(trade['net_pnl_jpy'], expected)
        self.assertAlmostEqual(state['cash_jpy'], 100000+expected)
        self.assertEqual(state['positions'], {})
        self.assertEqual([e['kind'] for e in state['outbox']], ['activation', 'entry', 'exit'])

    def test_take_profit_model_exit_and_time_exit(self):
        for now, probability, bid, ask, reason in [
            (NOW+300, .8, 103.,103.1,'take_profit_observed'),
            (NOW+300, .1, 100.,100.1,'model_exit'),
            (NOW+1800, .8,100.,100.1,'holding_horizon_elapsed')]:
            with self.subTest(reason=reason):
                state = step(step(), now, probability=probability, bid=bid, ask=ask)
                self.assertEqual(state['closed_trades'][0]['exit_reason'], reason)

    def test_same_candle_and_reruns_never_duplicate_entry(self):
        state = step()
        again = step(state)
        self.assertEqual(state['positions'], again['positions'])
        self.assertEqual(state['outbox'], again['outbox'])
        closed = step(state, NOW+1, bid=95., ask=95.1)
        rerun = step(closed, NOW+2)
        self.assertEqual(len(rerun['closed_trades']), 1)
        self.assertFalse(rerun['positions'])
        self.assertEqual(len(rerun['outbox']), 3)

    def test_strict_rejects_experimental_allows_same_signal(self):
        self.assertTrue(step()['positions'])
        self.assertFalse(step(create_state(NOW, 'strict'))['positions'])

    def test_stale_research_blocks_entries_but_still_exits(self):
        report, tickers, markets, book = fixture(NOW+1800)
        report['created_at'] = utc(NOW-3600)
        result = advance(step(), report, tickers, markets, book, NOW+1800)
        self.assertEqual(len(result['closed_trades']), 1)
        self.assertFalse(result['positions'])
        fresh = advance(create_state(NOW), report, tickers, markets, book, NOW+1800)
        self.assertFalse(fresh['positions'])

    def test_exits_only_ignores_fresh_buy_report(self):
        report, tickers, markets, book = fixture(NOW+1800)
        result = advance(step(), report, tickers, markets, book, NOW+1800, exits_only=True)
        self.assertEqual(len(result['closed_trades']), 1)
        self.assertFalse(result['positions'])

    def test_liquidity_stale_quotes_and_invalid_probability(self):
        for case in ('volume', 'wide_spread', 'offline', 'disabled', 'nan_probability'):
            with self.subTest(case=case):
                r, t, m, b = fixture()
                if case == 'volume': t['data'][0]['vol'] = 0
                if case == 'wide_spread': t['data'][0]['ask'] = 110
                if case == 'offline': m[0]['state'] = 'offline'
                if case == 'disabled': m[0]['api-trading'] = 'disabled'
                if case == 'nan_probability': r['results']['btcjpy_5m']['latest']['probability_positive_net_return'] = float('nan')
                self.assertFalse(advance(create_state(NOW), r, t, m, b, NOW)['positions'])
        r,t,m,b = fixture()
        t['ts'] -= 301000
        with self.assertRaisesRegex(ValueError, 'Fresh market'):
            advance(create_state(NOW),r,t,m,b,NOW)

    def test_depth_minimums_and_quote_precision(self):
        _, _, markets, _ = fixture()
        qty, value, fee = buy_fill([(100.,1.),(101.,100.)], 1000., markets[0], DEFAULTS)
        self.assertLessEqual(value+fee, 1000)
        self.assertEqual(round(qty,4), qty)
        self.assertGreater(value/qty, 100.)
        with self.assertRaisesRegex(ValueError, 'depth'):
            fill_cost([(100.,1.)], 2., .0005, False)
        with self.assertRaisesRegex(ValueError, 'minimum'):
            buy_fill([(100.,.00001)], 1000., markets[0], DEFAULTS)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            book_levels({'ts': (NOW-61)*1000, 'tick': {'bids': [[1,1]]}},'bids',NOW)

    def test_failed_exit_keeps_position_and_warns_once(self):
        state = step()
        r,t,m,_ = fixture(NOW+1800)
        bad_book = lambda _: {'ts': (NOW+1800)*1000, 'tick': {'bids': [[100.,.00001]]}}
        first = advance(state,r,t,m,bad_book,NOW+1800)
        again = advance(first,r,t,m,bad_book,NOW+1800)
        self.assertEqual(len(again['positions']),1)
        self.assertEqual(len(again['closed_trades']),0)
        self.assertEqual(len([e for e in again['outbox'] if e['kind']=='warning']),1)

    def test_one_position_per_symbol_and_pause_does_not_block_exit(self):
        r,t,m,b = fixture()
        other = copy.deepcopy(r['results']['btcjpy_5m'])
        other['source']['timeframe'] = '15m'
        other['latest']['candle'] = utc(NOW//900*900-900)
        r['results']['btcjpy_15m'] = other
        state = advance(create_state(NOW),r,t,m,b,NOW)
        self.assertEqual(len(state['positions']),1)
        state['paused'] = True
        state = step(state,NOW+5400,probability=.8)
        self.assertFalse(state['positions'])
        self.assertEqual(len(state['closed_trades']),1)

    def test_corrupt_ledger_and_backwards_time_fail_closed(self):
        state = step()
        state['cash_jpy'] += 100
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            step(state)
        with self.assertRaisesRegex(ValueError, 'backwards'):
            step(step(), NOW-1)


class NotificationTests(unittest.TestCase):
    def test_retry_and_durable_receipts_without_duplicate_trade(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'state.json'
            state = step()
            write_json(path,state)
            with patch('bittrade_paper.time.sleep'):
                def failed(_): raise RuntimeError('secret must not enter state')
                self.assertEqual(notify(path,failed),1)
                self.assertNotIn('secret must',path.read_text())
                messages=[]
                self.assertEqual(notify(path,lambda text: messages.append(text)),0)
                self.assertEqual(len(messages),2)
                notify(path,lambda text: messages.append(text))
                self.assertEqual(len(messages),2)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['positions'],state['positions'])
            self.assertTrue(all(e['sent_at'] for e in saved['outbox']))

    def test_telegram_failure_redacts_credentials(self):
        env={'BITTRADE_TELEGRAM_ENABLED':'1','TELEGRAM_TOKEN':'fixture-secret','TELEGRAM_CHAT_ID':'fixture-chat'}
        with patch.dict('os.environ',env), patch('bittrade_paper.urlopen',side_effect=URLError('fixture-secret')):
            with self.assertRaises(RuntimeError) as error:
                send_telegram('test')
            self.assertNotIn('fixture-secret',str(error.exception))


if __name__ == '__main__':
    unittest.main()
