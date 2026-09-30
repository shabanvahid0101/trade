import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bittrade_data import merge_candles, utc, validate_rows, collect
from bittrade_research import features, FEATURES, net_return, simulate, split_indices


def candle(t, price=100., volume=1.):
    return dict(id=t, open=price, high=price+1, low=price-1, close=price,
                amount=volume, vol=volume*price, count=int(volume))


class DataTests(unittest.TestCase):
    def test_closed_only_sorted_overlap_and_base_volume(self):
        rows, _ = merge_candles([], [candle(600), candle(300), candle(0)], 300, 750000)
        self.assertEqual([r['timestamp'] for r in rows], [utc(0), utc(300)])
        self.assertEqual(rows[0]['volume'], 1)
        self.assertEqual(rows[0]['quote_volume'], 100)
        merged, _ = merge_candles(rows, [candle(300, 105), candle(600)], 300, 900000)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[1]['close'], 105)

    def test_gap_report_and_staleness(self):
        rows, q = merge_candles([], [candle(0), candle(600)], 300, 900000)
        self.assertEqual(q['missing_candles'], 1)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            merge_candles([], [candle(0)], 300, 2000000)
        rows[0]['high'] = 1
        with self.assertRaisesRegex(ValueError, 'OHLC'):
            validate_rows(rows, 300)

    def test_collector_idempotent_and_source_guard(self):
        class Client:
            def get(self, endpoint, **params):
                if endpoint.endswith('symbols'):
                    return {'data': [{'symbol': 'btcjpy', 'state': 'online', 'api-trading': 'enabled'}]}
                if endpoint.endswith('tickers'):
                    return {'ts': 900000, 'data': []}
                return {'ch': 'market.btcjpy.kline.5min', 'ts': 900000,
                        'data': [candle(0), candle(300), candle(600)]}
        with tempfile.TemporaryDirectory() as d:
            first = collect(d, ['btcjpy'], ['5m'], Client())
            content = (Path(d)/'btcjpy_5m.csv').read_bytes()
            second = collect(d, ['btcjpy'], ['5m'], Client())
            self.assertFalse(first['errors'])
            self.assertFalse(second['errors'])
            self.assertEqual(content, (Path(d)/'btcjpy_5m.csv').read_bytes())
            (Path(d)/'btcjpy_5m.source.json').unlink()
            self.assertTrue(collect(d, ['btcjpy'], ['5m'], Client())['errors'])


class ResearchTests(unittest.TestCase):
    def test_purged_partition_labels_do_not_cross_boundaries(self):
        parts = split_indices(2000, 6)
        self.assertLess(parts['train'][1]-1+6, parts['validation'][0])
        self.assertLess(parts['validation'][1]-1+6, parts['test'][0])
        self.assertLess(parts['test'][1]-1+6, 2000)

    def test_features_are_causal(self):
        p = np.arange(100.)+100
        df = pd.DataFrame(dict(open=p, high=p+1, low=p-1, close=p, volume=p, quote_volume=p*p))
        original = features(df)
        df.loc[80:, 'close'] *= 3
        altered = features(df)
        pd.testing.assert_frame_equal(original.loc[:79, FEATURES], altered.loc[:79, FEATURES])

    def test_fills_use_next_open_and_both_fees(self):
        frame = pd.DataFrame(dict(open=[500.,100.,105.], close=[500.,103.,110.],
                                  low=[499.,99.,104.], volume=[1.,1.,1.]))
        metrics = simulate(frame, [1.], 2, .6, .001, .002)
        self.assertAlmostEqual(metrics['return_pct'], net_return(100.,110.,.001,.002)*100)
        self.assertEqual(metrics['closed_trades'], 1)
        cash = simulate(frame, [0.], 2, .6, .001, .002)
        self.assertEqual(cash['return_pct'], 0)
        frame.loc[1, 'volume'] = 0
        with self.assertRaisesRegex(ValueError, 'zero-volume'):
            simulate(frame, [1.], 2, .6, .001, .002)


if __name__ == '__main__':
    unittest.main()
